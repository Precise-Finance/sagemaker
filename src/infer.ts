import {
  SageMakerRuntime,
} from "@aws-sdk/client-sagemaker-runtime";
import {
  Logger,
  InferenceCallOptions,
  InferenceMetrics,
  SageMakerInferenceConfig,
  MetricsCallback,
  InferenceResponse,
} from "./interfaces";

export class SageMakerInference {
  private client: SageMakerRuntime;
  private logger: Logger;
  private config: SageMakerInferenceConfig;

  constructor(
    client: SageMakerRuntime,
    config: SageMakerInferenceConfig,
    logger: Logger
  ) {
    this.client = client;
    this.logger = logger;
    this.config = {
      ...config,
      retry: {
        maxAttempts: 3,
        timeoutMs: 30000,
        backoffMultiplier: 2,
        ...config.retry,
      },
      monitoring: {
        enabled: true,
        ...config.monitoring,
      },
      validation: {
        enabled: true,
        ...config.validation,
      },
      batch: {
        enabled: true,
        maxBatchSize: 10,
        concurrency: 3,
        ...config.batch,
      },
    };
  }

  private getEffectiveConfig(callOptions?: InferenceCallOptions) {
    return {
      retry: {
        ...this.config.retry,
        ...callOptions?.retry,
      },
      validation: {
        ...this.config.validation,
        ...callOptions?.validation,
      },
      monitoring: {
        ...this.config.monitoring,
        ...callOptions?.monitoring,
      },
    };
  }

  private getPayloadSize(payload: any): number {
    return Buffer.from(JSON.stringify(payload)).length;
  }

  private async executeWithRetry<T>(
    operation: () => Promise<T>,
    endpointName: string,
    callOptions?: InferenceCallOptions,
    retryCount = 0
  ): Promise<T> {
    const effectiveConfig = this.getEffectiveConfig(callOptions);
    const startTime = Date.now();

    try {
      this.logger.info(
        `Starting operation on endpoint: ${endpointName}, attempt: ${
          retryCount + 1
        }/${effectiveConfig.retry.maxAttempts}`
      );

      const result = await Promise.race([
        operation(),
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error("Timeout")),
            effectiveConfig.retry.timeoutMs
          )
        ),
      ]);

      const duration = Date.now() - startTime;
      this.logger.info(
        `Operation successful on endpoint: ${endpointName}, duration: ${duration}ms`
      );

      this.recordMetrics(
        {
          timestamp: startTime,
          latencyMs: duration,
          success: true,
          endpointName,
        },
        effectiveConfig.monitoring
      );

      return result as T;
    } catch (error) {
      const duration = Date.now() - startTime;

      if (retryCount < effectiveConfig.retry.maxAttempts) {
        this.logger.warn(
          `Operation failed on endpoint: ${endpointName}, ` +
            `attempt: ${retryCount + 1}/${
              effectiveConfig.retry.maxAttempts
            }, ` +
            `duration: ${duration}ms, error: ${error.message}`
        );

        const backoffTime =
          Math.pow(effectiveConfig.retry.backoffMultiplier!, retryCount) * 1000;
        this.logger.info(`Retrying in ${backoffTime}ms...`);

        await new Promise((resolve) => setTimeout(resolve, backoffTime));
        return this.executeWithRetry(
          operation,
          endpointName,
          callOptions,
          retryCount + 1
        );
      }

      this.logger.error(
        `All retry attempts failed for endpoint: ${endpointName}, ` +
          `final duration: ${duration}ms, error: ${error.message}`
      );

      this.recordMetrics(
        {
          timestamp: startTime,
          latencyMs: duration,
          success: false,
          endpointName,
          error: error.message,
        },
        effectiveConfig.monitoring
      );

      throw error;
    }
  }

  private recordMetrics(
    metrics: InferenceMetrics,
    monitoringConfig: { enabled: boolean; metricsCallback?: MetricsCallback }
  ): void {
    if (monitoringConfig?.enabled) {
      const callback =
        monitoringConfig.metricsCallback ||
        this.config.monitoring?.metricsCallback;
      if (callback) {
        callback(metrics);
      }
    }
  }

  private validateResponse(
    response: any,
    validationConfig: {
      enabled: boolean;
      customValidator?: (response: any) => boolean;
    }
  ): void {
    if (!validationConfig?.enabled) return;

    if (validationConfig.customValidator) {
      if (!validationConfig.customValidator(response)) {
        throw new Error("Custom validation failed for response");
      }
      return;
    }

    if (!response || (Array.isArray(response) && response.length === 0)) {
      throw new Error("Invalid response received from endpoint");
    }
  }

  public async invokeEndpoint(
    endpointName: string,
    payload: any,
    options: InferenceCallOptions = {}
  ): Promise<InferenceResponse> {
    const effectiveConfig = this.getEffectiveConfig(options);
    const {
      contentType = this.config.defaultContentType || "application/json",
      accept = this.config.defaultAccept || "application/json",
      customAttributes = this.config.defaultCustomAttributes || {},
      transformInput = (p: any) => p,
      transformOutput = (r: any) => r,
      // Add support for new targeting options
      targetModel,
      targetVariant,
      targetContainerHostname,
      inferenceId = `inf-${Date.now()}-${Math.random()
        .toString(36)
        .substr(2, 9)}`,
      enableExplanations,
      inferenceComponentName,
      sessionId,
    } = options;

    try {
      const transformedPayload = transformInput(payload);
      const payloadSize = this.getPayloadSize(transformedPayload);
      const jsonPayload = JSON.stringify(transformedPayload);
      const customAttributesString = Object.entries(customAttributes)
        .map(([key, value]) => `${key}=${value}`)
        .join(",");

      this.logger.info(
        `Invoking endpoint: ${endpointName}, ` +
          `payload size: ${payloadSize} bytes, ` +
          (targetModel ? `target model: ${targetModel}, ` : "") +
          (targetVariant ? `target variant: ${targetVariant}, ` : "") +
          (inferenceId ? `inference ID: ${inferenceId}, ` : "") +
          (sessionId ? `session ID: ${sessionId}, ` : "") +
          `content type: ${contentType}`
      );

      const response = await this.executeWithRetry(
        async () => {
          const result = await this.client.invokeEndpoint({
            EndpointName: endpointName,
            ContentType: contentType,
            Accept: accept,
            CustomAttributes: customAttributesString,
            Body: Buffer.from(jsonPayload),
            // Add new targeting options
            TargetModel: targetModel,
            TargetVariant: targetVariant,
            TargetContainerHostname: targetContainerHostname,
            InferenceId: inferenceId,
            EnableExplanations: enableExplanations,
            InferenceComponentName: inferenceComponentName,
            SessionId: sessionId,
          });

          const responseBody = await result.Body.transformToString();
          const parsedResponse = JSON.parse(responseBody);

          return {
            data: transformOutput(parsedResponse),
            inferenceId,
            targetModel,
            targetVariant,
            explanation: enableExplanations
              ? parsedResponse.explanation
              : undefined,
          };
        },
        endpointName,
        options
      );

      this.logger.debug(
        `Raw response from endpoint: ${endpointName}`,
        response
      );

      if (effectiveConfig.validation?.enabled) {
        this.logger.debug(`Validating response from endpoint: ${endpointName}`);
        this.validateResponse(response, effectiveConfig.validation);
      }

      const transformedResponse = transformOutput(response);
      this.logger.info(
        `Successfully processed response from endpoint: ${endpointName}, ` +
          `response size: ${this.getPayloadSize(transformedResponse)} bytes`
      );

      return transformedResponse;
    } catch (error) {
      this.logger.error(`Inference failed for endpoint ${endpointName}:`, {
        error: error.message,
        stack: error.stack,
        inferenceId,
        targetModel,
        targetVariant,
        sessionId,
        payload: this.getPayloadSize(payload),
        configuration: {
          contentType,
          accept,
          customAttributes,
          validation: effectiveConfig.validation,
          retry: effectiveConfig.retry,
        },
      });
      throw error;
    }
  }

  public async batchInvokeEndpoint(
    endpointName: string,
    payloads: any[],
    options: InferenceCallOptions = {}
  ): Promise<any[]> {
    if (!Array.isArray(payloads)) {
      throw new Error("Payloads must be an array");
    }

    const totalPayloads = payloads.length;
    if (totalPayloads === 0) {
      this.logger.warn(
        `Empty payload array provided for endpoint: ${endpointName}`
      );
      return [];
    }

    const batchConfig = this.config.batch;
    if (!batchConfig?.enabled) {
      throw new Error("Batch processing is not enabled");
    }

    const batchSize = batchConfig.maxBatchSize;
    const concurrency = batchConfig.concurrency || 1;
    const results: any[] = [];
    const startTime = Date.now();

    this.logger.info(
      `Starting batch operation on endpoint: ${endpointName}, ` +
        `total items: ${totalPayloads}, ` +
        `batch size: ${batchSize}, ` +
        `concurrency: ${concurrency}`
    );

    try {
      for (let i = 0; i < payloads.length; i += batchSize * concurrency) {
        const batchStartTime = Date.now();
        const batchNumber = Math.floor(i / (batchSize * concurrency)) + 1;
        const totalBatches = Math.ceil(
          totalPayloads / (batchSize * concurrency)
        );

        const currentBatchSize = Math.min(
          batchSize * concurrency,
          payloads.length - i
        );
        this.logger.info(
          `Processing batch ${batchNumber}/${totalBatches}, ` +
            `items: ${i + 1}-${i + currentBatchSize} of ${totalPayloads}`
        );

        const batch = payloads.slice(i, i + batchSize * concurrency);
        const batchPromises = batch.map((payload) =>
          this.invokeEndpoint(endpointName, payload, options)
        );

        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults);

        const batchDuration = Date.now() - batchStartTime;
        this.logger.info(
          `Completed batch ${batchNumber}/${totalBatches}, ` +
            `duration: ${batchDuration}ms, ` +
            `average time per item: ${batchDuration / currentBatchSize}ms`
        );
      }

      const totalDuration = Date.now() - startTime;
      this.logger.info(
        `Batch operation completed for endpoint: ${endpointName}, ` +
          `total duration: ${totalDuration}ms, ` +
          `average time per item: ${totalDuration / totalPayloads}ms`
      );

      return results;
    } catch (error) {
      const duration = Date.now() - startTime;
      this.logger.error(
        `Batch operation failed for endpoint: ${endpointName}, ` +
          `duration: ${duration}ms, ` +
          `processed: ${results.length}/${totalPayloads} items`,
        error
      );
      throw error;
    }
  }
}
