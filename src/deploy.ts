// src/lib/base-deployment.ts
import {
  SageMakerClient,
  CreateModelCommand,
  CreateEndpointConfigCommand,
  CreateEndpointCommand,
  UpdateEndpointCommand,
  DescribeEndpointCommand,
} from "@aws-sdk/client-sagemaker";
import { S3Client } from "@aws-sdk/client-s3";
import {
  DeploymentResult,
  FrameworkDeployConfig,
  Logger,
  MLFramework,
  ModelDeploymentInput,
  ServerlessConfig,
} from "./interfaces";
import path from "path";

export class ImageUriProvider {
  constructor(private readonly region: string) {}

  getDefaultImageUri(
    framework: MLFramework,
    frameworkVersion: string,
    pythonVersion: string,
    useGpu: boolean = false
  ): string {
    const processor = useGpu ? "gpu" : "cpu";
    switch (framework) {
      case MLFramework.PYTORCH:
        return `763104351884.dkr.ecr.${this.region}.amazonaws.com/pytorch-inference:${frameworkVersion}-${processor}-${pythonVersion}`;

      case MLFramework.TENSORFLOW:
        return `763104351884.dkr.ecr.${this.region}.amazonaws.com/tensorflow-inference:${frameworkVersion}-${processor}-${pythonVersion}`;

      case MLFramework.HUGGINGFACE:
        return `763104351884.dkr.ecr.${this.region}.amazonaws.com/huggingface-pytorch-inference:${frameworkVersion}-transformers-${pythonVersion}`;

      case MLFramework.XGBOOST:
        return `683313688378.dkr.ecr.${this.region}.amazonaws.com/sagemaker-xgboost-inference:${frameworkVersion}`;

      case MLFramework.SKLEARN:
        return `683313688378.dkr.ecr.${this.region}.amazonaws.com/sagemaker-scikit-learn-inference:${frameworkVersion}`;

      default:
        return "";
    }
  }
}

export abstract class BaseSageMakerDeployment {
  protected client: SageMakerClient;
  protected readonly framework: MLFramework;
  protected readonly config: FrameworkDeployConfig;
  protected readonly imageUriProvider: ImageUriProvider;
  protected logger: Logger;
  protected readonly service: string;
  protected readonly model: string;

  constructor(
    client: SageMakerClient,
    framework: MLFramework,
    config: FrameworkDeployConfig,
    logger: Logger,
    service: string,
    model: string
  ) {
    this.framework = framework;
    this.config = config;
    this.logger = logger;
    this.service = service;
    this.model = model;
    this.client = client;
    this.imageUriProvider = new ImageUriProvider(config.region);
  }

  protected abstract getFrameworkEnvironment(
    deployInput: ModelDeploymentInput
  ): Record<string, string>;

  protected generateResourceId(): string {
    return `${Date.now()}-${Math.floor(Math.random() * 9000 + 1000)}`;
  }

  protected getEndpointName(): string {
    return `${this.service}-${this.model}-endpoint`;
  }

  protected getModelName(resourceId: string): string {
    return `${this.service}-${this.model}-${resourceId}`;
  }

  protected getConfigName(resourceId: string): string {
    return `${this.service}-${this.model}-${resourceId}`;
  }

  protected getS3ModelPath(deployInput: ModelDeploymentInput): string {
    if (deployInput.modelPath) {
      if (deployInput.modelPath.startsWith("s3://")) {
        return deployInput.modelPath;
      }
      return `s3://${this.config.bucket}/${this.service}/${
        this.model
      }/models/${path.basename(deployInput.modelPath)}`;
    }

    if (!deployInput.trainingJobName) {
      throw new Error("Either modelPath or trainingJobName must be provided");
    }

    return `s3://${this.config.bucket}/${this.service}/${this.model}/${deployInput.trainingJobName}/output/model.tar.gz`;
  }

  protected async createModel(params: {
    modelName: string;
    modelPath: string;
    imageUri: string;
    environment: Record<string, string>;
  }) {
    this.logger.log(`Creating model with name: ${params.modelName}`);
    try {
      await this.client.send(
        new CreateModelCommand({
          ModelName: params.modelName,
          ExecutionRoleArn: this.config.role,
          PrimaryContainer: {
            Image: params.imageUri,
            ModelDataUrl: params.modelPath,
            Environment: params.environment,
          },
          Tags: [
            {
              Key: "Framework",
              Value: this.framework,
            },
          ],
        })
      );
    } catch (error) {
      if ((error as any).message.includes("already exist")) {
        this.logger.log(
          `Model ${params.modelName} already exists, continuing...`
        );
      } else {
        throw error;
      }
    }
  }

  public async deploy(
    deployInput: ModelDeploymentInput,
    serverlessConfig: ServerlessConfig,
    options: {
      waitForDeployment?: boolean;
      timeoutSeconds?: number;
    } = {}
  ): Promise<DeploymentResult> {
    try {
      const resourceId = this.generateResourceId();
      const modelName = `${
        deployInput.trainingJobName ?? this.getModelName(resourceId)
      }-model`;
      const configName = `${
        deployInput.trainingJobName ?? this.getConfigName(resourceId)
      }-config`;
      const endpointName = this.getEndpointName();
      const modelPath = this.getS3ModelPath(deployInput);

      await this.createModel({
        modelName,
        modelPath,
        imageUri:
          deployInput.imageUri ??
          this.imageUriProvider.getDefaultImageUri(
            this.framework,
            deployInput.frameworkVersion,
            deployInput.pythonVersion,
            deployInput.useGpu
          ),
        environment: {
          ...(this.getFrameworkEnvironment(deployInput) || {}),
          ...(this.config.environmentVariables || {}),
          ...(deployInput.environmentVariables || {}),
        },
      });

      this.logger.log(`Creating endpoint config with name: ${configName}`);
      try {
        await this.client.send(
          new CreateEndpointConfigCommand({
            EndpointConfigName: configName,
            ProductionVariants: [
              {
                VariantName: "AllTraffic",
                ModelName: modelName,
                ServerlessConfig: {
                  MemorySizeInMB: serverlessConfig.memorySizeInMb,
                  MaxConcurrency: serverlessConfig.maxConcurrency,
                },
              },
            ],
            Tags: [
              {
                Key: "Service",
                Value: this.service,
              },
              {
                Key: "Model",
                Value: this.model,
              },
            ],
          })
        );
      } catch (error) {
        if ((error as any).message.includes("already exist")) {
          this.logger.log(
            `Endpoint config ${configName} already exists, continuing...`
          );
        } else {
          throw error;
        }
      }

      const exists = await this.endpointExists(endpointName);

      if (exists) {
        this.logger.log(`Updating existing endpoint: ${endpointName}`);
        await this.client.send(
          new UpdateEndpointCommand({
            EndpointName: endpointName,
            EndpointConfigName: configName,
          })
        );

        return {
          modelName,
          endpointName,
          status: "Updated",
        };
      } else {
        this.logger.log(`Creating new endpoint: ${endpointName}`);
        await this.client.send(
          new CreateEndpointCommand({
            EndpointName: endpointName,
            EndpointConfigName: configName,
            Tags: [
              {
                Key: "Service",
                Value: this.service,
              },
              {
                Key: "Model",
                Value: this.model,
              },
            ],
          })
        );

        return {
          modelName,
          endpointName,
          status: "Created",
        };
      }
    } catch (error) {
      this.logger.error("Deployment failed:", error);
      throw error;
    }
  }

  private async endpointExists(endpointName: string): Promise<boolean> {
    try {
      await this.client.send(
        new DescribeEndpointCommand({
          EndpointName: endpointName,
        })
      );
      return true;
    } catch (error) {
      if (
        (error as any).name === "ResourceNotFound" ||
        (error as any).message.includes("Could not find endpoint")
      ) {
        return false;
      }
      throw error;
    }
  }
}

export class PyTorchDeployment extends BaseSageMakerDeployment {
  constructor(
    client: SageMakerClient,
    config: FrameworkDeployConfig,
    logger: Logger,
    service: string,
    model: string
  ) {
    super(client, MLFramework.PYTORCH, config, logger, service, model);
  }

  protected getFrameworkEnvironment(
    deployInput: ModelDeploymentInput
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: deployInput.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      SAGEMAKER_FRAMEWORK_VERSION: deployInput.frameworkVersion,
      SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.pytorch.serving:main",
      SAGEMAKER_PYTHON_VERSION: deployInput.pythonVersion,
    };
  }
}

export class TensorFlowDeployment extends BaseSageMakerDeployment {
  constructor(
    client: SageMakerClient,
    config: FrameworkDeployConfig,
    logger: Logger,
    service: string,
    model: string
  ) {
    super(client, MLFramework.TENSORFLOW, config, logger, service, model);
  }

  protected getFrameworkEnvironment(
    deployInput: ModelDeploymentInput
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: deployInput.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      SAGEMAKER_FRAMEWORK_VERSION: deployInput.frameworkVersion,
      SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.tensorflow.serving:main",
      SAGEMAKER_PYTHON_VERSION: deployInput.pythonVersion,
    };
  }
}

export class HuggingFaceDeployment extends BaseSageMakerDeployment {
  constructor(
    client: SageMakerClient,
    config: FrameworkDeployConfig,
    logger: Logger,
    service: string,
    model: string
  ) {
    super(client, MLFramework.HUGGINGFACE, config, logger, service, model);
  }

  protected getFrameworkEnvironment(
    deployInput: ModelDeploymentInput
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: deployInput.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      SAGEMAKER_FRAMEWORK_VERSION: deployInput.frameworkVersion,
      SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.huggingface.serving:main",
      SAGEMAKER_PYTHON_VERSION: deployInput.pythonVersion,
      SAGEMAKER_HF_TASK: "text-classification", // Can be made configurable
    };
  }
}
