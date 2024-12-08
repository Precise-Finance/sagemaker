// src/lib/base-deployment.ts
import {
  SageMakerClient,
  CreateModelCommand,
  CreateEndpointConfigCommand,
  CreateEndpointCommand,
  UpdateEndpointCommand,
  DescribeEndpointCommand,
} from "@aws-sdk/client-sagemaker";
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
  protected readonly config: FrameworkDeployConfig;
  protected readonly imageUriProvider: ImageUriProvider;
  protected logger: Logger;

  constructor(config: FrameworkDeployConfig, logger: Logger) {
    this.config = config;
    this.logger = logger;
    this.client = new SageMakerClient({
      region: config.region,
      credentials: config.credentials,
    });
    this.imageUriProvider = new ImageUriProvider(config.region);
  }

  protected abstract getFrameworkEnvironment(
    modelConfig: FrameworkDeployConfig
  ): Record<string, string>;

  protected generateResourceId(): string {
    return `${Date.now()}-${Math.floor(Math.random() * 9000 + 1000)}`;
  }

  protected getEndpointName(modelConfig: ModelDeploymentInput): string {
    return `${modelConfig.service}-${modelConfig.model}-endpoint`;
  }

  protected getModelName(
    modelConfig: ModelDeploymentInput,
    resourceId: string
  ): string {
    if (modelConfig.trainingJobName) {
      return `${modelConfig.service}-${modelConfig.model}-model-${modelConfig.trainingJobName}`;
    }
    return `${modelConfig.service}-${modelConfig.model}-model-${resourceId}`;
  }

  protected getConfigName(
    modelConfig: ModelDeploymentInput,
    resourceId: string
  ): string {
    if (modelConfig.trainingJobName) {
      return `${modelConfig.service}-${modelConfig.model}-config-${modelConfig.trainingJobName}`;
    }
    return `${modelConfig.service}-${modelConfig.model}-config-${resourceId}`;
  }

  protected getS3ModelPath(modelConfig: ModelDeploymentInput): string {
    if (modelConfig.modelPath) {
      if (modelConfig.modelPath.startsWith("s3://")) {
        return modelConfig.modelPath;
      }
      return `s3://${this.config.bucket}/${modelConfig.service}/${
        modelConfig.model
      }/models/${path.basename(modelConfig.modelPath)}`;
    }

    if (!modelConfig.trainingJobName) {
      throw new Error("Either modelPath or trainingJobName must be provided");
    }

    // Use the standard SageMaker training output path pattern
    return `s3://${this.config.bucket}/${modelConfig.service}/${modelConfig.model}/${modelConfig.trainingJobName}/output/model.tar.gz`;
  }

  protected async createModel(params: {
    modelName: string;
    modelPath: string;
    imageUri: string;
    environment: Record<string, string>;
  }) {
    this.logger.log(`Creating model with name: ${params.modelName}`);
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
            Value: this.config.framework,
          },
        ],
      })
    );
  }

  public async deploy(
    modelConfig: ModelDeploymentInput,
    serverlessConfig: ServerlessConfig,
    options: {
      waitForDeployment?: boolean;
      timeoutSeconds?: number;
    } = {}
  ): Promise<DeploymentResult> {
    try {
      const resourceId = this.generateResourceId();
      const modelName = this.getModelName(modelConfig, resourceId);
      const configName = this.getConfigName(modelConfig, resourceId);
      const endpointName = this.getEndpointName(modelConfig);
      const modelPath = this.getS3ModelPath(modelConfig);

      await this.createModel({
        modelName,
        modelPath,
        imageUri: this.imageUriProvider.getDefaultImageUri(
          this.config.framework,
          this.config.frameworkVersion,
          this.config.pythonVersion,
          modelConfig.useGpu
        ),
        environment: {
          ...this.config.environmentVariables,
          SAGEMAKER_PROGRAM: this.config.entryPoint,
        },
      });

      this.logger.log(`Creating endpoint config with name: ${configName}`);
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
              Value: modelConfig.service,
            },
            {
              Key: "Model",
              Value: modelConfig.model,
            },
          ],
        })
      );

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
                Value: modelConfig.service,
              },
              {
                Key: "Model",
                Value: modelConfig.model,
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
      if ((error as any).name === "ResourceNotFound") {
        return false;
      }
      throw error;
    }
  }
}

// src/frameworks/pytorch.ts
export class PyTorchDeployment extends BaseSageMakerDeployment {
  protected getFrameworkEnvironment(
    modelConfig: FrameworkDeployConfig
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: this.config.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      SAGEMAKER_FRAMEWORK_VERSION: this.config.frameworkVersion,
      SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.pytorch.serving:main",
      SAGEMAKER_PYTHON_VERSION: this.config.pythonVersion,
    };
  }
}

// src/frameworks/tensorflow.ts
export class TensorFlowDeployment extends BaseSageMakerDeployment {
  protected getFrameworkEnvironment(
    modelConfig: FrameworkDeployConfig
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: this.config.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      SAGEMAKER_FRAMEWORK_VERSION: this.config.frameworkVersion,
      SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.tensorflow.serving:main",
      SAGEMAKER_PYTHON_VERSION: this.config.pythonVersion,
    };
  }
}

// src/frameworks/huggingface.ts
export class HuggingFaceDeployment extends BaseSageMakerDeployment {
  protected getFrameworkEnvironment(
    modelConfig: FrameworkDeployConfig
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: this.config.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      SAGEMAKER_FRAMEWORK_VERSION: this.config.frameworkVersion,
      SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.huggingface.serving:main",
      SAGEMAKER_PYTHON_VERSION: this.config.pythonVersion,
      SAGEMAKER_HF_TASK: "text-classification", // Can be made configurable
    };
  }
}

// src/frameworks/custom.ts
export class CustomDeployment extends BaseSageMakerDeployment {
  private customEnvironment: Record<string, string>;

  constructor(
    config: FrameworkDeployConfig,
    logger: Logger,
    customEnvironment: Record<string, string>
  ) {
    super(config, logger);
    this.customEnvironment = customEnvironment;
  }

  protected getFrameworkEnvironment(
    modelConfig: FrameworkDeployConfig
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: this.config.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      ...this.customEnvironment,
    };
  }
}
