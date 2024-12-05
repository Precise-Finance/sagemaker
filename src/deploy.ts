// src/lib/base-deployment.ts
import {
  SageMakerClient,
  CreateModelCommand,
  CreateEndpointConfigCommand,
  CreateEndpointCommand,
  UpdateEndpointCommand,
  DescribeEndpointCommand,
} from "@aws-sdk/client-sagemaker";
import { DeploymentConfig, DeploymentResult, FrameworkModelConfig, Logger, MLFramework, ServerlessConfig } from "./interfaces";

export class ImageUriProvider {
  private readonly region: string;

  constructor(region: string) {
    this.region = region;
  }

  getDefaultImageUri(
    modelConfig: FrameworkModelConfig,
  ): string {
    const processor = modelConfig.useGpu ? "gpu" : "cpu";
    switch (modelConfig.framework) {
      case MLFramework.PYTORCH:
        return `763104351884.dkr.ecr.${this.region}.amazonaws.com/pytorch-inference:${modelConfig.frameworkVersion}-${processor}-${modelConfig.pythonVersion}`;

      case MLFramework.TENSORFLOW:
        return `763104351884.dkr.ecr.${this.region}.amazonaws.com/tensorflow-inference:${modelConfig.frameworkVersion}-${processor}-${modelConfig.pythonVersion}`;

      case MLFramework.HUGGINGFACE:
        return `763104351884.dkr.ecr.${this.region}.amazonaws.com/huggingface-pytorch-inference:${modelConfig.frameworkVersion}-transformers-${modelConfig.pythonVersion}`;

      case MLFramework.XGBOOST:
        return `683313688378.dkr.ecr.${this.region}.amazonaws.com/sagemaker-xgboost-inference:${modelConfig.frameworkVersion}`;

      case MLFramework.SKLEARN:
        return `683313688378.dkr.ecr.${this.region}.amazonaws.com/sagemaker-scikit-learn-inference:${modelConfig.frameworkVersion}`;

      default:
        return "";
    }
  }
}

export abstract class BaseSageMakerDeployment {
  protected client: SageMakerClient;
  protected readonly config: DeploymentConfig;
  protected logger: Logger;
  private readonly imageUriProvider: ImageUriProvider;

  constructor(config: DeploymentConfig, logger: Logger) {
    this.config = config;
    this.logger = logger;
    this.client = new SageMakerClient({
      region: config.region,
      credentials: {
        accessKeyId: config.credentials.accessKeyId,
        secretAccessKey: config.credentials.secretAccessKey,
        sessionToken: config.credentials.sessionToken,
      },
    });
    this.imageUriProvider = new ImageUriProvider(config.region);
  }

  protected abstract getFrameworkEnvironment(
    modelConfig: FrameworkModelConfig
  ): Record<string, string>;

  protected generateResourceId(): string {
    return `${Date.now()}-${Math.floor(Math.random() * 9000 + 1000)}`;
  }

  public async deploy(
    modelConfig: FrameworkModelConfig,
    serverlessConfig: ServerlessConfig,
    options: {
      waitForDeployment?: boolean;
      timeoutSeconds?: number;
    } = {}
  ): Promise<DeploymentResult> {
    try {
      const resourceId = this.generateResourceId();
      const modelName = `model-${resourceId}`;
      const configName = `config-${resourceId}`;

      this.logger.log(`Creating model with name: ${modelName}`);
      // Create model with framework-specific environment
      await this.client.send(
        new CreateModelCommand({
          ModelName: modelName,
          ExecutionRoleArn: modelConfig.role,
          PrimaryContainer: {
            Image: modelConfig.imageUri ?? this.imageUriProvider.getDefaultImageUri(modelConfig),
            ModelDataUrl: modelConfig.modelData,
            Environment: this.getFrameworkEnvironment(modelConfig),
          },
        })
      );

      this.logger.log(`Creating endpoint config with name: ${configName}`);
      // Create endpoint config
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
        })
      );

      const exists = await this.endpointExists();

      if (exists) {
        this.logger.log(
          `Updating existing endpoint: ${this.config.endpointName}`
        );
        await this.client.send(
          new UpdateEndpointCommand({
            EndpointName: this.config.endpointName,
            EndpointConfigName: configName,
          })
        );

        return {
          modelName,
          endpointName: this.config.endpointName,
          status: "Updated",
        };
      } else {
        this.logger.log(`Creating new endpoint: ${this.config.endpointName}`);
        await this.client.send(
          new CreateEndpointCommand({
            EndpointName: this.config.endpointName,
            EndpointConfigName: configName,
          })
        );

        return {
          modelName,
          endpointName: this.config.endpointName,
          status: "Created",
        };
      }
    } catch (error) {
      this.logger.error("Deployment failed:", error);
      throw error;
    }
  }

  private async endpointExists(): Promise<boolean> {
    try {
      await this.client.send(
        new DescribeEndpointCommand({
          EndpointName: this.config.endpointName,
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
    modelConfig: FrameworkModelConfig
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: modelConfig.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      SAGEMAKER_FRAMEWORK_VERSION: modelConfig.frameworkVersion,
      SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.pytorch.serving:main",
      SAGEMAKER_PYTHON_VERSION: modelConfig.pythonVersion,
    };
  }
}

// src/frameworks/tensorflow.ts
export class TensorFlowDeployment extends BaseSageMakerDeployment {
  protected getFrameworkEnvironment(
    modelConfig: FrameworkModelConfig
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: modelConfig.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      SAGEMAKER_FRAMEWORK_VERSION: modelConfig.frameworkVersion,
      SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.tensorflow.serving:main",
      SAGEMAKER_PYTHON_VERSION: modelConfig.pythonVersion,
    };
  }
}

// src/frameworks/huggingface.ts
export class HuggingFaceDeployment extends BaseSageMakerDeployment {
  protected getFrameworkEnvironment(
    modelConfig: FrameworkModelConfig
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: modelConfig.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      SAGEMAKER_FRAMEWORK_VERSION: modelConfig.frameworkVersion,
      SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.huggingface.serving:main",
      SAGEMAKER_PYTHON_VERSION: modelConfig.pythonVersion,
      SAGEMAKER_HF_TASK: "text-classification", // Can be made configurable
    };
  }
}

// src/frameworks/custom.ts
export class CustomDeployment extends BaseSageMakerDeployment {
  private customEnvironment: Record<string, string>;

  constructor(
    config: DeploymentConfig,
    logger: Logger,
    customEnvironment: Record<string, string>
  ) {
    super(config, logger);
    this.customEnvironment = customEnvironment;
  }

  protected getFrameworkEnvironment(
    modelConfig: FrameworkModelConfig
  ): Record<string, string> {
    return {
      SAGEMAKER_PROGRAM: modelConfig.entryPoint,
      SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
      ...this.customEnvironment,
    };
  }
}
