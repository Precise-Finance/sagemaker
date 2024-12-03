// src/lib/base-deployment.ts
import {
  SageMakerClient,
  CreateModelCommand,
  CreateEndpointConfigCommand,
  CreateEndpointCommand,
  UpdateEndpointCommand,
  DescribeEndpointCommand,
} from "@aws-sdk/client-sagemaker";
import { AWSCredentials, MLFramework } from "./sagemaker-training";

export interface ServerlessConfig {
  memorySizeInMb: number;
  maxConcurrency: number;
}

export interface BaseModelConfig {
  modelData: string;
  role: string;
  entryPoint: string;
  imageUri: string;
}

export interface FrameworkModelConfig extends BaseModelConfig {
  frameworkVersion: string;
  pythonVersion: string;
  framework: MLFramework;
}

export interface DeploymentConfig {
  region: string;
  endpointName: string;
  credentials: AWSCredentials;
}

export interface DeploymentResult {
  modelName: string;
  endpointName: string;
  status: "Created" | "Updated";
}

export abstract class BaseSageMakerDeployment {
  protected client: SageMakerClient;
  protected readonly config: DeploymentConfig;

  constructor(config: DeploymentConfig) {
    this.config = config;
    this.client = new SageMakerClient({
      region: config.region,
      credentials: {
        accessKeyId: config.credentials.accessKeyId,
        secretAccessKey: config.credentials.secretAccessKey,
        sessionToken: config.credentials.sessionToken,
      },
    });
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

      // Create model with framework-specific environment
      await this.client.send(
        new CreateModelCommand({
          ModelName: modelName,
          ExecutionRoleArn: modelConfig.role,
          PrimaryContainer: {
            Image: modelConfig.imageUri,
            ModelDataUrl: modelConfig.modelData,
            Environment: this.getFrameworkEnvironment(modelConfig),
          },
        })
      );

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
      console.error("Deployment failed:", error);
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
    customEnvironment: Record<string, string>
  ) {
    super(config);
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
