import {
  SageMakerClient,
  CreateModelCommand,
  CreateEndpointConfigCommand,
  CreateEndpointCommand,
  UpdateEndpointCommand,
  DescribeEndpointCommand,
} from '@aws-sdk/client-sagemaker';
import { AWSCredentials } from './sagemaker-training';

export interface ServerlessConfig {
  memorySizeInMb: number;
  maxConcurrency: number;
}

export interface ModelConfig {
  modelData: string;
  role: string;
  entryPoint: string;
  frameworkVersion: string;
  pyVersion: string;
  imageUri: string;
}

export interface DeploymentConfig {
  region: string;
  endpointName: string;
  credentials: AWSCredentials;
}

export interface DeploymentResult {
  modelName: string;
  endpointName: string;
  status: 'Created' | 'Updated';
}

export class SageMakerDeployment {
  private client: SageMakerClient;
  private readonly config: DeploymentConfig;

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

  private generateResourceId(): string {
    return `${Date.now()}-${Math.floor(Math.random() * 9000 + 1000)}`;
  }

  public async deploy(
    modelConfig: ModelConfig,
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

      // Create model
      await this.client.send(new CreateModelCommand({
        ModelName: modelName,
        ExecutionRoleArn: modelConfig.role,
        PrimaryContainer: {
          Image: modelConfig.imageUri,
          ModelDataUrl: modelConfig.modelData,
          Environment: {
            SAGEMAKER_PROGRAM: modelConfig.entryPoint,
            SAGEMAKER_SUBMIT_DIRECTORY: '/opt/ml/model/code',
            SAGEMAKER_FRAMEWORK_VERSION: modelConfig.frameworkVersion,
            SAGEMAKER_FRAMEWORK_MODULE: 'sagemaker.pytorch.serving:main',
            SAGEMAKER_PYTHON_VERSION: modelConfig.pyVersion,
          },
        },
      }));

      // Create endpoint config
      await this.client.send(new CreateEndpointConfigCommand({
        EndpointConfigName: configName,
        ProductionVariants: [{
          VariantName: 'AllTraffic',
          ModelName: modelName,
          ServerlessConfig: {
            MemorySizeInMB: serverlessConfig.memorySizeInMb,
            MaxConcurrency: serverlessConfig.maxConcurrency,
          },
        }],
      }));

      // Check if endpoint exists
      const exists = await this.endpointExists();

      if (exists) {
        await this.client.send(new UpdateEndpointCommand({
          EndpointName: this.config.endpointName,
          EndpointConfigName: configName,
        }));
        
        return {
          modelName,
          endpointName: this.config.endpointName,
          status: 'Updated',
        };
      } else {
        await this.client.send(new CreateEndpointCommand({
          EndpointName: this.config.endpointName,
          EndpointConfigName: configName,
        }));

        return {
          modelName,
          endpointName: this.config.endpointName,
          status: 'Created',
        };
      }
    } catch (error) {
      console.error('Deployment failed:', error);
      throw error;
    }
  }

  private async endpointExists(): Promise<boolean> {
    try {
      await this.client.send(new DescribeEndpointCommand({
        EndpointName: this.config.endpointName,
      }));
      return true;
    } catch (error) {
      if ((error as any).name === 'ResourceNotFound') {
        return false;
      }
      throw error;
    }
  }
}
