
import { SageMakerClient, CreateTrainingJobCommandInput } from '@aws-sdk/client-sagemaker';
import { S3Client } from '@aws-sdk/client-s3';

export abstract class SageMakerBase {
  protected sagemakerClient: SageMakerClient;
  protected s3Client: S3Client;
  
  constructor(protected readonly config: SageMakerConfig) {
    const sharedConfig = {
      region: config.region,
      credentials: config.credentials
    };
    
    this.sagemakerClient = config.sagemakerClient || new SageMakerClient(sharedConfig);
    this.s3Client = config.s3Client || new S3Client(sharedConfig);
  }

  // Core methods that can be overridden
  protected abstract validateConfig(): void;
  protected abstract createJobParams(): CreateTrainingJobCommandInput;
  
  // Utility methods with default implementations
  protected getJobName(prefix: string): string {
    return `${prefix}-${Date.now()}`;
  }
  
  protected async uploadToS3(data: Buffer | string, key: string): Promise<string> {
    // Default S3 upload implementation
    // ...existing code...
  }
}

// Flexible configuration with optional overrides
export interface SageMakerConfig {
  region: string;
  credentials: AWSCredentials;
  role: string;
  bucket: string;
  sagemakerClient?: SageMakerClient;
  s3Client?: S3Client;
  tags?: Record<string, string>;
}