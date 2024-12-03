import {
  SageMakerClient,
  CreateTrainingJobCommand,
  DescribeTrainingJobCommand,
  CreateTrainingJobCommandInput,
  TrainingJobStatus,
  TrainingInstanceType,
  Channel,
} from '@aws-sdk/client-sagemaker';
import { S3Client } from '@aws-sdk/client-s3';
import { Upload } from '@aws-sdk/lib-storage';
import * as fs from 'fs';
import archiver from 'archiver';
import * as path from 'path';

// Core framework enums and interfaces
export enum MLFramework {
  PYTORCH = 'pytorch',
  TENSORFLOW = 'tensorflow',
  SKLEARN = 'sklearn',
  XGBOOST = 'xgboost',
  HUGGINGFACE = 'huggingface',
}

export interface FrameworkConfig {
  framework: MLFramework;
  frameworkVersion: string;
  pythonVersion: string;
  imageUri: string;
}

export interface AWSCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
}

// Training configuration interfaces
export interface TrainingConfig {
  role: string;
  bucket: string;
  service: string;
  model: string;
  region: string;
  credentials: AWSCredentials;
}

export interface ResourceConfig {
  instanceCount: number;
  instanceType: TrainingInstanceType;
  volumeSizeGB: number;
  maxRuntimeSeconds?: number;
  maxPendingSeconds?: number;
}

export interface MetricDefinition {
  Name: string;
  Regex: string;
}

// Framework-specific configurations
export interface FrameworkSpecificConfig {
  contentType: string;
  environmentVariables: Record<string, string>;
  hyperparameterMapping: Record<string, string>;
}

// Training metadata and results
export interface TrainingMetadata {
  trainingJobName: string;
  modelOutputPath: string;
  hyperParameters: Record<string, any>;
  status: TrainingJobStatus;
  framework: MLFramework;
}

export enum DataFormat {
  JSON = 'application/json',
  CSV = 'text/csv',
  PARQUET = 'application/x-parquet',
  LIBSVM = 'application/x-libsvm',
  RECORDIO = 'application/x-recordio-protobuf',
  PROTOBUF = 'application/x-protobuf',
  NUMPY = 'application/x-npy',
  // Add other formats as needed
}

export interface InputDataConfig {
  // The actual data or reference to it
  data: Buffer | string | NodeJS.ReadableStream; // Can be data, file path, or S3 URI
  format: DataFormat;

  // Channel configuration
  channelName?: string; // Default to 'train' for first channel if not specified
  distributionType?: 'FullyReplicated' | 'ShardedByS3Key';
  s3DataType?: 'S3Prefix' | 'ManifestFile';

  // Optional metadata about the data structure
  // This can be used by the training script to understand the data layout
  schema?: Record<string, string>; // Flexible schema definition
}

/**
 * Handles framework-specific configurations and requirements
 */
class FrameworkHandler {
  private static readonly FRAMEWORK_CONFIGS: Record<
    MLFramework,
    FrameworkSpecificConfig
  > = {
    [MLFramework.PYTORCH]: {
      contentType: 'application/x-torch',
      environmentVariables: {
        SAGEMAKER_FRAMEWORK_MODULE: 'sagemaker.pytorch.serving:main',
      },
      hyperparameterMapping: {
        learningRate: 'lr',
        batchSize: 'batch_size',
        epochs: 'epochs',
      },
    },
    [MLFramework.TENSORFLOW]: {
      contentType: 'application/x-tensorflow',
      environmentVariables: {
        SAGEMAKER_FRAMEWORK_MODULE: 'sagemaker.tensorflow.serving:main',
      },
      hyperparameterMapping: {
        learningRate: 'learning_rate',
        batchSize: 'batch_size',
        epochs: 'epochs',
      },
    },
    [MLFramework.SKLEARN]: {
      contentType: 'text/csv',
      environmentVariables: {
        SAGEMAKER_FRAMEWORK_MODULE: 'sagemaker.sklearn.serving:main',
      },
      hyperparameterMapping: {
        maxDepth: 'max_depth',
        nEstimators: 'n_estimators',
      },
    },
    [MLFramework.XGBOOST]: {
      contentType: 'text/libsvm',
      environmentVariables: {
        SAGEMAKER_FRAMEWORK_MODULE: 'sagemaker.xgboost.serving:main',
      },
      hyperparameterMapping: {
        learningRate: 'eta',
        maxDepth: 'max_depth',
        nEstimators: 'n_estimators',
      },
    },
    [MLFramework.HUGGINGFACE]: {
      contentType: 'application/json',
      environmentVariables: {
        SAGEMAKER_FRAMEWORK_MODULE: 'sagemaker.huggingface.serving:main',
      },
      hyperparameterMapping: {
        learningRate: 'learning_rate',
        batchSize: 'per_device_train_batch_size',
        epochs: 'num_train_epochs',
      },
    },
  };

  static getFrameworkConfig(framework: MLFramework): FrameworkSpecificConfig {
    console.log(`Getting framework config for: ${framework}`);
    return this.FRAMEWORK_CONFIGS[framework];
  }
}

/**
 * Main SageMaker training class that supports multiple ML frameworks
 */
export class SageMakerTraining {
  private sagemakerClient: SageMakerClient;
  private s3Client: S3Client;
  private readonly config: TrainingConfig;
  private readonly frameworkConfig: FrameworkConfig;
  private readonly sourceDir: string;
  private readonly frameworkSpecificConfig: FrameworkSpecificConfig;

  constructor(
    config: TrainingConfig,
    frameworkConfig: FrameworkConfig,
    sourceDir: string,
  ) {
    console.log('Initializing SageMakerTraining with config:', config);
    console.log('Framework config:', frameworkConfig);
    console.log('Source directory:', sourceDir);
    this.config = config;
    this.frameworkConfig = frameworkConfig;
    this.sourceDir = sourceDir;
    this.frameworkSpecificConfig = FrameworkHandler.getFrameworkConfig(
      frameworkConfig.framework,
    );

    const sharedConfig = {
      region: config.region,
      credentials: {
        accessKeyId: config.credentials.accessKeyId,
        secretAccessKey: config.credentials.secretAccessKey,
        sessionToken: config.credentials.sessionToken,
      },
    };

    this.sagemakerClient = new SageMakerClient(sharedConfig);
    this.s3Client = new S3Client(sharedConfig);
  }

  private async prepareSourceDir(trainingJobName: string): Promise<string> {
    console.log(
      `Preparing source directory for training job: ${trainingJobName}`,
    );
    const zipPath = path.join(process.cwd(), 'tmp', 'source.tar.gz');
    fs.mkdirSync(path.dirname(zipPath), { recursive: true });

    const output = fs.createWriteStream(zipPath);
    const archive = archiver('tar', { gzip: true });

    return new Promise((resolve, reject) => {
      output.on('close', async () => {
        console.log('Source directory archived successfully.');
        try {
          const key = `${this.config.service}/${this.config.model}/${trainingJobName}/code/source.tar.gz`;
          const upload = new Upload({
            client: this.s3Client,
            params: {
              Bucket: this.config.bucket,
              Key: key,
              Body: fs.createReadStream(zipPath),
            },
          });

          await upload.done();
          console.log(
            `Source code uploaded to S3 at: s3://${this.config.bucket}/${key}`,
          );
          return resolve(`s3://${this.config.bucket}/${key}`);
        } catch (error) {
          console.error('Error uploading source code to S3:', error);
          reject(error);
        }
      });

      archive.on('error', reject);
      archive.pipe(output);
      archive.directory(this.sourceDir, false);
      archive.finalize();
    });
  }

  private async monitorTrainingJob(
    trainingJobName: string,
  ): Promise<TrainingJobStatus> {
    console.log(`Monitoring training job: ${trainingJobName}`);
    while (true) {
      const response = await this.sagemakerClient.send(
        new DescribeTrainingJobCommand({
          TrainingJobName: trainingJobName,
        }),
      );

      const status = response.TrainingJobStatus as TrainingJobStatus;
      console.log(`Training job status: ${status}`);
      if (['Completed', 'Failed', 'Stopped'].includes(status)) {
        return status;
      }

      console.log(`Training job is still in status: ${status}. Waiting...`);
      await new Promise((resolve) => setTimeout(resolve, 60000));
    }
  }

  protected async prepareInputData(
    inputData: InputDataConfig,
    trainingJobName: string,
  ): Promise<string> {
    console.log(`Preparing input data for training job: ${trainingJobName}`);
    console.log('Input data config:', inputData);
    // Handle S3 URI
    if (
      typeof inputData.data === 'string' &&
      inputData.data.startsWith('s3://')
    ) {
      return inputData.data;
    }

    // Generate a file name based on channel and format
    const extension = this.getFileExtension(inputData.format);
    const fileName = `${inputData.channelName || 'data'}.${extension}`;

    // Create the S3 key for this data
    const key = `${this.config.service}/${this.config.model}/${trainingJobName}/data/${fileName}`;

    // Upload the data
    const upload = new Upload({
      client: this.s3Client,
      params: {
        Bucket: this.config.bucket,
        Key: key,
        Body:
          inputData.data instanceof Buffer
            ? inputData.data
            : fs.createReadStream(inputData.data as string),
        ContentType: inputData.format,
      },
    });

    await upload.done();
    console.log(
      `Input data uploaded to S3 at: s3://${this.config.bucket}/${key}`,
    );
    return `s3://${this.config.bucket}/${key}`;
  }

  private getFileExtension(format: DataFormat): string {
    console.log(`Getting file extension for data format: ${format}`);
    const extensionMap: Record<DataFormat, string> = {
      [DataFormat.JSON]: 'json',
      [DataFormat.CSV]: 'csv',
      [DataFormat.PARQUET]: 'parquet',
      [DataFormat.LIBSVM]: 'libsvm',
      [DataFormat.RECORDIO]: 'recordio',
      [DataFormat.PROTOBUF]: 'pb',
      [DataFormat.NUMPY]: 'npy',
    };
    return extensionMap[format] || 'dat';
  }

  private createTrainingJobParams(
    sourceCodeLocation: string,
    trainingJobName: string,
    resourceConfig: ResourceConfig,
    hyperParameters: Record<string, any>,
    metricDefinitions: MetricDefinition[],
    inputChannels: Channel[],
  ): CreateTrainingJobCommandInput {
    console.log('Creating training job parameters...');
    console.log('Source code location:', sourceCodeLocation);
    console.log('Training job name:', trainingJobName);
    console.log('Resource config:', resourceConfig);
    console.log('Hyperparameters:', hyperParameters);
    console.log('Metric definitions:', metricDefinitions);
    console.log('Input channels:', inputChannels);
    const mappedHyperParameters = Object.entries(hyperParameters).reduce(
      (acc, [key, value]) => {
        const mappedKey =
          this.frameworkSpecificConfig.hyperparameterMapping[key] || key;
        acc[mappedKey] = value.toString();
        return acc;
      },
      {} as Record<string, string>,
    );

    return {
      TrainingJobName: trainingJobName,
      StoppingCondition: {
        MaxRuntimeInSeconds: resourceConfig.maxRuntimeSeconds || 86400,
        MaxPendingTimeInSeconds: resourceConfig.maxPendingSeconds || 3600,
      },
      AlgorithmSpecification: {
        TrainingImage: this.frameworkConfig.imageUri,
        TrainingInputMode: 'File',
        EnableSageMakerMetricsTimeSeries: true,
        MetricDefinitions: metricDefinitions,
      },
      RoleArn: this.config.role,
      InputDataConfig: inputChannels,
      OutputDataConfig: {
        S3OutputPath: `s3://${this.config.bucket}/${this.config.service}/${this.config.model}`,
      },
      ResourceConfig: {
        InstanceCount: resourceConfig.instanceCount,
        InstanceType: resourceConfig.instanceType,
        VolumeSizeInGB: resourceConfig.volumeSizeGB,
      },
      HyperParameters: {
        sagemaker_program: 'train.py',
        sagemaker_submit_directory: sourceCodeLocation,
        ...mappedHyperParameters,
        ...(hyperParameters.schema && {
          data_schema: JSON.stringify(hyperParameters.schema),
        }),
      },
      Tags: [
        {
          Key: 'Framework',
          Value: this.frameworkConfig.framework,
        },
      ],
    };
  }

  public async train(
    resourceConfig: ResourceConfig,
    hyperParameters: Record<string, any>,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions: MetricDefinition[] = [],
  ): Promise<TrainingMetadata> {
    console.log('Starting training job...');
    console.log('Resource config:', resourceConfig);
    console.log('Hyperparameters:', hyperParameters);
    console.log('Input data:', inputData);
    console.log('Metric definitions:', metricDefinitions);
    try {
      const trainingJobName = `${this.config.service}-${
        this.frameworkConfig.framework
      }-${Date.now()}`;
      const sourceCodeLocation = await this.prepareSourceDir(trainingJobName);

      // Handle multiple input channels
      const inputDataArray = Array.isArray(inputData) ? inputData : [inputData];

      // Process all input channels
      const inputChannels = await Promise.all(
        inputDataArray.map(async (input, index) => {
          const channelName =
            input.channelName || (index === 0 ? 'train' : `channel_${index}`);
          const s3Uri = await this.prepareInputData(
            {
              ...input,
              channelName,
            },
            trainingJobName,
          );

          return {
            ChannelName: channelName,
            DataSource: {
              S3DataSource: {
                S3DataType: input.s3DataType || 'S3Prefix',
                S3Uri: s3Uri,
                S3DataDistributionType:
                  input.distributionType || 'FullyReplicated',
              },
            },
            ContentType: input.format,
          };
        }),
      );

      // Create and start the training job
      const trainingJobParams = this.createTrainingJobParams(
        sourceCodeLocation,
        trainingJobName,
        resourceConfig,
        hyperParameters,
        metricDefinitions,
        inputChannels,
      );

      await this.sagemakerClient.send(
        new CreateTrainingJobCommand(trainingJobParams),
      );
      console.log(`Training job started with name: ${trainingJobName}`);

      const status = await this.monitorTrainingJob(trainingJobName);

      const modelOutputPath = `${trainingJobParams.OutputDataConfig.S3OutputPath}/${trainingJobName}/output/model.tar.gz`;

      console.log('Training job completed with status:', status);
      return {
        trainingJobName,
        modelOutputPath,
        hyperParameters,
        status,
        framework: this.frameworkConfig.framework,
      };
    } catch (error) {
      console.error('Error during training job:', error);
      throw error;
    }
  }
}
