import {
  SageMaker,
  CreateTrainingJobCommand,
  DescribeTrainingJobCommand,
  CreateTrainingJobCommandInput,
  TrainingJobStatus,
  TrainingInstanceType,
  Channel,
} from "@aws-sdk/client-sagemaker";
import { S3 } from "@aws-sdk/client-s3";
import { Upload } from "@aws-sdk/lib-storage";
import * as fs from "fs";
import archiver from "archiver";
import * as path from "path";
import {
  MLFramework,
  FrameworkSpecificConfig,
  TrainingConfig,
  FrameworkConfig,
  Logger,
  InputDataConfig,
  DataFormat,
  TrainingMetadata,
  ResourceConfig,
  MetricDefinition,
} from "./interfaces";

/**
 * Handles framework-specific configurations and requirements
 */
class FrameworkHandler {
  private static readonly FRAMEWORK_CONFIGS: Record<
    MLFramework,
    FrameworkSpecificConfig
  > = {
    [MLFramework.PYTORCH]: {
      contentType: "application/x-torch",
      environmentVariables: {
        SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.pytorch.serving:main",
      },
      hyperparameterMapping: {
        learningRate: "lr",
        batchSize: "batch_size",
        epochs: "epochs",
      },
    },
    [MLFramework.TENSORFLOW]: {
      contentType: "application/x-tensorflow",
      environmentVariables: {
        SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.tensorflow.serving:main",
      },
      hyperparameterMapping: {
        learningRate: "learning_rate",
        batchSize: "batch_size",
        epochs: "epochs",
      },
    },
    [MLFramework.SKLEARN]: {
      contentType: "text/csv",
      environmentVariables: {
        SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.sklearn.serving:main",
      },
      hyperparameterMapping: {
        maxDepth: "max_depth",
        nEstimators: "n_estimators",
      },
    },
    [MLFramework.XGBOOST]: {
      contentType: "text/libsvm",
      environmentVariables: {
        SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.xgboost.serving:main",
      },
      hyperparameterMapping: {
        learningRate: "eta",
        maxDepth: "max_depth",
        nEstimators: "n_estimators",
      },
    },
    [MLFramework.HUGGINGFACE]: {
      contentType: "application/json",
      environmentVariables: {
        SAGEMAKER_FRAMEWORK_MODULE: "sagemaker.huggingface.serving:main",
      },
      hyperparameterMapping: {
        learningRate: "learning_rate",
        batchSize: "per_device_train_batch_size",
        epochs: "num_train_epochs",
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
  private sagemakerClient: SageMaker;
  private s3Client: S3;
  private readonly config: TrainingConfig;
  private readonly sourceDir: string;
  private readonly frameworkSpecificConfig: FrameworkSpecificConfig;
  private logger: Logger;

  constructor(
    sagemakerClient: SageMaker,
    s3Client: S3,
    config: TrainingConfig,
    sourceDir: string,
    logger: Logger
  ) {
    this.logger = logger;
    this.logger.log("Initializing SageMakerTraining with config:", config);
    this.logger.log("Source directory:", sourceDir);
    this.config = config;
    this.sourceDir = sourceDir;
    this.frameworkSpecificConfig = FrameworkHandler.getFrameworkConfig(
      config.framework
    );

    this.sagemakerClient = sagemakerClient;
    this.s3Client = s3Client;
  }

  private cleanupTempFiles(filePath: string) {
    this.logger.log(`Cleaning up temporary path: ${filePath}`);
    try {
      if (fs.existsSync(filePath)) {
        // If it's a directory, remove it recursively
        if (fs.lstatSync(filePath).isDirectory()) {
          fs.rmSync(filePath, { recursive: true, force: true });
        } else {
          fs.unlinkSync(filePath);
        }
        this.logger.log("Cleanup completed successfully");
      }
    } catch (error) {
      this.logger.error("Error during cleanup:", error);
    }
  }

  private async prepareSourceDir(trainingJobName: string): Promise<string> {
    this.logger.log(
      `Preparing source directory for training job: ${trainingJobName}`
    );
    const tempDir = path.join(process.cwd(), "tmp", trainingJobName);
    fs.mkdirSync(tempDir, { recursive: true });
    const zipPath = path.join(tempDir, "source.tar.gz");

    const output = fs.createWriteStream(zipPath);
    const archive = archiver("tar", { gzip: true });

    return new Promise((resolve, reject) => {
      output.on("close", async () => {
        this.logger.log("Source directory archived successfully.");
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
          this.logger.log(
            `Source code uploaded to S3 at: s3://${this.config.bucket}/${key}`
          );
          this.cleanupTempFiles(tempDir);
          return resolve(`s3://${this.config.bucket}/${key}`);
        } catch (error) {
          this.cleanupTempFiles(tempDir);
          this.logger.error("Error uploading source code to S3:", error);
          reject(error);
        }
      });

      archive.on("error", (error) => {
        this.cleanupTempFiles(tempDir);
        reject(error);
      });

      archive.pipe(output);
      archive.directory(this.sourceDir, false);
      archive.finalize();
    });
  }

  private async monitorTrainingJob(
    trainingJobName: string
  ): Promise<TrainingJobStatus> {
    this.logger.log(`Monitoring training job: ${trainingJobName}`);
    while (true) {
      const response = await this.sagemakerClient.describeTrainingJob({
        TrainingJobName: trainingJobName,
      });

      const status = response.TrainingJobStatus as TrainingJobStatus;
      this.logger.log(`Training job status: ${status}`);
      if (["Completed", "Failed", "Stopped"].includes(status)) {
        return status;
      }

      this.logger.log(`Training job is still in status: ${status}. Waiting...`);
      await new Promise((resolve) => setTimeout(resolve, 60000));
    }
  }

  protected async prepareInputData(
    inputData: InputDataConfig,
    trainingJobName: string
  ): Promise<string> {
    this.logger.log(
      `Preparing input data for training job: ${trainingJobName}`
    );
    this.logger.log("Input data config:", inputData);
    // Handle S3 URI
    if (
      typeof inputData.data === "string" &&
      inputData.data.startsWith("s3://")
    ) {
      return inputData.data;
    }

    // Generate a file name based on channel and format
    const extension = this.getFileExtension(inputData.format);
    const fileName = `${inputData.channelName || "data"}.${extension}`;

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
    this.logger.log(
      `Input data uploaded to S3 at: s3://${this.config.bucket}/${key}`
    );
    return `s3://${this.config.bucket}/${key}`;
  }

  private getFileExtension(format: DataFormat): string {
    this.logger.log(`Getting file extension for data format: ${format}`);
    const extensionMap: Record<DataFormat, string> = {
      [DataFormat.JSON]: "json",
      [DataFormat.CSV]: "csv",
      [DataFormat.PARQUET]: "parquet",
      [DataFormat.LIBSVM]: "libsvm",
      [DataFormat.RECORDIO]: "recordio",
      [DataFormat.PROTOBUF]: "pb",
      [DataFormat.NUMPY]: "npy",
    };
    return extensionMap[format] || "dat";
  }

  private createTrainingJobParams(
    sourceCodeLocation: string,
    trainingJobName: string,
    resourceConfig: ResourceConfig,
    hyperParameters: Record<string, any>,
    metricDefinitions: MetricDefinition[],
    inputChannels: Channel[],
    frameworkConfig: FrameworkConfig,
    tags: { Key: string; Value: string }[]
  ): CreateTrainingJobCommandInput {
    this.logger.log("Creating training job parameters...");
    this.logger.log("Source code location:", sourceCodeLocation);
    this.logger.log("Training job name:", trainingJobName);
    this.logger.log("Resource config:", resourceConfig);
    this.logger.log("Hyperparameters:", hyperParameters);
    this.logger.log("Metric definitions:", metricDefinitions);
    this.logger.log("Input channels:", inputChannels);
    const mappedHyperParameters = Object.entries(hyperParameters).reduce(
      (acc, [key, value]) => {
        const mappedKey =
          this.frameworkSpecificConfig.hyperparameterMapping[key] || key;
        acc[mappedKey] = value.toString();
        return acc;
      },
      {} as Record<string, string>
    );

    return {
      TrainingJobName: trainingJobName,
      StoppingCondition: {
        MaxRuntimeInSeconds: resourceConfig.maxRuntimeSeconds || 86400,
        MaxPendingTimeInSeconds: resourceConfig.maxPendingSeconds || 3600,
      },
      AlgorithmSpecification: {
        TrainingImage: frameworkConfig.imageUri,
        TrainingInputMode: "File",
        EnableSageMakerMetricsTimeSeries: true,
        MetricDefinitions: metricDefinitions,
      },
      Environment: {
        ...this.frameworkSpecificConfig.environmentVariables,
        SAGEMAKER_REGION: this.config.region,
        SAGEMAKER_CONTAINER_LOG_LEVEL: "10",
        ...(resourceConfig.environmentVariables || {}),
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
        sagemaker_program: "train.py",
        sagemaker_submit_directory: sourceCodeLocation,
        "s3-output-path": `s3://${this.config.bucket}/${this.config.service}/${this.config.model}/${trainingJobName}/output`,
        ...mappedHyperParameters,
        ...(hyperParameters.schema && {
          data_schema: JSON.stringify(hyperParameters.schema),
        }),
      },
      Tags: tags,
    };
  }

  public async train(
    frameworkConfig: FrameworkConfig,
    resourceConfig: ResourceConfig,
    hyperParameters: Record<string, any>,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions: MetricDefinition[] = [],
    monitor: boolean = false,
    tags: { Key: string; Value: string }[] = []
  ): Promise<TrainingMetadata> {
    this.logger.log("Training job input parameters:", {
      frameworkConfig,
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions,
      monitor,
    });

    const defaultTags = [
      { Key: "Framework", Value: this.config.framework },
      { Key: "Service", Value: this.config.service },
      { Key: "Model", Value: this.config.model },
    ];
    const allTags = [...defaultTags, ...tags];

    try {
      const trainingJobName = `${this.config.service}-${
        this.config.model
      }-${Date.now()}`;
      const sourceCodeLocation = await this.prepareSourceDir(trainingJobName);

      // Handle multiple input channels
      const inputDataArray = Array.isArray(inputData) ? inputData : [inputData];

      // Process all input channels
      const inputChannels = await Promise.all(
        inputDataArray.map(async (input, index) => {
          const channelName =
            input.channelName || (index === 0 ? "train" : `channel_${index}`);
          const s3Uri = await this.prepareInputData(
            {
              ...input,
              channelName,
            },
            trainingJobName
          );

          return {
            ChannelName: channelName,
            DataSource: {
              S3DataSource: {
                S3DataType: input.s3DataType || "S3Prefix",
                S3Uri: s3Uri,
                S3DataDistributionType:
                  input.distributionType || "FullyReplicated",
              },
            },
            ContentType: input.format,
          };
        })
      );

      // Create and start the training job
      const trainingJobParams = this.createTrainingJobParams(
        sourceCodeLocation,
        trainingJobName,
        resourceConfig,
        hyperParameters,
        metricDefinitions,
        inputChannels,
        frameworkConfig,
        allTags
      );

      await this.sagemakerClient.createTrainingJob(trainingJobParams);
      this.logger.log(`Training job started with name: ${trainingJobName}`);

      const status = monitor
        ? await this.monitorTrainingJob(trainingJobName)
        : ("InProgress" as TrainingJobStatus);

      const modelOutputPath = `${trainingJobParams.OutputDataConfig.S3OutputPath}/${trainingJobName}/output/model.tar.gz`;

      this.logger.log("Training job completed with status:", status);
      return {
        trainingJobName,
        modelOutputPath,
        hyperParameters,
        status,
        framework: this.config.framework,
      };
    } catch (error) {
      this.logger.error("Error during training job:", error);
      throw error;
    }
  }
}
