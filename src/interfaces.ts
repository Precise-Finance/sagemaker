import { TrainingJobStatus, TrainingInstanceType } from '@aws-sdk/client-sagemaker';

// Core AWS interfaces
export interface AWSCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
}

// Base configurations
export interface BaseConfig {
  region: string;
  credentials: AWSCredentials;
}

export interface TrainingConfig extends BaseConfig {
  role: string;
  bucket: string;
  service: string;
  model: string;
}

export interface DeploymentConfig extends BaseConfig {
  endpointName: string;
  useGpu?: boolean;
}

// Framework related interfaces
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

export interface FrameworkSpecificConfig {
  contentType: string;
  environmentVariables: Record<string, string>;
  hyperparameterMapping: Record<string, string>;
}

// Training related interfaces
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
}

export interface InputDataConfig {
  data: Buffer | string | NodeJS.ReadableStream;
  format: DataFormat;
  channelName?: string;
  distributionType?: 'FullyReplicated' | 'ShardedByS3Key';
  s3DataType?: 'S3Prefix' | 'ManifestFile';
  schema?: Record<string, string>;
}

// Deployment related interfaces
export interface ServerlessConfig {
  memorySizeInMb: number;
  maxConcurrency: number;
}

export interface BaseModelConfig {
  modelData: string;
  role: string;
  entryPoint: string;
  imageUri?: string;
  useGpu?: boolean;
}

export interface FrameworkModelConfig extends BaseModelConfig {
  frameworkVersion: string;
  pythonVersion: string;
  framework: MLFramework;
}

export interface DeploymentResult {
  modelName: string;
  endpointName: string;
  status: "Created" | "Updated";
}

// Framework-specific hyperparameter interfaces
export interface BaseHyperParameters {
  [key: string]: any;
}

export interface PyTorchHyperParameters extends BaseHyperParameters {
  learningRate?: number;
  batchSize?: number;
  epochs?: number;
  optimizerName?: string;
  momentum?: number;
  weightDecay?: number;
  scheduleRate?: number;
}

export interface XGBoostHyperParameters extends BaseHyperParameters {
  learningRate?: number;
  maxDepth?: number;
  nEstimators?: number;
  minChildWeight?: number;
  subsample?: number;
  colsampleBytree?: number;
  gamma?: number;
  alpha?: number;
  lambda?: number;
}

export interface SklearnHyperParameters extends BaseHyperParameters {
  maxDepth?: number;
  nEstimators?: number;
  criterion?: string;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: string | number;
}

export interface HuggingFaceHyperParameters extends BaseHyperParameters {
  learningRate?: number;
  batchSize?: number;
  epochs?: number;
  warmupSteps?: number;
  weightDecay?: number;
  maxSeqLength?: number;
  modelName?: string;
}

export interface TensorFlowHyperParameters extends BaseHyperParameters {
  learningRate?: number;
  batchSize?: number;
  epochs?: number;
  optimizerName?: string;
  momentum?: number;
  weightDecay?: number;
}

// Utility interfaces
export interface Logger {
  log(message?: any, ...optionalParams: any[]): void;
  error(message?: any, ...optionalParams: any[]): void;
  warn(message?: any, ...optionalParams: any[]): void;
  info(message?: any, ...optionalParams: any[]): void;
  debug(message?: any, ...optionalParams: any[]): void;
}
