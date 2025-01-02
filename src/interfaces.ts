import {
  SageMakerClient,
  TrainingJobStatus,
  TrainingInstanceType,
} from "@aws-sdk/client-sagemaker";
import { S3Client } from "@aws-sdk/client-s3";

// Base configurations - AWS level only
export interface BaseConfig {
  region: string;
  bucket: string;
  role: string;
}

// Framework Configuration extends Base AWS config
export interface FrameworkDeployConfig extends BaseConfig {
  environmentVariables: Record<string, string>;
}

// Training specific configuration
export interface TrainingConfig extends BaseConfig {
  service: string; // Service/model specific
  model: string;
  framework: MLFramework;
}

// Dynamic Model Deployment Configuration
export interface ModelDeploymentInput {
  frameworkVersion: string;
  pythonVersion: string;
  entryPoint: string;
  trainingJobName?: string;
  modelPath?: string;
  useGpu?: boolean;
  imageUri?: string;
  environmentVariables?: Record<string, string>;
}

// Framework related interfaces
export enum MLFramework {
  PYTORCH = "pytorch",
  TENSORFLOW = "tensorflow",
  SKLEARN = "sklearn",
  XGBOOST = "xgboost",
  HUGGINGFACE = "huggingface",
}

export interface FrameworkConfig {
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
  environmentVariables?: Record<string, string>;
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
  JSON = "application/json",
  CSV = "text/csv",
  PARQUET = "application/x-parquet",
  LIBSVM = "application/x-libsvm",
  RECORDIO = "application/x-recordio-protobuf",
  PROTOBUF = "application/x-protobuf",
  NUMPY = "application/x-npy",
}

export interface InputDataConfig {
  data: Buffer | string | NodeJS.ReadableStream;
  format: DataFormat;
  channelName?: string;
  distributionType?: "FullyReplicated" | "ShardedByS3Key";
  s3DataType?: "S3Prefix" | "ManifestFile";
  schema?: Record<string, string>;
}

// Deployment related interfaces
export interface ServerlessConfig {
  memorySizeInMb: number;
  maxConcurrency: number;
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

// Inference related interfaces
export interface InferenceConfig {
  maxRetries?: number;
  timeoutMs?: number;
  batchSize?: number;
  validateResponse?: boolean;
  metricsEnabled?: boolean;
}

export interface InferenceRequest {
  contentType?: string;
  accept?: string;
  customAttributes?: Record<string, string>;
  transformInput?: (payload: any) => any;
  transformOutput?: (response: any) => any;
}

export interface InferenceMetrics {
  timestamp: number;
  latencyMs: number;
  success: boolean;
  endpointName: string;
  error?: string;
}

export type MetricsCallback = (metrics: InferenceMetrics) => void;

export interface SageMakerInferenceConfig extends BaseConfig {
  defaultContentType?: string;
  defaultAccept?: string;
  defaultCustomAttributes?: Record<string, string>;
  retry?: {
    maxAttempts: number;
    timeoutMs: number;
    backoffMultiplier?: number;
  };
  monitoring?: {
    enabled: boolean;
    metricsCallback?: MetricsCallback;
  };
  validation?: {
    enabled: boolean;
    customValidator?: (response: any) => boolean;
  };
  batch?: {
    enabled: boolean;
    maxBatchSize: number;
    concurrency?: number;
  };
}

export interface InferenceTargetConfig {
  targetModel?: string;          // For multi-model endpoints
  targetVariant?: string;        // For multiple production variants (A/B testing)
  targetContainerHostname?: string;  // For multi-container endpoints
  inferenceId?: string;          // Unique tracking ID
  enableExplanations?: string;   // For model explainability
  inferenceComponentName?: string;  // For multi-component endpoints
  sessionId?: string;            // For stateful inference
}

export interface InferenceCallOptions extends InferenceRequest, InferenceTargetConfig {
  retry?: {
    maxAttempts?: number;
    timeoutMs?: number;
    backoffMultiplier?: number;
  };
  validation?: {
    enabled?: boolean;
    customValidator?: (response: any) => boolean;
  };
  monitoring?: {
    enabled?: boolean;
    metricsCallback?: MetricsCallback;
  };
}

export interface InferenceResponse<T = any> {
  data: T;
  inferenceId?: string;
  explanation?: any;
  targetModel?: string;
  targetVariant?: string;
}
