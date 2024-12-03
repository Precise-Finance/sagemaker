import {
  SageMakerTraining,
  TrainingConfig,
  ResourceConfig,
  MetricDefinition,
  MLFramework,
  FrameworkConfig,
  TrainingMetadata,
  InputDataConfig,
  DataFormat,
} from './sagemaker-training';

/**
 * Base interface for framework-specific hyperparameters
 * This ensures type safety for each framework's unique parameters
 */
interface BaseHyperParameters {
  [key: string]: any;
}

/**
 * PyTorch-specific hyperparameters interface
 */
export interface PyTorchHyperParameters extends BaseHyperParameters {
  learningRate?: number;
  batchSize?: number;
  epochs?: number;
  optimizerName?: string;
  momentum?: number;
  weightDecay?: number;
  scheduleRate?: number;
}

/**
 * XGBoost-specific hyperparameters interface
 */
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

/**
 * Scikit-learn-specific hyperparameters interface
 */
export interface SklearnHyperParameters extends BaseHyperParameters {
  maxDepth?: number;
  nEstimators?: number;
  criterion?: string;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: string | number;
}

/**
 * HuggingFace-specific hyperparameters interface
 */
export interface HuggingFaceHyperParameters extends BaseHyperParameters {
  learningRate?: number;
  batchSize?: number;
  epochs?: number;
  warmupSteps?: number;
  weightDecay?: number;
  maxSeqLength?: number;
  modelName?: string;
}

/**
 * PyTorch Extension of SageMaker Training
 */
export class PyTorchTraining extends SageMakerTraining {
  constructor(
    config: TrainingConfig,
    sourceDir: string,
    frameworkVersion = '2.1',
    pythonVersion = 'py310',
  ) {
    const frameworkConfig: FrameworkConfig = {
      framework: MLFramework.PYTORCH,
      frameworkVersion,
      pythonVersion,
      imageUri: `763104351884.dkr.ecr.${config.region}.amazonaws.com/pytorch-training:${frameworkVersion}-gpu-${pythonVersion}`,
    };

    super(config, frameworkConfig, sourceDir);
  }

  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: PyTorchHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
  ): Promise<TrainingMetadata> {
    // Add PyTorch-specific metric definitions if none provided
    const defaultMetrics: MetricDefinition[] = [
      { Name: 'loss', Regex: 'Loss: ([0-9\\.]+)' },
      { Name: 'accuracy', Regex: 'Accuracy: ([0-9\\.]+)' },
      { Name: 'learning_rate', Regex: 'Learning Rate: ([0-9\\.]+)' },
    ];

    return super.train(
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
    );
  }
}

/**
 * XGBoost Extension of SageMaker Training
 */
export class XGBoostTraining extends SageMakerTraining {
  constructor(
    config: TrainingConfig,
    sourceDir: string,
    frameworkVersion = '1.5',
    pythonVersion = 'py310',
  ) {
    const frameworkConfig: FrameworkConfig = {
      framework: MLFramework.XGBOOST,
      frameworkVersion,
      pythonVersion,
      imageUri: `683313688378.dkr.ecr.${config.region}.amazonaws.com/sagemaker-xgboost:${frameworkVersion}`,
    };

    super(config, frameworkConfig, sourceDir);
  }

  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: XGBoostHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
  ): Promise<TrainingMetadata> {
    const defaultMetrics: MetricDefinition[] = [
      { Name: 'validation:rmse', Regex: 'validation-rmse:([0-9\\.]+)' },
      { Name: 'train:rmse', Regex: 'train-rmse:([0-9\\.]+)' },
    ];

    return super.train(
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
    );
  }
}

/**
 * Scikit-learn Extension of SageMaker Training
 */
export class SklearnTraining extends SageMakerTraining {
  constructor(
    config: TrainingConfig,
    sourceDir: string,
    frameworkVersion = '1.0',
    pythonVersion = 'py310',
  ) {
    const frameworkConfig: FrameworkConfig = {
      framework: MLFramework.SKLEARN,
      frameworkVersion,
      pythonVersion,
      imageUri: `683313688378.dkr.ecr.${config.region}.amazonaws.com/sagemaker-scikit-learn:${frameworkVersion}`,
    };

    super(config, frameworkConfig, sourceDir);
  }

  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: SklearnHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
  ): Promise<TrainingMetadata> {
    const defaultMetrics: MetricDefinition[] = [
      { Name: 'accuracy', Regex: 'accuracy: ([0-9\\.]+)' },
      { Name: 'f1_score', Regex: 'f1: ([0-9\\.]+)' },
    ];

    return super.train(
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
    );
  }
}

/**
 * HuggingFace Extension of SageMaker Training
 */
export class HuggingFaceTraining extends SageMakerTraining {
  constructor(
    config: TrainingConfig,
    sourceDir: string,
    frameworkVersion = '4.26',
    pythonVersion = 'py310',
  ) {
    const frameworkConfig: FrameworkConfig = {
      framework: MLFramework.HUGGINGFACE,
      frameworkVersion,
      pythonVersion,
      imageUri: `763104351884.dkr.ecr.${config.region}.amazonaws.com/huggingface-pytorch-training:${frameworkVersion}`,
    };

    super(config, frameworkConfig, sourceDir);
  }

  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: HuggingFaceHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
  ): Promise<TrainingMetadata> {
    const defaultMetrics: MetricDefinition[] = [
      { Name: 'loss', Regex: 'loss: ([0-9\\.]+)' },
      { Name: 'eval_loss', Regex: 'eval_loss: ([0-9\\.]+)' },
    ];

    return super.train(
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
    );
  }
}

// First, let's add a new interface for custom framework configuration
export interface CustomFrameworkConfig {
  imageUri: string;
  framework: MLFramework;
  frameworkVersion: string;
  pythonVersion: string;
  customEnvironmentVariables?: Record<string, string>;
}

// Now let's create a custom framework extension
export class CustomFrameworkTraining extends SageMakerTraining {
  constructor(
    config: TrainingConfig,
    sourceDir: string,
    customConfig: CustomFrameworkConfig,
  ) {
    const frameworkConfig: FrameworkConfig = {
      framework: customConfig.framework, // We can specify a base framework if needed
      frameworkVersion: customConfig.frameworkVersion,
      pythonVersion: customConfig.pythonVersion,
      imageUri: customConfig.imageUri,
    };

    super(config, frameworkConfig, sourceDir);
  }

  // We can add custom metrics specific to your framework
  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: Record<string, any>,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
  ): Promise<TrainingMetadata> {
    // Default metrics for NeuralForecast
    const defaultMetrics: MetricDefinition[] = [
      { Name: 'mse', Regex: 'MSE: ([0-9\\.]+)' },
      { Name: 'mase', Regex: 'MASE: ([0-9\\.]+)' },
    ];

    return super.train(
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
    );
  }
}

// Let's create a specific NeuralForecast implementation
export class NeuralForecastTraining extends CustomFrameworkTraining {
  constructor(
    config: TrainingConfig,
    sourceDir: string,
    accountId: string,
    version = '1.7.1',
  ) {
    const customConfig: CustomFrameworkConfig = {
      imageUri: `${accountId}.dkr.ecr.${config.region}.amazonaws.com/sagemaker-neuralforecast-training:${version}`,
      framework: MLFramework.PYTORCH,
      frameworkVersion: '2.1', // PyTorch base version
      pythonVersion: 'py310',
      customEnvironmentVariables: {
        // Add any NeuralForecast specific environment variables here
        SAGEMAKER_PROGRAM: 'train.py',
        SAGEMAKER_SUBMIT_DIRECTORY: '/opt/ml/model/code',
      },
    };

    super(config, sourceDir, customConfig);
  }

  // Add NeuralForecast specific hyperparameter interface
  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: {
      'max-steps'?: number;
      'context-length'?: number;
      [key: string]: any;
    },
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
  ): Promise<TrainingMetadata> {
    return super.train(
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions,
    );
  }

  async trainWithData(
    resourceConfig: ResourceConfig,
    hyperParameters: {
      'max-steps'?: number;
      'context-length'?: number;
      [key: string]: any;
    },
    data: Buffer | string,
    format: DataFormat = DataFormat.JSON,
  ): Promise<TrainingMetadata> {
    const inputConfig: InputDataConfig = {
      data,
      format,
      channelName: 'train_data',
    };

    return this.train(resourceConfig, hyperParameters, inputConfig);
  }
}
