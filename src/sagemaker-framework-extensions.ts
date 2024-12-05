import {
  SageMakerTraining,
} from './sagemaker-training';
import {
  PyTorchHyperParameters,
  TensorFlowHyperParameters,
  XGBoostHyperParameters,
  SklearnHyperParameters,
  HuggingFaceHyperParameters,
  TrainingConfig,
  Logger,
  FrameworkConfig,
  DataFormat,
  InputDataConfig,
  MetricDefinition,
  MLFramework,
  ResourceConfig,
  TrainingMetadata,
} from './interfaces';
/**
 * PyTorch Extension of SageMaker Training
 */
export class PyTorchTraining extends SageMakerTraining {
  constructor(
    config: TrainingConfig,
    sourceDir: string,
    logger: Logger,
    frameworkVersion = '2.1',
    pythonVersion = 'py310',
    useGpu = true,
  ) {
    const processor = useGpu ? 'gpu' : 'cpu';
    const frameworkConfig: FrameworkConfig = {
      framework: MLFramework.PYTORCH,
      frameworkVersion,
      pythonVersion,
      imageUri: `763104351884.dkr.ecr.${config.region}.amazonaws.com/pytorch-training:${frameworkVersion}-${processor}-${pythonVersion}`,
    };

    super(config, frameworkConfig, sourceDir, logger);
  }

  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: PyTorchHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false,
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
      monitor,
    );
  }
}

/**
 * TensorFlow Extension of SageMaker Training
 */
export class TensorFlowTraining extends SageMakerTraining {
  constructor(
    config: TrainingConfig,
    sourceDir: string,
    logger: Logger,
    frameworkVersion = '2.12',
    pythonVersion = 'py310',
    useGpu = true,
  ) {
    const processor = useGpu ? 'gpu' : 'cpu';
    const frameworkConfig: FrameworkConfig = {
      framework: MLFramework.TENSORFLOW,
      frameworkVersion,
      pythonVersion,
      imageUri: `763104351884.dkr.ecr.${config.region}.amazonaws.com/tensorflow-training:${frameworkVersion}-${processor}-${pythonVersion}`,
    };

    super(config, frameworkConfig, sourceDir, logger);
  }

  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: TensorFlowHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false,
  ): Promise<TrainingMetadata> {
    const defaultMetrics: MetricDefinition[] = [
      { Name: 'loss', Regex: 'loss: ([0-9\\.]+)' },
      { Name: 'accuracy', Regex: 'accuracy: ([0-9\\.]+)' },
      { Name: 'val_loss', Regex: 'val_loss: ([0-9\\.]+)' },
    ];

    return super.train(
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
      monitor,
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
    logger: Logger,
    frameworkVersion = '1.5',
    pythonVersion = 'py310',
  ) {
    const frameworkConfig: FrameworkConfig = {
      framework: MLFramework.XGBOOST,
      frameworkVersion,
      pythonVersion,
      imageUri: `683313688378.dkr.ecr.${config.region}.amazonaws.com/sagemaker-xgboost:${frameworkVersion}`,
    };

    super(config, frameworkConfig, sourceDir, logger);
  }

  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: XGBoostHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false,
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
      monitor,
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
    logger: Logger,
    frameworkVersion = '1.0',
    pythonVersion = 'py310',
  ) {
    const frameworkConfig: FrameworkConfig = {
      framework: MLFramework.SKLEARN,
      frameworkVersion,
      pythonVersion,
      imageUri: `683313688378.dkr.ecr.${config.region}.amazonaws.com/sagemaker-scikit-learn:${frameworkVersion}`,
    };

    super(config, frameworkConfig, sourceDir, logger);
  }

  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: SklearnHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false,
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
      monitor,
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
    logger: Logger,
    frameworkVersion = '4.28',
    pythonVersion = 'py310',
    useGpu = true,
  ) {
    const processor = useGpu ? 'gpu' : 'cpu';
    const frameworkConfig: FrameworkConfig = {
      framework: MLFramework.HUGGINGFACE,
      frameworkVersion,
      pythonVersion,
      imageUri: `763104351884.dkr.ecr.${config.region}.amazonaws.com/huggingface-pytorch-training:${frameworkVersion}-${processor}-${pythonVersion}`,
    };

    super(config, frameworkConfig, sourceDir, logger);
  }

  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: HuggingFaceHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false,
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
      monitor,
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
    logger: Logger,
  ) {
    const frameworkConfig: FrameworkConfig = {
      framework: customConfig.framework, // We can specify a base framework if needed
      frameworkVersion: customConfig.frameworkVersion,
      pythonVersion: customConfig.pythonVersion,
      imageUri: customConfig.imageUri,
    };

    super(config, frameworkConfig, sourceDir, logger);
  }

  // We can add custom metrics specific to your framework
  async train(
    resourceConfig: ResourceConfig,
    hyperParameters: Record<string, any>,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false,
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
      monitor,
    );
  }
}

// Let's create a specific NeuralForecast implementation
export class NeuralForecastTraining extends CustomFrameworkTraining {
  constructor(
    config: TrainingConfig,
    sourceDir: string,
    accountId: string,
    logger: Logger,
    version = '1.7.1',
  ) {
    const customConfig: CustomFrameworkConfig = {
      imageUri: `${accountId}.dkr.ecr.${config.region}.amazonaws.com/sagemaker-neuralforecast-training:${version}`,
      framework: MLFramework.PYTORCH,
      frameworkVersion: '2.1', // PyTorch base version
      pythonVersion: 'py310',
      customEnvironmentVariables: {
        // Add any NeuralForecast specific environment variables here
      },
    };

    super(config, sourceDir, customConfig, logger);
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
    monitor: boolean = false,
  ): Promise<TrainingMetadata> {
    return super.train(
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions,
      monitor,
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
    monitor: boolean = false,
  ): Promise<TrainingMetadata> {
    const inputConfig: InputDataConfig = {
      data,
      format,
      channelName: 'train_data',
    };

    return this.train(resourceConfig, hyperParameters, inputConfig, undefined, monitor);
  }
}
