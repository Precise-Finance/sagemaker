import { SageMakerTraining } from "./sagemaker-training";
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
} from "./interfaces";
/**
 * PyTorch Extension of SageMaker Training
 */
export class PyTorchTraining extends SageMakerTraining {
  private readonly defaultFrameworkConfig: FrameworkConfig;

  constructor(
    config: Omit<TrainingConfig, "framework">,
    sourceDir: string,
    logger: Logger,
    frameworkVersion = "2.1",
    pythonVersion = "py310",
    useGpu = true
  ) {
    const processor = useGpu ? "gpu" : "cpu";
    const imageUri = `763104351884.dkr.ecr.${config.region}.amazonaws.com/pytorch-training:${frameworkVersion}-${processor}-${pythonVersion}`;

    super({ framework: MLFramework.PYTORCH, ...config }, sourceDir, logger);

    this.defaultFrameworkConfig = {
      frameworkVersion,
      pythonVersion,
      imageUri,
    };
  }

  async train(
    frameworkConfig: FrameworkConfig,
    resourceConfig: ResourceConfig,
    hyperParameters: PyTorchHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false
  ): Promise<TrainingMetadata> {
    const defaultMetrics: MetricDefinition[] = [
      { Name: "loss", Regex: "Loss: ([0-9\\.]+)" },
      { Name: "accuracy", Regex: "Accuracy: ([0-9\\.]+)" },
      { Name: "learning_rate", Regex: "Learning Rate: ([0-9\\.]+)" },
    ];

    return super.train(
      { ...this.defaultFrameworkConfig, ...frameworkConfig },
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
      monitor
    );
  }
}

/**
 * TensorFlow Extension of SageMaker Training
 */
export class TensorFlowTraining extends SageMakerTraining {
  private readonly defaultFrameworkConfig: FrameworkConfig;

  constructor(
    config: Omit<TrainingConfig, "framework">,
    sourceDir: string,
    logger: Logger,
    frameworkVersion = "2.12",
    pythonVersion = "py310",
    useGpu = true
  ) {
    const processor = useGpu ? "gpu" : "cpu";
    const imageUri = `763104351884.dkr.ecr.${config.region}.amazonaws.com/tensorflow-training:${frameworkVersion}-${processor}-${pythonVersion}`;

    super({ framework: MLFramework.TENSORFLOW, ...config }, sourceDir, logger);

    this.defaultFrameworkConfig = {
      frameworkVersion,
      pythonVersion,
      imageUri,
    };
  }

  async train(
    frameworkConfig: FrameworkConfig,
    resourceConfig: ResourceConfig,
    hyperParameters: TensorFlowHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false
  ): Promise<TrainingMetadata> {
    const defaultMetrics: MetricDefinition[] = [
      { Name: "loss", Regex: "loss: ([0-9\\.]+)" },
      { Name: "accuracy", Regex: "accuracy: ([0-9\\.]+)" },
      { Name: "val_loss", Regex: "val_loss: ([0-9\\.]+)" },
    ];

    return super.train(
      { ...this.defaultFrameworkConfig, ...frameworkConfig },
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
      monitor
    );
  }
}

/**
 * XGBoost Extension of SageMaker Training
 */
export class XGBoostTraining extends SageMakerTraining {
  private readonly defaultFrameworkConfig: FrameworkConfig;

  constructor(
    config: Omit<TrainingConfig, "framework">,
    sourceDir: string,
    logger: Logger,
    frameworkVersion = "1.5",
    pythonVersion = "py310"
  ) {
    const imageUri = `683313688378.dkr.ecr.${config.region}.amazonaws.com/sagemaker-xgboost:${frameworkVersion}`;

    super({ framework: MLFramework.XGBOOST, ...config }, sourceDir, logger);

    this.defaultFrameworkConfig = {
      frameworkVersion,
      pythonVersion,
      imageUri,
    };
  }

  async train(
    frameworkConfig: FrameworkConfig,
    resourceConfig: ResourceConfig,
    hyperParameters: XGBoostHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false
  ): Promise<TrainingMetadata> {
    const defaultMetrics: MetricDefinition[] = [
      { Name: "validation:rmse", Regex: "validation-rmse:([0-9\\.]+)" },
      { Name: "train:rmse", Regex: "train-rmse:([0-9\\.]+)" },
    ];

    return super.train(
      { ...this.defaultFrameworkConfig, ...frameworkConfig },
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
      monitor
    );
  }
}

/**
 * Scikit-learn Extension of SageMaker Training
 */
export class SklearnTraining extends SageMakerTraining {
  private readonly defaultFrameworkConfig: FrameworkConfig;

  constructor(
    config: Omit<TrainingConfig, "framework">,
    sourceDir: string,
    logger: Logger,
    frameworkVersion = "1.0",
    pythonVersion = "py310"
  ) {
    const imageUri = `683313688378.dkr.ecr.${config.region}.amazonaws.com/sagemaker-scikit-learn:${frameworkVersion}`;

    super({ framework: MLFramework.SKLEARN, ...config }, sourceDir, logger);

    this.defaultFrameworkConfig = {
      frameworkVersion,
      pythonVersion,
      imageUri,
    };
  }

  async train(
    frameworkConfig: FrameworkConfig,
    resourceConfig: ResourceConfig,
    hyperParameters: SklearnHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false
  ): Promise<TrainingMetadata> {
    const defaultMetrics: MetricDefinition[] = [
      { Name: "accuracy", Regex: "accuracy: ([0-9\\.]+)" },
      { Name: "f1_score", Regex: "f1: ([0-9\\.]+)" },
    ];

    return super.train(
      { ...this.defaultFrameworkConfig, ...frameworkConfig },
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
      monitor
    );
  }
}

/**
 * HuggingFace Extension of SageMaker Training
 */
export class HuggingFaceTraining extends SageMakerTraining {
  private readonly defaultFrameworkConfig: FrameworkConfig;

  constructor(
    config: Omit<TrainingConfig, "framework">,
    sourceDir: string,
    logger: Logger,
    frameworkVersion = "4.28",
    pythonVersion = "py310",
    useGpu = true
  ) {
    const processor = useGpu ? "gpu" : "cpu";
    const imageUri = `763104351884.dkr.ecr.${config.region}.amazonaws.com/huggingface-pytorch-training:${frameworkVersion}-${processor}-${pythonVersion}`;

    super({ framework: MLFramework.HUGGINGFACE, ...config }, sourceDir, logger);

    this.defaultFrameworkConfig = {
      frameworkVersion,
      pythonVersion,
      imageUri,
    };
  }

  async train(
    frameworkConfig: FrameworkConfig,
    resourceConfig: ResourceConfig,
    hyperParameters: HuggingFaceHyperParameters,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false
  ): Promise<TrainingMetadata> {
    const defaultMetrics: MetricDefinition[] = [
      { Name: "loss", Regex: "loss: ([0-9\\.]+)" },
      { Name: "eval_loss", Regex: "eval_loss: ([0-9\\.]+)" },
    ];

    return super.train(
      { ...this.defaultFrameworkConfig, ...frameworkConfig },
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
      monitor
    );
  }
}

export class CustomFrameworkTraining extends SageMakerTraining {
  constructor(config: TrainingConfig, sourceDir: string, logger: Logger) {
    super(config, sourceDir, logger);
  }

  // We can add custom metrics specific to your framework
  async train(
    frameworkConfig: FrameworkConfig,
    resourceConfig: ResourceConfig,
    hyperParameters: Record<string, any>,
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false
  ): Promise<TrainingMetadata> {
    // Default metrics for NeuralForecast
    const defaultMetrics: MetricDefinition[] = [
      { Name: "mse", Regex: "MSE: ([0-9\\.]+)" },
      { Name: "mase", Regex: "MASE: ([0-9\\.]+)" },
    ];

    return super.train(
      frameworkConfig,
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions || defaultMetrics,
      monitor
    );
  }
}

// Let's create a specific NeuralForecast implementation
export class NeuralForecastTraining extends CustomFrameworkTraining {
  private readonly defaultFrameworkConfig: FrameworkConfig;
  constructor(
    config: Omit<TrainingConfig, 'framework'>,
    sourceDir: string,
    accountId: string,
    logger: Logger,
    version = "1.7.1"
  ) {
    super({ framework: MLFramework.PYTORCH, ...config }, sourceDir, logger);

    this.defaultFrameworkConfig = {
      imageUri: `${accountId}.dkr.ecr.${config.region}.amazonaws.com/sagemaker-neuralforecast-training:${version}`,
      frameworkVersion: "2.1", // PyTorch base version
      pythonVersion: "py310",
    };
  }

  // Add NeuralForecast specific hyperparameter interface
  async train(
    frameworkConfig: Partial<FrameworkConfig>,
    resourceConfig: ResourceConfig,
    hyperParameters: {
      "max-steps"?: number;
      "context-length"?: number;
      [key: string]: any;
    },
    inputData: InputDataConfig | InputDataConfig[],
    metricDefinitions?: MetricDefinition[],
    monitor: boolean = false
  ): Promise<TrainingMetadata> {
    return super.train(
      { ...this.defaultFrameworkConfig, ...frameworkConfig },
      resourceConfig,
      hyperParameters,
      inputData,
      metricDefinitions,
      monitor
    );
  }

  async trainWithData(
    frameworkConfig: Partial<FrameworkConfig>,
    resourceConfig: ResourceConfig,
    hyperParameters: {
      "max-steps"?: number;
      "context-length"?: number;
      [key: string]: any;
    },
    data: Buffer | string,
    metricDefinitions?: MetricDefinition[],
    format: DataFormat = DataFormat.JSON,
    monitor: boolean = false
  ): Promise<TrainingMetadata> {
    const inputConfig: InputDataConfig = {
      data,
      format,
      channelName: "train_data",
    };

    return this.train(
      { ...this.defaultFrameworkConfig, ...frameworkConfig },
      resourceConfig,
      hyperParameters,
      inputConfig,
      metricDefinitions,
      monitor
    );
  }
}
