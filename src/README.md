# SageMaker Inference Client

A powerful and flexible TypeScript client for making inference calls to Amazon SageMaker endpoints.

## Features

- Robust error handling and retries
- Configurable timeouts and backoff
- Batch processing support
- Response validation
- Metrics and monitoring
- Support for multiple target options (A/B testing, multi-model endpoints)
- Customizable input/output transformations

## Installation

```bash
npm install @precise-finance/sagemaker-utils
```

## Basic Usage

```typescript
import { SageMakerInference, SageMakerInferenceConfig } from '@precise-finance/sagemaker-utils';
import { SageMakerClient } from "@aws-sdk/client-sagemaker";

const config: SageMakerInferenceConfig = {
  region: 'us-east-1',
  sagemakerClient: new SageMakerClient({ region: 'us-east-1' }),
  s3Client: new S3Client({ region: 'us-east-1' }),
  retry: {
    maxAttempts: 3,
    timeoutMs: 30000
  }
};

const client = new SageMakerInference(config, console, new SageMakerClient({ region: 'us-east-1' }));

// Single inference
const result = await client.invokeEndpoint(
  'my-endpoint',
  { data: [1, 2, 3, 4] }
);

// Batch inference
const results = await client.batchInvokeEndpoint(
  'my-endpoint',
  [
    { data: [1, 2, 3, 4] },
    { data: [5, 6, 7, 8] }
  ]
);
```

## Advanced Usage

### Custom Validation and Transformation

```typescript
const response = await client.invokeEndpoint(
  'my-endpoint',
  { input: [1, 2, 3] },
  {
    transformInput: (payload) => ({
      ...payload,
      timestamp: Date.now()
    }),
    transformOutput: (response) => response.predictions,
    validation: {
      enabled: true,
      customValidator: (response) => response.predictions.length > 0
    }
  }
);
```

### A/B Testing and Multi-Model Endpoints

```typescript
const response = await client.invokeEndpoint(
  'my-endpoint',
  payload,
  {
    targetModel: 'model-v2',
    targetVariant: 'variant-b',
    inferenceId: 'custom-inference-id',
    sessionId: 'user-session-123'
  }
);
```

### Monitoring and Metrics

```typescript
const config: SageMakerInferenceConfig = {
  // ... other config ...
  monitoring: {
    enabled: true,
    metricsCallback: (metrics) => {
      console.log(`Inference latency: ${metrics.latencyMs}ms`);
      // Send metrics to monitoring system
    }
  }
};
```

### Batch Processing with Custom Configuration

```typescript
const config: SageMakerInferenceConfig = {
  // ... other config ...
  batch: {
    enabled: true,
    maxBatchSize: 10,
    concurrency: 3
  }
};

const payloads = generatePayloads(100); // Array of 100 items
const results = await client.batchInvokeEndpoint(
  'my-endpoint',
  payloads,
  {
    contentType: 'application/json',
    customAttributes: {
      'batch-id': 'batch-123'
    }
  }
);
```

### Error Handling

```typescript
try {
  const result = await client.invokeEndpoint(
    'my-endpoint',
    payload,
    {
      retry: {
        maxAttempts: 5,
        timeoutMs: 10000,
        backoffMultiplier: 1.5
      }
    }
  );
} catch (error) {
  console.error('Inference failed:', error);
}
```

## Interface Definitions

For complete type definitions, refer to the `interfaces.ts` file. Key interfaces include:

- `SageMakerInferenceConfig`
- `InferenceCallOptions`
- `InferenceResponse`
- `InferenceMetrics`

## Best Practices

1. Always configure appropriate timeouts and retry strategies
2. Use batch processing for high-throughput scenarios
3. Implement proper error handling
4. Enable monitoring for production deployments
5. Use custom validation for domain-specific response checking
6. Configure logging appropriately for your environment

## License

MIT