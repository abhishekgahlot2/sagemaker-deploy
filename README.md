# SageMaker Model Deployment

Deploy quantized language models on AWS SageMaker for cost-effective inference.

## What This Does

This project handles the deployment of HuggingFace models to SageMaker endpoints with 4-bit quantization. The main use case is running large language models for text generation (particularly SQL query generation) on GPU instances while keeping costs manageable.

## Prerequisites

- AWS account with SageMaker access
- IAM role with SageMaker permissions
- Python 3.12+
- AWS CLI configured

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your configuration:
```
AWS_REGION=us-east-1
SAGEMAKER_ROLE_NAME=your-sagemaker-role
MODEL_ID=your-huggingface-model-id
INSTANCE_TYPE=ml.g5.xlarge
MODEL_SERVER_TIMEOUT=600
MAX_CONTEXT_LENGTH=4096
MAX_NEW_TOKENS=50
```

## Usage

### Deploy a Model

```bash
python deploy_model.py
```

This creates a SageMaker endpoint with:
- 4-bit quantization for memory efficiency
- Custom inference code optimized for text generation
- Configurable instance types (g5.xlarge recommended for cost/performance)

### Test the Endpoint

Once deployed, test with:

```bash
python test_endpoint.py
```

### Monitor and Cleanup

Check endpoint status:
```bash
python check_current_endpoint.py
```

View logs:
```bash
python check_logs.py
```

Delete endpoints when done (important to avoid charges):
```bash
python cleanup.py
```

## Project Structure

- `deploy_model.py` - Main deployment script
- `code/inference.py` - Custom inference handler with quantization
- `test_endpoint.py` - Test deployed endpoints
- `cleanup.py` - Delete endpoints and clean up resources
- `check_logs.py` - Monitor endpoint logs
- `requirements.txt` - Python dependencies

## Cost Considerations

Always delete endpoints when not in use to avoid charges. Here are recommended instance types for different use cases:

### Recommended Instances for Inference

| Instance Type | GPUs | GPU Model | Memory | Price/Hour | Use Case |
|--------------|------|-----------|---------|------------|----------|
| ml.g4dn.xlarge | 1 | NVIDIA T4 | 16 GB | $0.736 | Budget inference |
| ml.inf2.xlarge | 1 | AWS Inferentia2 | 32 GB | $0.99 | Cost-optimized inference |
| ml.g5.xlarge | 1 | NVIDIA A10G | 24 GB | $1.408 | Standard inference |
| ml.g5.12xlarge | 4 | NVIDIA A10G | 96 GB | $7.09 | Multi-GPU inference |
| ml.g5.48xlarge | 8 | NVIDIA A10G | 192 GB | $20.36 | High-throughput inference |

### High-Performance Training/Fine-tuning

| Instance Type | GPUs | GPU Model | Memory | Price/Hour | Use Case |
|--------------|------|-----------|---------|------------|----------|
| ml.g6.4xlarge | 1 | NVIDIA L4 | 24 GB | $1.654 | Light fine-tuning |
| ml.p4d.24xlarge | 8 | NVIDIA A100 | 320 GB | $25.25 | Large model training |
| ml.p5.48xlarge | 8 | **NVIDIA H100** | 640 GB | $63.30 | Cutting-edge training |

### CPU Instances (Preprocessing/Light Tasks)

| Instance Type | vCPUs | Memory | Price/Hour | Use Case |
|--------------|-------|---------|------------|----------|
| ml.m5.xlarge | 4 | 16 GB | $0.23 | General compute |
| ml.c5.xlarge | 4 | 8 GB | $0.204 | Compute optimized |

**Notes:**
- The **H100-powered p5.48xlarge** offers the best performance for large models but at premium cost
- For most inference workloads, g5.xlarge provides excellent price/performance
- The 4-GPU g5.12xlarge is ideal for serving larger models that don't fit on single GPU
- inf2 instances offer the lowest inference cost but require model optimization

Use `check_sagemaker_pricing.py` to review current pricing in your region.

## Technical Details

The deployment uses:
- BitsAndBytes for 4-bit quantization
- Flash Attention 2 for faster inference
- Custom inference script to handle model loading and prediction
- Configurable parameters for context length and token generation

## Common Issues

1. **Out of memory errors**: Reduce MAX_CONTEXT_LENGTH or use a larger instance
2. **Timeout during deployment**: Increase MODEL_SERVER_TIMEOUT in .env
3. **Role permissions**: Ensure your IAM role has full SageMaker access

## Development

To modify the inference behavior, edit `code/inference.py`. The main functions:
- `model_fn()` - Loads the model with quantization config
- `predict_fn()` - Handles text generation
- `input_fn()/output_fn()` - Parse requests and format responses
