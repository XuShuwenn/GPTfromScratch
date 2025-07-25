# GPT Model Inference Guide

## Quick Start

The `inference.sh` script provides an easy way to run inference with your trained GPT models.

### Basic Usage

```bash
# Use default settings with GPT2-29M model
./scripts/inference.sh

# Specify a different model
./scripts/inference.sh -m GPT2-49M

# Custom prompt and generation settings
./scripts/inference.sh -m GPT2-14M -p "Once upon a time in a magical forest" -l 100 -t 0.9
```

### Available Models

- **GPT2-29M**: 29 million parameters (256 hidden size, 4 layers, 4 heads)
- **GPT2-49M**: 49 million parameters (384 hidden size, 6 layers, 6 heads)  
- **GPT2-14M**: 14 million parameters (128 hidden size, 4 layers, 4 heads)

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Model to use (GPT2-14M, GPT2-29M, GPT2-49M) | GPT2-14M |
| `-p, --prompt` | Input prompt for generation | "Once upon a time" |
| `-l, --max-length` | Maximum generation length | 100 |
| `-t, --temperature` | Sampling temperature (higher = more random) | 0.7 |
| `-k, --top-k` | Top-k sampling parameter | 20 |
| `--model-path` | Specific checkpoint path (overrides auto-selection) | Auto-selected |
| `--list-ckpts` | List available checkpoints for a model | - |
| `-h, --help` | Show help message | - |

### Examples

```bash
# List available checkpoints for a specific model
./scripts/inference.sh --list-ckpts -m GPT2-29M

# Generate a short story with high creativity
./scripts/inference.sh -m GPT2-49M -p "The wizard opened the ancient book and discovered" -l 150 -t 1.0

# Generate text with low randomness (more deterministic)
./scripts/inference.sh -m GPT2-14M -p "The scientific explanation for this phenomenon is" -l 80 -t 0.3

# Use a specific checkpoint file
./scripts/inference.sh --model-path logs/GPT2-29M/ckpts/model_epoch9_valloss0.0386.pth -p "Hello world"
```

### Output Example

```
================================
GPT Model Inference Configuration
================================
Model: GPT2-49M
Checkpoint: logs/GPT2-49M/ckpts/model_epoch7_valloss0.0389.pth
Prompt: "Once upon a time"
Max Length: 100
Temperature: 0.7
Top-K: 20
Model Config:
  - Vocab Size: 50257
  - D Model: 384
  - N Layers: 6
  - N Heads: 6
  - D FF: 1536
  - Max Seq Len: 256
================================

==================================================
Generated Text:
==================================================
Once upon a time, there was a little girl named Lucy...
==================================================
```

### Tips for Better Generation

1. **Temperature**: 
   - Lower (0.1-0.5): More deterministic, coherent text
   - Higher (0.8-1.2): More creative, diverse text

2. **Top-k**: 
   - Lower values (5-20): More focused vocabulary
   - Higher values (50-100): More diverse word choices

3. **Prompt Engineering**:
   - Use specific, detailed prompts for better results
   - Start with complete sentences when possible
   - For dialogue, use proper formatting

4. **Model Selection**:
   - GPT2-29M: Fastest, good for simple tasks
   - GPT2-49M: Balanced speed and quality
   - GPT2-14M: Lightweight, fastest inference

### Troubleshooting

**Error: "No checkpoints found"**
- Make sure you've trained the model first
- Check that the checkpoint directory exists: `logs/MODEL_NAME/ckpts/`

**Error: "Checkpoint file does not exist"**
- Use `--list-ckpts` to see available checkpoints
- Specify a valid checkpoint path with `--model-path`

**Poor generation quality**
- Try different temperature values
- Ensure the model has been trained sufficiently
- Use more specific prompts
