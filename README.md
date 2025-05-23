# Universal LLM Interface

This module provides wrapper classes for interacting with various Large Language Models (LLMs) in a unified way.

## Installation

You can install this package directly from PyPI:

```bash
pip install unillm
```

To install from source:

```bash
git clone https://github.com/BrachioLab/universal_llm.git
cd universal_llm
pip install -e .
```

## Classes

### OurLLM

A wrapper for local LLM models, particularly those from Hugging Face like Llama and Qwen.

**Features:**
- Supports Llama-3.2, Llama-3.3, and Qwen models.
- Handles both text and image inputs.
- Provides consistent interface for model inference.

**Usage:**
```python
from src.llm_models import OurLLM

# Initialize the model
model = OurLLM(model_name="meta-llama/Llama-3.2-90B-Vision-Instruct")

# Use the model
response = model.chat(prompt, sampling_params, use_tqdm=False)
```

### APIModel

A wrapper for API-based LLM services like Claude, Gemini, and GPT.

**Features:**
- Supports Claude, Gemini, and GPT models.
- Handles API authentication and retries.
- Provides consistent interface for model inference.

**Usage:**
```python
from src.llm_models import APIModel

# Initialize the model
model = APIModel(model_name="claude-3-opus-20240229")

# Use the model
response = model.chat(prompt, sampling_params, use_tqdm=False)
```

## Common Interface

Both classes implement a common interface with the following methods:

- `__init__(model_name)`: Initialize the model with the specified model name
- `chat(prompt, sampling_params, use_tqdm)`: Generate a response from the model

## Dependencies

- transformers
- torch
- openai
- boto3
- anthropic
- python-dotenv
- google-genai
- pillow

## Building and Distribution

To build the package:

```bash
pip install build
python -m build
```

This will create both source distribution (.tar.gz) and wheel (.whl) packages in the `dist/` directory.

To upload to PyPI (requires PyPI credentials):

```bash
pip install twine
twine upload dist/*
```

## Notes

- API keys should be configured securely, not hardcoded in the source code.
  Please create a `.env` directory which contains your API keys.
- The module is designed to be used with the vLLM framework for local models.