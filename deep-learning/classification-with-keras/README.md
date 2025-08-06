# ✍️ Handwritten digit classification with keras

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![Keras](https://img.shields.io/badge/Keras-used-d00000.svg?logo=keras)
![TensorFlow](https://img.shields.io/badge/TensorFlow-used-ff6f00.svg?logo=tensorflow)
![Streamlit UI](https://img.shields.io/badge/User%20Interface-Streamlit-ff4b4b.svg?logo=streamlit)

</div>

## 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

# Overview

This project demonstrates how to perform image classification, specifically for handwritten digits, using TensorFlow and the MNIST (Modified National Institute of Standards and Technology) dataset of handwritten digits. The MNIST dataset consists of a collection of handwritten digits from 0 to 9.

---

# Project Structure

```
├── configs/
│   └── config.yaml                                                   # Configuration management
├── demo/
│   ├── streamlit/                                                    # Streamlit UI for deployment
│   │   ├── assets/                                                   # Logo assets
│   │   ├── main.py                                                   # Streamlit application
│   │   └── ...                                                       # Additional Streamlit files
├── docs/
│   └── streamlit-ui-handwritten-digit-classification.pdf             # UI screenshot
│   └── streamlit-ui-handwritten-digit-classification.png             # UI screenshot
│   └── swagger-ui-handwritten-digit-classification.pdf               # Swagger screenshot
│   └── swagger-ui-handwritten-digit-classification.png               # Swagger screenshot
├── notebooks/
│   └── register-model.ipynb                                          # Notebook for registering trained models to MLflow
│   └── run-workflow.ipynb                                            # Notebook for executing the pipeline using custom inputs and configurations
├── src/
│   └── utils.py                                                      # Utility functions for configuration and helpers
├── README.md                                                         # Project documentation
```

---

# Setup

### 0 ▪ Minimum Hardware Requirements

Ensure your environment meets the minimum compute requirements for smooth image classification performance:

- **RAM**: 16 GB
- **VRAM**: 4 GB
- **GPU**: NVIDIA GPU

### 1 ▪ Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### 2 ▪ Set Up a Workspace

- Choose **Deep Learning** as the base image.

### 3 ▪ Clone the Repository

1. Clone the GitHub repository:

   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. Ensure all files are available after workspace creation.

---

# Usage

### 1 ▪ Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
run-workflow.ipynb
```

This will:

- Load and preprocess the MNIST data
- Create the model architecture
- Train the model

### 2 ▪ Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
register-model.ipynb
```

This will:

- Logg Model to MLflow
- Fetch the Latest Model Version from MLflow
- Load the Model and Running Inference

### 3 ▪ Deploy the Handwritten digit classification with keras Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and **GPU** it's **not necessary**.
- Choose the workspace.
- Start the deployment.
- Note: This is a local deployment running on your machine. As a result, if API processing takes more than a few minutes, it may return a timeout error. If you need to work with inputs that require longer processing times, we recommend using the provided notebook in the project files instead of accessing the API via Swagger or the web app UI.

### 3 ▪ Swagger / raw API

Once deployed, access the **Swagger UI** via the Service URL.

Paste a payload like:

```
{
  "inputs": {
    "digit": [
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL 8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APAACzBVBJJwAO9dnp/wm8damu6Dw5dRjGf9IKw/+hkVPffCnWNJa7XVNV0Kxa1hErrNe/M2cnYqgElsAHpjkc1wlAODkV694W8c654t8M6n4TuvEctrrFw0cun3c0/lq+3AMJcDK5AyOeTkd+fPvGFn4gsvEtzF4m89tUG1ZJJjuMgUBVYN/EMKOe9YVXtK0bUtdvVs9LsZ7y4YgbIULYycZPoPc8V6lpfwh0/w7p66z8RdXj0y2z8llC4aWQ+mRn8lz9RXPfE3x1pvi46TYaPZTQadpMJghluWDSyrhQM9SMBe5Oc5NcBV7Tda1XRZJJNK1O8sXkG12tZ2iLD0JUjNQ3l9eahN517dT3MvTfNIXb16n6mq9Ff/2Q=="
    ]
  },
  "params": {}
}

```

And as response:

```
{
  "predictions": [
    9
  ]
}
```

### 4 ▪ Launch the Streamlit UI

1. To launch the Streamlit UI, follow the instructions in the README file located in the `demo/streamlit` folder.

2. Navigate to the shown URL and view the handwritten classification.

### Successful UI demo

- Streamlit
  ![Handwritten Digit Classification Streamlit UI](docs/streamlit-ui-handwritten-digit-classification.png)

---

# 🔄 ONNX Model Export

This project includes utilities to automatically convert your trained models to ONNX format during MLflow logging, making them ready for deployment on inference servers like Triton.

## How It Works

### 1. ModelExportConfig Class

Use the `ModelExportConfig` class to configure how each model should be exported:

```python
from src.onnx_utils import ModelExportConfig

# Configure your model for ONNX export
config = ModelExportConfig(
    model_path="my_model.keras",           # Path to your trained model
    model_name="mnist_classifier",         # Name for the exported model
    input_shape=(1, 28, 28, 1),           # Input shape (required for Keras/TensorFlow)
)
```

**Key Parameters:**
- `model_path`: Path to your model file (`.keras`, `.nemo`)
- `model_name`: Name that will be used for the ONNX file
- `input_shape`: Required for TensorFlow/Keras models (use `input_sample` for PyTorch/NeMo)

### 2. Export Models During MLflow Logging

Pass a list of `ModelExportConfig` objects to the custom `log_model` function:

```python
from src.onnx_utils import log_model, ModelExportConfig

# Configure models for export
models_to_export = [
    ModelExportConfig(
        model_path="mnist_model.keras",
        model_name="mnist_classifier",
        input_shape=(1, 28, 28, 1)
    )
]

# Log model with automatic ONNX conversion
log_model(
    artifact_path="mnist_pipeline",
    python_model=YourMLflowModelClass(),
    models_to_convert_onnx=models_to_export  # This triggers ONNX conversion
)
```

### 3. What Gets Created

**Direct ONNX File**
```
MLflow Artifacts:
├── mnist_classifier.onnx       # ONNX file
```

### 4. Multiple Models

You can export multiple models at once:

```python
models_to_export = [
    ModelExportConfig(
        model_path="encoder.keras",
        model_name="image_encoder",
        input_shape=(1, 224, 224, 3)
    ),
    ModelExportConfig(
        model_path="classifier.keras", 
        model_name="digit_classifier",
        input_shape=(1, 128)
    )
]

log_model(
    artifact_path="multi_model_pipeline",
    python_model=YourPipelineModel(),
    models_to_convert_onnx=models_to_export
)
```

### 5. Skip ONNX Export

If you don't want ONNX conversion, simply omit the `models_to_convert_onnx` parameter:

```python
# Regular MLflow logging without ONNX conversion
log_model(
    artifact_path="mnist_pipeline",
    python_model=YourMLflowModelClass()
    # No models_to_convert_onnx = automatic ONNX conversion is skipped
)
```

### Supported Model Types

- **TensorFlow/Keras**: `.keras`, `.h5`, SavedModel directories
- **NeMo**: `.nemo` files (requires `input_sample`)
- **Hugging Face**: Model identifiers or local paths (`translation only`)

# Contact and Support

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).