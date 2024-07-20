# Car Detection Model 

## Overview

This project provides a service to detect if an image contains a car or not. The service is built using BentoML and ONNX runtime.


## Project Structure

- `bentofile.yaml`: BentoML configuration file.
- `model_converter.py`: Script for converting and saving the model.
- `models/`: Directory containing the ONNX models.
- `onnx_converter.py`: Script for handling ONNX model conversion.
- `requirements.txt`: List of dependencies required for the project.
- `service.py`: Main service file that defines the BentoML service and its API.

## Setup

### Prerequisites

- Python 3.8 or higher
- `pip` for managing Python packages

### Installation

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Ensure you have the ONNX models in the `models/` directory.**

## Configuration

The service is configured using the `bentofile.yaml` file:

```yaml
service: "service:svc"
labels:
  owner: Hassn
  stage: demo
include:
  - "*.py"
python:
  requirements_txt: "./requirements.txt"
```

## Usage
Running the Service
Start the BentoML service:
```
bentoml serve service.py:svc
```
Send a prediction request:

Use an HTTP client like curl or Postman to send an image to the service:

```
curl -X POST -H "Content-Type: application/json" \
    --data-binary "@path_to_your_image_file" \
    http://127.0.0.1:5000/predict_image
```