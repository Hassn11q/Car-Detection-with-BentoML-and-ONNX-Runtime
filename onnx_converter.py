import torch
import torchvision.models as models
import torch.nn as nn

# Model and ONNX file paths
model_path = "/Users/hassn-/Desktop/Car-detection/models/car_classifier.pth"
onnx_path = "/Users/hassn-/Desktop/Car-detection/models/car_classifier.onnx"

# Load the model
def load_car_classifier(model_path, device):
    car_classifier = models.resnet18(pretrained=True)
    num_ftrs_car = car_classifier.fc.in_features
    car_classifier.fc = nn.Linear(num_ftrs_car, 2) 
    car_classifier.load_state_dict(torch.load(model_path, map_location=device))
    car_classifier = car_classifier.to(device)
    car_classifier.eval()
    return car_classifier

# Set the device to CPU
device = torch.device("cpu")

car_model = load_car_classifier(model_path, device)

# Dummy input tensor with the same size as the input to the model
model_input = torch.randn(1, 3, 224, 224).to(device)

# Convert the model to ONNX
torch.onnx.export(
    car_model,
    model_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

print(f"Model has been converted to ONNX format and saved to {onnx_path}")
