import bentoml
import onnx

# Path to the ONNX model file
car_model_path = "/Users/hassn-/Desktop/Car-detection/models/car_classifier.onnx"

# Load the ONNX model
model = onnx.load(car_model_path)

# Save the ONNX model with BentoML
bentoml_model = bentoml.onnx.save_model("car_classifier", model)

print(f"Model has been saved with BentoML: {bentoml_model}")
