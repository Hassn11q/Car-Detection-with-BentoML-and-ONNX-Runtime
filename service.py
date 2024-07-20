# Dependencies
import bentoml
from bentoml.io import JSON
from bentoml.io import Image 
import numpy as np 
import torch.nn.functional as F
import torch 


# Define preprocessing function 
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224,224))
    img = np.array(img).astype(np.float32)
    img = img / 255.0  
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


# Load the ONNX Model 
runner = bentoml.onnx.get("car_classifier:5s335zb33o3i4twx").to_runner()

# Define the service
svc = bentoml.Service("car_classifier", runners=[runner])


# Define the API endpoint
@svc.api(input=Image(), output=JSON())
async def predict(img: Image):
    # Preprocess the input image
    arr = preprocess_image(img)
    # Make predictions
    preds = await runner.async_run(arr)
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(torch.tensor(preds), dim=1).numpy()
    car_prob = probabilities[0, 0]  
    
    # Determine if the image is classified as a car
    threshold = 0.60
    is_car = car_prob >= threshold
    # Return the result
    return {
        "is_car": bool(is_car),
        "probability": float(car_prob)
    }
