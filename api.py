from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import json

app = FastAPI()

# Function to load a specified model
def load_model(model_name):
    model = YOLO(f'models/{model_name}')
    return model

# Load shape identification model
shape_model = load_model('shape_best_v8.pt')

class ImageData(BaseModel):
    base64_image: str

def decode_base64_image(data: str) -> np.ndarray:
    """Decode a base64 image string into a numpy array."""
    try:
        # Decode base64 string to bytes
        img_bytes = base64.b64decode(data)
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        # Convert numpy array to image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")
        
@app.post("/base64image-to-json/")
async def detect_objects(image_data: ImageData):
    """Receive a base64 encoded image, decode it, detect objects using YOLOv8, and return results."""
    # Decode the image
    image = decode_base64_image(image_data.base64_image)
    
    # Perform object detection
    shape = json.loads(shape_model(image)[0].tojson())
    print(shape)
    
    return shape

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)