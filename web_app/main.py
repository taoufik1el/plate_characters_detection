import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import io
import logging

from pydantic.types import conlist

logging.basicConfig(level=logging.DEBUG)
app = FastAPI()


# Define Pydantic models for request and response
class ImageData(BaseModel):
    images: List[List[List[conlist(int, min_items=3, max_items=3)]]]


class PredictionResponse(BaseModel):
    results: List[str]


@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html") as f:
        return f.read()


@app.post("/upload/", response_model=PredictionResponse)
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    image_np = np.array(image)
    logging.debug(f"Image shape: {image_np.shape}")
    logging.debug(f"Image dtype: {image_np.dtype}")

    # Convert to list for JSON serialization
    image_list = image_np.tolist()
    data = {"images": [image_list]}

    try:
        response = requests.post("http://ml_project:8001/predict/", json=data)
        logging.debug(f"Response status code: {response.status_code}")
        logging.debug(f"Response content: {response.text}")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to inference service: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Error connecting to inference service"})

    result = response.json()
    return PredictionResponse(results=result["results"])
