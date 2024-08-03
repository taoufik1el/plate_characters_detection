import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import logging
import onnxruntime as ort
from pydantic.types import conlist

from utils.batching import get_image_details, get_raw_batch_plate_detection, get_plate_crops, \
    get_raw_character_detection, apply_batch_nms, apply_batch_construct_lines

logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

plate_detector = ort.InferenceSession("best_nms_extended.onnx")
character_detector = ort.InferenceSession("yolo_v3_onnx_model.onnx")
with open("metadata.yaml", "r") as yaml_file:
    metadata = yaml.safe_load(yaml_file)


class ImageData(BaseModel):
    images: List[List[List[conlist(int, min_items=3, max_items=3)]]]


class PredictionResponse(BaseModel):
    results: List[str]


def prediction_pipeline(
        images: List[np.ndarray],
) -> List[str]:
    # Your implementation
    np_images = [np.array(img, dtype=np.uint8) for img in images]
    image_details = get_image_details(np_images)
    raw_boxes, raw_scores, image_indexes = get_raw_batch_plate_detection(plate_detector, image_details)
    plate_crops, detection_scores = get_plate_crops(raw_boxes, raw_scores, image_details, image_indexes)
    raw_tensor_output = get_raw_character_detection(character_detector, plate_crops)
    processed_outputs = apply_batch_nms(raw_tensor_output, metadata)
    constructed_lines = apply_batch_construct_lines(processed_outputs)
    return constructed_lines


@app.get("/")
async def root():
    return {"message": "Hello from inference service"}


@app.post("/predict/", response_model=PredictionResponse)
async def predict(image_data: ImageData):
    # logging.debug(f"Received data: {image_data}")
    # Convert lists back to numpy arrays
    images = [np.array(image) for image in image_data.images]

    results = prediction_pipeline(images)
    return PredictionResponse(results=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
