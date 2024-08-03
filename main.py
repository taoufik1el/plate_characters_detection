from typing import List, Dict, Any

import cv2
import numpy as np
import yaml
import onnxruntime as ort
import numpy.typing as npt

from plate_characters_detection.inference_utils import get_image_details, get_raw_batch_plate_detection, get_plate_crops, \
    get_raw_character_detection, apply_batch_nms, apply_batch_construct_lines

with open("backup/yolo_model/metadata.yaml", "r") as yaml_file:
    metadata = yaml.safe_load(yaml_file)


def prediction_pipeline(
        character_detector: ort.InferenceSession,
        plate_detector: ort.InferenceSession,
        images: List[npt.NDArray[np.uint8]],
        metadata: Dict[str, Any],
) -> List[str]:
    image_details = get_image_details(images)
    raw_boxes, raw_scores, image_indexes = get_raw_batch_plate_detection(plate_detector, image_details)
    plate_crops, detection_scores = get_plate_crops(raw_boxes, raw_scores, image_details, image_indexes)
    raw_tensor_output = get_raw_character_detection(character_detector, plate_crops)
    processed_outputs = apply_batch_nms(raw_tensor_output, metadata)
    constructed_lines = apply_batch_construct_lines(processed_outputs)
    return constructed_lines


plate_detector = ort.InferenceSession("/home/taoufik/Personalspace/yolov8/onnx_folder/best_nms_extended.onnx")
character_detector = ort.InferenceSession("/home/taoufik/Personalspace/yolov8/yolo_v3_onnx_model.onnx")
img_paths = [
    # "/home/taoufik/Desktop/452878733_450489637801547_5225122658314717743_n.jpg",
    "/home/taoufik/Personalspace/plate_characters_detection/images/car.jpg",
    # "/home/taoufik/Desktop/00239aec49e6069a2feee464179ee57e.jpg",
    # "/home/taoufik/Desktop/fisker-maroc.jpg",
    # "/home/taoufik/Desktop/Iconic image for city streets.jpg",
    # "/home/taoufik/Desktop/multiple_images.png"
]
batch_img = [cv2.imread(fi) for fi in img_paths]
string = prediction_pipeline(character_detector, plate_detector, batch_img, metadata)
print(string)
