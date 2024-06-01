from typing import List, Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf
from onnxruntime import InferenceSession

from synthetic_data.classes import Bbox
from yolo3.model import yolo_eval
import numpy.typing as npt


class ImageDetail:
    def __init__(self, image_shape: Tuple[int, ...], image_raw: npt.NDArray[np.uint8]):
        self.image_raw = image_raw
        self.image_shape = image_shape


class OutputBox:
    def __init__(self, box: List[float], score: float, character: str):
        self.bbox = Bbox(round(box[1]), round(box[0]), round(box[2] - box[0]), round(box[3] - box[1]))
        self.score = float(score)
        self.character = character


def preprocess_image_for_plate_detection(image, input_shape=(640, 640), normalise_factor=255.0):
    processed_image = cv2.resize(image.copy(), input_shape)
    processed_image = processed_image.astype(np.float32) / normalise_factor
    processed_image = np.transpose(processed_image, (2, 0, 1))  # Change data layout from HWC to CHW
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    return processed_image


def get_crop(raw_box: List[float],
             img_detail: ImageDetail,
             crop_size: Tuple[int, ...] = (416, 128),
             normalise_factor: float = 300.0):
    x, y, w, h = raw_box
    image_height, image_weight = img_detail.image_shape
    x, w = round(x * image_weight / 640), round(w * image_weight / 640)
    y, h = round(y * image_height / 640), round(h * image_height / 640)
    y_offset = int(h/2 + 0.3*h)
    x_offset = int(w/2 + 0.4*h)
    crop = img_detail.image_raw[y - y_offset:y + y_offset, x - x_offset:x + x_offset].copy()
    crop = cv2.resize(crop, crop_size)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) / normalise_factor
    return crop


def get_structured_output(boxes_tensor: tf.Tensor, scores_tensor: tf.Tensor, classes_tensor: tf.Tensor,
                          id_to_label: Dict[int, str]):
    bboxes = []
    for box_tensor, score_tensor, classe_tensor in zip(boxes_tensor.numpy(), scores_tensor.numpy(),
                                                       classes_tensor.numpy()):
        bboxes.append(
            OutputBox(box_tensor, score_tensor, id_to_label[int(classe_tensor)])
        )
    return bboxes


def construct_lines(bboxes: List[OutputBox], fraction: float = 1.0):
    first_box = min(bboxes, key=lambda x: x.bbox.y_min)
    h = first_box.bbox.height
    y_range = (first_box.bbox.centroid[1] - (h * fraction) // 2, first_box.bbox.centroid[1] + (h * fraction) // 2)
    first_aligned_boxes = sorted([box for box in bboxes if y_range[0] <= box.bbox.centroid[1] <= y_range[1]],
                                 key=lambda x: x.bbox.x_min)
    string = " ".join([box.character for box in first_aligned_boxes])
    box_coordinates = [box.bbox.coordinates for box in first_aligned_boxes]
    return string, box_coordinates


def get_plate(img, plate_finder):
    img = cv2.resize(img, (832, 832))
    (c, s, b) = plate_finder.detect(img)
    n = np.argmax(s)
    img = img[b[n, 1] :b[n, 1],
          b[n, 0]:b[n, 0]]
    return cv2.resize(img, (416, 128))


def prediction_pipeline(model: InferenceSession, plate_finder: InferenceSession, img, metadata):
    plate_img = get_plate(img, plate_finder)
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) / 300  # convet img to grayscale
    yolo_outputs = model(np.array([plate_img, plate_img]))
    boxes_tensor, scores_tensor, classes_tensor = yolo_eval(yolo_outputs,
                                                            np.array(metadata["anchors"]),
                                                            metadata["num_classes"],
                                                            (metadata["input_height"], metadata["input_width"]),
                                                            max_boxes=12,
                                                            score_threshold=.5,
                                                            iou_threshold=.5
                                                            )
    bboxes = get_structured_output(boxes_tensor, scores_tensor, classes_tensor, metadata["id_to_label"])
    plate_img = plate_img
    for box in bboxes:
        plate_img = cv2.rectangle(plate_img, (box.bbox.x_min, box.bbox.y_min), (box.bbox.x_max, box.bbox.y_max),
                                  (255, 0, 0), 2)
    cv2.imshow("show", plate_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    string, box_coordinates = construct_lines(bboxes)
    return string


