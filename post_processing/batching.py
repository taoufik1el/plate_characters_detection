from typing import List

import numpy as np
import numpy.typing as npt

from model.yolo3 import yolo_eval
from post_processing.tools import preprocess_image_for_plate_detection, ImageDetail, get_crop, get_structured_output
import tensorflow as tf

def get_image_details(images: List[npt.NDArray[np.uint8]]) -> List[ImageDetail]:
    return [ImageDetail(img.shape[:2], img) for img in images]


def get_raw_batch_plate_detection(plate_detector, image_details: List[ImageDetail]):
    input_images = [preprocess_image_for_plate_detection(image_detail.image_raw) for image_detail in image_details]
    input_tensor = np.vstack(input_images)
    raw_boxes, raw_scores, _, _, image_indexes = plate_detector.run(None, {"images": input_tensor})
    return raw_boxes, raw_scores, image_indexes


def get_plate_crops(raw_boxes: npt.NDArray[np.float64], raw_scores: npt.NDArray[np.float32], image_details: List[ImageDetail], image_indexes):
    duplicated_images = [image_details[_id] for _id in image_indexes.tolist()]
    crops = []
    for raw_box, img_detail in zip(raw_boxes[0].tolist(), duplicated_images):
        crops.append(get_crop(raw_box, img_detail))
    return np.array(crops), raw_scores[0].tolist()


def get_raw_character_detection(character_detector, plate_crops: List[npt.NDArray[np.float64]]):
    boxes_tensor, scores_tensor, classes_tensor = character_detector.run(["conv2d_58", "conv2d_66", "conv2d_74"],
                                                                 {"image_input": plate_crops})
    return boxes_tensor, scores_tensor, classes_tensor

def apply_batch_nms(raw_tensor_output, metadata):
    bvl_boxes = []
    for boxes, scores, classes in zip(raw_tensor_output[0], raw_tensor_output[1], raw_tensor_output[2]):
        boxes_tensor, scores_tensor, classes_tensor = yolo_eval([
            tf.convert_to_tensor(np.expand_dims(boxes, axis=0)),
            tf.convert_to_tensor(np.expand_dims(scores, axis=0)),
            tf.convert_to_tensor(np.expand_dims(classes, axis=0))
        ],
                                                                np.array(metadata["anchors"]),
                                                                metadata["num_classes"],
                                                                (metadata["input_height"], metadata["input_width"]),
                                                                max_boxes=12,
                                                                score_threshold=.5,
                                                                iou_threshold=.5
                                                                )
        bboxes = get_structured_output(boxes_tensor, scores_tensor, classes_tensor, metadata["id_to_label"])
        bvl_boxes.append(bboxes)
    return bvl_boxes


# TensorShape([1, 4, 13, 81])


