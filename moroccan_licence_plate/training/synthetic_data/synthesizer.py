import json
import random

import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from moroccan_licence_plate.training.synthetic_data.classes import (
    ImageGenerator,
    Bbox,
    OcrObjects,
    CharacterAnnotation,
)
from moroccan_licence_plate.training.synthetic_data.utils import (
    black_area_intersection,
    write_bool_on_background,
    write_negative_bool_on_image,
    get_random_bool_choice,
)


def draw_shape(
    crop: npt.NDArray[np.any],
    bbox_annotation: CharacterAnnotation,
    background: npt.NDArray[np.any],
    dirt_object: ImageGenerator,
):
    """draw a character in the background in a box given by its coordinates
    in xy and with a value alpha and variance sigma"""
    crop_h, crop_w = crop.shape[:2]
    bbox = bbox_annotation.bbox
    # resize image
    scale = min(bbox.width / crop_w, bbox.height / crop_h)
    new_crop_w = int(crop_w * scale)
    new_crop_h = int(crop_h * scale)
    x_offset = (bbox.width - new_crop_w) // 2
    y_offset = (bbox.height - new_crop_h) // 2
    scaled_crop = cv2.resize(crop, (new_crop_w, new_crop_h))
    white_background = np.ones((bbox.height, bbox.width, 3)) * 255
    white_background[
        y_offset : y_offset + new_crop_h, x_offset : x_offset + new_crop_w, :
    ] = scaled_crop
    if random.choices([True, False], [0.05, 0.95])[0]:
        dirt_img = cv2.resize(dirt_object.generate_image(), (bbox.width, bbox.height))
        white_background = (
            255 * black_area_intersection(white_background, dirt_img) + white_background
        )
    crop = write_bool_on_background(
        white_background, background[bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max]
    )
    crop = crop + write_negative_bool_on_image(
        white_background,
        np.random.normal(
            bbox_annotation.alpha, bbox_annotation.sigma, (bbox.height, bbox.width, 3)
        ),
    )
    background[bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max] = crop
    return background


def generate_aligned_boxes(
    ocr_objects: OcrObjects,
    false_characters: OcrObjects,
    background_shape: tuple[int, ...],
) -> tuple[list[CharacterAnnotation], list[CharacterAnnotation]]:
    """generate boxes with random dimensions and positions, and associating labels to them."""
    background_h, background_w = background_shape[:2]
    bbox_h = random.randrange(44, 60)
    bbox_w = int(bbox_h * 73 / 111)
    dy = int((background_h - bbox_h) / 2)
    dx = int((background_w - 11 * bbox_w) / 2)
    character_annotations = []
    false_chars = []
    for i in range(6):
        character_annotations.append(
            CharacterAnnotation(
                Bbox(dx + i * bbox_w, dy, bbox_h, bbox_w),
                ocr_objects.choose_ocr_object(),
                random.randrange(0, 70),
                random.randrange(5, 30),
            )
        )
    false_char = false_characters.choose_ocr_object()
    for e in [7, 9, 10]:
        character_annotations.append(
            CharacterAnnotation(
                Bbox(dx + e * bbox_w, dy, bbox_h, bbox_w),
                ocr_objects.choose_ocr_object(),
                random.randrange(0, 70),
                random.randrange(5, 30),
            )
        )
    r1 = random.randrange(dx + 6 * bbox_w, dx + 7 * bbox_w - int(5 * bbox_h / 12))
    r2 = random.randrange(dx + 8 * bbox_w, dx + 9 * bbox_w - int(5 * bbox_h / 12))
    for r in [r1, r2]:
        false_chars.append(
            CharacterAnnotation(
                Bbox(r, dy, bbox_h, int(5 * bbox_h / 12)),
                false_char,
                random.randrange(30, 60),
                random.randrange(5, 30),
            )
        )
    return character_annotations, false_chars


def generate_random_boxes(
    ocr_objects: OcrObjects,
    false_characters: OcrObjects,
    background_shape: tuple[int, ...],
    n: int,
) -> tuple[list[CharacterAnnotation], list[CharacterAnnotation]]:
    """generate boxes with random dimensions and positions, and associating labels to them."""
    background_h, background_w = background_shape[:2]
    character_annotations = []
    false_chars = []
    for i in range(n):
        h, w = random.randrange(50, 150), random.randrange(20, 150)
        bbox = Bbox(
            random.randrange(
                int(i * background_w / n), int((i + 1) * background_w / n) - 20
            ),
            random.randrange(20, background_h - 70),
            h,
            w,
            safe_coordinates=(0, 0, int((i + 1) * background_w / n), background_h - 20),
        )
        character_annotations.append(
            CharacterAnnotation(
                bbox,
                ocr_objects.choose_ocr_object(),
                random.randrange(0, 70),
                random.randrange(5, 30),
            )
        )
        if get_random_bool_choice(0.1, 0.9) and bbox.x_min - 20 > 3:
            false_chars.append(
                CharacterAnnotation(
                    Bbox(bbox.x_min - 20, bbox.y_min, bbox.height, 15),
                    false_characters.choose_ocr_object(),
                    random.randrange(0, 70),
                    random.randrange(5, 30),
                ),
            )
    return character_annotations, false_chars


def create_image_and_labels(
    ocr_objects: OcrObjects,
    false_characters: OcrObjects,
    backgrounds: ImageGenerator,
    dirt_object: ImageGenerator,
) -> tuple[npt.NDArray[np.any], list[CharacterAnnotation]]:
    background = backgrounds.generate_image()
    background = cv2.resize(background, (450, 170))
    if get_random_bool_choice(0.5, 0.5):
        character_annotations, false_chars = generate_aligned_boxes(
            ocr_objects, false_characters, background.shape
        )
    else:
        n = random.randrange(5, 12)
        character_annotations, false_chars = generate_random_boxes(
            ocr_objects, false_characters, background.shape, n
        )
    for false_character in false_chars:
        background = draw_shape(
            false_character.ocr_object.generate_random_transparent_crop(),
            false_character,
            background,
            dirt_object,
        )
    for annotation in character_annotations:
        background = draw_shape(
            annotation.ocr_object.generate_random_transparent_crop(),
            annotation,
            background,
            dirt_object,
        )
    return background, character_annotations


def generate(
    n,
    ocr_objects: OcrObjects,
    false_characters: OcrObjects,
    backgrounds: ImageGenerator,
    dirt_object: ImageGenerator,
    save=False,
):
    images = []
    annotation_dicts = []
    for image_id in tqdm(range(n), desc="Generating synthetic data ..."):
        image, image_labels = create_image_and_labels(
            ocr_objects, false_characters, backgrounds, dirt_object
        )
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        annotation_dicts.append(
            {
                "image_id": image_id,
                "bboxes": [
                    {
                        "label_id": label.ocr_object.label_id,
                        "coordinates": label.bbox.coordinates,
                    }
                    for label in image_labels
                ],
            }
        )
        print([lab.ocr_object.character for lab in image_labels])
        cv2.imshow("hakawaaa", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("  ")

    if save:
        np.save("images_numpy_v2", np.array(images))
        with open("labels.json", "w") as f:
            json.dump(annotation_dicts, f)
    else:
        return images, annotation_dicts
