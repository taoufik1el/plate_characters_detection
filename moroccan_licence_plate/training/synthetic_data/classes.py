import random

from pathlib import Path

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig

from moroccan_licence_plate.training.synthetic_data.utils import (
    generate_from_font,
    generate_from_crop,
)

FONTS = "fonts"
CROPS = "crops"
DATA = Path("data")


class OcrObject:
    def __init__(
        self,
        character: str,
        path_info: dict[str, list[str]],
        label_id: int | None = None,
    ):
        self.character = character
        self.font_paths = [
            DATA / FONTS / font_path for font_path in path_info.get("font_paths", [])
        ]
        self.crop_paths = [
            DATA / CROPS / crop_path for crop_path in path_info.get("crop_paths", [])
        ]
        self.weight: float = path_info.get("weight", 1.0)
        self.label_id = label_id
        self.font_proba: float = 0.0
        self.crop_proba: float = 0.0
        if self.font_paths and self.crop_paths:
            self.font_proba = 0.5
            self.crop_proba = 0.5
        if self.font_paths and not self.crop_paths:
            self.font_proba = 1.0
        elif not self.font_paths and self.crop_paths:
            self.crop_proba = 1.0

    def generate_random_transparent_crop(self):
        source = random.choices(
            [FONTS, CROPS], [self.font_proba, self.crop_proba], k=1
        )[0]
        if source == FONTS:
            font: Path = random.choice(self.font_paths)
            font_path = random.choice(list(font.rglob("*.ttf")))
            return generate_from_font(font_path, self.character)
        crop_path: Path = random.choice(self.crop_paths)
        crop_file = random.choice(list(crop_path.iterdir()))
        return generate_from_crop(crop_file)


class OcrObjects:
    def __init__(self, ocr_objects: list[OcrObject]):
        self.ocr_objects = ocr_objects

    @classmethod
    def from_json(cls, character_infos: DictConfig):
        sorted_characters = sorted(character_infos.keys())
        ocr_objects = []
        for label_id, char in enumerate(sorted_characters):
            ocr_objects.append(
                OcrObject(char, character_infos[char], label_id=label_id)
            )
        return cls(ocr_objects)

    def choose_ocr_object(self):
        return random.choices(
            self.ocr_objects, [obj.weight for obj in self.ocr_objects]
        )[0]

    @property
    def all_labels(self) -> list[int]:
        return [obj.label_id for obj in self.ocr_objects]

    @property
    def num_classes(self):
        return len(self.all_labels)


class ImageGenerator:
    def __init__(self, images_path: Path):
        self.image_paths = list(images_path.iterdir())

    def generate_image(self) -> npt.NDArray[np.uint8]:
        image_path = random.choice(self.image_paths)
        return generate_from_crop(image_path)


class Bbox:
    def __init__(
        self,
        x_min,
        y_min,
        height,
        width,
        safe_coordinates: tuple[int, int, int, int] | None = None,
    ):
        self.x_min = max(0, x_min)
        self.y_min = max(0, y_min)
        self.height = height
        self.width = width
        if safe_coordinates:
            safe_xmin, safe_ymin, safe_xmax, safe_ymax = safe_coordinates
            self.x_min = max(self.x_min, safe_xmin)
            self.y_min = max(self.y_min, safe_ymin)
            self.height = min(self.y_min + self.height, safe_ymax) - self.y_min
            self.width = min(self.x_min + self.width, safe_xmax) - self.x_min

    @property
    def x_max(self):
        return self.x_min + self.width

    @property
    def y_max(self):
        return self.y_min + self.height

    @property
    def coordinates(self):
        return self.x_min, self.x_max, self.y_min, self.y_max

    @property
    def centroid(self):
        return (self.x_max + self.x_min) // 2, (self.y_max + self.y_min) // 2


class CharacterAnnotation:
    def __init__(self, bbox: Bbox, ocr_object: OcrObject, alpha: int, sigma: int):
        self.bbox = bbox
        self.ocr_object = ocr_object
        self.alpha = alpha
        self.sigma = sigma
