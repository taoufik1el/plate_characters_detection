import random

import cv2
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import ImageFont, Image, ImageDraw


def generate_from_crop(crop_path: Path) -> npt.NDArray[np.uint8]:
    img = cv2.imread(str(crop_path))
    return img


def generate_from_font(font_path: Path, character: str) -> npt.NDArray[np.uint8]:
    frame = np.ones((500, 1000, 3), dtype=np.uint8) * 255
    font = ImageFont.truetype(str(font_path), 150)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((500, 250), character, font=font, fill=(0, 0, 0), anchor="mm")
    img = np.array(img_pil)
    img_thresh = cv2.threshold(img[:, :, 0], 20, 255, cv2.THRESH_BINARY)[1]
    img_binary = cv2.bitwise_not(img_thresh)
    x1, y1, w, h = cv2.boundingRect(img_binary)
    safe_y = max(0, y1 - 3)
    safe_x = max(0, x1 - 3)
    return img[safe_y : y1 + h + 3, safe_x : x1 + w + 3]


def get_random_bool_choice(true_weight: float, false_weight: float):
    return random.choices([True, False], [true_weight, false_weight])[0]


def bool_img(img: npt.NDArray[np.any]) -> npt.NDArray[np.any]:
    return img / 255


def negative_bool_img(img: npt.NDArray[np.any]) -> npt.NDArray[np.any]:
    return 1 - bool_img(img)


def black_area_intersection(
    img1: npt.NDArray[np.any], img2: npt.NDArray[np.any]
) -> npt.NDArray[np.any]:
    assert img1.shape == img2.shape, "img1 and img2 must have the same shape."
    return negative_bool_img(img1) * negative_bool_img(img2)


def write_bool_on_background(
    img: npt.NDArray[np.any], background: npt.NDArray[np.any]
) -> npt.NDArray[np.any]:
    assert img.shape == background.shape, (
        f"img and background must have the same shape,"
        f" get img.shape={img.shape} and background.shape={background.shape}"
    )
    return background * bool_img(img)


def write_negative_bool_on_image(
    img: npt.NDArray[np.any], background: npt.NDArray[np.any]
) -> npt.NDArray[np.any]:
    assert img.shape == background.shape, "img and background must have the same shape"
    return background * negative_bool_img(img)
