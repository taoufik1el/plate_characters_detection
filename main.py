"""Retrain the YOLO model for your own dataset."""
import os

import hydra
import numpy as np
import tensorflow.keras.backend as K
from omegaconf import DictConfig
from tensorflow.keras.layers import Input

from model.model import yolo_body, tiny_yolo_body, define_loss
from model.utils import DataGenerator, train_model


def get_anchors(anchors_line):
    anchors = [float(x) for x in anchors_line.split(",")]
    return np.array(anchors).reshape(-1, 2)


def create_model(anchors, num_classes):
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 1))
    num_anchors = len(anchors)

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print(f"Create YOLOv3 model with {num_anchors} anchors and {num_classes} classes.")

    return model_body


def create_tiny_model(anchors, num_classes):
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 1))
    num_anchors = len(anchors)

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print(
        f"Create Tiny YOLOv3 model with {num_anchors} anchors and {num_classes} classes."
    )

    return model_body


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(cfg: DictConfig):
    anchors = np.array(cfg.anchors)

    input_shape = (cfg.input_shape.height, cfg.input_shape.width)
    is_tiny_version = len(anchors) == 6

    train_data_generator = DataGenerator(
        n_batches=cfg.n_batches,
        anchors=anchors,
        batch_size=cfg.batch_size,
        input_shape=input_shape,
        shuffle=True,
        **cfg.data,
    )
    test_data_generator = DataGenerator(
        n_batches=cfg.n_batches // 10,
        anchors=anchors,
        batch_size=cfg.batch_size,
        input_shape=input_shape,
        shuffle=True,
        **cfg.data,
    )
    num_classes = train_data_generator.num_classes
    if is_tiny_version:
        model = create_tiny_model(anchors, num_classes)
    else:
        model = create_model(anchors, num_classes)  # make sure you know what you freeze

    loss_object = define_loss(anchors, num_classes, ignore_thresh=0.5)

    train_model(
        model,
        cfg.epochs,
        cfg.optimizer.lr,
        loss_object,
        train_data_generator,
        test_data_generator,
        cfg.optimizer.reduce_lr,
        cfg.optimizer.early_stopping,
    )
    if cfg.save_model.save:
        model.save_weights(os.path.join(cfg.save_model.save_path, "model"))
    return model


if __name__ == "__main__":
    model = train()
