"""
Retrain the YOLO model for your own dataset.
"""
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input

from yolo3.model import yolo_body, tiny_yolo_body, define_loss
from yolo3.utils import DataGenerator, train_model


def get_anchors(anchors_line):
    anchors = [float(x) for x in anchors_line.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(anchors, num_classes):
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 1))
    num_anchors = len(anchors)

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    return model_body


def create_tiny_model(anchors, num_classes):
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 1))
    num_anchors = len(anchors)

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    return model_body


def train(images: np.array, df: pd.DataFrame):
    anchors = np.array([[59, 23], [74, 28], [79, 47], [89, 22], [96, 32], [118, 58], [124, 39], [125, 22], [58, 36]])

    num_classes = 22

    input_shape = (128, 416)  # multiple of 32, hw
    is_tiny_version = len(anchors) == 6  # default setting

    if is_tiny_version:
        model = create_tiny_model(anchors, num_classes)
    else:
        model = create_model(anchors, num_classes)  # make sure you know what you freeze

    reduce_lr = {'factor': 0.5, 'patience': 3}
    early_stopping = {'patience': 5}

    val_split = 0.5
    annotation_indexes = [i for i in range(len(df))]
    np.random.seed(1024)
    np.random.shuffle(annotation_indexes)
    np.random.seed(None)
    num_val = int(len(annotation_indexes) * val_split)
    train_indexes = annotation_indexes[num_val:]
    test_indexes = annotation_indexes[:num_val]

    train_data_generator = DataGenerator(list_IDs=train_indexes, images=images, df=df, anchors=anchors,
                                         num_classes=num_classes,
                                         batch_size=32, input_shape=input_shape, shuffle=True)
    test_data_generator = DataGenerator(list_IDs=test_indexes, images=images, df=df, anchors=anchors,
                                        num_classes=num_classes,
                                        batch_size=32, input_shape=input_shape, shuffle=True)

    loss_object = define_loss(anchors, num_classes, ignore_thresh=.5)

    # train_model(model,5, 0.001, loss_object, train_data_generator, test_data_generator, reduce_lr, early_stopping)

    # for i in range(len(model.layers)):
    #    model.layers[i].trainable = True
    # model.compile(optimizer=Adam(lr=1e-3))

    train_model(model, 75, 0.001, loss_object, train_data_generator, test_data_generator, reduce_lr, early_stopping)

    return model


if __name__ == '__main__':
    df = pd.read_csv('fileeeeeeee')
    image = pd.load('fileeeee')
    model = train(image, df)
    model.save_weights("yolo_chars")
