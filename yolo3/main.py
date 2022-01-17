"""
Retrain the YOLO model for your own dataset.
"""
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from utils import get_random_data


def get_anchors(anchors_line):
    anchors = [float(x) for x in anchors_line.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=2):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 1))
    num_anchors = len(anchors)

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    return model_body


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=2):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 1))
    num_anchors = len(anchors)

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    return model_body


def _main():
    anchor_line = '80,80, 80,50, 50,80, 50,50, 80,20, 20,80, 50,20, 20,50, 20,20'
    anchors = np.array([[59, 23], [74, 28], [79, 47], [89, 22], [96, 32], [118, 58], [124, 39], [125, 22], [58, 36]])
    # get_anchors(anchor_line)

    num_classes = 22

    input_shape = (128, 416)  # multiple of 32, hw
    yolo_loss = define_loss(anchors, num_classes, ignore_thresh=.5, print_loss=False)
    is_tiny_version = len(anchors) == 6  # default setting

    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2)
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2)  # make sure you know what you freeze

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

    loss_object = define_loss(anchors, num_classes, ignore_thresh=.5, print_loss=False)

    # train_model(model,5, 0.001, loss_object, train_data_generator, test_data_generator, reduce_lr, early_stopping)

    # for i in range(len(model.layers)):
    #    model.layers[i].trainable = True
    # model.compile(optimizer=Adam(lr=1e-3))

    train_model(model, 75, 0.001, loss_object, train_data_generator, test_data_generator, reduce_lr, early_stopping)

    return model


if __name__ == '__main__':
    model = _main()
    model.save_weights("yolo_chars")
