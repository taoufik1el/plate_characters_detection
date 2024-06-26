import random
from functools import reduce

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


def box_transform(b, matrix):
    P1 = np.matmul(matrix, np.array([b[1], b[0], 1]))
    P2 = np.matmul(matrix, np.array([b[1], b[2], 1]))
    P3 = np.matmul(matrix, np.array([b[3], b[0], 1]))
    P4 = np.matmul(matrix, np.array([b[3], b[2], 1]))
    x1, y1 = P1[:2] / P1[2]
    x2, y2 = P2[:2] / P2[2]
    x3, y3 = P3[:2] / P3[2]
    x4, y4 = P4[:2] / P4[2]

    nx_min, ny_min = min([x1, x2, x3, x4]), min([y1, y2, y3, y4])
    nx_max, ny_max = max([x1, x2, x3, x4]), max([y1, y2, y3, y4])

    return int(ny_min), int(nx_min), int(ny_max), int(nx_max)


def gauss_noise(image, mean=0, var=0.001):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.0
    else:
        low_clip = 0.0
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(images, df, annotation_index, input_shape, max_boxes=12):
    """random preprocessing for real-time data augmentation"""
    annotation_line = df[df.id == annotation_index]["annotations"][annotation_index]
    image = images[annotation_index]
    line = annotation_line.split()  # df.loc[l,'annotation']
    ih, iw = image.shape  # image.shape
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(",")))) for box in line[0:]])

    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = random.randrange(0, w - nw + 1)
    dy = random.randrange(0, h - nh + 1)

    image = cv2.resize(image, (nw, nh))
    new_image = np.ones((h, w)) * random.uniform(1, 255)
    new_image[dy : dy + nh, dx : dx + nw] = image

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box[:, [0, 2]] = box[:, [0, 2]] * scale + dy
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dx
        box_data[: len(box)] = box

    if random.choices([True, False], [4 / 5, 1 / 5])[0]:
        r = [rand(0, 30) for i in range(4)]
        R = [rand(0, 70) for i in range(4)]

        pts1 = np.float32([[dx, dy], [dx, dy + nh], [dx + nw, dy + nh], [dx + nw, dy]])
        pts2 = np.float32(
            [
                [dx + R[0], dy + r[0]],
                [dx + R[1], dy + nh - r[1]],
                [dx + nw - R[2], dy + nh - r[2]],
                [dx + nw - R[3], dy + r[3]],
            ]
        )

        # Apply Perspective Transform Algorithm
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        new_image = cv2.warpPerspective(new_image, matrix, (w, h))

        for i in range(len(box)):
            box_data[i, 0:4] = box_transform(box_data[i, 0:4], matrix)

    # Apply blure
    if random.choices([True, False], [4 / 5, 1 / 5])[0]:
        k = random.randrange(1, 8, 2)
        new_image = cv2.blur(new_image, (k, k), cv2.BORDER_DEFAULT)

    # Apply Gaussian Noise
    if random.choices([True, False], [4 / 5, 1 / 5])[0]:
        v = rand(0, 0.0015)
        new_image = gauss_noise(new_image, mean=0, var=v)

    image_data = np.array(new_image) / 255

    return image_data, box_data


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (
        true_boxes[..., 4] < num_classes
    ).all(), "class id must be less than num_classes"
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = (
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    )

    true_boxes = np.array(true_boxes, dtype="float32")
    input_shape = np.array(input_shape, dtype="int32")
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape
    true_boxes[..., 2:4] = boxes_wh / input_shape
    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [
        np.zeros(
            (
                m,
                grid_shapes[l][0],
                grid_shapes[l][1],
                len(anchor_mask[l]),
                5 + num_classes,
            ),
            dtype="float32",
        )
        for l in range(num_layers)
    ]
    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.0
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.0
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][0]).astype(
                        "int32"
                    )
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][1]).astype(
                        "int32"
                    )
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype("int32")
                    y_true[l][b, i, j, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, i, j, k, 4] = 1
                    y_true[l][b, i, j, k, 5 + c] = 1

    return y_true


class DataGenerator(Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        list_IDs,
        images,
        df,
        anchors,
        num_classes,
        batch_size=32,
        input_shape=(128, 416),
        shuffle=True,
    ):
        "Initialization"
        self.list_IDs = list_IDs
        self.images = images
        self.df = df
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        image_data = []
        box_data = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image, box = get_random_data(
                self.images, self.df, list_IDs_temp[i], self.input_shape
            )
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(
            box_data, self.input_shape, self.anchors, self.num_classes
        )

        return image_data, y_true


def train_model(
    model,
    EPOCHS,
    lr0,
    loss_object,
    train_data_generator,
    test_data_generator,
    reduce_lr,
    early_stopping,
):
    optimizer = tf.keras.optimizers.Adam(lr0)

    @tf.function
    def train_step(images, labels, lr):
        optimizer.lr.assign(lr)
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss = loss_object(
                [tf.convert_to_tensor(x) for x in output],
                [tf.convert_to_tensor(x) for x in labels],
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def test_step(images, labels):
        output = model(images, training=True)
        t_loss = loss_object(
            [tf.convert_to_tensor(x) for x in output],
            [tf.convert_to_tensor(x) for x in labels],
        )
        return t_loss

    best_score = 100000
    best_epoch = 1
    best_weights = model.get_weights()
    lr = lr0
    epoch = 1
    while epoch <= EPOCHS and epoch - best_epoch <= early_stopping["patience"]:
        print(f"Epoch {epoch}, ")
        if epoch - best_epoch >= reduce_lr["patience"]:
            lr = lr * reduce_lr["factor"]
            print("learning rate reduced to ", lr)
        train_loss, n_train = 0, 0
        test_loss, n_test = 0, 0
        for images, labels in train_data_generator:
            train_loss += train_step(images, labels, lr)
            n_train += 1

        for test_images, test_labels in test_data_generator:
            test_loss += test_step(test_images, test_labels)
            n_test += 1
        if test_loss / n_test < best_score:
            best_score = test_loss / n_test
            best_epoch = epoch
            best_weights = model.get_weights()

        epoch += 1

        print(f"Loss: {train_loss / n_train}, " f"Test Loss: {test_loss / n_test}, ")
    model.set_weights(best_weights)
