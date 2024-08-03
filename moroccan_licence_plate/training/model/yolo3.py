from functools import wraps

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import (
    Conv2D,
    Add,
    ZeroPadding2D,
    UpSampling2D,
    Concatenate,
    MaxPooling2D,
)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from moroccan_licence_plate.training.model.utils import compose


@wraps(Conv2D)
def darknet_conv2_d(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {
        "kernel_regularizer": l2(5e-4),
        "padding": "valid" if kwargs.get("strides") == (2, 2) else "same",
    }
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def darknet_conv2_d_bn_leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {"use_bias": False}
    no_bias_kwargs.update(kwargs)
    return compose(
        darknet_conv2_d(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
    )


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = darknet_conv2_d_bn_leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            darknet_conv2_d_bn_leaky(num_filters // 2, (1, 1)),
            darknet_conv2_d_bn_leaky(num_filters, (3, 3)),
        )(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """Darknent body having 52 Convolution2D layers"""
    x = darknet_conv2_d_bn_leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(
        darknet_conv2_d_bn_leaky(num_filters, (1, 1)),
        darknet_conv2_d_bn_leaky(num_filters * 2, (3, 3)),
        darknet_conv2_d_bn_leaky(num_filters, (1, 1)),
        darknet_conv2_d_bn_leaky(num_filters * 2, (3, 3)),
        darknet_conv2_d_bn_leaky(num_filters, (1, 1)),
    )(x)
    y = compose(
        darknet_conv2_d_bn_leaky(num_filters * 2, (3, 3)),
        darknet_conv2_d(out_filters, (1, 1)),
    )(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(darknet_conv2_d_bn_leaky(256, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(darknet_conv2_d_bn_leaky(128, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    """Create Tiny YOLO_v3 model CNN body in keras."""
    x1 = compose(
        darknet_conv2_d_bn_leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        darknet_conv2_d_bn_leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        darknet_conv2_d_bn_leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        darknet_conv2_d_bn_leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        darknet_conv2_d_bn_leaky(256, (3, 3)),
    )(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        darknet_conv2_d_bn_leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        darknet_conv2_d_bn_leaky(1024, (3, 3)),
        darknet_conv2_d_bn_leaky(256, (1, 1)),
    )(x1)
    y1 = compose(
        darknet_conv2_d_bn_leaky(512, (3, 3)),
        darknet_conv2_d(num_anchors * (num_classes + 5), (1, 1)),
    )(x2)

    x2 = compose(darknet_conv2_d_bn_leaky(128, (1, 1)), UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        darknet_conv2_d_bn_leaky(256, (3, 3)),
        darknet_conv2_d(num_anchors * (num_classes + 5), (1, 1)),
    )([x2, x1])

    return Model(inputs, [y1, y2])


# ________________________________________________________________________________________________________________________________
# ________________________________________________________________________________________________________________________________


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(
        K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1],
    )
    grid_x = K.tile(
        K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1],
    )
    grid = K.concatenate([grid_y, grid_x])
    grid = K.cast(grid, K.dtype(feats))
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5]
    )
    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape, K.dtype(feats))
    box_wh = (
        K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape, K.dtype(feats))
    )
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes"""
    box_yx = box_xy
    box_hw = box_wh
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2.0 / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.0)
    box_maxes = box_yx + (box_hw / 2.0)
    boxes = K.concatenate(
        [
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2],  # x_max
        ]
    )

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(
        feats, anchors, num_classes, input_shape
    )
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(
    yolo_outputs,
    anchors,
    num_classes,
    image_shape,
    max_boxes=12,
    score_threshold=0.6,  # 0.5
    iou_threshold=0.5,
):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = (
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    )  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(
            yolo_outputs[l],
            anchors[anchor_mask[l]],
            num_classes,
            input_shape,
            image_shape,
        )
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype="int32")
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold
        )
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, "int32") * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def box_iou(b1, b2):
    """Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.0
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.0
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def define_loss(anchors, num_classes, ignore_thresh=0.5):
    def yolo_loss(yolo_outputs, y_true):
        num_layers = len(anchors) // 3  # default setting

        anchor_mask = (
            [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            if num_layers == 3
            else [[3, 4, 5], [1, 2, 3]]
        )
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
        grid_shapes = [
            K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0]))
            for l in range(num_layers)
        ]
        loss = 0
        m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
        mf = K.cast(m, K.dtype(yolo_outputs[0]))

        for l in range(num_layers):
            object_mask = y_true[l][..., 4:5]
            true_class_probs = y_true[l][..., 5:]

            grid, raw_pred, pred_xy, pred_wh = yolo_head(
                yolo_outputs[l],
                anchors[anchor_mask[l]],
                num_classes,
                input_shape,
                calc_loss=True,
            )
            pred_box = K.concatenate([pred_xy, pred_wh])

            # Darknet raw box to calculate loss.
            raw_true_xy = y_true[l][..., :2] * grid_shapes[l] - grid
            raw_true_wh = K.log(
                y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape
            )
            raw_true_wh = K.switch(
                object_mask, raw_true_wh, K.zeros_like(raw_true_wh)
            )  # avoid log(0)=-inf
            box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

            # Find ignore mask, iterate over each of batch.
            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = K.cast(object_mask, "bool")

            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(
                    y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0]
                )
                iou = box_iou(pred_box[b], true_box)
                best_iou = K.max(iou, axis=-1)
                ignore_mask = ignore_mask.write(
                    b, K.cast(best_iou < ignore_thresh, K.dtype(true_box))
                )
                return b + 1, ignore_mask

            _, ignore_mask = tf.while_loop(
                lambda b, *args: b < m, loop_body, [0, ignore_mask]
            )
            ignore_mask = ignore_mask.stack()
            ignore_mask = K.expand_dims(ignore_mask, -1)

            xy_loss = (
                object_mask
                * box_loss_scale
                * K.binary_crossentropy(
                    raw_true_xy, raw_pred[..., 0:2], from_logits=True
                )
            )
            wh_loss = (
                object_mask
                * box_loss_scale
                * 0.5
                * K.square(raw_true_wh - raw_pred[..., 2:4])
            )
            confidence_loss = (
                object_mask
                * K.binary_crossentropy(
                    object_mask, raw_pred[..., 4:5], from_logits=True
                )
                + (1 - object_mask)
                * K.binary_crossentropy(
                    object_mask, raw_pred[..., 4:5], from_logits=True
                )
                * ignore_mask
            )
            class_loss = object_mask * K.binary_crossentropy(
                true_class_probs, raw_pred[..., 5:], from_logits=True
            )

            xy_loss = K.sum(xy_loss) / mf
            wh_loss = K.sum(wh_loss) / mf
            confidence_loss = K.sum(confidence_loss) / mf
            class_loss = K.sum(class_loss) / mf
            loss += xy_loss + wh_loss + 2 * confidence_loss + class_loss

        return loss

    return yolo_loss
