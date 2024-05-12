import cv2
import yaml
import tensorflow as tf

from post_processing.tools import prediction_pipeline

net = cv2.dnn.readNet('plate_detection_models/yolov3_plates_final.weights',
                      'plate_detection_models/yolov3-license-plates.cfg')
plate_finder = cv2.dnn_DetectionModel(net)
plate_finder.setInputParams(size=(832, 832), scale=1 / 255)

with open("yolo_model/metadata.yaml", "r") as yaml_file:
    metadata = yaml.safe_load(yaml_file)

model = tf.keras.models.load_model("yolo_model/model")
img = cv2.imread("images/car.jpg")
string = prediction_pipeline(model, plate_finder, img, metadata)
print(string)
cv2.imshow(string, cv2.resize(img, (900, 900)))
cv2.waitKey(0)
cv2.destroyAllWindows()
