# CHARACTER DETECTION PIPLINE OF CAR LICENCE PLATES

In this repository you will find a method to recognize the numbers of a car plate in street view images.

If you want to build a license plate detection model in your country, but couldn't find a pre-trained model or an annotated dataset (bounding boxes on both plates and characters inside the plates) because you have characters in a different language (like Arabic, Chinese, Hindi...). This repository can be useful to you.

The used pipeline for character recognition from car images can be summarized in three steps:
- find the bounding box of the full licence plate in an image using a trained yolov3.
- find the bounding box of each character in the plate using a customized yolov3 trained on synthesized data.
- reconstruct the plate number based on the bounding box positions.

<img src="https://github.com/taoufik1el/plate_characters_detection/blob/main/images/car.jpg" width="400" height="400">

<img src=https://github.com/taoufik1el/plate_characters_detection/blob/main/images/plate.png width="400" height="100">

<img src=https://github.com/taoufik1el/plate_characters_detection/blob/main/images/plate_with_boxes.png width="400" height="100">


There is a possibility to use classical computer vision methods (canny, thresholding ...) to detect the plate and the characters, but this solutions are not robust,
for example the images can contain, different kinds of noise and backgrounds. Also OCR solutions like [Tesseract](https://github.com/tesseract-ocr/tesseract) does not work for noisy images.

This is why a neural network based solutions can be a good solution, by training it on a varied dataset.

YOLO (You Only Look Once) is one efficient NN model in object detection, it's performant in terms of accuracy and inference time.

For The first step, we will use a trained yolo model to detect the licence plates in images, the pretrained models can be found [here](https://github.com/ThorPham/License-plate-detection) and [here](https://github.com/oublalkhalid/MoroccoAI-Data-Challenge).

But For the second step, didn't find any annotated dataset or any trained model annotated dataset that can detect digits and arabic characters of the moroccan plates, so we will create our dataset and train a custumize yolo.

For the creation of datasets, we just take random images as backgrounds and write on top of them characters with different [fonts](https://fonts.google.com/) and sizes, then we can use some artifacts (rotations, perspectives, blur, noise ...) in the online data auguementation, then we feed those images to the model.
