# CHARACHTER DETECTION PIPLINE OF CAR LICENCE PLATES

In this repository you will find a method to recognize the numbers of a car plate in street view images.

If you want to build a license plate detection model in your country, but couldn't find a pre-trained model or an annotated dataset (bounding boxes on both plates and characters inside the plates) because you have characters in a different language (like Arabic, Chinese, Hindi...). This repository can be useful to you.

We can summerize the pipline in three steps:
- find the bounding box of the full plate in an image using a trained yolov3.
- find the bounding box of each charachter in the plate using a customized yolov3 trained on synthesized data.
- reconstruct the plate number based on the bounding box positions.

There is a possibility to use classical computer vision methods (canny, thresholding ...) to detect the plate and the charachters, but this solutions are not robust,
for example the images can containe different kinds of noise and backgrounds. This is why a neural network based solutions can be a good solution, by training it on a varied dataset.

YOLO (You Only Look Once) is one efficient NN model in object detection, it's performant in terms of accuracy and inference time.

We can use a trained yolo model to detect the plates in images, but I did'nt find any trained model that can detect digits and arabic charachters of the moroccan plates, so we will train a custumize yolo in our case. Also, there is no dataset of charachter detection of mixed LATIN-ARABIC charachters in a noisy backgroung, for this problem we will create a sythesized data by just taking random images as backgrounds and write on top of them charachters with different fonts and sizes, then we can apply in the online data auguementation some artifacts (rotations, perspectives, blur, noise ...), then we feed those images to the model.



The [yolo3](https://github.com/taoufik1el/PLATE_CHARACHTER_DETECTION/tree/main/yolo3) file is partialy copied from [this repository](https://github.com/qqwweee/keras-yolo3) and modified.
