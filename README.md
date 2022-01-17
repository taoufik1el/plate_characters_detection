# CHARACHTER DETECTION PIPLINE OF CAR LICENCE PLATES

In this repository I will give method to recognize the numbers of a car plate in images of steet views.

We can summerize the pipline as the following:
- find the bounding box of the full plate in an image
- find the bounding box of each charachter in the plate
- reconstruct the plate number based on the bounding box positions

There is a possibility to use classical computer vision methods (canny, thresholding ...) to detect the plate and the charachters, but this solutions are not robust,
for example the images can containe noise and different levels of colors and oppacity. This is why a neural network based solutions can handle this problem.

One efficient NN model in object detection is YOLO, that is performant in terms of accuracy and inference time.

We can use a trained yolo model to detect the plates in images, but I did'nt find any trained model that can detect digits and arabic charachters, so we will train a custumize yolo in our case. Also, there is no dataset of charachter detection of mixed LATIN-ARABIC charachters, for this problem we will create a sythesized data by just taking random images as backgrounds and write on top of them charachters with different fonts and sizes, then we can apply in the online data auguementation some artifacts (rotations, perspectives, blur, noise ...).


The yolo3 file is partialy copied from https://github.com/qqwweee/keras-yolo3 and modified.
