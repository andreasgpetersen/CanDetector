# CanDetector
Aluminium can detector developed for completion of course 02456 - Deep Learning at DTU.

Detecting and tracking objects in real-time is still a rather demanding task on computationally limited systems. In recent years the speed of object detection networks have become faster and faster given more data and better GPUs to train with. For various industrial applications a package of such a machine and a video camera could be the replacement of an on-site human eye and so a more convenient solution on smaller computers might be desirable. We take advantage of progress in deep learning, by using existing state-of-the-art convolutional neural networks (CNNs) to propose an end-to-end fast object detection and tracking model.


We collected video of beer (Grøn Tuborg) and cola (Coca Cola) cans as the data for both training and testing. The data set consists of 3389 frames from two video recordings.
The video was partitioned into frames that were manually labelled with boxes according to presence of cans. These annotations are processed, such that frames without annotations are disregarded to decrease subsequent processing time. Followingly the processed data can be fed into either a Faster R-CNN network with a variation of backbones (ResNet50, MobileNetV3, etc.) or a Single- Shot-Detection CNN (YOLOv5s).
This network is then trained and tested on a randomized 80/20 partition of the data set. Sub- sequently the performance of the network is evaluated on a video for its accuracy and real-time performance both on CPU and GPU to conclude on its industrial feasibility.


## Results

The accuracy and inference time of the different CNN models used for training on the entire dataset. 

 **Backbone**          | **Accuracy (mAP)** | **Inference Time [s]** 
-----------------------|--------------------|------------------------
 MobileNetv3Large      | 87.5\%             | 0.0664                 
 MobileNetv3Large\_320 | 85.0\%             | 0.0496                 
 ResNet50              | 86.8\%             | 0.3238                 
 YOLOv5s               | 85.1\%             | 0.0333                 

YOLOv5s performs the best overall with a compromise on both accuracy and inference time.
On the other spectrum is the ResNet50-network which lags behind in inference time per- formance, since it’s generally a larger network.

The proposed algorithm uses centroid tracking to maintain an overview of the current objects in a given frame. Centroid tracking is an easy way to implement object tracking, while not being the fastest or most reliable. For objects that have intersecting trajectories with near contact the centroid tracking algorithm can fail to detect the objects correctly as it uses Euclidean distances.

