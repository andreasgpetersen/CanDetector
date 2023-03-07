# import the necessary packages
from torch._C import device
from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils

import torch
import time
import cv2
import os
import matplotlib.pyplot as plt


########################################################
################## USER DEFINED INPUT ##################
# SET NAME OF MODEL AND VIDEO YOU WANT TO USE FROM /models AND /data
MODEL_NAME = "mobilenetv3_15epochs_entire_dataset.pth"
########################################################


CONFIDENCE = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['background','beer','cola']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

MODEL_PATH = os.getcwd() + "/models/"
VIDEO_PATH = os.getcwd() + "/data/"
MODEL_SOURCE = MODEL_PATH + MODEL_NAME
VID_SOURCE1 = VIDEO_PATH + "video1/test/"
VID_SOURCE2 = VIDEO_PATH + "video2/test/"
imgs1 = list(sorted(os.listdir(os.path.join(VID_SOURCE1,"frames"))))
imgs2 = list(sorted(os.listdir(os.path.join(VID_SOURCE2,"frames"))))
VID_SRC1 = VID_SOURCE1 +"frames/"
VID_SRC2 = VID_SOURCE2 +"frames/"

N = len(imgs1) + len(imgs2)



model = torch.load(MODEL_SOURCE,map_location=torch.device('cpu'))
model = model.to(DEVICE)
model.eval()
print("Using",DEVICE)

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")



time.sleep(2.0)
fps = FPS().start()
i = 0
# loop over the frames from the video stream
src = VID_SRC1
imgs = imgs1
plt.figure()
while True:
# grab the frame from the threaded video stream and resize it
# to have a maximum width of 480 pixels
    if src == VID_SRC2 and i==len(imgs2):
            break
    if i==len(imgs1):
        src = VID_SRC2
        imgs = imgs2
        i = 0
    
    img = src + imgs[i]
    frame = cv2.imread(img)

    if frame is None:
        break
    frame = imutils.resize(frame, width=480)
    orig = frame.copy()
    # convert the frame from BGR to RGB channel ordering and change
    # the frame from channels last to channels first ordering
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    # add a batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the frame to a floating point tensor
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame)
    # send the input to the device and pass the it through the
    # network to get the detections and predictions
    frame = frame.to(DEVICE)
    detections = model(frame)[0]

    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > CONFIDENCE:
            # extract the index of the class label from the
            # detections, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box and label on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(orig, (startX, startY), (endX, endY),
            COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output frame
    #cv2.destroyAllWindows()
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(30)
    if key == ord("q"):
       break
    #plt.imshow(orig)
    
    # update the FPS counter
    fps.update()
    i += 1
# key = cv2.waitKey(1) & 0xFF
# # if the 'q' key was pressed, break from the loop
# if key == ord("q"):
#     break
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
