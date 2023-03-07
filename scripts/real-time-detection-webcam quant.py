# import the necessary packages
from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils

import torch
import torch.quantization


import time
import cv2
import os

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

########################################################
################## USER DEFINED INPUT ##################
# SET NAME OF MODEL YOU WANT TO USE FROM /models
#cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
MODEL_NAME = "mobilenetv3_large_25_epochs_quantized.pth"
########################################################


CONFIDENCE = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"
CLASSES = ['background','beer','cola']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

MODEL_PATH = os.getcwd() + "/models/"

MODEL_SOURCE = MODEL_PATH + MODEL_NAME
DEVICE = "cpu"
if DEVICE != "cpu":
    model = torch.load(MODEL_SOURCE)
    model = model.to(DEVICE)
    print("Using CUDA")
else:
    print("Using CPU")
    model = torch.load(MODEL_SOURCE, map_location=torch.device('cpu'))#,map_location=torch.device('cpu'))

model = torch.quantization.QuantWrapper(model)
'''
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

print_size_of_model(quantized_model)
print_size_of_model(model)




'''




#model = quantized_model







model.eval()


# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")


vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
# grab the frame from the threaded video stream and resize it
# to have a maximum width of 400 pixels
    frame = vs.read()
    timer = cv2.getTickCount()
    frame = imutils.resize(frame, width=400)
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
    compute_time = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    cv2.putText(orig, "FPS : " + str(int(compute_time)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()
    
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
#'''
