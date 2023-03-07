from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils

import torch
import time
import cv2

ct = CentroidTracker()

CONFIDENCE = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['background','beer','cola']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# PARSE YOUR MODEL HERE:
MODEL_PATH = "C:/Users/Philip/02456-project/models/mobilenetv3_15epochs_entire_dataset.pth"
model = torch.load(MODEL_PATH)#,map_location=torch.device('cpu'))
model = model.to(DEVICE)
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
    rects = []	

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
            rects.append(box.astype("int"))
            # draw the bounding box and label on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(orig, (startX, startY), (endX, endY),
            COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
    objects = ct.update(rects)
	
	# loop over the tracked objects
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(orig, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(orig, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	

    # show the output frame
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