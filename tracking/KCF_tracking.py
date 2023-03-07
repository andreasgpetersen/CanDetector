from __future__ import print_function
import sys
import cv2
from random import randint
import torch
import numpy as np



MODEL_PATH = "/home/andreasgp/MEGAsync/DTU/9. Semester/Deep Learning/object-tracking-project/02456-project/models/mobilenetv3_15epochs_entire_dataset.pth"
VID_SOURCE = "/home/andreasgp/MEGAsync/DTU/9. Semester/Deep Learning/object-tracking-project/02456-project/data/2021_10_28_12_49_00.avi"#C:/Users/Philip/02456-project/data/2021_10_28_12_47_19.avi" # 0 is webcam
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE = 0.8
COLORS = [(0,0,0),(0,200,0),(255,0,0)]
CLASSES = ['background','beer','coke']
#load model
print(DEVICE)

if DEVICE != "cpu":
    model = torch.load(MODEL_PATH)
    model = model.to(DEVICE)
    print("Using CUDA")
else:
    print("Using CPU")
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))#,map_location=torch.device('cpu'))

model.eval()

# Create a video capture object to read videos
cap = cv2.VideoCapture(VID_SOURCE)


success, frame = cap.read()

# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)


## Select boxes
bboxes = []
colors = [] 

## image resize dim

width = 480
height = 320



# Specify the tracker type
trackerType = "KCF"    

# Create MultiTracker object
multiTracker = cv2.legacy.MultiTracker_create()
count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    # PERFORM DETECTION


    
    # Fix image
    frame = cv2.resize(frame, (width,height))
    orig = frame.copy()
    if count==0:
        print("Detection")
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
        frame = frame.to("cuda")
        detections = model(frame)[0]


        id = 0
        ids = []
    
        boxes = []
        # loop over the detections
        for i in range(0, len(detections["boxes"])):
            confidence = detections["scores"][i]

            if confidence > CONFIDENCE:

                idx = int(detections["labels"][i])
                box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                bboxes.append(box.astype("int"))
                # draw the bounding box and label on the frame
                '''
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(orig, (startX, startY), (endX, endY),
                COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(orig, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                '''

                # PERFORM TRACKING


    
        # Initialize MultiTracker
        for bbox in bboxes:
            multiTracker.add(cv2.legacy.TrackerMOSSE_create(), orig, bbox)



    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(orig)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(orig, p1, p2, (255,255,100), 2, 1)
    
    # show frame
    cv2.imshow('MultiTracker', orig)
    
    # quit on ESC button
    if cv2.waitKey(30) & 0xFF == 27:  # Esc pressed
        break
    count += 1

    if count == 10:
        count =0
