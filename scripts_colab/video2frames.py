# Program To Read video and Extract Frames from https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/

import cv2
import os

# image dimensions
width = 640
height = 480
dim = (width, height)
  
# Function to extract frames
def saveFrame(paths):
    # Path to video file
    for i in range(len(paths)):
        path = paths[i]
        vidObj = cv2.VideoCapture(path)
        destination = root + "video" + str(i+1) + "/frames/"

        print("Saving video",i+1)
        success, image = vidObj.read()
        count = 0

        while success:

            image = cv2.resize(image, dim)
            idx = "%06d"%count
            filename = destination + "frame_"+idx+".jpg"
            
            cv2.imwrite(filename, image)
            
            success, image = vidObj.read()
            count += 1
  
# Driver Code
if __name__ == '__main__':
  
    # Calling the function
    root = "data/"
    vid1_path = os.path.join(root, "video1.avi")
    vid2_path = os.path.join(root, "video2.avi")
    paths = [vid1_path, vid2_path]

    print("Resizing and saving following files:")
    print(vid1_path)
    print(vid2_path)
    print("Starting...")
    saveFrame(paths)
    print("Success")
