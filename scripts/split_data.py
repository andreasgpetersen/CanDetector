########## split data into train and test folder ##################
import os
import shutil 

#Use only this method if running directly from scripts folder otherwise comment out the 
# following change.
# os.chdir('..')

cwd = os.getcwd()
data_path = cwd
# Choose boundingbox path appropriately (video1 or video2)
video = 'video1'
bb_path = "/data/" + video + "/boundingboxes/"
data = os.listdir(data_path + bb_path)


# Split files into train and test
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=1)

# Create test and train directory
train_path = cwd + "/data/" + video + "/train/"
test_path = cwd + "/data/" + video + "/test/"
img_path = cwd + "/data/" + video + "/frames/"


train_box = train_path + "/boundingboxes"
train_frames = train_path + "/frames"
test_box = test_path + "/boundingboxes"
test_frames = test_path + "/frames"

try:
   os.mkdir(train_path)
   os.mkdir(test_path)
   os.mkdir(train_box)
   os.mkdir(train_frames)
   os.mkdir(test_box)
   os.mkdir(test_frames)
except FileExistsError:
   pass


# Move train files to train folder
for file_name in train:
   #Boundingboxes folder:
   shutil.copy(os.path.join(data_path + bb_path, file_name), train_box)
   #Images folder:
   shutil.copy(os.path.join(img_path, file_name[:-4] + '.jpg'), train_frames)


# Move test files to test folder
for file_name in test:
   shutil.copy(os.path.join(data_path + bb_path, file_name), test_box)
   shutil.copy(os.path.join(img_path, file_name[:-4] + '.jpg'), test_frames)

print("Size train data: ", len(train), "\n","Size test data: ", len(test))
print(len(data))

