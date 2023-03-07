import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
#fake comment

class CansDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root,"video1/train2/images"))))
        self.bbox = list(sorted(os.listdir(os.path.join(root,"video1/train2/boundingboxes"))))
    
    def __getitem__(self, idx):
        # load images and bboxes
        img_path = os.path.join(self.root, "video1/train2/images", self.imgs[idx])
        bbox_path = os.path.join(self.root, "video1/train2/boundingboxes", self.bbox[idx])
        img = Image.open(img_path).convert("RGB")
        print("img_path:",img_path)
        print("bbox_path:",bbox_path)


        bbox = []
        label = []
        with open(bbox_path, 'r') as f:
            for line in f:
                line = line.split(" ")
                id = line[0] # class label, 0=beer, 1=cola
                id = 0 if id == 'beer' else 1
                xmin = int(line[1])
                ymin = int(line[2])
                xmax = int(line[3])
                ymax = int(line[4])
                bbox.append([xmin, ymin, xmax, ymax])
                label.append(id)
        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        labels = torch.as_tensor(label, dtype=torch.float32) #torch.ones((num_objs, ), dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])

        target = {}
        target["boxes"] = bbox
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area


        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    print(os.getcwd())
    root = os.getcwd()+'/data/'
    dataset = CansDataset(root)

    #print(dataset.bbox)
    #print(dataset.imgs)

    img,target = dataset.__getitem__(907)
    print("img",img)
    print("target",target)


