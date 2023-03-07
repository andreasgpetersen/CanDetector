# From Github user: FairyOnIce/ObjectDetectionYolo/backend.py
#
import xml.etree.ElementTree as ET
import os

def parse_annotation(ann_dir, img_dir, labels=[]):
    '''
    output:
    - Each element of the train_image is a dictionary containing the annoation infomation of an image.
    - seen_train_labels is the dictionary containing
            (key, value) = (the object class, the number of objects found in the images)
    '''
    # This function iterates all annotation files for their contents using the ElementTree library 
    # for XML-parsing.
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        if "xml" not in ann:
            continue
        img = {'object':[]}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                path_to_image = img_dir + elem.text[:-4] + '.jpg'
                if not os.path.exists(path_to_image):
                    assert False, "file does not exist!\n{}".format(path_to_image)
                # print(path_to_image)
                img['filename'] = elem.text[:-4]
                ## make sure that the image exists:
                    
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:

                        obj['name'] = attr.text

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]



                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']]  = 1



                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels

def img2txtfile(all_imgs):
    # Changing current path to the directory
    cwd = os.getcwd()
    os.chdir(cwd + '/data/video1/boundingboxes/')

    # Creates a text file with the same name as the png and then adds one line pr bounding box: 
    # 'label xmin ymin xmax ymax\n'
    for frame in all_imgs:
        f = open(frame['filename'] + ".txt", "a")
        for can in frame['object']:
            boxCoords = "{} {} {} {} {}".format(can['name'],can['xmin'],can['ymin'],can['xmax'],can['ymax'])
            f.write(boxCoords + '\n')
        
        f.close()



LABELS = ['beer','cola','coke']



### The location where the VOC2012 data is saved.
# Run script from root directory: (otherwise from scripts use os.chdir('..') first) 
cwd = os.getcwd()
frame_path = '/data/video1/frames/'
annot_path = frame_path
train_image_folder = cwd +  frame_path

train_annot_folder = cwd + frame_path



train_image, seen_train_labels = parse_annotation(train_annot_folder,
                                                  train_image_folder,
                                                  labels=LABELS)

print("N train = {}".format(len(train_image)))
# print(train_image[833]['object'][0]['xmin'])
print(seen_train_labels)
img2txtfile(train_image)



## Parse annotations
# train_image, seen_train_labels = parse_annotation(train_annot_folder,train_image_folder, labels=LABELS)
# print("N train = {}".format(len(train_image)))
