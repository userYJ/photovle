import cv2 as cv
import numpy as np
import json
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=2.5, help='epsilon pixel to approximate the polygons')
parser.add_argument('--input', type=str, default="C:\Photovle\PixelAnnotationTool-master\images_test\Abbey_Road_watershed_mask.png", help='image mask input to compute all polygons')
parser.add_argument('--config', type=str, default="C:\Photovle\PixelAnnotationTool-master\config.json", help='config file content labels informations')
opt = parser.parse_args()

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
    
img = cv.imread(opt.input)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
config = json.load(open(opt.config))['labels']

shape = (gray.shape[0],gray.shape[1],1)
gray = gray.reshape(shape)
images = to_categorical(gray).reshape((gray.shape[0],gray.shape[1],-1))

out = open('C:\Photovle\PixelAnnotationTool-master\images_test\Abbey_Road.json','w')
data = {}
data["version"] = "3.6.16" # images_test에 Abbey_Road.json 파일은 3.4.9.33으로 설정해서 만들었습니다! 
data["flags"] = {}
data["shapes"] = []
data["lineColor"]= [0,255,0,128]
data["fillColor"]= [255,0,0,128]
data["imagePath"]= os.path.basename(opt.input).replace("_watershed_mask.png",".jpg")
data["imageData"]= None
data["imageHeight"] = img.shape[0]
data["imageWidth"] = img.shape[1]
  
for label in config : 
    person = images[:,:,config[label]['id']]
    person[person>0] = 255

    person = person.astype('uint8')
    
    # check OpenCV version
    major = cv.__version__.split('.')[0]
    # print(major)
    if major == '3':
        ret, contours, hierarchy = cv.findContours(person, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv.findContours(person, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # print('!!',hierarchy)
    # im, contours , hierarchy = cv.findContours(person, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i,c in enumerate(contours) :
        contours[i] = cv.approxPolyDP(c,opt.epsilon,True)
        d = {}
        d['label'] = label
        d["line_color"] = None
        d["fill_color"] = config[label]['color']
        if contours[i].shape[0] < 3:
            continue
        d["points"] = contours[i].reshape(contours[i].shape[0],contours[i].shape[2]).tolist()
        d["shape_type"] = "polygon"
        data["shapes"].append(d)

    # color = config[label]['color']
    # cv.drawContours(img, contour, -1, color, 1)
    
# cv.imshow("person",img)
# cv.waitKey(0)

json.dump(data,out,indent=2)