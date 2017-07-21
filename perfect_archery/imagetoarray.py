import cv2
import numpy as np

def img_to_array(filename):
    img = cv2.imread( filename )
    return img



im = img_to_array('./dataset/sample1.jpg')

print(type(im))
print(im.shape)