'''
Created on 2017/07/06

@author: toru
'''

import cv2
import os

if __name__ == '__main__':
    PRJHOME = "../../../"
    lena = cv2.imread(PRJHOME + "material/sample/lena.jpg")
    cv2.imshow('test', lena)
    cv2.waitKey(0)
    