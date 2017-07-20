'''
Created on 2017/07/20

@author: toru
'''

import cv2
import numpy as np
import os

SZ=20
bin_n = 16 # Number of bins

svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

# しきい値で"す"か"つ"を決定
def isSu(su, tsu, x):
    # サンプルのすに近いか、つに近いかで判定する
    return np.linalg.norm(hog(su) - hog(x)) < np.linalg.norm(hog(tsu) - hog(x))


PRJHOME = "../../../"
SAMPLEHOME =  PRJHOME + "material/好色一代男/simple/"

suList = os.listdir(SAMPLEHOME + "su/")
tsuList = os.listdir(SAMPLEHOME + "tsu/")
originalsu = cv2.imread(SAMPLEHOME + "su/" + suList[0])
originaltsu = cv2.imread(SAMPLEHOME + "tsu/" + tsuList[11])

sum = 0
for su in suList:
    suimg = cv2.imread(SAMPLEHOME + "su/" + su)
    if isSu(originalsu, originaltsu, suimg):
        sum+=1

for tsu in tsuList:
    tsuimg = cv2.imread(SAMPLEHOME + "tsu/" + tsu)
    if not isSu(originalsu, originaltsu, tsuimg):
        sum+=1
        
print(sum / (len(suList) + len(tsuList)) * 100) 