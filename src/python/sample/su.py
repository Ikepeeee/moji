'''
Created on 2017/07/13

@author: toru
'''
import cv2
import time

if __name__ == '__main__':
    PRJHOME = "../../../"
    img1 = cv2.cvtColor(cv2.imread(PRJHOME + "material/好色一代男/characters/U+3059/U+3059_200003076_00032_1_X1143_Y1012.jpg"), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(PRJHOME + "material/好色一代男/characters/U+3059/U+3059_200003076_00024_1_X0617_Y2565.jpg"), cv2.COLOR_BGR2GRAY)
#     img1 = cv2.imread(PRJHOME + "material/sample/lena.jpg")
#     img2 = cv2.imread(PRJHOME + "material/sample/lena.jpg")
    
    # A-KAZE検出器の生成
    akaze = cv2.AKAZE_create()                                
    
    # 特徴量の検出と特徴量ベクトルの計算
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    
    # Brute-Force Matcher生成
    bf = cv2.BFMatcher()
    
    # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
    matches = bf.knnMatch(des1, des2, k=2)
    
    # データを間引きする
    ratio = 1.0
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])
    
    # 対応する特徴点同士を描画
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    
    # 画像表示
    cv2.imshow('img', img3)
    
    # キー押下で終了
    cv2.waitKey(0)
    time.sleep(10)
    cv2.destroyAllWindows()
    