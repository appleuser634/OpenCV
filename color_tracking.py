import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#ノイズ処理の強度
kernel = np.ones((5,5),np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    #画像の取得
    ret, frame = cap.read()
    
    #img_trim = img[y:y+h, x:x+w]    
    #画像のリサイズ
    w,h = frame.shape[:2]
    small_size = (int(h/2),int(w/2))
    frame = cv2.resize(frame,small_size)

    #色空間の変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #指定色の範囲指定
    lo_orange = np.array([0,150,50])
    hi_orange = np.array([10,255,255])

    #マスク処理
    orange_mask = cv2.inRange(hsv, lo_orange, hi_orange)
    #orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

    merge_orange = cv2.bitwise_and(frame,frame,mask=orange_mask)

    #輪郭情報の取得
    contours, hieralchy = cv2.findContours(orange_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 

    #面積のリストを作成
    area_list = [cv2.contourArea(cnt) for cnt in contours]
    if len(area_list) == 0:
        continue
    
    #最大面積の輪郭を取得
    max_area_index = np.argmax(area_list)
    max_coutours = contours[max_area_index]

    #取得したエリアの重心座標を求める
    M = cv2.moments(max_coutours)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    #print("CX:",cx)
    #print("CY:",cy)

    #座標の描画処理
    frame = cv2.circle(frame,(cx,cy), 5, (0,255,0), -1)

    draw_text = "X=" + str(cx) + " Y=" + str(cy)
    cv2.putText(frame,draw_text,(cx,cy), font, 2,(255,0,255),2,cv2.LINE_AA)

    cv2.imshow("show image!",frame)
    cv2.imshow("show orange_mask!",merge_orange)

    #cv2.imshow("show gray!",gray)

    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cv2.destroyAllWindows()
