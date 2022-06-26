import cv2
import numpy as np
from overlay import CvOverlayImage



face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
#カメラの読み込み
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
#laugh_man = cv2.imread("laugh_man.png",cv2.IMREAD_UNCHANGED)
#動画終了まで繰り返し
while(cap.isOpened()):
        ret, frame = cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
        
                laugh_man = cv2.imread("laugh_man.png",cv2.IMREAD_UNCHANGED)
                fx = 1.8*w/520
                laugh_man = cv2.resize(laugh_man,dsize=None,fx=fx,fy=fx)
                frame = CvOverlayImage.overlay(frame, laugh_man,(x+int(w/2-(520*fx/2)),y+int(h/2-(480*fx/2))))

                cv2.imshow("Frame",frame)
                cv2.waitKey(5)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()    




