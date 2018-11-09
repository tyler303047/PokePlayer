from cv2 import *
import time

cap = VideoCapture(0)

i = 0

PicNum = 500

while(True):
    ret, frame = cap.read()

    if i % 5 == 0:
        imshow('frame', frame)
    if(waitKey(1) & 0xFF == ord('q')):
        break
    imwrite("filename_4_5_" + str(i) + ".jpg", frame)
    time.sleep(.1)
    if i == PicNum:
        break
    else:
        i = i+1

cap.release()
destroyAllWindows()
