from handModel.utils.timer import Timer
from handModel.handDetector import *
import cv2

testHand = HandDetector()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        to_show = frame
        dets = testHand.detectHand(to_show)
        for i in range(dets.shape[0]):
            cv2.rectangle(to_show, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), [0, 0, 255], 3)
            # print(str(dets[i][0]) + " " + str(dets[i][1]) + " " + str(dets[i][2]) + " " + str(dets[i][3]))
            cv2.putText(to_show, str(dets[i][4]), (int(dets[i][0]) + 5, int(dets[i][1]) + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', to_show)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        cv2.waitKey(1)
    else:
        break
cap.release()
cv2.destroyAllWindows()
