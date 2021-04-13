from threading import Thread
import cleaningObj
from cleaningObj import *
import cv2
import numpy as np
from cuboid import *
from cuboid_pnp_solver import *
from detector import *
import datetime


class webcamThread:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        (self.ret, self.img) = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.ret, self.img) = self.cap.read()

    def read(self):
        # return the frame most recently read
        return self.img

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class dopeThread:
    def __init__(self, models1, pnp1, pub1,color1, configdetect1,cleaningObj1):
        self.models = models1
        self.pnp = pnp1,
        self.pub = pub1,
        self.color = color1
        self.configdetect = configdetect1
        self.cleaning = cleaningObj1
        self.output = np.zeros((640,480,3),np.uint8)
        self.stopped = False

    def start(self,img1):
        tD = Thread(target=self.update,name="DOPE",args=(img1))
        tD.daemon = True
        tD.start()
        return self

    def update(self,img):
        for m in self.models:
            results = ObjectDetector.detect_object_in_image(self.models[m].net, self.pnp[m], img, self.configdetect)
            for i_r, result in enumerate(results):
                if result["location"] is None:
                    continue
                loc = result["location"]
                ori = result["quaternion"]
                score = result["score"]
                if None not in result['projected_points']:
                    points2d = []
                    for pair in result['projected_points']:
                        points2d.append(tuple(pair))

                    testHand = [tuple([150, 150]), tuple([150, 250]), tuple([250, 250]), tuple([250, 150])]
                    tempImg = cleaningObj.drawCompleteMask(points2d,self.cleaning,ori,testHand,img)
                else:
                    print("NOTHING")
        self.output = tempImg

