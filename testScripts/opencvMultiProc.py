from threading import Thread
import cv2
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


class FPS:
    def __init__(self):
        self.start = None
        self.end = None
        self.numFrames = 0

    def start(self):
        self.start = datetime.datetime.now()
        return self

    def stop(self):
        self.end = datetime.datetime.now()

    def update(self):
        self.numFrames += 1

    def elapsed(self):
        return (self.end - self.start).total_seconds()

    def fps(self):
        return self.numFrames / self.elapsed()
