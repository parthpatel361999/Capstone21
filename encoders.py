import random
import pandas as pd
import numpy as np


def listEncoder(points,quat,score):
    encoded = []
    for tuple in points:
        for item in tuple:
            encoded.append(item)
    for val in quat:
        encoded.append(val)
    encoded.append(score)
    return encoded


def listDecoder(encodedList):
    score = encodedList.pop()
    quat = []
    for i in range(0,3):
        quat.append(encodedList.pop())
    if len(encodedList) % 2 != 0:
        raise Exception("ERROR WITH ENCODED POINT LIST")
    pointList = []
    tempTuple = []
    for item in encodedList:
        if item == -999.999:
            break
        tempTuple.append(item)
        if len(tempTuple) == 2:
            pointList.append(tuple(tempTuple))
            tempTuple = []
    return [pointList,quat,score]


def imgEncoder(img):
    #x is 640 y is 480
    tempList = []
    tempTuple = []
    for y in img:
        for x in y:
            for val in x:
                tempList.append(val)
    return tempList

def imgDecoder(img,x,y,channel):
    tempX = []
    tempY = []
    tempTuple = []
    for val in img:
        tempTuple.append(val)
        if len(tempTuple) == channel:
            tempX.append(tempTuple)
            tempTuple = []
        if len(tempX) == x:
            tempY.append(tempX)
            tempX = []
    if len(tempY) == y:
        return tempY
    else:
        return -1
