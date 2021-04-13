import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from shapely.geometry import Polygon

import random

class cleaningObj:
    def __init__(self,objId,vertices, cleanedMask):
        self.objID = objId
        self.vertices = vertices
        self.cleanedMask = cleanedMask


def DrawLine(point1, point2, lineColor, lineWidth):
    global draw_main
    global draw_sub
    if not point1 is None and point2 is not None:
        draw_main.line([point1, point2], fill=lineColor, width=lineWidth)
        #draw_sub.line([point1, point2], fill=tuple([lineColor[0],lineColor[1],lineColor[2],255]), width=lineWidth)


cap = cv2.VideoCapture(0)

#[top left, bottom left, bottom right, top right]
#points = [tuple([200, 200]), tuple([200, 300]), tuple([300, 300]), tuple([300, 200]), tuple([150, 150]), tuple([150, 250]), tuple([250, 250]), tuple([250, 150])]
points = [tuple([200, 200]), tuple([200, 300]), tuple([300, 350]), tuple([300, 150]), tuple([150, 150]), tuple([150, 250]), tuple([250, 250]), tuple([250, 150])]

maskTest = np.array([])
randVal = 0
for i in range(1000):
    randVal = random.randint(0, 255)
    maskTest = np.append(maskTest, randVal)

imgTest = cv2.imread('/home/eolanday/Pictures/XDXXV.png')

while True:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_sub = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    img_copy = img.copy()
    im = Image.fromarray(img_copy)
    im_sub = Image.fromarray(img_sub)

    draw_main = ImageDraw.Draw(im, mode='RGBA')
    draw_sub = ImageDraw.Draw(im_sub, mode='RGBA')
    #draw_main.line([(100,100), (200,200)], fill = tuple([255, 0, 0, 255]), width = 2)
    DrawLine(points[0], points[1], tuple([255, 0, 255, 150]), 2)
    DrawLine(points[1], points[2], tuple([255, 0, 255, 150]), 2)
    DrawLine(points[2], points[3], tuple([255, 0, 255, 150]), 2)
    DrawLine(points[0], points[3], tuple([255, 0, 255, 150]), 2)

    DrawLine(points[4], points[5], tuple([255, 255, 0, 150]), 2)
    DrawLine(points[5], points[6], tuple([255, 255, 0, 150]), 2)
    DrawLine(points[6], points[7], tuple([255, 255, 0, 150]), 2)
    DrawLine(points[4], points[7], tuple([255, 255, 0, 150]), 2)

    open_cv_image = np.array(im)
    #open_cv_image_sub = np.array(im_sub)
    open_cv_image_sub = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    open_cv_image_sub = cv2.cvtColor(open_cv_image_sub, cv2.COLOR_RGB2BGR)

    #MASKTEST
    maskTest = np.zeros((open_cv_image.shape[0],open_cv_image.shape[1]),np.uint8)
    cv2.fillPoly(maskTest,[np.int32([points[0],points[1],points[2],points[3]])],(255,255,255))
    maskTest2 = maskTest.copy()
    maskTest = cv2.bitwise_not(maskTest)


    #WARPTEST --> Going from static object to output object
    rectArray = np.array([[0,0],[imgTest.shape[1],0],[imgTest.shape[1],imgTest.shape[0]],[0,imgTest.shape[0]]],dtype="float32")
    dstArray = np.array([points[0],points[3],points[2],points[1]],dtype="float32")
    perspective = cv2.getPerspectiveTransform(rectArray,dstArray)

#    imgTest = np.zeros((img))

    warp = cv2.warpPerspective(imgTest,perspective,(img.shape[1],img.shape[0]))

    open_cv_image2 = cv2.bitwise_and(open_cv_image, open_cv_image, mask=maskTest)
    open_cv_image2 = cv2.bitwise_or(warp,open_cv_image)

    #WARPTEST --> Going from output object to static object
#    cropMask = np.zeros(img.shape[1],img.shape[0] )

    cropImg = cv2.bitwise_and(open_cv_image_sub, open_cv_image_sub, mask=maskTest2)
    reversePerspective = cv2.getPerspectiveTransform(dstArray,rectArray)
    reverseWarpTest = cv2.warpPerspective(open_cv_image_sub,reversePerspective,(imgTest.shape[1],imgTest.shape[0]))


    #INTERSECTION TEST (utilizing both warptests)
    face = Polygon(points[0:4])
    hand = Polygon(points[4:])
    intersectionPoly = []
    if(face.intersects(hand)):
        intersectionPoly = face.intersection(hand)
        intersectionPoly = list(intersectionPoly.exterior.coords)
    intersectionMask = np.zeros((open_cv_image.shape[0],open_cv_image.shape[1]),np.uint8)
    cv2.fillPoly(intersectionMask,[np.int32(intersectionPoly)],(255,255,255))
    tempColor = np.zeros((open_cv_image.shape[0],open_cv_image.shape[1],3),np.uint8)
    tempColor[:] = (0,255,255)
    intersectionMask = cv2.bitwise_and(tempColor,tempColor, mask = intersectionMask)
    intersectionMask = cv2.warpPerspective(intersectionMask,reversePerspective,(imgTest.shape[1],imgTest.shape[0]))

    reverseWarpTest = cv2.addWeighted(reverseWarpTest, 1.0, intersectionMask, 0.25, 0)

    #FINAL TEST
    open_cv_image3 = open_cv_image.copy()
    warp2 = cv2.warpPerspective(reverseWarpTest, perspective, (img.shape[1], img.shape[0]))
    open_cv_image3 = open_cv_image3 + warp2

    #OUTPUT
    cv2.imshow('maskTest',maskTest)
    cv2.imshow('Open_cv_image2  ', open_cv_image2)
    cv2.imshow('imgTest',imgTest)
    cv2.imshow('warp',warp)
    cv2.imshow('cropImg',cropImg)
    cv2.imshow('reverseTest',reverseWarpTest)
    cv2.imshow('intersectionMask',intersectionMask)
    cv2.imshow('warp2',warp2)
    cv2.imshow('open_cv_image3', open_cv_image3)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
