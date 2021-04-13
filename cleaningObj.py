import numpy as np
import cv2, shapely, math
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
[memSize_X, memSize_Y] = [500, 500]


class CleaningObject:
    def __init__(self, objId1, vertices1, quarternion1, color1, dimensions):
        img_sub = np.zeros((memSize_X, memSize_Y, 3), np.uint8)
        img_sub = cv2.cvtColor(img_sub, cv2.COLOR_RGB2BGR)
        blankMask = np.zeros((memSize_X, memSize_Y, 3), np.uint8)
        self.objId = objId1
        self.vertices = vertices1
        self.quaternion = quarternion1
        self.color = tuple(color1)
        self.frontPoints = [self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]]  # good
        self.backPoints = [self.vertices[4], self.vertices[5], self.vertices[6], self.vertices[7]]
        self.topPoints = [self.vertices[0], self.vertices[1], self.vertices[5], self.vertices[4]]  # good
        self.bottomPoints = [self.vertices[3], self.vertices[2], self.vertices[6], self.vertices[7]]  # test
        self.leftPoints = [self.vertices[4], self.vertices[0], self.vertices[3], self.vertices[7]]  # good
        self.rightPoints = [self.vertices[2], self.vertices[1], self.vertices[5], self.vertices[6]]  # good
        self.frontFace = blankMask.copy()
        self.backFace = blankMask.copy()
        self.topFace = blankMask.copy()
        self.bottomFace = blankMask.copy()
        self.leftFace = blankMask.copy()
        self.rightFace = blankMask.copy()
        self.frontExposed = False
        self.backExposed = False
        self.topExposed = False
        self.bottomExposed = False
        self.rightExposed = False
        self.leftExposed = False
        diagonaLen = np.sqrt(dimensions[0]**2 + dimensions[1]**2 + dimensions[2]**2)
        handLen = 18.0
        self.acceptableRatio = handLen/diagonaLen

    def updateFaces(self, vertices1):
        self.frontPoints = [vertices1[0], vertices1[1], vertices1[2], vertices1[3]]
        self.backPoints = [vertices1[4], vertices1[5], vertices1[6], vertices1[7]]
        self.topPoints = [vertices1[0], vertices1[1], vertices1[5], vertices1[4]]
        self.bottomPoints = [vertices1[3], vertices1[2], vertices1[6], vertices1[7]]
        self.leftPoints = [vertices1[4], vertices1[0], vertices1[3], vertices1[7]]
        self.rightPoints = [vertices1[2], vertices1[1], vertices1[5], vertices1[6]]

    def updateExposed(self, quat1):
        rotationQ = Rotation.from_quat([quat1[0], quat1[1], quat1[2], quat1[3]])  # (X,Y,Z,W)

        vecX = np.array([1, 0, 0])
        vecY = np.array([0, 1, 0])
        vecZ = np.array([0, 0, 1])
        rotatedX = rotationQ.apply(vecX)
        rotatedY = rotationQ.apply(vecY)
        rotatedZ = rotationQ.apply(vecZ)
        #        print("X:",rotatedX)
        #        print("Y:",rotatedY)
        #        print("Z:",rotatedZ)
        # left/right test (x positive is left)
        # print(rotatedX[2])
        if rotatedX[2] < -0.2:
            print("LEFT EXPOSED")
            self.rightExposed = False
            self.leftExposed = True
        elif rotatedX[2] > 0.2:
            print("RIGHT EXPOSED")
            self.rightExposed = True
            self.leftExposed = False
        else:
            print("")
            self.rightExposed = False
            self.leftExposed = False
        # top/bottom test (y positive is left)
        if rotatedY[2] < -0.2:
            print("BOTTOM EXPOSED")
            self.topExposed = False
            self.bottomExposed = True
        elif rotatedY[2] > 0.2:
            print("TOP EXPOSED")
            self.topExposed = True
            self.bottomExposed = False
        else:
            print("")
            self.topExposed = False
            self.bottomExposed = False
        # front/back test (z positive is left)
        if rotatedZ[2] < -0.2:
            print("FRONT EXPOSED")
            self.frontExposed = True
            self.backExposed = False
        elif rotatedZ[2] > 0.2:
            print("BACK EXPOSED")
            self.frontExposed = False
            self.backExposed = True
        else:
            print("")
            self.frontExposed = False
            self.backExposed = False

        return True

    def clean(self, handPoints, mainImg):
        cleanWeight = 0.01
        hand = Polygon(handPoints)
        if self.frontExposed:
            face = Polygon(self.frontPoints)
            try:
                if face.intersects(hand):
                    intersectionPoly = face.intersection(hand)  # IMPOSSIBLE GEOMETRY PROBLEM
                    intersectionPoly = list(intersectionPoly.exterior.coords)
                    intersectionMask = np.zeros((mainImg.shape[0], mainImg.shape[1]), np.uint8)
                    cv2.fillPoly(intersectionMask, [np.int32(intersectionPoly)], (255, 255, 255))
                    tempColor = np.zeros((mainImg.shape[0], mainImg.shape[1], 3), np.uint8)
                    tempColor[:] = self.color
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.frontPoints, dtype="float32"),
                                                                       np.array(
                                                                           [[0, 0], [memSize_X, 0],
                                                                            [memSize_X, memSize_Y], [0, memSize_Y]],
                                                                           dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective,
                                                           (memSize_X, memSize_Y))
                    self.frontFace = cv2.addWeighted(self.frontFace, 1.0, intersectionMask, cleanWeight, 0)
            except shapely.errors.TopologicalError:
                pass
        if self.backExposed:
            face = Polygon(self.backPoints)
            try:
                if face.intersects(hand):
                    intersectionPoly = face.intersection(hand)
                    intersectionPoly = list(intersectionPoly.exterior.coords)
                    intersectionMask = np.zeros((mainImg.shape[0], mainImg.shape[1]), np.uint8)
                    cv2.fillPoly(intersectionMask, [np.int32(intersectionPoly)], (255, 255, 255))
                    tempColor = np.zeros((mainImg.shape[0], mainImg.shape[1], 3), np.uint8)
                    tempColor[:] = self.color
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.backPoints, dtype="float32"),
                                                                       np.array(
                                                                           [[0, 0], [memSize_X, 0],
                                                                            [memSize_X, memSize_Y], [0, memSize_Y]],
                                                                           dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective,
                                                           (memSize_X, memSize_Y))
                    self.backFace = cv2.addWeighted(self.backFace, 1.0, intersectionMask, cleanWeight, 0)
            except shapely.errors.TopologicalError:
                pass
        if self.topExposed:
            face = Polygon(self.topPoints)
            try:
                if face.intersects(hand):
                    intersectionPoly = face.intersection(hand)
                    intersectionPoly = list(intersectionPoly.exterior.coords)
                    intersectionMask = np.zeros((mainImg.shape[0], mainImg.shape[1]), np.uint8)
                    cv2.fillPoly(intersectionMask, [np.int32(intersectionPoly)], (255, 255, 255))
                    tempColor = np.zeros((mainImg.shape[0], mainImg.shape[1], 3), np.uint8)
                    tempColor[:] = self.color
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.topPoints, dtype="float32"),
                                                                       np.array(
                                                                           [[0, 0], [memSize_X, 0],
                                                                            [memSize_X, memSize_Y], [0, memSize_Y]],
                                                                           dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective,
                                                           (memSize_X, memSize_Y))
                    self.topFace = cv2.addWeighted(self.topFace, 1.0, intersectionMask, cleanWeight, 0)
            except shapely.errors.TopologicalError:
                pass
        if self.bottomExposed:
            face = Polygon(self.bottomPoints)
            try:
                if face.intersects(hand):
                    intersectionPoly = face.intersection(hand)
                    intersectionPoly = list(intersectionPoly.exterior.coords)
                    intersectionMask = np.zeros((mainImg.shape[0], mainImg.shape[1]), np.uint8)
                    cv2.fillPoly(intersectionMask, [np.int32(intersectionPoly)], (255, 255, 255))
                    tempColor = np.zeros((mainImg.shape[0], mainImg.shape[1], 3), np.uint8)
                    tempColor[:] = self.color
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.bottomPoints, dtype="float32"),
                                                                       np.array(
                                                                           [[0, 0], [memSize_X, 0],
                                                                            [memSize_X, memSize_Y], [0, memSize_Y]],
                                                                           dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective,
                                                           (memSize_X, memSize_Y))
                    self.bottomFace = cv2.addWeighted(self.bottomFace, 1.0, intersectionMask, cleanWeight, 0)
            except shapely.errors.TopologicalError:
                pass
        if self.leftExposed:
            face = Polygon(self.leftPoints)
            try:
                if face.intersects(hand):
                    intersectionPoly = face.intersection(hand)
                    intersectionPoly = list(intersectionPoly.exterior.coords)
                    intersectionMask = np.zeros((mainImg.shape[0], mainImg.shape[1]), np.uint8)
                    cv2.fillPoly(intersectionMask, [np.int32(intersectionPoly)], (255, 255, 255))
                    tempColor = np.zeros((mainImg.shape[0], mainImg.shape[1], 3), np.uint8)
                    tempColor[:] = self.color
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.leftPoints, dtype="float32"),
                                                                       np.array(
                                                                           [[0, 0], [memSize_X, 0],
                                                                            [memSize_X, memSize_Y], [0, memSize_Y]],
                                                                           dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective,
                                                           (memSize_X, memSize_Y))
                    self.leftFace = cv2.addWeighted(self.leftFace, 1.0, intersectionMask, cleanWeight, 0)
            except shapely.errors.TopologicalError:
                pass
        if self.rightExposed:
            face = Polygon(self.rightPoints)
            try:
                if face.intersects(hand):
                    intersectionPoly = face.intersection(hand)
                    intersectionPoly = list(intersectionPoly.exterior.coords)
                    intersectionMask = np.zeros((mainImg.shape[0], mainImg.shape[1]), np.uint8)
                    cv2.fillPoly(intersectionMask, [np.int32(intersectionPoly)], (255, 255, 255))
                    tempColor = np.zeros((mainImg.shape[0], mainImg.shape[1], 3), np.uint8)
                    tempColor[:] = self.color
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.rightPoints, dtype="float32"),
                                                                       np.array(
                                                                           [[0, 0], [memSize_X, 0],
                                                                            [memSize_X, memSize_Y], [0, memSize_Y]],
                                                                           dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective,
                                                           (memSize_X, memSize_Y))
                    self.rightFace = cv2.addWeighted(self.rightFace, 1.0, intersectionMask, cleanWeight, 0)
            except shapely.errors.TopologicalError:
                pass

    def drawFaces(self, mainImg):
        temp = np.zeros((mainImg.shape[0], mainImg.shape[1], 3), np.uint8)
        if self.frontExposed:
            transformPerspective = cv2.getPerspectiveTransform(
                np.array([[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"),
                np.array(self.frontPoints, dtype="float32"))
            cleanedMask = cv2.warpPerspective(self.frontFace, transformPerspective,
                                              (mainImg.shape[1], mainImg.shape[0]))
            temp = cv2.add(cleanedMask, temp)
        if self.backExposed:
            transformPerspective = cv2.getPerspectiveTransform(
                np.array([[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"),
                np.array(self.backPoints, dtype="float32"))
            cleanedMask = cv2.warpPerspective(self.backFace, transformPerspective,
                                              (mainImg.shape[1], mainImg.shape[0]))
            temp = cv2.add(cleanedMask, temp)
        if self.bottomExposed:
            transformPerspective = cv2.getPerspectiveTransform(
                np.array([[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"),
                np.array(self.bottomPoints, dtype="float32"))
            cleanedMask = cv2.warpPerspective(self.bottomFace, transformPerspective,
                                              (mainImg.shape[1], mainImg.shape[0]))
            temp = cv2.add(cleanedMask, temp)
        if self.topExposed:
            transformPerspective = cv2.getPerspectiveTransform(
                np.array([[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"),
                np.array(self.topPoints, dtype="float32"))
            cleanedMask = cv2.warpPerspective(self.topFace, transformPerspective,
                                              (mainImg.shape[1], mainImg.shape[0]))
            temp = cv2.add(cleanedMask, temp)
        if self.leftExposed:
            transformPerspective = cv2.getPerspectiveTransform(
                np.array([[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"),
                np.array(self.leftPoints, dtype="float32"))
            cleanedMask = cv2.warpPerspective(self.leftFace, transformPerspective,
                                              (mainImg.shape[1], mainImg.shape[0]))
            temp = cv2.add(cleanedMask, temp)
        if self.rightExposed:
            transformPerspective = cv2.getPerspectiveTransform(
                np.array([[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"),
                np.array(self.rightPoints, dtype="float32"))
            cleanedMask = cv2.warpPerspective(self.rightFace, transformPerspective,
                                              (mainImg.shape[1], mainImg.shape[0]))
            temp = cv2.add(cleanedMask, temp)
        return temp


def drawCompleteMask(points2D_1, cleaningObject_1,orientation_1, score1, handCords, mainImg_1):
    lineWidth = 2
    diagThreshold = 0.5
    tempImg = np.zeros((mainImg_1.shape[0],mainImg_1.shape[1],3),np.uint8)
    tempImg = cv2.cvtColor(tempImg,cv2.COLOR_BGR2RGB)
    tempImg = Image.fromarray(tempImg)
    tempPILDraw = ImageDraw.Draw(tempImg, mode='RGBA')
    tempCVMask = np.zeros((mainImg_1.shape[0],mainImg_1.shape[1],3),np.uint8)
    cleaningObject_1.updateExposed(orientation_1)
    cleaningObject_1.updateFaces(points2D_1)
    print("Score:",score1)
    if handCords is None:
        hand = None
    else:
        if len(handCords) >= 4:
            hand = np.copy(handCords)
        else:
            hand = None

    if len(points2D_1) >= 8 and score1 > 0.25:

        xLen = np.linalg.norm(np.array(points2D_1[0]) - np.array(points2D_1[1]))
        yLen = np.linalg.norm(np.array(points2D_1[0]) - np.array(points2D_1[3]))
        zLen = np.linalg.norm(np.array(points2D_1[0]) - np.array(points2D_1[4]))
        curDiag = np.sqrt(xLen**2 + yLen**2 + zLen**2)

        if hand is not None:
            handWidth = np.abs(hand[0] - hand[2])
            handLength = np.abs(hand[2] - hand[3])
            handDiagonal = np.linalg.norm(np.array(hand[0],hand[1]) - np.array(hand[2],hand[3]))
            handPoints = [tuple([hand[0],hand[1]]), tuple([hand[0]+handWidth,hand[1]]), tuple([hand[2],hand[3]]), tuple([hand[0],hand[1]+handLength])]

            compDiag = handDiagonal/curDiag
            compResult = np.abs(compDiag - cleaningObject_1.acceptableRatio)
            print("Distance Differece:",compResult)
            if compResult <= diagThreshold:
                print("CLEANING")
                cleaningObject_1.clean(handPoints=handPoints,mainImg=mainImg_1)
        tempCVMask = cleaningObject_1.drawFaces(mainImg=mainImg_1)
        #Front Face
        tempPILDraw.line([points2D_1[0], points2D_1[1]], fill=cleaningObject_1.color, width=lineWidth)
        tempPILDraw.line([points2D_1[1], points2D_1[2]], fill=cleaningObject_1.color, width=lineWidth)
        tempPILDraw.line([points2D_1[3], points2D_1[2]], fill=cleaningObject_1.color, width=lineWidth)
        tempPILDraw.line([points2D_1[3], points2D_1[0]], fill=cleaningObject_1.color, width=lineWidth)
        #Back Face
        tempPILDraw.line([points2D_1[4], points2D_1[5]], fill=cleaningObject_1.color, width=lineWidth)
        tempPILDraw.line([points2D_1[6], points2D_1[5]], fill=cleaningObject_1.color, width=lineWidth)
        tempPILDraw.line([points2D_1[6], points2D_1[7]], fill=cleaningObject_1.color, width=lineWidth)
        tempPILDraw.line([points2D_1[4], points2D_1[7]], fill=cleaningObject_1.color, width=lineWidth)
        #Side Faces
        tempPILDraw.line([points2D_1[0], points2D_1[4]], fill=cleaningObject_1.color, width=lineWidth)
        tempPILDraw.line([points2D_1[7], points2D_1[3]], fill=cleaningObject_1.color, width=lineWidth)
        tempPILDraw.line([points2D_1[5], points2D_1[1]], fill=cleaningObject_1.color, width=lineWidth)
        tempPILDraw.line([points2D_1[2], points2D_1[6]], fill=cleaningObject_1.color, width=lineWidth)
    tempImg = np.array(tempImg)
    tempImg = cv2.cvtColor(tempImg, cv2.COLOR_RGB2BGR)
    tempImg = cv2.add(tempCVMask,tempImg)
    return tempImg
