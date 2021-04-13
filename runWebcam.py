# CREDIT: https://github.com/Abdul-Mukit/dope_exp/blob/master/demo/live_dope_webcam.py

import numpy as np
import queue, yaml, threading, time, cv2, math
from cuboid import *
from detector import *
from cuboid_pnp_solver import *
from PIL import Image
from PIL import ImageDraw
from shapely.geometry import Polygon
import shapely
from pyrr import quaternion
from scipy.spatial.transform import Rotation
#[top left, bottom left, bottom right, top right]
[memSize_X, memSize_Y] = [500, 500]

class CleaningObject:
    def __init__(self, objId1, vertices1, img1, quarternion1, color1):
        img_sub = np.zeros((memSize_X,memSize_Y,3),np.uint8)
        img_sub = cv2.cvtColor(img_sub, cv2.COLOR_RGB2BGR)
        blankMask = np.zeros((memSize_X,memSize_Y,3), np.uint8)
        self.objId = objId1
        self.vertices = vertices1
        self.quaternion = quarternion1
        self.color = color1
        self.frontPoints = [self.vertices[0],self.vertices[1],self.vertices[2],self.vertices[3]] #good
        self.backPoints = [self.vertices[4], self.vertices[5], self.vertices[6], self.vertices[7]]
        self.topPoints = [self.vertices[0],self.vertices[1],self.vertices[5],self.vertices[4]] #good
        self.bottomPoints = [self.vertices[3], self.vertices[2], self.vertices[6], self.vertices[7]] #test
        self.leftPoints = [self.vertices[4],self.vertices[0],self.vertices[3],self.vertices[7]] #good
        self.rightPoints = [self.vertices[2],self.vertices[1],self.vertices[5],self.vertices[6]] #good
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

    def updateFaces(self, vertices1):
        self.frontPoints = [vertices1[0],vertices1[1],vertices1[2],vertices1[3]]
        self.backPoints = [vertices1[4], vertices1[5], vertices1[6], vertices1[7]]
        self.topPoints = [vertices1[0],vertices1[1],vertices1[5],vertices1[4]]
        self.bottomPoints = [vertices1[3], vertices1[2], vertices1[6], vertices1[7]]
        self.leftPoints = [vertices1[4],vertices1[0],vertices1[3],vertices1[7]]
        self.rightPoints = [vertices1[2],vertices1[1],vertices1[5],vertices1[6]]
#        print(self.frontPoints)

    def updateExposed(self, quat1):
#        quat = quaternion.create(x=quat1[0], y=quat1[1], z=quat1[2], w=quat1[3])
        rotationQ = Rotation.from_quat([quat1[0],quat1[1],quat1[2],quat1[3]]) #(X,Y,Z,W)
#       rotation = rotationQ.as_euler('ZXY',degrees=True)
        R_x = quat1[0]/math.sqrt(10)

        vecX = np.array([1,0,0])
        vecY = np.array([0,1,0])
        vecZ = np.array([0,0,1])
        rotatedX = rotationQ.apply(vecX)
        rotatedY = rotationQ.apply(vecY)
        rotatedZ = rotationQ.apply(vecZ)
#        print("X:",rotatedX)
#        print("Y:",rotatedY)
#        print("Z:",rotatedZ)
        #left/right test (x positive is left)
        print(rotatedX[2])
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
        #top/bottom test (y positive is left)
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
        #front/back test (z positive is left)
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

    def cleanFront(self, handPoints, mainImg):
        # Input: [Vertices of Tracked Hand], [mainImg (maybe we can make it into just the shape)], perspective from output image to memory image
        if (not self.frontExposed):
            return False
        face = Polygon(self.frontPoints)
        hand = Polygon(handPoints)
        try:
            if((not face.intersects(hand))):
                return False
        except shapely.geos.TopologyException:
            return False
        intersectionPoly = face.intersection(hand) #IMPOSSIBLE GEOMETRY PROBLEM
        intersectionPoly = list(intersectionPoly.exterior.coords)
        intersectionMask = np.zeros((mainImg.shape[0],mainImg.shape[1]), np.uint8)
        cv2.fillPoly(intersectionMask,[np.int32(intersectionPoly)],(255,255,255))
        tempColor = np.zeros((mainImg.shape[0], mainImg.shape[1],3),np.uint8)
        tempColor[:] = self.color
        transformPerspective = cv2.getPerspectiveTransform(np.array(self.frontPoints, dtype="float32"), np.array([[0,0],[memSize_X,0],[memSize_X,memSize_Y],[0,memSize_Y]],dtype="float32"))
        intersectionMask = cv2.bitwise_and(tempColor,tempColor, mask=intersectionMask)
        intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective, (memSize_X,memSize_Y))
        self.frontFace = cv2.addWeighted(self.frontFace, 1.0, intersectionMask, 0.1, 0)

    def clean(self, handPoints, mainImg):
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
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.frontPoints, dtype="float32"), np.array(
                        [[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective, (memSize_X, memSize_Y))
                    self.frontFace = cv2.addWeighted(self.frontFace, 1.0, intersectionMask, 0.1, 0)
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
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.backPoints, dtype="float32"), np.array(
                        [[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective, (memSize_X, memSize_Y))
                    self.backFace = cv2.addWeighted(self.backFace, 1.0, intersectionMask, 0.1, 0)
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
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.topPoints, dtype="float32"), np.array(
                        [[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective, (memSize_X, memSize_Y))
                    self.topFace = cv2.addWeighted(self.topFace, 1.0, intersectionMask, 0.1, 0)
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
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.bottomPoints, dtype="float32"), np.array(
                        [[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective, (memSize_X, memSize_Y))
                    self.bottomFace = cv2.addWeighted(self.bottomFace, 1.0, intersectionMask, 0.1, 0)
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
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.leftPoints, dtype="float32"), np.array(
                        [[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective, (memSize_X, memSize_Y))
                    self.leftFace = cv2.addWeighted(self.leftFace, 1.0, intersectionMask, 0.1, 0)
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
                    transformPerspective = cv2.getPerspectiveTransform(np.array(self.rightPoints, dtype="float32"), np.array(
                        [[0, 0], [memSize_X, 0], [memSize_X, memSize_Y], [0, memSize_Y]], dtype="float32"))
                    intersectionMask = cv2.bitwise_and(tempColor, tempColor, mask=intersectionMask)
                    intersectionMask = cv2.warpPerspective(intersectionMask, transformPerspective, (memSize_X, memSize_Y))
                    self.rightFace = cv2.addWeighted(self.rightFace, 1.0, intersectionMask, 0.1, 0)
            except shapely.errors.TopologicalError:
                pass


    def drawFront(self, mainImg):
        if (not self.frontExposed):
            return False
        transformPerspective = cv2.getPerspectiveTransform(np.array([[0,0],[memSize_X,0],[memSize_X,memSize_Y],[0,memSize_Y]],dtype="float32"),np.array(self.frontPoints, dtype="float32"))
        cleanedMask = cv2.warpPerspective(self.frontFace,transformPerspective, (mainImg.shape[1],mainImg.shape[0]))

        temp = np.zeros((mainImg.shape[0],mainImg.shape[1],3), np.uint8)
        tempImg = cv2.bitwise_or(temp, mainImg)
        return cleanedMask

    def drawFaces(self,mainImg):
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
            print("left draw")
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

def DrawLine(point1, point2, lineColor, lineWidth):
    global draw_main
    if not point1 is None and point2 is not None:
        draw_main.line([point1, point2], fill=lineColor, width=lineWidth)


def DrawDot(point, pointColor, pointRadius):
    global draw_main
    if point is not None:
        xy = [
            point[0] - pointRadius,
            point[1] - pointRadius,
            point[0] + pointRadius,
            point[1] + pointRadius
        ]
        draw_main.ellipse(xy,
                          fill=pointColor,
                          outline=pointColor
                          )


def DrawCube(points, color=(255, 0, 0)):
    lineWidthForDrawing = 2

    # draw front
    DrawLine(points[0], points[1], color, lineWidthForDrawing)
    DrawLine(points[1], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[0], color, lineWidthForDrawing)

    # draw back
    DrawLine(points[4], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[7], color, lineWidthForDrawing)
    DrawLine(points[4], points[7], color, lineWidthForDrawing)

    # draw sides
    DrawLine(points[0], points[4], color, lineWidthForDrawing)
    DrawLine(points[7], points[3], color, lineWidthForDrawing)
    DrawLine(points[5], points[1], color, lineWidthForDrawing)
    DrawLine(points[2], points[6], color, lineWidthForDrawing)

    # draw dots
#    DrawDot(points[0], pointColor=(255,0,0), pointRadius=4) #top left from front face
#    DrawDot(points[1], pointColor=(0, 0, 255), pointRadius=4) #top right from front face
#    DrawDot(points[2], pointColor=(0, 255, 0), pointRadius=4)  # bottom right from front face
#    DrawDot(points[3], pointColor=(255, 255, 255), pointRadius=4)  # bottom left from front face

#    DrawDot(points[0], pointColor=(255,0,0), pointRadius=4) #top right from left face
#    DrawDot(points[4], pointColor=(0, 0, 255), pointRadius=4) #top left from left face
#    DrawDot(points[7], pointColor=(0, 255, 0), pointRadius=4)  #bottom left  from left face
#    DrawDot(points[3], pointColor=(255, 255, 255), pointRadius=4)  # bottom right from left face

#    DrawDot(points[0], pointColor=(255,0,0), pointRadius=4) #top left  from top face
#    DrawDot(points[1], pointColor=(0, 0, 255), pointRadius=4) #top right  from top face
#    DrawDot(points[4], pointColor=(0, 255, 0), pointRadius=4)  #bottom left   from top face
#    DrawDot(points[5], pointColor=(255, 255, 255), pointRadius=4)  #bottom right   from top face

#    DrawDot(points[5], pointColor=(255,0,0), pointRadius=4) # bottom right from right face
#    DrawDot(points[1], pointColor=(0, 0, 255), pointRadius=4) #top right from right face
#    DrawDot(points[2], pointColor=(0, 255, 0), pointRadius=4)  #top left from right face
#    DrawDot(points[6], pointColor=(255, 255, 255), pointRadius=4)  #bottom left from right face

#    DrawDot(points[4], pointColor=(255,0,0), pointRadius=4) # top left from back face
#    DrawDot(points[5], pointColor=(0, 0, 255), pointRadius=4) #top right from back face
#    DrawDot(points[6], pointColor=(0, 255, 0), pointRadius=4)  #bottom right from back face
#    DrawDot(points[7], pointColor=(255, 255, 255), pointRadius=4)  #bottom left from back face

#    DrawDot(points[2], pointColor=(255,0,0), pointRadius=4) #top right  from bottom face
#    DrawDot(points[3], pointColor=(0, 0, 255), pointRadius=4) #top left from bottom face
#    DrawDot(points[6], pointColor=(0, 255, 0), pointRadius=4)  #bottom right from bottom face
#    DrawDot(points[7], pointColor=(255, 255, 255), pointRadius=4)  #bottom left from bottom face

    # draw x on the top
#    DrawLine(points[0], points[5], color, lineWidthForDrawing)
#    DrawLine(points[1], points[4], color, lineWidthForDrawing)




def testHandAnimated(countFrame):
        testHand = [tuple([150, 150]), tuple([150, 250]), tuple([250, 250]), tuple([250, 150])]
        #testHand = [tuple([countFrame, 150]), tuple([countFrame, 250]), tuple([countFrame+100, 250]), tuple([countFrame+100, 150])]
        DrawLine(testHand[0], testHand[1], tuple([255, 0, 255, 255]), 2)
        DrawLine(testHand[1], testHand[2], tuple([255, 0, 255, 255]), 2)
        DrawLine(testHand[2], testHand[3], tuple([255, 0, 255, 255]), 2)
        DrawLine(testHand[0], testHand[3], tuple([255, 0, 255, 255]), 2)
        return testHand
        #print(points)
    #    DrawFilling([points[0], points[1], points[2], points[3]], (255, 0, 255, 100))

# Settings
exposure_val = 166

##From YAML loader in original code
yaml_path = 'webcam_config.yaml'
with open(yaml_path, 'r') as stream:
    try:
        print("Loading DOPE parameters from '{}'...".format(yaml_path))
        params = yaml.load(stream)
        print('    Parameters loaded.')
    except yaml.YAMLError as exc:
        print(exc)

    models = {}
    pnp_solvers = {}
    pub_dimension = {}
    draw_colors = {}

    # Initialize parameters
    matrix_camera = np.zeros((3, 3))
    matrix_camera[0, 0] = params["camera_settings"]['fx']
    matrix_camera[1, 1] = params["camera_settings"]['fy']
    matrix_camera[0, 2] = params["camera_settings"]['cx']
    matrix_camera[1, 2] = params["camera_settings"]['cy']
    matrix_camera[2, 2] = 1
    dist_coeffs = np.zeros((4, 1))

    if "dist_coeffs" in params["camera_settings"]:
        dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = 0.5
    config_detect.softmax = 1000
    config_detect.thresh_angle = params['thresh_angle']
    config_detect.thresh_map = params['thresh_map']
    config_detect.sigma = params['sigma']
    config_detect.thresh_points = params["thresh_points"]

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model in params['weights']:
        models[model] = \
            ModelData(
                model,
                "weights/" + params['weights'][model]
            )
        models[model].load_net_model()

        draw_colors[model] = tuple(params["draw_colors"][model])

        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                matrix_camera,
                Cuboid3d(params['dimensions'][model]),
                dist_coeffs=dist_coeffs
            )

def angleSolver(inputVec):
    x = [1,0,0]
    y = [0,1,0]
    z = [0,0,1]
    x_angle = np.degrees(np.arccos(np.dot([x[0],0,x[2]],inputVec)))
#    x_angle = np.degrees(np.arcsin(np.abs(inputVec[0])/(np.linalg.norm(inputVec))))
    y_angle = np.degrees(np.arccos(np.dot(y,inputVec)))
#    y_angle = np.degrees(np.arcsin(np.abs(inputVec[1]) / (np.linalg.norm(inputVec))))
    z_angle = np.degrees(np.arccos(np.dot(z,inputVec)))
#    z_angle = np.degrees(np.arcsin(np.abs(inputVec[2]) / (np.linalg.norm(inputVec))))
    return [x_angle,y_angle,z_angle]

cap = cv2.VideoCapture(0)
countFrame = 0

testCleaningObject = CleaningObject(objId1=1, vertices1=np.zeros(8), img1=0, quarternion1=0, color1=(255,255,255))
test = np.zeros((480,640,3),np.uint8)
subtest = np.zeros((480,640,3),np.uint8)
handFrameCount = -100
while True:
    test = np.zeros((480, 640, 3), np.uint8)
    handFrameCount += 10
    # Reading image from camera
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Copy and draw image
    img_copy = img.copy()
    im = Image.fromarray(img_copy)
    draw_main = ImageDraw.Draw(im, mode='RGBA')

    for m in models:
        if countFrame % 5 == 0:
            results = ObjectDetector.detect_object_in_image(models[m].net, pnp_solvers[m], img, config_detect)
#            print(results)
        for i_r, result in enumerate(results):
            if result["location"] is None:
                continue
            loc = result["location"]
            ori = result["quaternion"]
            if None not in result['projected_points']:
                points2d = []
                for pair in result['projected_points']:
                    points2d.append(tuple(pair))

                testHand = [tuple([150, 150]), tuple([150, 250]), tuple([250, 250]), tuple([250, 150])]
                testCleaningObject.updateExposed(ori)
                testCleaningObject.updateFaces(points2d)
                if len(points2d) >= 8:
                    testCleaningObject.clean(handPoints=testHand,mainImg=img_copy)
                    test = testCleaningObject.drawFaces(mainImg=img_copy)
                else:
                    test = np.zeros((480, 640, 3), np.uint8)
                    subtest = np.zeros((480, 640, 3), np.uint8)
                img_copy = cv2.add(img_copy,test)
                im = Image.fromarray(img_copy)
                draw_main = ImageDraw.Draw(im, mode='RGBA')
                DrawCube(points2d, draw_colors[m])
                testHand = testHandAnimated(handFrameCount)
            else:
                print("NOTHING")


    open_cv_image = np.array(im)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
#    open_cv_image = cv2.add(open_cv_image,test)

    cv2.imshow('Open_cv_image', open_cv_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    countFrame += 1
    countFrame = 5
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
