# CREDIT: https://github.com/Abdul-Mukit/dope_exp/blob/master/demo/live_dope_webcam.py

import numpy as np
import queue, yaml, threading, time, cv2
from cuboid import *
from detector import *
from cuboid_pnp_solver import *
from PIL import Image
from PIL import ImageDraw




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


def DrawFilling(pointSet, fillColor):
    global draw_main
    if pointSet is not None:
        draw_main.polygon(pointSet, fill=fillColor)

        #length = np.linalg.norm(pointSet[0]-pointSet[1])
       # width = np.linalg.norm(pointSet[1],pointSet[2])
       # tempMask = Image.new('RGBA',(length,width),"black")
        #tempPxls = tempMask.load()
        #for i in range(tempMask.size[0]):
        #    for j in range(tempMask.size[1]):
                #tempPxls[i,j] = (i,j,100)
        #draw_main.bitmap(pointSet,tempPxls,fill=fillColor)


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
    DrawDot(points[0], pointColor=color, pointRadius=4)
    DrawDot(points[1], pointColor=color, pointRadius=4)

    # draw x on the top
    DrawLine(points[0], points[5], color, lineWidthForDrawing)
    DrawLine(points[1], points[4], color, lineWidthForDrawing)

    #print(points)
    DrawFilling([points[0], points[1], points[2], points[3]], (255, 0, 255, 100))


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


def cameraStart():
    print("Something")

def inferPoints():
    print("Hello")


if __name__ == '__main__':
    print("Hello")







cap = cv2.VideoCapture(0)
countFrame = 0
while True:
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
                DrawCube(points2d, draw_colors[m])

    open_cv_image = np.array(im)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    cv2.imshow('Open_cv_image', open_cv_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    countFrame += 1
    countFrame = 5
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
