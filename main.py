import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import cleaningObj
import imutils
import yaml
from cleaningObj import *
from cuboid import *
from cuboid_pnp_solver import *
from detector import *
from handModel.handDetector import *
from imutils.video import WebcamVideoStream
from imutils.video import FPS

# Constants
[memSize_X, memSize_Y] = [500, 500]
STOP_EXECUTION = "230495872345"

# Settings
exposure_val = 166

# YAML Loader for DOPE
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
    dimensions = {}
    mainObjects = {}
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
        dimensions[model] = tuple(params['dimensions'][model])
        mainObjects[model] = cleaningObj.CleaningObject(objId1=1, vertices1=np.zeros(8), quarternion1=0, color1=params['draw_colors'][model], dimensions=dimensions[model])
detectHand_net = HandDetector()
cap = WebcamVideoStream(src=0).start()
frameCtrl = FPS().start()
isHandDetected = False
initialRun = True
frameCount = 0
frameThres = 10
while True:
    frame = cap.read()
    frameDOPE = frame.copy()
    frameDOPE = cv2.cvtColor(frameDOPE, cv2.COLOR_BGR2RGB)
    frameHand = frame.copy()
    cleaningMask = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
    if frameCount % frameThres == 0:
        handCoords = detectHand_net.detectHand(frameHand)
    for m in models:
        if initialRun:
            results = ObjectDetector.detect_object_in_image(models[m].net, pnp_solvers[m], frameDOPE, config_detect)
            handCoords = None
            initialRun = False
        else:
            if handCoords is None and (frameCount % frameThres == 0):
                results = ObjectDetector.detect_object_in_image(models[m].net, pnp_solvers[m], frameDOPE, config_detect)
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

                #testHand = [tuple([150, 150]), tuple([150, 250]), tuple([250, 250]), tuple([250, 150])]
                cleaningMask = cleaningObj.drawCompleteMask(points2d, mainObjects[m], ori, score, handCoords, frame)
            else:
                print("NOTHING")
    #img = cv2.cvtColor(frameDOPE, cv2.COLOR_RGB2BGR)
    img = cv2.add(frame, cleaningMask)
    if handCoords is not None and len(handCoords) >= 4:
        cv2.rectangle(img, (handCoords[0],handCoords[1]),(handCoords[2],handCoords[3]),[0,0,255],3)
    cv2.imshow('Cleaning Output', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frameCtrl.update()
    if frameCount % frameThres == 0:
        frameCount = 0
    frameCount+=1
frameCtrl.stop()
cap.stop()
cv2.destroyAllWindows()

