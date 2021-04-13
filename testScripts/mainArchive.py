import cleaningObj
import multiprocessing
import yaml
from cleaningObj import *
from cuboid import *
from cuboid_pnp_solver import *
from detector import *
from encoders import *

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




def mainCameraIn(inQueue):
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return


def mainCameraOut(inQueue, outImg, updatedC):
    img = np.zeros((640, 480, 3), np.uint8)
    while True:
        print("cam out")
        print(list(outImg))
        queueVal = inQueue.get()
        if queueVal == STOP_EXECUTION:
            break
        if updatedC.value == 1:
            subImg = list(outImg)
            subImg = imgDecoder(subImg, 640, 480, 3)
            img = np.asarray(subImg, dtype=np.uint8)
            updatedC.value = 0
        #cv2.imshow('Cleaning Output', img)
        #cv2.waitKey(0)
    return


def mainCamera(inQueue, inImg, outImg, updated, updatedC):
    img = np.zeros((640, 480, 3), np.uint8)
    while True:
        print("cam run")
        queueVal = inQueue.get()
        if queueVal == STOP_EXECUTION:
            break
        for i in range(len(inImg)):
            inImg.pop()
        for i in range(len(outImg)):
            outImg.pop()
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        subImg = img.tolist()
        subImg = imgEncoder(subImg)
        for item in subImg:
            inImg.append(item)
        updatedC.value = 1
        if updated == 1:
            updated = 0
        if updatedC.value == 1:
            subImg = list(outImg)
            subImg = imgDecoder(subImg, 640, 480, 3)
            img = np.asarray(subImg, dtype=np.uint8)
            updatedC.value = 0
        cv2.imshow('Cleaning Output', img)
        cv2.waitKey(1)

    return


def mainDOPE(inQueue, inImg, outImg, updated):
    while True:
        queueVal = inQueue.get()
        if queueVal == STOP_EXECUTION:
            break
        for m in models:
            if countFrame % 5 == 0:
                results = ObjectDetector.detect_object_in_image(models[m].net, pnp_solvers[m], img, config_detect)
            #            print(results)
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
                    tempImg = cleaningObj.drawCompleteMask(points2d, testCleaningObject, ori, testHand, img)
                else:
                    print("NOTHING")
    return


def mainHand():
    return


# Camera to DOPE --> Img
# Camera to Hand --> Img
# DOPE to Camera --> Points, Orientation, maybe Score?
# Hand to Camera --> Points

def main():
    manager = multiprocessing.Manager()
    mainQueue = manager.Queue()
    listDOPE = manager.list()
    listCamera_in = manager.list()
    listCamera_out = manager.list()
    listHand = manager.list()
    updatedMem = manager.Value('i', 0)
    updatedCam = manager.Value('i', 0)

    processPool = multiprocessing.Pool()

    processCamera = processPool.apply_async(mainCamera,
                                            (mainQueue, listCamera_in, listCamera_out, updatedMem, updatedCam))
#    cameraOutTest = processPool.apply_async(mainCameraOut, (mainQueue, listCamera_in, updatedCam))
    #    processDOPE = processPool.apply_async(mainDOPE,())
    #    processHand = processPool.apply_async(mainHand,())

    while True:
        # print("idk")
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        mainQueue.put(0)

    mainQueue.put(STOP_EXECUTION)
    processPool.close()
    processPool.join()

    return


#if __name__ == '__main__':
    #main()
#cap.release()
#cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
frameRate = 30
prev = 0
countFrame = 0

testCleaningObject = cleaningObj.CleaningObject(objId1=1, vertices1=np.zeros(8), quarternion1=0, color1=(0, 255, 0))
handFrameCount = -100
while True:
    handFrameCount += 10
    # Reading image from camera
    timeElapsed = time.time() - prev
    ret, img = cap.read()
    if not timeElapsed > 1.0/frameRate:
        continue
    prev = time.time()
    tempImg = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    for m in models:
        if countFrame % 5 == 0:
            results = ObjectDetector.detect_object_in_image(models[m].net, pnp_solvers[m], img, config_detect)
            countFrame = 0
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
                tempImg = cleaningObj.drawCompleteMask(points2d,testCleaningObject,ori,testHand,img)
            else:
                print("NOTHING")

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.add(img,tempImg)

    cv2.imshow('Cleaning Output', img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    countFrame += 1
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

