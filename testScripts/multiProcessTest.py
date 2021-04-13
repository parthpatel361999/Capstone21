import multiprocessing
from encoderTest import *

import time

STOP_EXECUTION = "230495872345"

def test1(inQueue,pointList1, pointUpdated1):
    while True:
        print("Hello1_______________________________________________________")
#        inQueue.clear()
#        del pointList[:]
        point = inQueue.get()
        if point == STOP_EXECUTION:
            break
        if pointUpdated1 == 2:
            continue

        inputList1 = createTestSamples(9)
        inputList12 = listEncoder(inputList1)
#        pointList1.clear()
        for i in range(len(pointList1)):
            pointList1.pop()
        for item in inputList12:
            pointList1.append(item)
        time.sleep(2)
        pointUpdated1.value = 1
        """

        for count in range(len(output1)):
            if(count < len(inputList1)):
                output1[count] = inputList1[count]
            else:
                output1[count] = -999.999
        """

def test2(inQueue,pointList2, pointUpdated2):
    tempList = []
    while True:
        #print("Expected:", pointCount.value)
        #print("Actual:", len(pointList2))
        point = inQueue.get()
        if point == STOP_EXECUTION:
            break
        if pointUpdated2.value == 1:
            print("_______________________________")
            pointUpdated2.value = 2
            tempList = []
            tester2 = listDecoder(pointList2).copy()
            for item in pointList2:
                tempList.append(item)
            pointUpdated2.value = 0
        for item in tempList:
            print(item)

def main():
    print("HelloMain")

    manager = multiprocessing.Manager()
    pointQueue = manager.Queue()
    pointList = manager.list()
    pointCount = manager.Value('i', 0)

    processPool = multiprocessing.Pool()

    testProc1 = processPool.apply_async(test1, (pointQueue, pointList, pointCount))
    testProc2 = processPool.apply_async(test2, (pointQueue, pointList, pointCount))

    while True:
        pointQueue.put(0)
        time.sleep(0.5)


    pointQueue.put(STOP_EXECUTION)
    processPool.close()
    processPool.join()
    return

if __name__ == '__main__':
    main()

"""
    mainTest1 = multiprocessing.Array('d', 18)
    mainTest2 = multiprocessing.Array('d', 3)
    for count in range(len(mainTest1)):
        mainTest1[count] = -999.999
    while True:
        print("NEW")
        proc1 = multiprocessing.Process(target=test1, args=(mainTest1, 0))
        proc2 = multiprocessing.Process(target=test2, args=(mainTest2, mainTest1))
        proc1.start()
        proc2.start()
"""