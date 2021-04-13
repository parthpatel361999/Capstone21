import random
import pandas as pd
import numpy as np


def createTestSamples(count):
    tempNum = -1
    tempTuple = []
    mainList = []
    for i in range(count):
        for a in range(2):
            randRaw = random.random()
            randShift = random.randrange(4)
            randNeg = random.choice([1, -1])
            tempNum = randRaw * (10 ** randShift) * randNeg
            tempTuple.append(tempNum)
        mainList.append(tuple(tempTuple))
        tempTuple = []
    return mainList

def listEncoder(rawList):
    encoded = []
    for tuple in rawList:
        for item in tuple:
            encoded.append(item)
    return encoded
def listDecoder(encodedList):
    if len(encodedList) % 2 != 0:
        raise Exception("ERROR WITH ENCODED POINT LIST")
    decodedList = []
    tempTuple = []
    for item in encodedList:
        if item == -999.999:
            break
        tempTuple.append(item)
        if len(tempTuple) == 2:
            decodedList.append(tuple(tempTuple))
            tempTuple = []
    return decodedList

if __name__ == '__main__':
    raw = createTestSamples()
    encode = listEncoder(raw)
    decode = listDecoder(encode)
    print(raw)
    print()
    print(decode)