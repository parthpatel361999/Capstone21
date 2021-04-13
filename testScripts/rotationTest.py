x1 = 0.06121995097107449
y1 = 0.9387979585933692
z1 =  -0.3388127152624723
w1 = -0.010791409569312625

import numpy as np
from pyrr import quaternion
from scipy.spatial.transform import Rotation

q1 = quaternion.create(x=x1, y=y1, z=z1, w=w1)

r1 = Rotation.from_quat([q1[0], q1[1], q1[2], q1[3]])
r2 = Rotation.from_quat([ 0.12779437,  0.95200852, -0.22087233,  0.16900827]) #front, left, and bottom exposed
print(r1.as_euler('xyz', degrees=True))
print(r1.as_euler('XYZ', degrees=True))
print(r1.as_euler('zyx', degrees=True))
print(r1.as_euler('ZYX', degrees=True))
print()
print(r1.as_euler('ZXY',degrees=True))
print(r2.as_euler('ZXY',degrees=True))
# if Y is positive, then front face is exposed; otherwise, back face is exposed
# if Z is positive, then top is exposed
# if X is positive, then right face is exposed
print()

vec = np.array([1,0,0])
rotated = r1.apply(vec)
print(rotated)
