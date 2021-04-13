import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon

p1 = Polygon([(0,0), (0,1), (1,1), (1,0)])
p2 = Polygon([(0.5,0), (1.5,0), (1.5,1.5),(0.5,1.5)])
print(p1.intersects(p2))