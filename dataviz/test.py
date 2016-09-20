import numpy as np
from matplotlib import *
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


with open('data.json') as jsonData:
    data = json.load(jsonData)

    size = data['vocab_size']
    mapping = data['word_to_id']
    points = data['embed']

    i = 0
    for point in points:
        if i < 10:
            ax.scatter(xs=point[0]*1000, ys=point[1]*1000, zs=point[2]*1000, c='b', marker='o')
            label = list(mapping.keys())[list(mapping.values()).index(i)]
            ax.text(point[0]*1000, point[1]*1000, point[2]*1000, label, size=14, zorder=1, color='k')


        i+=1



# ax.scatter(xs=1, ys=0, zs=0, c='b', marker='o')
# ax.text(1, 1, 1, '0', size=14, zorder=1, color='k')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()