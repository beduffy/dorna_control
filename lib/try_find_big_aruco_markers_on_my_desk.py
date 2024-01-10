from cv2 import aruco
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl


# marker_length = 0.0275
# marker_length = 0.0935  # big marker

# marker_separation = 0.06  # TODO OMFG it is supposed to be 6 milimetres not 6 cm
# marker_separation = 0.006
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
#  'DICT_4X4_100', 'DICT_4X4_1000', 'DICT_4X4_250', 'DICT_4X4_50', 'DICT_5X5_100', 'DICT_5X5_1000', 
# 'DICT_5X5_250', 'DICT_5X5_50', 'DICT_6X6_100', 'DICT_6X6_1000', 'DICT_6X6_250', 'DICT_6X6_50', 
# 'DICT_7X7_100', 'DICT_7X7_1000', 'DICT_7X7_250', 'DICT_7X7_50',
parameters = aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
# parameters.adaptiveThreshWinSizeMin = 3
# parameters.adaptiveThreshWinSizeStep = 4  # todo test more and see if it makes worse/better
nx = 4
ny = 3
# nx = 6
# ny = 6
# board = cv2.aruco.GridBoard_create(ny, nx, marker_length, marker_separation, aruco_dict)

fig = plt.figure()

for i in range(1, nx * ny + 1):
    ax = fig.add_subplot(ny, nx, i)
    img = aruco.drawMarker(aruco_dict, i, 700)
    plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
    ax.axis("off")

fig.tight_layout()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig("data/markers.pdf")
plt.show()

# import pdb;pdb.set_trace()