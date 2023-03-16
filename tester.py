import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image

print("libraries imported")

#Conversion
for file in os.listdir():
    filename, extension = os.path.splitext(file)
    if extension == ".pgm":
        new_file = "{}.jpg".format(filename)
        with Image.open(file) as im:
            im.save(new_file)
            print(filename + ".jpg" + " has been saved")

mapImg = cv2.imread("MapSpin.jpg")
mapImg = cv2.resize(mapImg, (0,0), fx=1, fy=1)
imgplot = plt.imshow(mapImg, cmap="gray")
plt.show()

grayscaled = cv2.cvtColor(mapImg,cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(grayscaled, 100, 255, cv2.THRESH_BINARY_INV)
retval2, threshold2 = cv2.threshold(grayscaled, 230, 255, cv2.THRESH_BINARY_INV)
plt.figure()
f, axarr = plt.subplots(1,2)
axarr[0].imshow(threshold, cmap="gray")
axarr[1].imshow(threshold2, cmap="gray")

plt.imshow(threshold2)

edged = cv2.Canny(threshold2, 0, 10)
plt.imshow(edged)
contours, hierarchy = cv2.findContours(threshold2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
co = cv2.drawContours(threshold2 ,contours, -1, (255, 0, 0), 2)
plt.imshow(co, cmap="Paired")
cv2.imshow('Contours', co)
cv2.waitKey(0)
cv2.destroyAllWindows()