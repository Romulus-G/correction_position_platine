import numpy as np
import matplotlib.pyplot as plt
import cv2

# On utilise la méthode Scale Invariant Feature Transform (SIFT) pour obtenir des points clé (keypoints) auxquels sont associés des descripteurs. 
# A partir de ces points clé sur les deux images et leurs descripteurs, il est possible de trouver des correspondances sur les deux images. 

# read images
img1 = cv2.imread('SE2.tif')  
img2 = cv2.imread('SE3.tif') 

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialisation de la méthode SIFT
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

# On fait correspondre les keypoints par méthode force brute
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
plt.imshow(img3),plt.show()

# On récupère les paires de points associés sur les chaque image. 
pairs = [[(keypoints_1[match.queryIdx].pt, keypoints_2[match.trainIdx].pt)] for match in matches]
print(pairs)


