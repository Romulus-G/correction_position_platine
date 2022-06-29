import numpy as np
import matplotlib.pyplot as plt
import cv2

# On utilise la méthode Scale Invariant Feature Transform (SIFT) pour obtenir des points clé (keypoints) auxquels sont associés des descripteurs. 
# A partir de ces points clé sur les deux images et leurs descripteurs, il est possible de trouver des correspondances sur les deux images. 

# read images
im_ref = str(input("Entrer le nom de l'image de référence : "))
im_trans = str(input("Entrer le nom de l'image translatée : "))

img1 = cv2.imread(im_ref)
img2 = cv2.imread(im_trans)

# On applique un flou gaussien léger pour atténuer le bruit éventuel de l'image
img1 = cv2.GaussianBlur(img1, [0,0], 1.5)
img2 = cv2.GaussianBlur(img2, [0,0], 1.5)

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
pairs = np.array([[keypoints_1[match.queryIdx].pt, keypoints_2[match.trainIdx].pt] for match in matches])

# On récupère la translation en x et y pour chacun des points
trans_mat = np.array([[pairs[i,1][0] - pairs[i,0][0], pairs[i,1][1] - pairs[i,0][1]] for i in range(len(pairs))])

# On élimine les translations aberrantes, i.e celles qui sont trop éloignées de la moyenne. 
avg_trans = np.average(trans_mat, axis = 0)
std_trans = np.std(trans_mat, axis = 0) # écart-type


non_ab_trans_mat = []
for i in range(len(trans_mat)):
    if np.abs(avg_trans[0] - trans_mat[i, 0]) < std_trans[0]:
        if np.abs(avg_trans[1] - trans_mat[i, 1]) < std_trans[1]:
            non_ab_trans_mat.append(trans_mat[i])
non_ab_trans_mat = np.array(non_ab_trans_mat)

# Clustering pour trouver la translation optimale dans l'espace des translations en 2D
best_count = 0
best_list = []
opt_tr = np.zeros((2,))
for tr in non_ab_trans_mat:
    count = 0
    list = []
    for elt in non_ab_trans_mat:
        if np.linalg.norm(tr - elt) < 10 :
            count += 1
            list.append(elt)
    if count > best_count:
        best_count = count
        best_list = list
        opt_tr = tr
best_list = np.array(best_list)

fig, ax = plt.subplots()
circle = plt.Circle(opt_tr, 10, fill = False, edgecolor = 'red')
ax.add_patch(circle)
ax.scatter(non_ab_trans_mat[:, 0], non_ab_trans_mat[:, 1], marker='x')
ax.set_xlabel("Translation selon x")
ax.set_ylabel("Translation selon y")
plt.show()

print(f"Image de référence : {im_ref}")
print(f"Image translatée : {im_trans}")
print(f"la translation optimale est {np.round(np.average(best_list, axis = 0))}")