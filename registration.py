import numpy as np
import cv2 as cv

def find_translation(ref, toTranslate, filenames=True):  

    """
    On utilise la methode Scale Invariant Feature Transform (SIFT) pour obtenir des points cle (keypoints) auxquels sont associes des descripteurs. 
    A partir de ces points cle sur les deux images et leurs descripteurs, il est possible de trouver des correspondances sur les deux images. 
    
    Arguments de find_translation :

    ref : numpy array ou string ; image de référence

    toTranslate : numpy array ou string ; image à translater
    
    filenames : bool ; 
    indique si les arguments ref et toTranslate sont des noms de fichier,
    dans le cas contraire ref et toTranslate doivent être des array numpy de dtype np.uint8
    """
    
    # read images
    ref = cv.imread(ref, cv.IMREAD_GRAYSCALE)
    toTranslate = cv.imread(toTranslate, cv.IMREAD_GRAYSCALE)
    
    # On applique un flou gaussien léger pour atténuer le bruit éventuel de l'image
    ref = cv.GaussianBlur(ref, [0,0], 1.5)
    toTranslate = cv.GaussianBlur(toTranslate, [0,0], 1.5)

    # Initialisation de la méthode SIFT
    sift = cv.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(ref, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(toTranslate, None)

    # On fait correspondre les keypoints par méthode force brute
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)

    # On récupère les paires de points associés sur les chaque image. 
    pairs = np.array([[keypoints_1[match.queryIdx].pt, keypoints_2[match.trainIdx].pt] for match in matches])

    # On récupère la translation en x et y pour chacun des points
    translations = pairs[:,0] - pairs[:,1]


    # Clustering pour trouver la translation optimale dans l'espace des translations en 2D
    # On cherche la zone de l'espace des translations qui est très dense en points
    
    # Pour cela, on cherche la translation T0 qui a le plus grand nombre de voisins dans 
    # un rayon de 8 pixels autour d'elle

    x_trans, y_trans = translations.T
    
    # matrice des distances au carré
    dist_squared = (x_trans - x_trans.reshape(-1,1))**2 + (y_trans - y_trans.reshape(-1,1))**2
    
    mask = dist_squared < 64
    most_neighbours_index = mask.sum(axis=0).argmax()
    
    # on détermine la translation entre les deux images en faisant la moyenne 
    # des translations voisines de T0
    neighbours_mask = mask[most_neighbours_index]
    neighbours_translations = translations[neighbours_mask]

    best_translation = neighbours_translations.mean(axis=0).round().astype(np.int32)

    return best_translation


if __name__=="__main__":
    help(find_translation)
    print(find_translation("Nickel/Ref1.tif", "Nickel/ToBeAligned1.tif"))