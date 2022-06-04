import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Pour faire un recalage d'image, il faut d'abord trouver des points de repères sur les images.
# On peut se servir de points "saillants", i.e. des points où le contraste est élevé sur l'image (contours, délimitations, irrégularités).
# Une façon de trouver ces points est d'appliquer la méthode de différence de gradients.
# Elle consiste en la soustraction de deux images obtenues en appliquant des flous gaussiens de deux intensités différentes à l'image à traiter.


# On définit la fonction de différence de gaussiennes
def DoG(im, radius_inf = 9., radius_sup = 10.):
    im_inf = cv2.GaussianBlur(im, [0,0], radius_inf)
    im_sup = cv2.GaussianBlur(im, [0,0], radius_sup)
    # On applique un autre flou gaussien au résultat pour limiter le bruit sur l'image
    ret = cv2.GaussianBlur(im_inf - im_sup, [0,0], 3)
    return ret

# On définit désormais la fonction mask qui renvoie un masque des points les plus saillants.
def mask(im, tolerance_threshold = 10):
    DoG_image = DoG(im)
    max = np.max(DoG_image)
    mask = (DoG_image > max - tolerance_threshold).astype(np.uint8)
    return mask

# On définit une fonction first_guess qui renvoie une première approximation du vecteur de translation à appliquer à partir de la position
# moyenne des points saillants sur les deux images. Rq, cette méthode ne fonctionne pas très bien ...
def first_guess(im_ref, im, tolerance_threshold = 10):
    m_ref, m = mask(im_ref, tolerance_threshold), mask(im, tolerance_threshold)
    x_ref, y_ref = np.where(m_ref == 1)
    x, y = np.where(m == 1)
    avg_x_ref = np.average(x_ref)
    avg_y_ref = np.average(y_ref)
    avg_x = np.average(x)
    avg_y = np.average(y)
    print(x_ref, y_ref, x, y)
    print(avg_x_ref, avg_y_ref, avg_x, avg_y)
    return (avg_x_ref - avg_x, avg_y_ref - avg_y)

# Une fois que l'on a obtenu le masque avec les points saillants, on détecte les contours des motifs formés. 
# On code une fonction boundingbox qui renvoie une liste des rectangles contenant les différents motifs.
def boundingbox(im):
    '''im doit être une image binaire/noir et blanc'''
    contours, hierarchy = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
    # RETR_LIST demande à la fonction de ne pas établir de hiérarchie entre les contours, CHAIN_APPROX_NONE demande à la fonction
    # de renvoyer absolument tous les points qui constituent le contour.
    ret = []
    for elt in contours:
        ret.append(cv2.boundingRect(elt))
    return ret


if __name__ == "__main__":

    ref = cv2.cvtColor(cv2.imread("SE3.tif"), cv2.COLOR_BGR2GRAY)
    to_translate = cv2.cvtColor(cv2.imread("SE2.tif"), cv2.COLOR_BGR2GRAY)

    mask_ref = mask(ref, tolerance_threshold=30)
    mask_trans = mask(to_translate, tolerance_threshold=30)

    bbox_ref = boundingbox(mask_ref)
    bbox_trans = boundingbox(mask_trans)

    fig, ax = plt.subplots(1,4)
    ax[0].imshow(mask_ref)
    ax[1].imshow(mask_trans)
    ax[2].imshow(DoG(ref))
    ax[3].imshow(DoG(to_translate))

    for elt in bbox_ref:
        x, y, w, h = elt
        ax[0].add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor = 'red'))

    for elt in bbox_trans:
        x, y, w, h = elt
        ax[1].add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor = 'red'))

    
    plt.show()


