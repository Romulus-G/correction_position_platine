# La méthode de différence de gaussienne (Difference of Gaussians ou encore DoG) 
# dépend de plusieurs paramètres. 
# Pour avoir une idée desquels il faut prendre, voici un script qui lance 
# une fenêtre interactive, avec des sliders pour faire bouger les paramêtres 
# et voir en temps réel le résultat de la différence de gaussiennes.

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib.widgets import Slider


# Difference of Gaussians : 
# - faire la différence de deux flous gaussiens (flou gaussien = convolution de l'image avec une gaussienne en 2D)
# - appliquer un flou gaussien sur le résultat (la différence)
# - faire un masque pour prendre que les valeurs assez hautes (assez proches du max de l'image)


# L'idée de la différence de gaussienne est d'obtenir la convolution de l'image par une
# fonction "Mexican hat" (la différence des convolutions est la convolution de la différence).
# Cette fonction est un passe-bande spatial, qui ne laisse passer que certaines fréquences
# spatiales. Les fréquences spatiales caractérisent la vitesse de variation des pixels sur une image.
# En choisissant le bon passe-bande, on peut garder les motifs qui correspondent au petites
# poussières et impuretés sur notre image, et supprimer le reste. On simplifie ainsi notre
# image tout en gardant assez d'information pour pouvoir déterminer la translation.



# Chaque flou gaussien dépend de 2 paramètres :
# - la taille du kernel (le kernel est la fonction gaussienne en 2D discrétisée)
# - l'écart type de la gaussienne

# le masque final dépend d'une paramètre : le seuil de tolérance

def DoG(im, 
        kernel_shape, sigma_inf, sigma_sup, 
        last_blur_kernel_shape, last_blur_sigma,
        tolerance_threshold):

    im_inf = cv.GaussianBlur(im, kernel_shape, sigmaX=sigma_inf, sigmaY=sigma_inf)
    im_sup = cv.GaussianBlur(im, kernel_shape, sigmaX=sigma_sup, sigmaY=sigma_sup)
    
    diff = im_inf - im_sup
    DoG_im = cv.GaussianBlur(diff , last_blur_kernel_shape, last_blur_sigma)
    
    mask = DoG_im > (np.max(DoG_im) - tolerance_threshold)

    return mask.astype(np.uint8) #, DoG_im 



# Tout ce qui est dessous concerne la mise en place de 
# la fenetre interactive (pas interessant à lire)
if __name__ == "__main__":

    im = [cv.imread(f"SE{i}.tif", cv.IMREAD_UNCHANGED) for i in range(1,6)]

    # init parameters--------------
    kernel_shape = (9, 9)
    sigma_inf = 19
    sigma_diff = 1
    sigma_sup = sigma_inf + sigma_diff
    
    last_blur_sigma = 5.5

    tolerance_threshold = 17000
    # -----------------------------

    mask1 = DoG(im[0], kernel_shape, sigma_inf, sigma_sup, (0,0), last_blur_sigma, tolerance_threshold)
    mask2 = DoG(im[1], kernel_shape, sigma_inf, sigma_sup, (0,0), last_blur_sigma, tolerance_threshold)

    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle("Les paramètres initiaux nous semblent pas mal")
    fig.tight_layout()
    plot_im1 = ax[0].imshow(mask1)
    contours1, hierarchy1 = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    ax[0].set_title(f"DoG sur image 1 : {len(contours1)} composantes connexes")

    plot_im2 = ax[1].imshow(mask2) ; ax[1].set_title("DoG sur image 2")
    contours2, hierarchy2 = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    ax[1].set_title(f"DoG sur image 2 : {len(contours2)} composantes connexes")

    plt.subplots_adjust(bottom=0.25)

    ax_last = plt.axes([0.7, 0.05, 0.25, 0.03])
    last_blur_slider = Slider(
        ax=ax_last,
        label='écart-type du flou gaussien appliqué à la fin',
        valmin=1,
        valmax=30,
        valinit=last_blur_sigma,
    )

    ax_sigma_inf = plt.axes([0.15, 0.11, 0.2, 0.03])
    sigma_inf_slider = Slider(
        ax=ax_sigma_inf,
        label="sigma_inf",
        valmin=1,
        valmax=100,
        valinit=sigma_inf,
    )

    ax_sigma_diff = plt.axes([0.15, 0.05, 0.2, 0.03])
    sigma_diff_slider = Slider(
        ax=ax_sigma_diff,
        label='sigma_sup - sigma_inf',
        valmin=0.01,
        valmax=15,
        valinit=sigma_diff,
    )

    ax_tol = plt.axes([0.7, 0.17, 0.25, 0.03])
    threshold_slider = Slider(
        ax=ax_tol,
        label='seuil de tolérance du masque (voir le code de la fonction DoG)',
        valmin=10,
        valmax=40000,
        valinit=tolerance_threshold,
    )

    ax_ksize = plt.axes([0.7, 0.11, 0.25, 0.03])
    ksize_slider = Slider(
        ax=ax_ksize,
        label='taille du kernel (le kernel est carré)',
        valmin=3,
        valmax=60,
        valinit=kernel_shape[0],
    )

    def update(val):
        i = int(ksize_slider.val)
        kernel_shape = (i + (i%2==0),)*2

        mask1 = DoG(im[0], 
                kernel_shape, sigma_inf_slider.val, sigma_inf_slider.val + sigma_diff_slider.val, 
                (0,0), last_blur_slider.val, 
                threshold_slider.val)    
        contours1, hierarchy1 = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        ax[0].set_title(f"DoG sur image 1 : {len(contours1)} composantes connexes")
        plot_im1.set_data(mask1)
        
        
        mask2 = DoG(im[1], 
                kernel_shape, sigma_inf_slider.val, sigma_inf_slider.val + sigma_diff_slider.val, 
                (0,0), last_blur_slider.val, 
                threshold_slider.val)
        contours2, hierarchy2 = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        ax[1].set_title(f"DoG sur image 2 : {len(contours2)} composantes connexes")
        plot_im2.set_data(mask2)
        
        fig.canvas.draw_idle()

    last_blur_slider.on_changed(update)
    sigma_inf_slider.on_changed(update)
    sigma_diff_slider.on_changed(update)
    threshold_slider.on_changed(update)
    ksize_slider.on_changed(update)

    plt.show()


