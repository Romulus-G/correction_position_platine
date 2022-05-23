import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


# Pour commencer, on localise le motif approximativement. Pour cela, on remarque que le 
# motif peut être caractérisé : c'est le seul endroit sur les photos avec des zones de
# pixels noirs purs (couleur = 0) juste à côté de pixels blancs purs (couleur = 0xffff).  

def locate_cross(im, visualization=None):
    # pixels noirs
    I_black, J_black = np.where(im == 0) # (on pourrait prendre une condition moins restrictive, c'est un paramètre à expérimenter)

    # parmi ces pixels, lesquels sont dans une zone très noire, mais contenant au moins un pixel blanc ? 
    # Ceux là appartiennent à coup sûr au motif. ("zone", "noir", "blanc" sont des notions expressément vagues)
    I, J = [], []
    for i,j in zip(I_black, J_black):
        close_neighborhood = im[i-1:i+2, j-1:j+2]
        great_neighborhood = im[i-6:i+6, j-6:j+6]
        if (close_neighborhood == 0).sum() >= 7 and (great_neighborhood == 0xffff).any(): 
            # La taille des zones, les couleurs retenues par les filtres et les seuils après filtrages sont des paramètres 
            # sur lesquels on peut jouer. Il semble y avoir de la marge sur les valeurs acceptables (qui permettent de 
            # bien caractériser les pixels du motif) grâce au contraste élevé entre le motif et le reste de l'image
            I.append(i)
            J.append(j)

    # Bordure approximative du motifs (bounding box approximative)
    mi, Mi = min(I), max(I)
    mj, Mj = min(J), max(J)

    # Pour visualiser ce qu'il se passe (faire des plots)
    if visualization != None:
        On_the_cross_pixels = np.zeros((Mi - mi + 1, Mj - mj + 1), dtype=bool)
        On_the_cross_pixels[I - mi, J - mj] = True
        visualization.imshow(On_the_cross_pixels)

    return mi, Mi, mj, Mj



# Construction de la sous-image de référence :
# on agrandit la bounding box approximative pour être sûr qu'elle contient le motif en (quasi-)entièreté. (étape pas très intéressante)
def get_ref_rect(im, mi, Mi, mj, Mj):
    Delta_i, Delta_j = Mi - mi, Mj - mj

    # valeurs d'agrandissement tout à fait empiriques, basées sur le fait que la bounding box approximative 
    # se trouve être une très bonne approximation en pratique (avec les paramètres utilisées plus hauts)
    height = Delta_i + Delta_i//3      # hauteur
    width  = Delta_j + Delta_j//3      # largeur

    di = Delta_i//6
    dj = Delta_j//6

    top, left = (mi - di), (mj - dj)    # coin en haut à gauche

    return top, left, height, width, di, dj, im[top:top+height, left:left+width]



# Détermination de la translation optimale par méthode force brute à partir d'une bonne approximation : 
# on possède les locations approximatives des motifs de 2 photos, l'une étant considérée comme celle de référence,
# donc on possède une première approximation de la transaltion recherchée, qu'on affine en la perturbant un peu.
def optimal_translation(to_compare_ref_rect, height, width, top, left, im, visualization=False):
    best_top, best_left = None, None
    min_dist = 1e9

    if visualization:
        fig.suptitle("Faut désactiver les animations si on veut que le programme termine en un instant \n i.e mettre visualization=False dans l'appel de optimal_translation")
        fig.canvas.draw() ; renderer = fig.canvas.renderer
    
    # on fait varier notre première approximation pour trouver la translation la plus exacte possible :
    # l'intervalle des perturbations est un paramètre modifiable, on a pris un peu large ici : sur les photos exemples 
    # la première approximation de translation est à environ 4 - 5 pixels d'écart (écart horizontal + vertical)
    for i in range(-10, 10):
        for j in range(-10, 10):
            new_top, new_left = top+i, left+j
            if new_left+width >= im.shape[1] or new_left < 0 or new_top+height >= im.shape[0] or new_top < 0: 
                continue 
            
            # nouvelle sous-image à comparer avec la sous-image de référence
            rect = im[new_top:new_top+height, new_left:new_left+width]
            
            # identification des pixels qui seront utilisés pour la compraison dans la nouvelle sous-image :
            # la comparaison sera effectuée que sur ces pixels filtrés (sensibilité des filtres est un paramètre),
            # ici on ne retient que les pixels noirs purs et blancs purs.
            black_pixels_mask = (rect == 0).astype(np.int8)
            white_pixels_mask = (rect == 0xffff).astype(np.int8) 

            # matrice "comparable"
            to_compare = black_pixels_mask - white_pixels_mask

            # calcul de la distance associée à la norme 1 entre les deux matrices
            diff = abs(to_compare_ref_rect - to_compare)            
            dist = diff.sum()

            if dist < min_dist:
                min_dist = dist
                best_top = new_top
                best_left = new_left

                if visualization:
                    ax[0,2].cla() ; ax[0,2].set_title(f"Best so far : {i},{j}") ; ax[0,2].imshow(diff) ; ax[0,2].draw(renderer) ; plt.pause(0.05)
    
            if visualization:
                ax[1,2].cla() ; ax[1,2].set_title(f"{i},{j}") ; ax[1,2].imshow(diff) ; ax[1,2].draw(renderer) ; plt.pause(0.05)

    return best_top, best_left
            

if __name__ == "__main__":
    # Setup pour la visualisation
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    # traitement de l'image de référence
    ref_name = "SE3.tif"
    print(f"Image de référence : {ref_name}")
    ref_im = imread(ref_name)
    ref_mi, ref_Mi, ref_mj, ref_Mj = locate_cross(ref_im, visualization=ax[0,0])
    ref_top, ref_left, height, width, di, dj, ref_rect = get_ref_rect(ref_im, ref_mi, ref_Mi, ref_mj, ref_Mj)

    # identification des pixels du motif dans la sous-image de référence
    black_pixels_mask = (ref_rect == 0).astype(np.int8)
    white_pixels_mask = (ref_rect == 0xffff).astype(np.int8) 

    # matrice utile pour la comparaison avec les sous-images des images à translater (matrice "comparable")
    to_compare_ref_rect = black_pixels_mask - white_pixels_mask

    # image à translater
    im_name = "SE2.tif"
    print(f"Image à translater : {im_name}")
    im = imread(im_name)
    mi, Mi, mj, Mj = locate_cross(im, visualization=ax[1,0])
    top, left = mi - di, mj - dj

    # visualisation
    ax[0,0].set_title("image Ref: pixels detectés comme appartenant au motif", fontsize=8)
    ax[1,0].set_title("image à translater : pixels détectés comme \n appartenants au motif", fontsize=8)
    ax[0,1].imshow(ref_im[ref_top:ref_top+height, ref_left:ref_left+width])
    ax[0,1].set_title("image Ref : cette zone devrait contenir le motif", fontsize=8)
    ax[1,1].imshow(im[top:top+height, left:left+width])
    ax[1,1].set_title("image à translater : cette zone devrait contenir le motif", fontsize=8)

    print(f"première approximation de la translation : ({ref_top - top}, {ref_left - left})")

    # recherche de la meilleure translation possible à l'aide de la première approximation
    new_top, new_left = optimal_translation(to_compare_ref_rect, height, width, top, left, im, visualization=True)
    translate_i, translate_j = ref_top - new_top, ref_left - new_left
    print(f"Translation optimale : ({translate_i}, {translate_j})")

    plt.show()