- `Translate.py` est notre première méthode pour trouver la translation entre deux images. Elle se base sur la détection du motif noir en forme de croix.

- `DoG_interactive_parameters.py` est une deuxième approche, plus générale, dont le but est de trouver des zones particulières dans l'image : poussières, impuretés. On se base sur le filtre "Difference of Gaussians" ou "DoG". Ce script permet de faire varier les paramètre de la DoG et de voir en temps réels les conséquences sur l'image filtrée.

- `SIFT_registration.py` est un programme qui permet de trouver des points-clé (keypoints) grâce à la méthode SIFT (Scale Invariant Feature Transform) dont le principe repose sur la méthode DoG du programme précédent. On se sert pour cela du module OpenCV-cv2 et plus particulièrement de la fonction cv2.SIFT_create(). Une fois les keypoints trouvés, on cherche les correspondances entre l'image de référence et l'image translatée avec une méthode force brute. On élimine les valeurs de translation aberrantes et on calcule la moyenne sur les données restantes. On obtient alors une bonne approximation de la translation entre les deux images. 
