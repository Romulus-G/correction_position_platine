- `registration.py` est un fichier contenant une seule fonction : `find_translation`. Celle ci prend en argument 2 images, l'image de référence et l'image à aligner renvoie la translation $(T_x,T_y)$ permettant de passer de l'une à l'autre: <br> <br>

    $ref[x,y] = toTranslate[x-T_x,y-T_y]$
    <br> <br>
    où l'axe des x est horizontal dirigé vers la droite et l'axe des y est vertical dirigé vers le bas.

    Pour utliser `find_translation` dans un autre fichier : `from registration import find_translation` (`registration.py` doit être présent dans le répertoire dudit fichier)

- `Translate.py` est notre première méthode pour trouver la translation entre deux images. Elle se base sur la détection du motif noir en forme de croix.

- `DoG_interactive_parameters.py` est une deuxième approche, plus générale, dont le but est de trouver des zones particulières dans l'image : poussières, impuretés. On se base sur le filtre "Difference of Gaussians" ou "DoG". Ce script permet de faire varier les paramètre de la DoG et de voir en temps réels les conséquences sur l'image filtrée.

- `SIFT_registration.py` est un programme qui permet de trouver des points-clé (keypoints) grâce à la méthode SIFT (Scale Invariant Feature Transform) dont le principe repose sur la méthode DoG du programme précédent. On se sert pour cela du module OpenCV-cv2 et plus particulièrement de la fonction cv2.SIFT_create(). Une fois les keypoints trouvés, on cherche les correspondances entre l'image de référence et l'image translatée avec une méthode force brute. On élimine les valeurs de translation aberrantes en cherchant dans l'espace des translations un cluster de points, qui correspondent aux correspondances de keypoints les plus fiables et robustes. Pour utiliser le programme, entrer les noms des deux fichiers image (référence et à translater) si dans le même dossier, les chemins sinon.
