- `Translate.py` est notre première méthode pour trouver la translation entre deux images. Elle se base sur la détection du motif noir en forme de croix.

- `DoG_interactive_parameters.py` est une deuxième approche, plus générale, dont le but est de trouver des zones particulières dans l'image : poussières, impuretés. On se base sur le filtre "Difference of Gaussians" ou "DoG". Ce script permet de faire varier les paramètre de la DoG et de voir en temps réels les conséquences sur l'image filtrée.

- `SIFT_registration.py` 