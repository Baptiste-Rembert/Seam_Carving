# Seam Carving

Projet de redimensionnement d'images et suppression d'objets par la méthode "Seam Carving".

## Structure du Répertoire

L'ensemble du code source en Python se trouve à la racine du projet. 

- **Fichiers exécutables (Démos Interactives & Principales) :**
  - `objet_removal.py` : Script interactif dédié à la suppression ciblée d'objets dans une image.
  - `multisize_images.py` : Démo de redimensionnement interactif (généralement sur 1 axe).
  - `multisize_images_improved.py` (ou `_copy.py`) : Démo avancée de redimensionnement interactif cohérent en 2D (utilisant l'algorithme hongrois).
  - *Autres scripts utilitaires* : `compare_energy_improved.py`, `batch_evaluate.py`, `eval_metrics.py` (scripts d'analyse, comparaison d'énergie et évaluation des métriques de qualité).
- **Bibliothèque et Cœur Algorithmique :**
  - `seam_carving.py` : Contient les fonctions maîtresses de l'algorithme (calcul des gradients, méthodes d'énergie (e1, eHOG), et recherche de chemins minimaux). Il est sollicité par les scripts ci-dessus.

## Prérequis et Installation

Pour exécuter le code, **Python 3.x** est nécessaire accompagné des dépendances listées dans le fichier `requirements.txt`.

Dépendances clés : `numpy`, `scipy`, `opencv-python`, `Pillow`, `matplotlib`, `scikit-image`.

Pour configurer votre environnement et installer les dépendances, effectuez la commande suivante à la racine :

```bash
python -m pip install -r requirements.txt
```

*(Nous vous recommandons fortement de le faire dans un environnement virtuel `.venv`).*

## Comment exécuter le code

Le point fort de ce projet repose sur ses interfaces interactives. Voici comment lancer les fonctionnalités principales :

### 1. Suppression d'Objet (Object Removal)
Lance une interface où vous pouvez détourer un objet à la volée pour l'effacer intelligemment de l'image.
```bash
python objet_removal.py chemin/vers/mon_image.jpg
```

### 2. Redimensionnement Interactif (Multisize 1D)
Permet de réduire ou d'agrandir l'image interactivement avec les flèches du clavier, avec calcul préalable d'une "Index Map".
```bash
python multisize_images.py chemin/vers/mon_image.jpg
```

### 3. Redimensionnement Cohérent 2D (Multisize Improved)
Utilise une approche matricielle poussée (algorithme hongrois) pour adapter la hauteur et la largeur en croisant l'image avec cohérence.
```bash
python multisize_images_improved.py chemin/vers/mon_image.jpg
```

### 4. Agrandissement d'Image (Image Enlarging)
Permet d'agrandir une image au-delà de sa taille d'origine en dupliquant intelligemment les coutures (seams) d'énergie minimale. Opère par fractions successives pour éviter les effets d'étirement.
```bash
python image_enlarging.py chemin/vers/mon_image.jpg resultat.jpg --dw 200 --dh 100
```

### 5. Amplification de Contenu (Content Amplification)
Amplifie les traits majeurs du contenu (effet loupe) tout en conservant exactement les dimensions de base de l'image. Implémente un agrandissement géométrique classique suivi d'une réduction par Seam Carving.
```bash
python content_amplification.py chemin/vers/mon_image.jpg resultat.jpg --scale 1.2
```

*(Note : D'autres scripts d'analyse de performances ou d'évaluation tels que `eval_metrics.py`, `compare_energy_improved.py` et `batch_evaluate.py` sont également lançables en ligne de commande. Vous pouvez utiliser l'argument `-h` pour voir leurs paramètres, par exemple `python compare_energy_improved.py -h`).*