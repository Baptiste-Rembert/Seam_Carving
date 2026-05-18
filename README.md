# Seam Carving

Projet de redimensionnement d'images et suppression d'objets par la méthode "Seam Carving".

## Structure du Répertoire

L'ensemble du code source en Python se trouve à la racine du projet. 

- **Fichiers exécutables (Scripts principaux) :**
  - `batch_evaluate.py` / `batch_compare_improved.py` : Scripts de traitement par lots permettant d'évaluer divers algorithmes d'énergie.
  - `compare_energy_improved.py` : Script pour comparer les approches de redimensionnement selon différentes méthodes de calcul d'énergie.
  - `eval_metrics.py` : Script pour calculer et évaluer des métriques sur le rendu final.
  - `objet_removal.py` : Script dédié à la suppression ciblée d'objets dans une image.
  - `multisize_images.py` : Script pour redimensionner vers de multiples dimensions.
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

Les scripts du projet s'exécutent en ligne de commande et utilisent souvent `argparse` pour récupérer leurs paramètres en arguments (comme les chemins des images ou facteurs de réduction). 

**Exemples de commandes :**

1. Visualiser les paramètres attendus par un script :
   ```bash
   python compare_energy_improved.py -h
   ```

2. Exemple d'exécution basique (à adapter selon les arguments requis par l'outil) :
   ```bash
   python eval_metrics.py [vos_arguments]
   ```
   ```bash
   python batch_evaluate.py [vos_arguments]
   ```