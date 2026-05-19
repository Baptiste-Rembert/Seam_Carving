import cv2 as cv
import numpy as np
import argparse
from seam_carving import find_k_seams_vert, find_k_seams_hor, add_k_seams_vert, add_k_seams_hor, e1

def enlarge_dimension(image, target_size, is_width=True, energy_func=e1, max_fraction=0.5):
    """
    Agrandit l'image selon une dimension par étapes pour prévenir les artefacts d'étirement.
    is_width=True pour la largeur (coutures verticales), False pour la hauteur (coutures horizontales).
    """
    current_image = image.copy()
    
    while True:
        rows, cols = current_image.shape[:2]
        current_size = cols if is_width else rows
        delta = target_size - current_size
        
        if delta <= 0:
            break
            
        # Ne pas agrandir de plus de la fraction maximale (max_fraction) en une seule étape
        step = min(delta, max(1, int(current_size * max_fraction)))
            
        print(f"Etape d'agrandissement de {step} pixels (taille actuelle: {current_size}, cible: {target_size})")
        
        if is_width:
            seams = find_k_seams_vert(current_image, energy_func, step)
            current_image = add_k_seams_vert(current_image, seams)
        else:
            seams = find_k_seams_hor(current_image, energy_func, step)
            current_image = add_k_seams_hor(current_image, seams)
            
    return current_image

def enlarge_image(image, target_rows, target_cols, energy_func=e1, max_fraction=0.5):
    rows, cols = image.shape[:2]
    result = image.copy()
    
    # Agrandir d'abord la largeur
    if target_cols > cols:
        print(f"Agrandissement de la largeur de {cols} à {target_cols}...")
        result = enlarge_dimension(result, target_cols, is_width=True, energy_func=energy_func, max_fraction=max_fraction)
        
    # Ensuite, agrandir la hauteur
    if target_rows > rows:
        print(f"Agrandissement de la hauteur de {rows} à {target_rows}...")
        result = enlarge_dimension(result, target_rows, is_width=False, energy_func=energy_func, max_fraction=max_fraction)
        
    return result

def main():
    parser = argparse.ArgumentParser(description="Agrandissement d'image par Seam Carving (Content-Aware)")
    parser.add_argument("input", help="Chemin vers l'image d'entrée")
    parser.add_argument("output", help="Chemin vers l'image de sortie")
    parser.add_argument("--dw", type=int, default=0, help="Pixels à ajouter à la largeur")
    parser.add_argument("--dh", type=int, default=0, help="Pixels à ajouter à la hauteur")
    parser.add_argument("--max_fraction", type=float, default=0.5, help="Fraction maximale d'agrandissement par étape (défaut 0.5)")
    
    args = parser.parse_args()
    
    image = cv.imread(args.input)
    if image is None:
        print(f"Erreur de chargement de l'image : {args.input}")
        exit(1)
        
    rows, cols = image.shape[:2]
    target_rows = rows + args.dh
    target_cols = cols + args.dw
    
    if target_rows <= rows and target_cols <= cols:
        print("Aucun agrandissement demandé. Veuillez spécifier --dw ou --dh (> 0).")
        exit(0)
        
    print(f"Taille d'origine : {cols}x{rows}")
    print(f"Taille cible : {target_cols}x{target_rows}")
    
    result = enlarge_image(image, target_rows, target_cols, max_fraction=args.max_fraction)
    
    cv.imwrite(args.output, result)
    print(f"Image agrandie sauvegardée dans {args.output}")

if __name__ == '__main__':
    main()
