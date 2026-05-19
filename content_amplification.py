import cv2 as cv
import numpy as np
import argparse
from seam_carving import e1, compute_energy, optimal_seam_vert, optimal_seam_hor, remove_seam_vert, remove_seam_hor

def amplify_content(image, scale_factor, energy_func=e1):
    """
    Amplifie le contenu de l'image tout en préservant sa taille d'origine.
    Combine un redimensionnement classique (scaling) suivi d'un seam carving pour revenir à la taille initiale.
    """
    if scale_factor <= 1.0:
        print("Le facteur d'échelle doit être > 1.0 pour l'amplification.")
        return image
    
    orig_rows, orig_cols = image.shape[:2]
    
    # 1. Redimensionnement standard pour agrandir l'image
    scaled_cols = int(orig_cols * scale_factor)
    scaled_rows = int(orig_rows * scale_factor)
    
    print(f"1. Redimensionnement standard de {orig_cols}x{orig_rows} à {scaled_cols}x{scaled_rows}...")
    # Utilisation d'INTER_CUBIC, généralement performant pour l'upsampling
    scaled_image = cv.resize(image, (scaled_cols, scaled_rows), interpolation=cv.INTER_CUBIC)
    
    result = scaled_image.copy()
    
    # 2. Application du seam carving pour réduire à la taille d'origine
    delta_cols = scaled_cols - orig_cols
    delta_rows = scaled_rows - orig_rows
    
    print(f"2. Seam carving vers la taille d'origine...")
    print(f"   Suppression de {delta_cols} coutures verticales...")
    for i in range(delta_cols):
        energy = compute_energy(result, energy_func)
        seam, _ = optimal_seam_vert(energy)
        result = remove_seam_vert(result, seam)
        if (i + 1) % 10 == 0 or (i + 1) == delta_cols:
            print(f"     Suppression de {i+1}/{delta_cols} coutures verticales")
            
    print(f"   Suppression de {delta_rows} coutures horizontales...")
    for i in range(delta_rows):
        energy = compute_energy(result, energy_func)
        seam, _ = optimal_seam_hor(energy)
        result = remove_seam_hor(result, seam)
        if (i + 1) % 10 == 0 or (i + 1) == delta_rows:
            print(f"     Suppression de {i+1}/{delta_rows} coutures horizontales")
            
    return result

def main():
    parser = argparse.ArgumentParser(description="Amplification de contenu par Seam Carving")
    parser.add_argument("input", help="Chemin vers l'image d'entrée")
    parser.add_argument("output", help="Chemin vers l'image de sortie")
    parser.add_argument("--scale", type=float, default=1.2, help="Multiplicateur d'échelle (ex: 1.2 pour 20%% d'amplification) (défaut: 1.2)")
    
    args = parser.parse_args()
    
    image = cv.imread(args.input)
    if image is None:
        print(f"Erreur de chargement de l'image : {args.input}")
        exit(1)
        
    if args.scale <= 1.0:
        print("Le facteur d'échelle doit être supérieur à 1.0")
        exit(1)
        
    result = amplify_content(image, args.scale)
    
    cv.imwrite(args.output, result)
    print(f"Image amplifiée sauvegardée dans {args.output}")

if __name__ == '__main__':
    main()
