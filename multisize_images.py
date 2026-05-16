import cv2 as cv
import numpy as np
from seam_carving import (e1, optimal_seam_vert, optimal_seam_hor, 
                          remove_seam_vert, remove_seam_hor, 
                          _remove_seam_vert_2d, _remove_seam_hor_2d,
                          find_k_seams_vert, add_seam_vert)

def prepare_multisize_image(img, max_expansion=0, energy_func=e1):
    """
    Pré-calcule l'évolution (Index Map) pour la réduction ET l'agrandissement.
    Retourne la représentation compacte: I_max (image maximale) et V_max (carte d'indices).
    """
    rows, cols, _ = img.shape
    V_orig = np.zeros((rows, cols), dtype=np.int32)
    original_indices = np.tile(np.arange(cols), (rows, 1))
    current_img = img.copy()
    
    for t in range(1, cols):
        energy = energy_func(current_img)
        seam, _ = optimal_seam_vert(energy)
        for i in range(rows):
            V_orig[i, original_indices[i, seam[i]]] = t
        current_img = remove_seam_vert(current_img, seam)
        original_indices = _remove_seam_vert_2d(original_indices, seam)
        
    if max_expansion <= 0:
        return img.copy(), V_orig
        
    # Agrandissement: on insère des coutures avec indices négatifs (-1, -2...)
    seams_to_add = find_k_seams_vert(img, energy_func, max_expansion)
    I_max = img.copy()
    V_max = V_orig.copy()
    todo = [s.astype(int).copy() for s in seams_to_add]
    
    k_index = -1
    while todo:
        s = todo.pop(0)
        I_max = add_seam_vert(I_max, s)
        
        # Insérer l'index négatif pour la nouvelle couture
        new_V = np.zeros((rows, V_max.shape[1] + 1), dtype=np.int32)
        for i in range(rows):
            j = s[i]
            new_V[i, :j+1] = V_max[i, :j+1]
            new_V[i, j+1] = k_index
            new_V[i, j+2:] = V_max[i, j+1:]
        V_max = new_V
        
        for other in todo:
            other[other >= s] += 1
        k_index -= 1
        
    return I_max, V_max

def compute_vertical_index_map(img, energy_func=e1):
    """
    Pré-calcule l'évolution de l'image (Index Map V) pour le redimensionnement en largeur.
    Chaque pixel retiré lors de la t-ième itération se voit attribuer la valeur t.
    """
    rows, cols, _ = img.shape
    V = np.zeros((rows, cols), dtype=np.int32)
    
    # On garde une trace des indices originaux des colonnes pour chaque ligne
    original_indices = np.tile(np.arange(cols), (rows, 1))
    
    current_img = img.copy()
    
    # On retire cols-1 pixels pour créer la carte d'indices
    # (ou on s'arrête avant selon les besoins, mais pour la carte complète :)
    for t in range(1, cols):
        energy = energy_func(current_img)
        seam, _ = optimal_seam_vert(energy)
        
        # Enregistrer l'indice de suppression t dans la carte V 
        # aux emplacements d'origine correspondants
        for i in range(rows):
            orig_j = original_indices[i, seam[i]]
            V[i, orig_j] = t
            
        # Supprimer le chemin dans l'image courante et dans le tableau des indices
        current_img = remove_seam_vert(current_img, seam)
        original_indices = _remove_seam_vert_2d(original_indices, seam)
        
        # Pour aller plus vite, on peut s'arrêter si on a juste besoin d'un redimensionnement maximal donné.
        # Mais pour la carte complète, on continue jusqu'à ce qu'il ne reste qu'une colonne.
        
    return V

def fast_resize_width(I_max, V_max, original_width, target_width):
    """
    Retaillage (réduction ou agrandissement) via la Carte d'Indices.
    Pour une réduction m-m', on filtre > m-m'
    Pour une expansion +k, on filtre >= -k
    """
    rows, cols, channels = I_max.shape
    seams_to_remove = original_width - target_width
    
    new_img = np.zeros((rows, target_width, channels), dtype=I_max.dtype)
    
    for i in range(rows):
        if seams_to_remove > 0:
            # Réduction: on garde les coutures "fortes" (> seams_to_remove)
            # et le pixel qui ne bouge jamais (qui a reçu la valeur 0)
            mask = (V_max[i] > seams_to_remove) | (V_max[i] == 0)
        else:
            # Agrandissement: on garde tout l'original (>= 0) plus les 
            # coutures synthétiques ajoutées (>= seams_to_remove)
            mask = V_max[i] >= seams_to_remove
            
        extracted = I_max[i][mask]
        new_img[i] = extracted[:target_width]
        
    return new_img

def simple_row_removal(img, target_height):
    """
    Suppression de lignes droites dégénérées pour l'autre direction.
    Au lieu de calculer un vrai seam horizontal, on supprime simplement
    les lignes du bas (ou celles de moindre énergie globale).
    """
    rows = img.shape[0]
    lines_to_remove = rows - target_height
    if lines_to_remove <= 0: return img
    
    # Exemple dégénéré: on enlève juste le bas
    return img[:target_height, :, :]

def multisize_resize(I_max, V_max, original_width, target_width, target_height):
    """
    Combine les deux méthodes : vrai seam carving pour la largeur (avec Index Map, supporte l'agrandissement)
    et simple suppression de lignes pour la hauteur.
    """
    # 1. Redimensionnement rapide en largeur via la carte V_max
    img_resized_w = fast_resize_width(I_max, V_max, original_width, target_width)
    
    # 2. Suppression dégénérée (colonnes pleines) en hauteur (ou ajout dégénéré simple)
    if target_height < img_resized_w.shape[0]:
        final_img = simple_row_removal(img_resized_w, target_height)
    else:
        # Si on agrandit la hauteur, ajout de padding pour éviter crash (lignes dégénérées)
        h_diff = target_height - img_resized_w.shape[0]
        padding = np.zeros((h_diff, target_width, img_resized_w.shape[2]), dtype=img_resized_w.dtype)
        final_img = np.vstack([img_resized_w, padding])
    
    return final_img

if __name__ == '__main__':
    img_path = 'images-20100824/Woman.png'
    img = cv.imread(img_path)
    if img is None:
        print(f"Erreur : Impossible de lire l'image {img_path}")
        exit()

    # On réduit la taille pour le test afin que le pré-calcul aille vite
    # img = cv.resize(img, (300, int(300 * img.shape[0] / img.shape[1])))
    
    print("Pré-calcul de l'Index Map en cours (cela peut prendre du temps)...")
    original_h, original_w = img.shape[:2]
    max_expansion = 50 
    I_max, V_max = prepare_multisize_image(img, max_expansion=max_expansion)
    print("Pré-calcul terminé !")

    window_name = 'Multisize - ->: +large, <-: -large, v: -haut, ESC: quitter'
    cv.namedWindow(window_name)
    
    target_width = original_w
    target_height = original_h

    print("Commandes:")
    print(" - Flèche Droite  : Agrandir la largeur")
    print(" - Flèche Gauche  : Réduire la largeur")
    print(" - Flèche Bas     : Réduire la hauteur (dégénéré)")
    print(" - Echap          : Quitter")

    while True:
        resized = multisize_resize(I_max, V_max, original_w, target_width, target_height)
        cv.imshow(window_name, resized)
        
        key = cv.waitKeyEx(0)
        if key == 27 or key == ord('q'): # ESC
            break
        elif key in (65363, 2555904, 83): # Right Arrow (Linux/Windows/Mac)
            if target_width < original_w + max_expansion:
                target_width += 5
                print(f"Nouvelle largeur: {target_width}")
        elif key in (65361, 2424832, 81): # Left Arrow
            if target_width > 50: 
                target_width -= 5
                print(f"Nouvelle largeur: {target_width}")
        elif key in (65364, 2621440, 84): # Down Arrow
            if target_height > 50:
                target_height -= 5
                print(f"Nouvelle hauteur: {target_height}")
        elif key in (65362, 2490368, 82): # Up Arrow
            # Agrandir en hauteur
            target_height += 5
            print(f"Nouvelle hauteur: {target_height}")

    cv.destroyAllWindows()


