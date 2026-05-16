import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from seam_carving import e1

def compute_consistent_maps(img, energy_func=e1):
    rows, cols, _ = img.shape
    energy = energy_func(img)
    
    # 1. Calcul des coutures verticales (0-connectées)
    v_paths = np.zeros((rows, cols), dtype=np.int32)
    v_paths[0, :] = np.arange(cols) # Initialisation à la première ligne
    
    # Stocker les arêtes diagonales utilisées: set de ((y, x), (y+1, x'))
    used_diagonals_v = set()
    
    print("Calcul des coutures verticales (Algorithme Hongrois)...")
    for y in range(rows - 1):
        # Matrice de coût
        cost_matrix = np.full((cols, cols), 1e9, dtype=np.float32)
        for x in range(cols):
            # Voisins valides
            for dx in [-1, 0, 1]:
                x_next = x + dx
                if 0 <= x_next < cols:
                    cost_matrix[x, x_next] = energy[y+1, x_next]
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        v_paths[y+1, col_ind] = v_paths[y, row_ind]
        
        # Enregistrer les diagonales utilisées
        for i in range(cols):
            x1 = row_ind[i]
            x2 = col_ind[i]
            if x1 != x2:
                used_diagonals_v.add(((y, x1), (y+1, x2)))

    # Organiser et trier les coutures verticales pour obtenir la V-Map
    v_seam_energies = np.zeros(cols)
    for c in range(cols):
        mask = (v_paths == c)
        v_seam_energies[c] = energy[mask].sum()
        
    removal_order_v = np.argsort(v_seam_energies) 
    v_order_to_time = {seam_id: t+1 for t, seam_id in enumerate(removal_order_v)}
    
    V_map = np.zeros((rows, cols), dtype=np.int32)
    for y in range(rows):
        for x in range(cols):
            seam_id = v_paths[y, x]
            V_map[y, x] = v_order_to_time[seam_id]

    # 2. Calcul des coutures horizontales avec contrainte de consistance
    h_paths = np.zeros((rows, cols), dtype=np.int32)
    h_paths[:, 0] = np.arange(rows)
    
    print("Calcul des coutures horizontales (Algorithme Hongrois)...")
    for x in range(cols - 1):
        cost_matrix = np.full((rows, rows), 1e9, dtype=np.float32)
        for y in range(rows):
            for dy in [-1, 0, 1]:
                y_next = y + dy
                if 0 <= y_next < rows:
                    # check de la consistance (pas de croisement diagonal)
                    # La verticale a utilisé (x, y_next) -> (x+1, y) ?
                    conflict = False
                    if dy == 1:
                        # H-seam: (y, x) -> (y+1, x+1). V-seam diagonale croisée: (y, x+1) -> (y+1, x)
                        if ((y, x+1), (y+1, x)) in used_diagonals_v:
                            conflict = True
                    elif dy == -1:
                        # H-seam: (y, x) -> (y-1, x+1). V-seam diagonale croisée: (y-1, x) -> (y, x+1)
                        if ((y-1, x), (y, x+1)) in used_diagonals_v:
                            conflict = True
                            
                    if not conflict:
                        cost_matrix[y, y_next] = energy[y_next, x+1]
                        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        h_paths[col_ind, x+1] = h_paths[row_ind, x]
        
    h_seam_energies = np.zeros(rows)
    for r in range(rows):
        mask = (h_paths == r)
        h_seam_energies[r] = energy[mask].sum()
        
    removal_order_h = np.argsort(h_seam_energies)
    h_order_to_time = {seam_id: t+1 for t, seam_id in enumerate(removal_order_h)}
    
    H_map = np.zeros((rows, cols), dtype=np.int32)
    for y in range(rows):
        for x in range(cols):
            seam_id = h_paths[y, x]
            H_map[y, x] = h_order_to_time[seam_id]

    return V_map, H_map

def instant_resize_2d(img, V_map, H_map, target_w, target_h):
    rows, cols, _ = img.shape
    seams_to_remove_w = cols - target_w
    seams_to_remove_h = rows - target_h
    
    # 1. Traiter la largeur (réduction ou agrandissement)
    temp_img = np.zeros((rows, target_w, 3), dtype=img.dtype)
    temp_H_map = np.zeros((rows, target_w), dtype=np.int32)
    
    for y in range(rows):
        row_pixels = img[y]
        row_h = H_map[y]
        row_v = V_map[y]
        
        if seams_to_remove_w > 0:
            # Réduction : on garde > seams_to_remove
            mask_v = row_v > seams_to_remove_w
            ext_img = row_pixels[mask_v]
            ext_h = row_h[mask_v]
        else:
            # Agrandissement : On duplique les k premiers pixels (ordre d'énergie)
            k = -seams_to_remove_w
            ext_img = []
            ext_h = []
            for x in range(cols):
                ext_img.append(row_pixels[x])
                ext_h.append(row_h[x])
                # Duplication si le pixel fait partie des "premières" coutures
                if row_v[x] <= k:
                    ext_img.append(row_pixels[x])
                    ext_h.append(row_h[x])
            ext_img = np.array(ext_img)
            ext_h = np.array(ext_h)

        valid_w = min(target_w, len(ext_img))
        temp_img[y, :valid_w] = ext_img[:valid_w]
        temp_H_map[y, :valid_w] = ext_h[:valid_w]
        
    # 2. Traiter la hauteur sur l'image temporaire
    new_img = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    
    for x in range(target_w):
        col_pixels = temp_img[:, x]
        col_h = temp_H_map[:, x]
        
        if seams_to_remove_h > 0:
            mask_h = col_h > seams_to_remove_h
            ext_img2 = col_pixels[mask_h]
        else:
            # Agrandissement
            k = -seams_to_remove_h
            ext_img2 = []
            for y in range(rows):
                ext_img2.append(col_pixels[y])
                if col_h[y] <= k:
                    ext_img2.append(col_pixels[y])
            ext_img2 = np.array(ext_img2)

        valid_h = min(target_h, len(ext_img2))
        new_img[:valid_h, x] = ext_img2[:valid_h]
        
    return new_img

if __name__ == '__main__':
    img_path = 'images-20100824/Woman.png'
    img = cv.imread(img_path)
    # Réduction forte de l'image de base car l'algorithme hongrois sur des matrices N*N est très lourd
    #img = cv.resize(img, (300, int(300 * img.shape[0] / img.shape[1])))
    
    print(f"Dimension : {img.shape}")
    V_map, H_map = compute_consistent_maps(img)
    
    original_h, original_w = img.shape[:2]
    target_width = original_w
    target_height = original_h

    window_name = 'Multisize 2D - Fleches: redimensionner, ESC: quitter'
    cv.namedWindow(window_name)

    print("Commandes:")
    print(" - Flèche Droite  : Agrandir la largeur")
    print(" - Flèche Gauche  : Réduire la largeur")
    print(" - Flèche Haut    : Agrandir la hauteur")
    print(" - Flèche Bas     : Réduire la hauteur")
    print(" - Echap ou 'q'   : Quitter")

    while True:
        resized = instant_resize_2d(img, V_map, H_map, target_width, target_height)
        cv.imshow(window_name, resized)
        
        key = cv.waitKeyEx(0)
        if key == 27 or key == ord('q'): # ESC
            break
        elif key in (65363, 2555904, 83): # Right Arrow
            target_width += 5
            print(f"Nouvelle largeur: {target_width}")
        elif key in (65361, 2424832, 81): # Left Arrow
            if target_width > 20: 
                target_width -= 5
                print(f"Nouvelle largeur: {target_width}")
        elif key in (65364, 2621440, 84): # Down Arrow
            if target_height > 20:
                target_height -= 5
                print(f"Nouvelle hauteur: {target_height}")
        elif key in (65362, 2490368, 82): # Up Arrow
            target_height += 5
            print(f"Nouvelle hauteur: {target_height}")

    cv.destroyAllWindows()
