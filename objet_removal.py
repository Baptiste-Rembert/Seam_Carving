import cv2 as cv
import numpy as np
from seam_carving import (e1, optimal_seam_vert, optimal_seam_hor, 
                          remove_seam_vert, remove_seam_hor, 
                          _remove_seam_vert_2d, _remove_seam_hor_2d,
                          add_k_seams_vert, find_k_seams_vert,
                          add_k_seams_hor, find_k_seams_hor)

# Variables globales pour le dessin
drawing = False
mode = 'remove' # 'remove' ou 'protect'
ix, iy = -1, -1

def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing, mode, remove_mask, protect_mask, img_display

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            if mode == 'remove':
                cv.circle(remove_mask, (x, y), 5, 255, -1)
                cv.circle(img_display, (x, y), 5, (0, 0, 255), -1) # Vert
            else:
                cv.circle(protect_mask, (x, y), 5, 255, -1)
                cv.circle(img_display, (x, y), 5, (0, 255, 0), -1) # Rouge
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

def interactive_object_removal(image_path):
    global remove_mask, protect_mask, img_display, mode
    
    img = cv.imread(image_path)
    if img is None: return
    
    img_display = img.copy()
    rows, cols, _ = img.shape
    remove_mask = np.zeros((rows, cols), dtype=np.uint8)
    protect_mask = np.zeros((rows, cols), dtype=np.uint8)

    window_name = 'Interface: r=Suppr(Vert), k=Proteger(Rouge), ESPACE=Lancer'
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, draw_mask)

    while True:
        cv.imshow(window_name, img_display)
        key = cv.waitKey(1) & 0xFF
        if key == 32: # Touche ESPACE pour valider
            break
        elif key == ord('r'): # Appui sur 'r' pour remove
            mode = 'remove'
            print("Mode remove sélectionné (suppression)")
        elif key == ord('p'): # Appui sur 'p' pour protect
            mode = 'protect'
            print("Mode protect sélectionné (conservation)")
        elif key == 27: # ESC pour quitter
            cv.destroyAllWindows()
            return

    # Lancement du processus de suppression
    print("Calcul de la suppression d'objet...")
    result = process_removal(img, remove_mask > 0, protect_mask > 0)
    
    # Optionnel : Remise à la taille originale (Section 4.6 du papier)
    final_size = restore_size(result, img.shape[0], img.shape[1])
    
    cv.imshow('Resultat Final', final_size)
    cv.waitKey(0)
    cv.destroyAllWindows()

def process_removal(img, r_mask, p_mask):
    out = img.copy()
    curr_r = r_mask.copy()
    curr_p = p_mask.copy()

    while np.any(curr_r):
        # Modification de l'énergie selon imret.pdf
        energy = e1(out)
        energy[curr_p] = 100000.0   # Tres élevé pour protéger
        energy[curr_r] = -100000.0  # Tres faible pour supprimer[cite: 1]

        # On choisit la direction selon le plus petit diamètre[cite: 1]
        h_diam = np.sum(np.any(curr_r, axis=0))
        v_diam = np.sum(np.any(curr_r, axis=1))

        if h_diam <= v_diam:
            seam, _ = optimal_seam_vert(energy)
            out = remove_seam_vert(out, seam)
            curr_r = _remove_seam_vert_2d(curr_r, seam)
            curr_p = _remove_seam_vert_2d(curr_p, seam)
        else:
            seam, _ = optimal_seam_hor(energy)
            out = remove_seam_hor(out, seam)
            curr_r = _remove_seam_hor_2d(curr_r, seam)
            curr_p = _remove_seam_hor_2d(curr_p, seam)
    return out

def restore_size(img, target_h, target_w):
    """Réinsère des seams pour retrouver la taille d'origine."""
    out = img.copy()
    if out.shape[1] < target_w:
        k = target_w - out.shape[1]
        seams = find_k_seams_vert(out, e1, k)
        out = add_k_seams_vert(out, seams)
    if out.shape[0] < target_h:
        k = target_h - out.shape[0]
        seams = find_k_seams_hor(out, e1, k)
        out = add_k_seams_hor(out, seams)
    return out

if __name__ == "__main__":
    interactive_object_removal('images-20100824/family.png')