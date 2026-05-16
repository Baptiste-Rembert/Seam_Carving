import cv2 as cv
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix

def e1(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    return np.hypot(sobelx, sobely)

def ehog(image):
    energy = e1(image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hog = cv.HOGDescriptor()
    h = hog.compute(gray)
    h_max = h.max()
    if h_max > 0:
        energy = energy / h_max
    return energy

def compute_energy(image, function):
    return function(image)

def optimal_seam_vert(energy):
    rows, cols = energy.shape
    M = energy.copy()
    for i in range(1, rows):
        prev = M[i - 1]
        left         = np.roll(prev,  1); left[0]   = np.inf
        right        = np.roll(prev, -1); right[-1] = np.inf
        M[i] += np.minimum(prev, np.minimum(left, right))
    seam = np.empty(rows, dtype=np.int32)
    seam[-1] = np.argmin(M[-1])
    for i in range(rows - 2, -1, -1):
        j = seam[i + 1]
        lo = max(0, j - 1)
        hi = min(cols, j + 2)
        seam[i] = lo + np.argmin(M[i, lo:hi])
    costs = M[np.arange(rows), seam]
    return seam, costs

def optimal_seam_hor(energy):
    seam, costs = optimal_seam_vert(energy.T)
    return seam, costs

def remove_seam_vert(image, seam):
    rows, cols, ch = image.shape
    mask = np.ones((rows, cols), dtype=bool)
    mask[np.arange(rows), seam] = False
    return image[mask].reshape(rows, cols - 1, ch)

def remove_seam_hor(image, seam):
    return np.ascontiguousarray(
    remove_seam_vert(image.transpose(1, 0, 2), seam).transpose(1, 0, 2)
    )

def add_seam_vert(image, seam):
    rows, cols, ch = image.shape
    output = np.empty((rows, cols + 1, ch), dtype=image.dtype)
    for i in range(rows):
        j = seam[i]
        output[i, :j + 1] = image[i, :j + 1]
        left  = image[i, j - 1] if j > 0        else image[i, j]
        right = image[i, j + 1] if j < cols - 1 else image[i, j]
        output[i, j + 1] = ((left.astype(np.float64) +
                            image[i, j].astype(np.float64) +
                            right.astype(np.float64)) / 3).astype(image.dtype)
        output[i, j + 2:] = image[i, j + 1:]
    return output

def add_seam_hor(image, seam):
    return np.ascontiguousarray(
    add_seam_vert(image.transpose(1, 0, 2), seam).transpose(1, 0, 2)
    )

def _remove_seam_vert_2d(array, seam):
    rows, cols = array.shape
    mask = np.ones((rows, cols), dtype=bool)
    mask[np.arange(rows), seam] = False
    return array[mask].reshape(rows, cols - 1)

def _remove_seam_hor_2d(array, seam):
    return _remove_seam_vert_2d(array.T, seam).T

def find_k_seams_vert(image, energy_function, k):
    if k <= 0:
        return []
    working = image.copy()
    rows, cols, _ = working.shape
    if k > cols:
        raise ValueError("k > image width")
    index_map = np.tile(np.arange(cols), (rows, 1))
    seams = []
    for _ in range(k):
        seam = optimal_seam_vert(compute_energy(working, energy_function))[0]
        seams.append(index_map[np.arange(rows), seam].copy())
        working   = remove_seam_vert(working, seam)
        index_map = _remove_seam_vert_2d(index_map, seam)
    return seams

def find_k_seams_hor(image, energy_function, k):
    if k <= 0:
        return []
    working = image.copy()
    rows, cols, _ = working.shape
    if k > rows:
        raise ValueError("k > image height")
    index_map = np.tile(np.arange(rows)[:, None], (1, cols))
    seams = []
    for _ in range(k):
        seam = optimal_seam_hor(compute_energy(working, energy_function))[0]
        seams.append(index_map[seam, np.arange(cols)].copy())
        working   = remove_seam_hor(working, seam)
        index_map = _remove_seam_hor_2d(index_map, seam)
    return seams

def add_k_seams_vert(image, seams):
    output = image.copy()
    todo = [s.astype(int).copy() for s in seams]
    while todo:
        s = todo.pop(0)
        output = add_seam_vert(output, s)
        for other in todo:
            other[other >= s] += 1
    return output

def add_k_seams_hor(image, seams):
    output = image.copy()
    todo = [s.astype(int).copy() for s in seams]
    while todo:
        s = todo.pop(0)
        output = add_seam_hor(output, s)
        for other in todo:
            other[other >= s] += 1
    return output

def add_k_seams_index(seams, matrice_index):
    todo = [s.astype(int).copy() for s in seams]
    index = 1
    while todo:
        s = todo.pop(0)
        matrice_index[s, np.arange(matrice_index.shape[1])] = index
        index += 1
    return matrice_index

def transport_map(function, image, r, c):
    T = np.full((r + 1, c + 1), np.inf)
    M = np.zeros((r + 1, c + 1), dtype=np.int8)
    T[0, 0] = 0.0

    img_v = image.copy()
    vert_costs = []
    for _ in range(r):
        e = compute_energy(img_v, function)
        seam, costs = optimal_seam_vert(e)
        vert_costs.append(float(costs.sum()))
        img_v = remove_seam_vert(img_v, seam)

    img_h = image.copy()
    hor_costs = []
    for _ in range(c):
        e = compute_energy(img_h, function)
        seam, costs = optimal_seam_hor(e)
        hor_costs.append(float(costs.sum()))
        img_h = remove_seam_hor(img_h, seam)

    for i in range(1, r + 1):
        T[i, 0] = T[i - 1, 0] + vert_costs[i - 1]
        M[i, 0] = 0

    for j in range(1, c + 1):
        T[0, j] = T[0, j - 1] + hor_costs[j - 1]
        M[0, j] = 1

    for i in range(1, r + 1):
        for j in range(1, c + 1):
            cost_v = T[i - 1, j] + vert_costs[i - 1]
            cost_h = T[i, j - 1] + hor_costs[j - 1]
            if cost_v <= cost_h:
                T[i, j] = cost_v; M[i, j] = 0
            else:
                T[i, j] = cost_h; M[i, j] = 1

    return T, M

def optimal_order(M):
    r, c = M.shape[0] - 1, M.shape[1] - 1
    order = []
    while r > 0 or c > 0:
        if r == 0:
            order.append("hor"); c -= 1
        elif c == 0:
            order.append("vert"); r -= 1
        elif M[r, c] == 0:
            order.append("vert"); r -= 1
        else:
            order.append("hor"); c -= 1
    return order[::-1]

def image_resize_down_naive(image, energy_function, height, length):
    out = image.copy()
    rows, cols, _ = out.shape
    for _ in range(rows - height):
        out = remove_seam_hor(out, optimal_seam_hor(compute_energy(out, energy_function))[0])
    for _ in range(cols - length):
        out = remove_seam_vert(out, optimal_seam_vert(compute_energy(out, energy_function))[0])
    return out

def image_resize_down(image, energy_function, height, length):
    out = image.copy()
    r = image.shape[0] - height
    c = image.shape[1] - length
    if r == 0 and c == 0:
        return out
    _, M = transport_map(energy_function, out, r, c)
    for direction in optimal_order(M):
        energy_map = compute_energy(out, energy_function)
        if direction == "vert":
            out = remove_seam_vert(out, optimal_seam_vert(energy_map)[0])
        else:
            out = remove_seam_hor(out, optimal_seam_hor(energy_map)[0])
    return out

def image_resize_up(image, energy_function, height, length, max_step_ratio=0.5):
    out = image.copy()
    while out.shape[0] < height or out.shape[1] < length:
        rows, cols, _ = out.shape
        if max_step_ratio <= 0:
            raise ValueError("max_step_ratio must be > 0")
        add_rows = max(0, min(height - rows,
                              max(1, int(np.floor(rows * max_step_ratio)))))
        add_cols = max(0, min(length - cols,
                              max(1, int(np.floor(cols * max_step_ratio)))))
        if add_cols > 0:
            out = add_k_seams_vert(out, find_k_seams_vert(out, energy_function, add_cols))
        if add_rows > 0:
            out = add_k_seams_hor(out, find_k_seams_hor(out, energy_function, add_rows))
    return out

def image_amplification(image, energy_function, scale_factor=1.5):
    image_height = image.shape[0]
    image_length = image.shape[1]

    #up = image_resize_up(image, energy_function, height, length)

    new_w = int(image_length * scale_factor)
    new_h = int(image_height * scale_factor)

    up = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    return image_resize_down(up, energy_function, image_height, image_length)

def image_resize(image, energy_function, height, length):
    if height < image.shape[0] and length < image.shape[1]:
        return image_resize_down(image, energy_function, height, length)
    elif height > image.shape[0] and length > image.shape[1]:
        return image_resize_up(image, energy_function, height, length)
    raise ValueError("Mixed up/down resizing is not supported.")

def compute_divergence(gx, gy):
    """Calcule la divergence d'un champ de vecteurs (gx, gy)."""
    h, w = gx.shape
    div = np.zeros((h, w), dtype=np.float64)
    # Différences arrières pour correspondre au Laplacien
    div[:, 1:] += gx[:, 1:] - gx[:, :-1]
    div[:, 0] += gx[:, 0]
    div[1:, :] += gy[1:, :] - gy[:-1, :]
    div[0, :] += gy[0, :]
    return div

def poisson_reconstruct_channel(gx, gy, boundary_image):
    """Reconstruit un canal (2D) à partir de ses gradients via Poisson."""
    h, w = gx.shape
    div = compute_divergence(gx, gy)
    
    N = h * w
    A = lil_matrix((N, N))
    b = np.zeros(N, dtype=np.float64)
    
    # Remplissage du système linéaire
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            # Conditions aux limites (pixels sur les bords)
            if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                A[idx, idx] = 1
                b[idx] = boundary_image[i, j]
            else:
                # Équation de Poisson pour les pixels internes
                A[idx, idx] = -4
                A[idx, idx - w] = 1  # Voisin du haut
                A[idx, idx + w] = 1  # Voisin du bas
                A[idx, idx - 1] = 1  # Voisin de gauche
                A[idx, idx + 1] = 1  # Voisin de droite
                b[idx] = div[i, j]
    
    # Résolution
    print("Résolution du système de Poisson (cela peut prendre quelques secondes)...")
    x = spsolve(A.tocsr(), b)
    
    # Reformater le vecteur solution en image et cliper les valeurs valides
    result = np.clip(x.reshape((h, w)), 0, 255).astype(np.uint8)
    return result

def seam_carving_gradient_down(image, energy_function, target_h, target_w):
    """
    Implémentation complète de la section 4.5 :
    Retire les coutures dans le domaine du gradient et reconstruit l'image.
    """
    out = image.copy()
    r = image.shape[0] - target_h
    c = image.shape[1] - target_w
    
    if r == 0 and c == 0:
        return out

    # 1. Extraire et calculer les gradients (différences finies) pour chaque canal
    channels = [out[:, :, i].astype(np.float64) for i in range(3)]
    grads_x = []
    grads_y = []
    
    for ch in channels:
        gx = np.zeros_like(ch)
        gy = np.zeros_like(ch)
        # Gradient X = pixel de droite - pixel actuel
        gx[:, :-1] = ch[:, 1:] - ch[:, :-1]
        # Gradient Y = pixel du bas - pixel actuel
        gy[:-1, :] = ch[1:, :] - ch[:-1, :]
        grads_x.append(gx)
        grads_y.append(gy)
        
    # 2. Trouver l'ordre optimal de suppression
    _, M = transport_map(energy_function, out, r, c)
    order = optimal_order(M)
    
    # 3. Boucle de suppression
    for direction in order:
        energy_map = compute_energy(out, energy_function)
        
        if direction == "vert":
            seam, _ = optimal_seam_vert(energy_map)
            out = remove_seam_vert(out, seam)
            # Retirer la couture des gradients de CHAQUE canal
            for i in range(3):
                grads_x[i] = _remove_seam_vert_2d(grads_x[i], seam)
                grads_y[i] = _remove_seam_vert_2d(grads_y[i], seam)
        else:
            seam, _ = optimal_seam_hor(energy_map)
            out = remove_seam_hor(out, seam)
            # Retirer la couture des gradients de CHAQUE canal
            for i in range(3):
                grads_x[i] = _remove_seam_hor_2d(grads_x[i], seam)
                grads_y[i] = _remove_seam_hor_2d(grads_y[i], seam)
                
    # 4. Reconstruction finale via le solveur de Poisson pour chaque canal
    print(f"Reconstruction de l'image ({target_h}x{target_w})...")
    result = np.zeros_like(out)
    for i in range(3):
        print(f"  Traitement du canal {['Bleu', 'Vert', 'Rouge'][i]}...")
        # out[:,:,i] sert ici de condition aux limites (boundary_image)
        result[:, :, i] = poisson_reconstruct_channel(grads_x[i], grads_y[i], out[:, :, i])
        
    return result

# Seam carving in the gradient domain
def image_amplification_gradient(image, energy_function):
    image_height = image.shape[0]
    image_length = image.shape[1]

    #up = image_resize_up(image, energy_function, height, length)

    new_w = image_length - 200
    new_h = image_height - 200

    down = seam_carving_gradient_down(image, energy_function, new_h, new_w)
    return image_resize_up(down, energy_function, image_height, image_length)

if __name__ == "__main__":
    image = cv.imread('images-20100824/family.png')
    if image is None:
        raise FileNotFoundError("image not found")
    new_w = image.shape[1] - 150
    new_h = image.shape[0] - 150
    print("Début du redimensionnement dans le domaine du gradient...")
    result_gradient = seam_carving_gradient_down(image, ehog, new_h, new_w)
    print("Redimensionnement terminé.")
    cv.imshow('Original', image)
    cv.imshow('Amplified', result_gradient)
    cv.waitKey(0)
    cv.destroyAllWindows()
