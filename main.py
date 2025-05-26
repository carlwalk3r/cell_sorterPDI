import cv2
import numpy as np
from skimage.util import invert

def reconstruccion_morfologica(marker, mask, kernel):
    prev = np.zeros_like(marker)
    curr = marker.copy()
    while True:
        prev[:] = curr
        dil = cv2.dilate(curr, kernel, iterations=1)
        curr = cv2.bitwise_and(dil, mask)
        if np.array_equal(curr, prev):
            break
    return curr


# -----------------------------
# ITEM 1: Generar Imagen A
# -----------------------------
img = cv2.imread('imagen_entrada.jpg', cv2.IMREAD_GRAYSCALE)
_, bin_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
h, w = bin_img.shape

marker = np.zeros_like(bin_img)
marker[0, :]   = bin_img[0, :]
marker[-1, :]  = bin_img[-1, :]
marker[:, 0]   = bin_img[:, 0]
marker[:, -1]  = bin_img[:, -1]

B = np.array([[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]], dtype=np.uint8)

bordes_recon = reconstruccion_morfologica(marker, bin_img, B)
not_bordes   = invert(bordes_recon)
img_A_bin    = bin_img & not_bordes       
mask_A       = (1 - img_A_bin) * 255       
cv2.imwrite('imagen_A.png', mask_A.astype(np.uint8))


# -----------------------------
# ITEM 2: Generar Imagen B (solo citoplasmas)
# -----------------------------
A_bin = (mask_A // 255).astype(np.uint8)   

marker2 = np.zeros_like(A_bin)
marker2[0, :]   = A_bin[0, :]
marker2[-1, :]  = A_bin[-1, :]
marker2[:, 0]   = A_bin[:, 0]
marker2[:, -1]  = A_bin[:, -1]

background_recon = reconstruccion_morfologica(marker2, A_bin, B)
not_background   = invert(background_recon)
holes            = A_bin & not_background  

img_B = (holes * 255).astype(np.uint8)
img_B = cv2.bitwise_not(img_B)
cv2.imwrite('imagen_B.png', img_B)


# -----------------------------
# ITEM 3: Generar Imagen C (células agujereadas)
# -----------------------------
A_bin       = (255 - mask_A) // 255     
B_holes_bin = (255 - img_B)   // 255     

recon_C = reconstruccion_morfologica(B_holes_bin, A_bin, B)
img_C   = (1 - recon_C) * 255           
cv2.imwrite('imagen_C.png', img_C.astype(np.uint8))


# -----------------------------
# ITEM 4: Generar Imagen D (células sin agujeros, Tipo 1)
# -----------------------------
A_bin = (255 - mask_A) // 255   
C_bin = (255 - img_C)   // 255  

D_bin = cv2.bitwise_and(A_bin, cv2.bitwise_not(C_bin))  
mask_D = (1 - D_bin) * 255  
cv2.imwrite('imagen_D.png', mask_D.astype(np.uint8))


# -----------------------------
# ITEM 5: Generar Imagen E (citoplasmas cerrados)
# -----------------------------
B_bin = (img_B // 255).astype(np.uint8)  
marker_bg = np.zeros_like(B_bin)
marker_bg[0, :] = B_bin[0, :]
marker_bg[-1, :] = B_bin[-1, :]
marker_bg[:, 0] = B_bin[:, 0]
marker_bg[:, -1] = B_bin[:, -1]

fondo_recon_B = reconstruccion_morfologica(marker_bg, B_bin, B)
citoplasmas_cerrados = B_bin & invert(fondo_recon_B)  

img_E = (citoplasmas_cerrados * 255).astype(np.uint8)  # Blanco = citoplasmas cerrados
img_E = cv2.bitwise_not(img_E)
cv2.imwrite('imagen_E.png', img_E)


# -----------------------------
# Mostrar resultados
# -----------------------------
cv2.imshow('Original', img)
cv2.imshow('A Celulas completas', mask_A.astype(np.uint8))
cv2.imshow('B Citoplasmas (huecos)', img_B)
cv2.imshow('C Celulas agujereadas', img_C.astype(np.uint8))
cv2.imshow('D Celulas sin agujeros (Tipo 1)', mask_D.astype(np.uint8))
cv2.imshow('E Citoplasmas cerrados (Tipo 2 y 4)', img_E)
cv2.waitKey(0)
cv2.destroyAllWindows()