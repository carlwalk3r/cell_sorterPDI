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
marker[0, :] = bin_img[0, :]
marker[-1, :] = bin_img[-1, :]
marker[:, 0] = bin_img[:, 0]
marker[:, -1] = bin_img[:, -1]

B = np.array([[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]], dtype=np.uint8)

bordes_recon = reconstruccion_morfologica(marker, bin_img, B)

not_bordes = invert(bordes_recon)
img_A_bin = bin_img & not_bordes

mask_A = (1 - img_A_bin) * 255
cv2.imwrite('imagen_A.png', mask_A.astype(np.uint8))

# -----------------------------
# ITEM 2: Generar Imagen B
# -----------------------------

A_bin = (mask_A // 255).astype(np.uint8)

marker2 = np.zeros_like(A_bin)
marker2[0, :] = A_bin[0, :]
marker2[-1, :] = A_bin[-1, :]
marker2[:, 0] = A_bin[:, 0]
marker2[:, -1] = A_bin[:, -1]

background_recon = reconstruccion_morfologica(marker2, A_bin, B)
not_background = invert(background_recon)
holes = A_bin & not_background

img_B = (holes * 255).astype(np.uint8)
img_B = cv2.bitwise_not(img_B)
cv2.imwrite('imagen_B.png', img_B)

# -----------------------------
# ITEM 3: Obtener células con agujeros (Tipos 2, 3, 4)
# -----------------------------

# 1) Convertir Imagen A a binaria: 1 = células negras, 0 = fondo blanco
A_bin = (255 - mask_A) // 255  # célula = 1, fondo = 0

# 2) Convertir Imagen B a binaria: 1 = agujeros (negros en B), 0 = resto
B_holes_bin = (255 - img_B) // 255  # agujeros = 1, resto = 0

# 3) Usar los agujeros como marcador y A_bin como máscara para crecer hasta el cuerpo celular
marker_C = B_holes_bin.copy()
mask_C = A_bin.copy()

# 4) Reconstrucción morfológica desde los agujeros, limitados a la célula
reconstruido = reconstruccion_morfologica(marker_C, mask_C, B)

# 5) Invertimos para que fondo y agujeros sean blancos, células negras
img_C = (1 - reconstruido) * 255

# 6) Guardar Imagen C
cv2.imwrite('imagen_C.png', img_C.astype(np.uint8))

# -----------------------------
# Mostrar resultados (opcional)
# -----------------------------
cv2.imshow('Imagen Original', img)
cv2.imshow('Imagen A - Celulas completas', mask_A.astype(np.uint8))
cv2.imshow('Imagen B - Agujeros (citoplasma)', img_B)
cv2.imshow('Imagen C - Celulas agujereadas', img_C.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
