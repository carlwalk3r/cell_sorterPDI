import cv2
import numpy as np
from skimage.util import invert
from skimage.measure import label, regionprops 

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
# ITEM 5: Generar Imagen E – aislar SOLO los NÚCLEOS de Tipo 4
# -----------------------------

# Paso 5.1: Convertir img_B a binaria 0/1
# img_B: 0 = membrana+célula, 255 = fondo+citoplasma
# B_bin = 1 donde hay blanco (fondo+citoplasma), 0 donde está la membrana
B_bin = (img_B // 255).astype(np.uint8)

# Paso 5.2: Crear semilla (marker) del fondo exterior
# Marcamos con 1 todos los píxeles de B_bin que tocan cualquiera de los 4 bordes
marker_bg = np.zeros_like(B_bin)
marker_bg[0, :]   = B_bin[0, :]   # fila superior
marker_bg[-1, :]  = B_bin[-1, :]  # fila inferior
marker_bg[:, 0]   = B_bin[:, 0]   # columna izquierda
marker_bg[:, -1]  = B_bin[:, -1]  # columna derecha

# Paso 5.3: Reconstrucción morfológica del fondo
# Reconstruimos a partir del marker_bg, pero sin salirnos de las regiones permitidas por B_bin
fondo_recon_B = reconstruccion_morfologica(marker_bg, B_bin, B)

# Paso 5.4: Detectar citoplasmas completamente cerrados
# citoplasmas_cerrados = 1 en huecos de B_bin que NO fueron alcanzados por la reconstrucción del fondo
citoplasmas_cerrados = B_bin & invert(fondo_recon_B)

# Paso 5.5: Generar img_E (invertir para que núcleos queden en negro)
# Primero pasamos citoplasmas_cerrados de 1→255, 0→0
img_E = (citoplasmas_cerrados * 255).astype(np.uint8)
# Luego invertimos: 255→0 (los núcleos), 0→255 (el resto)
img_E = cv2.bitwise_not(img_E)

# Paso 5.6: Guardar Imagen E
# La salida contiene en NEGRO únicamente los píxeles correspondientes a los núcleos de Tipo 4
cv2.imwrite('imagen_E.png', img_E)


# -----------------------------
# ITEM 6: Generar Imagen F (células de Tipo 4 completas)
# -----------------------------

# Paso 6.1: Definir función de reconstrucción morfológica mejorada
def reconstruccion_morfologica(marker, mask, kernel):
    prev = np.zeros_like(marker)
    curr = marker.copy()
    
    while not np.array_equal(curr, prev):
        prev = curr.copy()
        dilated = cv2.dilate(curr, kernel, iterations=1)
        curr = np.minimum(dilated, mask)
    
    return curr

# Paso 6.2: Preparar los núcleos como semillas (marker)
# La imagen E tiene los núcleos en NEGRO (0), resto en blanco (255)
E_nucleos = cv2.bitwise_not(img_E) // 255  # Invertimos para que núcleos sean 1
E_nucleos = E_nucleos.astype(np.uint8)

# Paso 6.3: Preparar la máscara de células (de imagen A)
# Necesitamos todas las células como 1 y el fondo como 0
A_celulas = (255 - mask_A) // 255
A_celulas = A_celulas.astype(np.uint8)

# Paso 6.4: Dilatar ligeramente los núcleos (mejor punto de partida)
kernel_dil = np.ones((5,5), np.uint8)  # Kernel más grande para mejor propagación
nucleos_dil = cv2.dilate(E_nucleos, kernel_dil, iterations=2)

# Paso 6.5: Realizar la reconstrucción morfológica
celulas_tipo4 = reconstruccion_morfologica(nucleos_dil, A_celulas, B)

# Paso 6.6: Guardar resultado final (con colores invertidos)
img_F = ((1 - celulas_tipo4) * 255).astype(np.uint8)
cv2.imwrite('imagen_F.png', img_F)

# -----------------------------
# ITEM 7: Generar Imagen G –  
#         células de Tipo 2 y Tipo 3
# -----------------------------

# 7.1) Binarizar las imágenes A, D y F:
#     A_bin = 1 en cualquier célula (Tipos 1–4), 0 en fondo
#     D_bin = 1 en células Tipo 1, 0 en resto
#     F_bin = 1 en células Tipo 4 completas, 0 en resto
A_bin = ((255 - mask_A) // 255).astype(np.uint8)
D_bin = ((255 - mask_D) // 255).astype(np.uint8)
F_bin = ((255 - img_F) // 255).astype(np.uint8)  # Células Tipo 4 completas

# 7.2) Excluir Tipo 1:
#     A_no1 = células A menos las Tipo 1
A_no1 = cv2.bitwise_and(A_bin, cv2.bitwise_not(D_bin))

# 7.3) Excluir Tipo 4:
#     G_bin = A_no1 menos las células Tipo 4
G_bin = cv2.bitwise_and(A_no1, cv2.bitwise_not(F_bin))

# 7.4) Formatear a 0/255:
#     255 = fondo, 0 = células Tipo 2 y 3
img_G = (1 - G_bin) * 255

# 7.5) Guardar Imagen G
cv2.imwrite('imagen_G.png', img_G.astype(np.uint8))


# -----------------------------
# ITEM 8: Diferenciar Tipo 2 vs Tipo 3
# -----------------------------

# 8.1) Binariza G para etiquetar:
#     img_G: 0=célula,255=fondo -> invertimos a 1=célula,0=fondo
_, G_bin = cv2.threshold(img_G, 127, 1, cv2.THRESH_BINARY_INV)

# 8.2) Binariza los huecos de citoplasma (B):
#     img_B: 0=membrana,255=fondo+citoplasma -> holes=1 en citoplasma
B_holes_bin = ((255 - img_B) // 255).astype(np.uint8)

# 8.3) Etiqueta componentes conectados en G_bin
num_labels, labels = cv2.connectedComponents(G_bin, connectivity=8)

# 8.4) Prepara lienzos blancos para Tipo2 y Tipo3
tipo2_img = np.ones_like(img_G) * 255
tipo3_img = np.ones_like(img_G) * 255

# 8.5) Umbral en píxeles para separar hueco “pequeño” vs “grande”
UMBRAL_SMALL = 150  # ajusta entre ~50–300 según tu escala

# 8.6) Recorre cada célula (etiqueta 1..num_labels-1)
for lab in range(1, num_labels):
    mask_cell = (labels == lab)               # máscara booleana de la región
    hole_region = B_holes_bin * mask_cell     # 1 sólo donde hay hueco dentro de esa región

    area_hole = int(hole_region.sum())        # tamaño del hueco

    # Clasifica:
    if area_hole >= UMBRAL_SMALL:
        # Tipo 3: hueco grande
        tipo3_img[mask_cell] = 0
    else:
        # Tipo 2: hueco pequeño
        tipo2_img[mask_cell] = 0

# 8.7) Guarda y muestra
cv2.imwrite('imagen_tipo2.png', tipo2_img)
cv2.imwrite('imagen_tipo3.png', tipo3_img)






# -----------------------------
# Mostrar resultados (opcional)
# -----------------------------
cv2.imshow('Original', img)
cv2.imshow('A Celulas completas', mask_A.astype(np.uint8))
cv2.imshow('B Citoplasmas (huecos)', img_B)
cv2.imshow('C Celulas agujereadas', img_C.astype(np.uint8))
cv2.imshow('D Celulas sin agujeros (Tipo 1)', mask_D.astype(np.uint8))
cv2.imshow('E Nucleos Tipo 4', img_E)
cv2.imshow('F Celulas Tipo 4 completas', img_F)
cv2.imshow('G Celulas Tipo 2 y 3', img_G.astype(np.uint8))
cv2.imshow('Tipo 2', tipo2_img)
cv2.imshow('Tipo 3', tipo3_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
