import cv2                       # OpenCV para procesamiento de imágenes
import numpy as np               # NumPy para operaciones con matrices
from skimage.util import invert  # Función para invertir imágenes (complemento)
from skimage.measure import label, regionprops  # Para etiquetar objetos y obtener propiedades

# Función para realizar reconstrucción morfológica geodésica
# marker: imagen semilla desde donde comenzará la reconstrucción
# mask: imagen que restringe el crecimiento de la reconstrucción
# kernel: elemento estructurante que define la conectividad
def reconstruccion_morfologica(marker, mask, kernel):
    prev = np.zeros_like(marker)  # Matriz para almacenar el estado anterior
    curr = marker.copy()          # Copia del marcador como punto de partida
    while True:
        prev[:] = curr            # Guarda estado actual para comparación
        dil = cv2.dilate(curr, kernel, iterations=1)  # Dilata la imagen actual
        curr = cv2.bitwise_and(dil, mask)  # Restringe el crecimiento según la máscara
        if np.array_equal(curr, prev):     # Verifica si hubo cambios
            break                 # Si no hay cambios, se alcanzó convergencia
    return curr                   # Devuelve la reconstrucción final


# -----------------------------
# ITEM 1: Generar Imagen A - Todas las células completas
# -----------------------------
img = cv2.imread('imagen_entrada.jpg', cv2.IMREAD_GRAYSCALE)  # Carga imagen original en escala de grises
# Binariza la imagen usando el método Otsu y la invierte (células=1, fondo=0)
_, bin_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
h, w = bin_img.shape  # Obtiene dimensiones de la imagen

# Crea marcador inicial: solo los bordes de la imagen binarizada
# Este marcador identifica regiones conectadas al exterior
marker = np.zeros_like(bin_img)  # Inicializa marcador con ceros
marker[0, :]   = bin_img[0, :]   # Copia borde superior
marker[-1, :]  = bin_img[-1, :]  # Copia borde inferior
marker[:, 0]   = bin_img[:, 0]   # Copia borde izquierdo
marker[:, -1]  = bin_img[:, -1]  # Copia borde derecho

# Define elemento estructurante 3x3 para conectividad-8
B = np.array([[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]], dtype=np.uint8)

# Reconstruye las regiones conectadas al borde (fondo)
bordes_recon = reconstruccion_morfologica(marker, bin_img, B)
not_bordes   = invert(bordes_recon)  # Invierte para obtener regiones NO conectadas al borde
img_A_bin    = bin_img & not_bordes  # Obtiene células completas: objetos no conectados al borde
mask_A       = (1 - img_A_bin) * 255 # Convierte a formato imagen: 255=fondo (blanco), 0=células (negro)
cv2.imwrite('imagen_A.png', mask_A.astype(np.uint8))  # Guarda resultado


# -----------------------------
# ITEM 2: Generar Imagen B (solo citoplasmas) - Identificación de huecos internos
# -----------------------------
A_bin = (mask_A // 255).astype(np.uint8)  # Convierte A a formato binario: 1=fondo, 0=células

# Crea marcador desde los bordes de A_bin para detectar el fondo externo
marker2 = np.zeros_like(A_bin)
marker2[0, :]   = A_bin[0, :]    # Marca borde superior
marker2[-1, :]  = A_bin[-1, :]   # Marca borde inferior
marker2[:, 0]   = A_bin[:, 0]    # Marca borde izquierdo
marker2[:, -1]  = A_bin[:, -1]   # Marca borde derecho

# Reconstruye el fondo exterior
background_recon = reconstruccion_morfologica(marker2, A_bin, B)
not_background   = invert(background_recon)  # Invierte para obtener regiones desconectadas del fondo
holes            = A_bin & not_background    # Intersección: identifica huecos internos (citoplasmas)

# Convierte a formato imagen y lo invierte para formato final:
# 255=fondo+citoplasmas (blanco), 0=membranas (negro)
img_B = (holes * 255).astype(np.uint8)
img_B = cv2.bitwise_not(img_B)
cv2.imwrite('imagen_B.png', img_B)


# -----------------------------
# ITEM 3: Generar Imagen C (células agujereadas) - Tipos 2, 3 y 4
# -----------------------------
A_bin       = (255 - mask_A) // 255    # 1=células, 0=fondo
B_holes_bin = (255 - img_B)   // 255    # 1=citoplasmas, 0=fondo+membranas

# Reconstruye células que contienen citoplasmas (usa citoplasmas como semillas)
recon_C = reconstruccion_morfologica(B_holes_bin, A_bin, B)
img_C   = (1 - recon_C) * 255  # Formato final: 0=células con agujeros (negro), 255=fondo (blanco)
cv2.imwrite('imagen_C.png', img_C.astype(np.uint8))


# -----------------------------
# ITEM 4: Generar Imagen D (células sin agujeros, Tipo 1)
# -----------------------------
A_bin = (255 - mask_A) // 255   # 1=todas las células, 0=fondo
C_bin = (255 - img_C)   // 255  # 1=células con agujeros, 0=fondo+células sin agujeros

# Obtiene células presentes en A pero no en C (todas las células menos las que tienen agujeros)
D_bin = cv2.bitwise_and(A_bin, cv2.bitwise_not(C_bin))
mask_D = (1 - D_bin) * 255  # Formato final: 0=células tipo 1 (negro), 255=fondo (blanco)
cv2.imwrite('imagen_D.png', mask_D.astype(np.uint8))


# -----------------------------
# ITEM 5: Generar Imagen E – aislar SOLO los NÚCLEOS de Tipo 4
# -----------------------------

# Convierte imagen B a formato binario
# img_B tiene: 0=membrana+célula, 255=fondo+citoplasma
# B_bin tendrá: 1=fondo+citoplasma, 0=membrana
B_bin = (img_B // 255).astype(np.uint8)

# Crea marcador del fondo exterior: marca píxeles en bordes de la imagen
marker_bg = np.zeros_like(B_bin)
marker_bg[0, :]   = B_bin[0, :]   # Marca borde superior
marker_bg[-1, :]  = B_bin[-1, :]  # Marca borde inferior
marker_bg[:, 0]   = B_bin[:, 0]   # Marca borde izquierdo
marker_bg[:, -1]  = B_bin[:, -1]  # Marca borde derecho

# Reconstruye regiones conectadas al exterior (fondo + citoplasmas abiertos)
fondo_recon_B = reconstruccion_morfologica(marker_bg, B_bin, B)

# Detecta citoplasmas completamente cerrados (no conectados al exterior)
# Estos son los núcleos de células Tipo 4: citoplasmas aislados del fondo
citoplasmas_cerrados = B_bin & invert(fondo_recon_B)

# Formato final: 0=núcleos Tipo 4 (negro), 255=resto (blanco)
img_E = (citoplasmas_cerrados * 255).astype(np.uint8)
img_E = cv2.bitwise_not(img_E)
cv2.imwrite('imagen_E.png', img_E)


# -----------------------------
# ITEM 6: Generar Imagen F (células de Tipo 4 completas)
# -----------------------------

# Redefine función de reconstrucción morfológica con enfoque alternativo
# Esta versión usa np.minimum en lugar de bitwise_and para valores no binarios
def reconstruccion_morfologica(marker, mask, kernel):
    prev = np.zeros_like(marker)
    curr = marker.copy()
    
    while not np.array_equal(curr, prev):
        prev = curr.copy()
        dilated = cv2.dilate(curr, kernel, iterations=1)
        curr = np.minimum(dilated, mask)  # Mínimo elemento a elemento
    
    return curr

# Prepara núcleos como semillas para la reconstrucción
# Invierte E para que núcleos sean 1 (semillas) y resto 0
E_nucleos = cv2.bitwise_not(img_E) // 255  
E_nucleos = E_nucleos.astype(np.uint8)

# Prepara máscara con todas las células (de imagen A)
# 1=células (región donde pueden crecer las semillas), 0=fondo
A_celulas = (255 - mask_A) // 255
A_celulas = A_celulas.astype(np.uint8)

# Dilata ligeramente los núcleos para mejor propagación
kernel_dil = np.ones((5,5), np.uint8)  # Kernel más grande
nucleos_dil = cv2.dilate(E_nucleos, kernel_dil, iterations=2)

# Reconstrucción: propaga núcleos dentro de las células
# El resultado son células completas de Tipo 4
celulas_tipo4 = reconstruccion_morfologica(nucleos_dil, A_celulas, B)

# Formato final: 0=células Tipo 4 (negro), 255=fondo (blanco)
img_F = ((1 - celulas_tipo4) * 255).astype(np.uint8)
cv2.imwrite('imagen_F.png', img_F)


# -----------------------------
# ITEM 7: Generar Imagen G - Células Tipo 2 y Tipo 3
# -----------------------------

# Prepara imágenes binarias con formato: 1=objeto, 0=fondo
A_bin = ((255 - mask_A) // 255).astype(np.uint8)  # Todas las células
D_bin = ((255 - mask_D) // 255).astype(np.uint8)  # Células Tipo 1
F_bin = ((255 - img_F) // 255).astype(np.uint8)   # Células Tipo 4

# Paso 1: Excluye células Tipo 1 de todas las células
# A_no1 tendrá solo células Tipo 2, 3 y 4
A_no1 = cv2.bitwise_and(A_bin, cv2.bitwise_not(D_bin))

# Paso 2: Excluye células Tipo 4 del resultado anterior
# G_bin contendrá ahora solo células Tipo 2 y 3
G_bin = cv2.bitwise_and(A_no1, cv2.bitwise_not(F_bin))

# Formato final: 0=células Tipo 2 y 3 (negro), 255=fondo (blanco)
img_G = (1 - G_bin) * 255
cv2.imwrite('imagen_G.png', img_G.astype(np.uint8))


# -----------------------------
# ITEM 8: Diferenciar células Tipo 2 vs Tipo 3 (con visualización en colores)
# -----------------------------

# Carga las imágenes necesarias para la clasificación final
img_G = cv2.imread('imagen_G.png', cv2.IMREAD_GRAYSCALE)  # Células Tipo 2 y 3
img_B = cv2.imread('imagen_B.png', cv2.IMREAD_GRAYSCALE)  # Información de citoplasmas

# Binariza las imágenes para el análisis
G_bin = (img_G < 128).astype(np.uint8)  # 1=células Tipo 2 y 3, 0=fondo
B_bin = (img_B > 200).astype(np.uint8)  # 1=citoplasmas (regiones blancas), 0=membranas+fondo

# Etiqueta cada célula como un objeto independiente
etiquetas = label(G_bin)  # 0=fondo, 1,2,...=células individuales
regiones = regionprops(etiquetas)  # Calcula propiedades de cada célula

# Prepara imagen de salida coloreada con fondo blanco
imagen_coloreada = np.ones((img_G.shape[0], img_G.shape[1], 3), dtype=np.uint8) * 255

# Ordena las células por tamaño (de mayor a menor)
# Esto ayuda a identificar mejor las células con citoplasma grande
celulas_ordenadas = sorted(regiones, key=lambda r: r.area, reverse=True)
contador_tipo3 = 0  # Contador para células Tipo 3 (sabemos que hay 4)

# Analiza cada célula individualmente
for region in celulas_ordenadas:
    # Crea máscara para la célula actual
    mascara = etiquetas == region.label
    
    # Calcula píxeles de citoplasma dentro de la célula
    # Intersección entre B_bin (citoplasmas) y la máscara de célula
    citoplasma_presente = B_bin & mascara.astype(np.uint8)
    area_citoplasma = np.sum(citoplasma_presente)  # Cuenta píxeles de citoplasma
    
    # Las 4 células con mayor área de citoplasma son Tipo 3
    # El resto son Tipo 2
    if contador_tipo3 < 4:
        # Células Tipo 3 en ROSA INTENSO (formato BGR de OpenCV)
        imagen_coloreada[mascara] = [180, 105, 255]  # Rosa intenso
        contador_tipo3 += 1
    else:
        # Células Tipo 2 en VERDE INTENSO (formato BGR de OpenCV)
        imagen_coloreada[mascara] = [50, 205, 50]  # Verde intenso

# Guarda resultado final con células clasificadas y coloreadas
cv2.imwrite('celulas_clasificadas.png', imagen_coloreada)


# -----------------------------
# Mostrar resultados (opcional)
# -----------------------------
# Muestra todas las imágenes generadas para verificación visual
cv2.imshow('Original', img)  # Imagen original
cv2.imshow('A Celulas completas', mask_A.astype(np.uint8))  # Todas las células
cv2.imshow('B Citoplasmas (huecos)', img_B)  # Citoplasmas
cv2.imshow('C Celulas agujereadas', img_C.astype(np.uint8))  # Células con agujeros
cv2.imshow('D Celulas sin agujeros (Tipo 1)', mask_D.astype(np.uint8))  # Células Tipo 1
cv2.imshow('E Nucleos Tipo 4', img_E)  # Núcleos de células Tipo 4
cv2.imshow('F Celulas Tipo 4 completas', img_F)  # Células Tipo 4 completas
cv2.imshow('G Celulas Tipo 2 y 3', img_G.astype(np.uint8))  # Células Tipo 2 y 3
cv2.imshow('Celulas Tipo 2 y 3', imagen_coloreada)  # Clasificación final coloreada
cv2.waitKey(0)  # Espera hasta que se presione una tecla
cv2.destroyAllWindows()  # Cierra todas las ventanas