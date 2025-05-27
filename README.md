# cell_sorterPDI
Algoritmo diseñado en Python para clasificar celulas pertenecientes a una misma colonia en sus distintas etapas de vida.


Descripción del Algoritmo de Clasificación Celular
Este algoritmo procesa imágenes de colonias celulares para clasificar las células según sus características morfológicas. Utiliza técnicas de procesamiento digital de imágenes, especialmente operaciones de morfología matemática como la reconstrucción geodésica.

Funcionamiento por Etapas
ITEM 1: Aislamiento de Células Completas (Imagen A)
Detecta y extrae todas las células completas de la imagen de entrada, separándolas del fondo. Utiliza binarización adaptativa (Otsu) y reconstrucción morfológica para identificar objetos no conectados con los bordes de la imagen.

ITEM 2: Detección de Citoplasmas (Imagen B)
Identifica los huecos internos en las células (citoplasmas) mediante técnicas de detección de regiones desconectadas del fondo exterior. La imagen resultante muestra en blanco los citoplasmas y en negro las membranas celulares.

ITEM 3: Identificación de Células Agujereadas (Imagen C)
Detecta todas las células que contienen citoplasmas (células de Tipo 2, 3 y 4). Utiliza los citoplasmas como semillas para una reconstrucción que identifica las células completas que los contienen.

ITEM 4: Aislamiento de Células sin Agujeros (Imagen D)
Extrae células Tipo 1 (sin citoplasma visible) mediante operaciones lógicas entre todas las células y las células con agujeros identificadas previamente.

ITEM 5: Detección de Núcleos Tipo 4 (Imagen E)
Aísla específicamente los núcleos de células Tipo 4, identificando citoplasmas completamente cerrados (desconectados del fondo exterior) que representan una característica distintiva de este tipo celular.

ITEM 6: Extracción de Células Tipo 4 Completas (Imagen F)
Reconstruye las células Tipo 4 completas utilizando los núcleos identificados como semillas y propagándolos dentro de las regiones celulares.

ITEM 7: Aislamiento de Células Tipo 2 y 3 (Imagen G)
Obtiene las células de Tipo 2 y 3 mediante la exclusión de las células Tipo 1 y Tipo 4 del conjunto total de células.

ITEM 8: Clasificación Final y Visualización
Diferencia entre células Tipo 2 y Tipo 3 basándose en el área de citoplasma. Genera una visualización a color donde:

Células Tipo 3: Rosa intenso (4 células con mayor área de citoplasma)
Células Tipo 2: Verde intenso (resto de células con citoplasma)
El algoritmo genera imágenes intermedias para cada etapa del proceso y una imagen final coloreada que permite visualizar la clasificación completa.

Características de las Células por Tipo
Tipo 1: Células sin citoplasma visible
Tipo 2: Células con citoplasma pequeño
Tipo 3: Células con citoplasma grande
Tipo 4: Células con núcleo completamente cerrado y aislado