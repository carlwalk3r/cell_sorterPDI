# cell_sorterPDI
Algoritmo diseñado en Python para clasificar celulas pertenecientes a una misma colonia en sus distintas etapas de vida.

## Descripción del Algoritmo de Clasificación Celular
Este algoritmo procesa imágenes de colonias celulares para clasificar las células según sus características morfológicas. Utiliza técnicas de procesamiento digital de imágenes, especialmente operaciones de morfología matemática como la reconstrucción geodésica.

## Instalación y Requisitos

### Requisitos previos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instalación de dependencias
Para instalar todas las dependencias necesarias, ejecute el siguiente comando:

pip install -r requirements.txt


### Ejecución del programa

1. Coloque la imagen a analizar como imagen_entrada.jpg en el mismo directorio que el script

2. Ejecute el programa principal:

    python main.py



### RESPUESTAS A LAS PREGUNTAS

Item 1: Generar Imagen A sin piezas ni células truncadas

Lo hacemos de manera totalmente automática aplicando solo reconstrucción morfológica, operaciones lógicas y la inversión de imagen. Primero binarizamos (células=1, fondo=0) con Otsu+INV. Luego construimos un marcador copiando únicamente los píxeles de los cuatro bordes de la máscara binaria. A partir de ese marcador, ejecutamos una reconstrucción morfológica por dilatación iterativa (conectividad‑8) limitada por la máscara original. El resultado es el “fondo” unido al perímetro. Invertimos esa reconstrucción y hacemos un AND lógico con la máscara primaria: con ello “restamos” todas las regiones conectadas al borde, quedando solo las células completas. Finalmente invertimos para fondo blanco (255) y células negras (0).


Item 2: Generar Imagen B – todos los agujeros de las células con citoplasma

Partiendo de Imagen A (fondo blanco, células negras), la invertimos a binario (fondo=1, célula=0). Construimos un nuevo marcador con las regiones de fondo que tocan los bordes y reconstruimos esos píxeles sobre la máscara de fondo. Al invertir la reconstrucción obtenemos las cavidades desconectadas del exterior: los huecos internos (citoplasmas). Hacemos un AND con la imagen binaria original y, tras convertir a 0/255, invertimos para que los huecos queden en negro sobre fondo blanco.



Item 3: Generar Imagen C – células agujereadas (Tipos 2, 3 y 4)

Usamos como marcador la máscara de citoplasmas (Imagen B) y como máscara la imagen binaria de todas las células (invertida de A). Por reconstrucción morfológica, hacemos crecer esas semillas de hueco hasta llenar solo el interior de las células que efectivamente los contenían. El resultado, invertido a 0=célula agujereada sobre 255=fondo, nos da Imagen C con las membranas de todas las células que tienen citoplasma (Tipos 2–4).


Item 4: Generar Imagen D – células sin agujeros (Tipo 1)

A partir de la binaria de todas las células y de la binaria de las que tienen huecos (Imagen C), aplicamos una operación lógica AND con la negación de C. Es decir, D = A_bin AND NOT(C_bin). De este modo eliminamos de A aquellas células que tenían citoplasma (2–4), quedando sólo las no agujereadas (Tipo 1). Convertimos a fondo blanco y células negras para la salida.


Item 5: Generar Solo los nucleos de las células Tipo 4 completas

En la Imagen E vemos únicamente esos pequeñísimos núcleos que caracterizan a las células de Tipo 4: tras identificar en Imagen B todos los huecos (citoplasmas) abiertos y cerrados, construimos un “marcador de fondo” a partir de los píxeles que tocan los bordes, lo que nos deja sólo las cavidades comunicadas con el exterior. Al invertir esa reconstrucción y cruzarla con la máscara original, eliminamos por completo cualquier citoplasma abierto y aislamos únicamente los huecos sellados por membranas: los núcleos. Finalmente, convertimos esas regiones a negro sobre fondo blanco, de modo que lo único que permanece en negro son las regiones internas perfectamente encerradas por membrana y citoplasma—es decir, los núcleos “sueltos” de las células Tipo 4.


Item 6: Generar Imagen F – células Tipo 4 completas

Ahora sí utilizamos, además de reconstrucción e inversión, dilatación (no condicional) y erosión permitidas. Tomamos Imagen E como semillas (1 en núcleo) y la máscara de todas las células (de A) como contención. Dilatamos ligeramente los núcleos para asegurar un punto de partida amplio y luego ejecutamos reconstrucción morfológica: esto expande la semilla hasta llenar todo el cuerpo celular que la rodea (citoplasma+membrana). El resultado invertido en 0=células Tipo 4, 255=fondo nos da Imagen F con las células Tipo 4 completas.


Item 7: Generar Imagen G – células Tipo 2 y 3

A partir de las máscaras binarias de A (todas las células), D (Tipo 1) y F (Tipo 4), hacemos dos AND lógicos sucesivos: primero “todas menos Tipo 1” (A AND NOT D), luego “eso menos Tipo 4” (… AND NOT F). Con esto queda G = A ∧ ¬D ∧ ¬F, que selecciona exactamente las células de Tipo 2 y 3 unidas. Invertimos para fondo blanco y objetos negros y guardamos Imagen G.


Item 8: Diferenciar células de Tipo 2 vs. Tipo 3

Sobre la imagen rellena de células 2+3 (mask23 = C_bin ∧ ¬F_bin) etiquetamos por componentes conectados y obtenemos, para cada célula, su región completa. Con la máscara de citoplasmas (B_holes_bin = (img_B==0)), contamos cuántos píxeles de hueco hay dentro de cada región. Al comparar cada área de hueco con un umbral (dinámico o predefinido), clasificamos:

Si es pequeño, Tipo 2 (fundamentalmente membrana sin gran citoplasma).

Si es grande, Tipo 3 (citoplasma voluminoso).
Finalmente dibujamos dos máscaras: una con solo Tipo 2 y otra con solo Tipo 3.