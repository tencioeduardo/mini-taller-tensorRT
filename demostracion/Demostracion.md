# Demo técnica: Piedra, Papel o Tijera con LiteRT

Esta demo muestra el recorrido de un modelo de aprendizaje automático desde su entrenamiento hasta su ejecución en un equipo Linux con **LiteRT**. Un mismo clasificador de **Piedra, Papel o Tijera** se convierte a cuatro formatos (FP32, FP16, INT8 dinámico e INT8 completo), se mide su comportamiento y finalmente se ejecuta en vivo con la cámara web.

La demostración tiene tres partes:

1. **El artefacto:** los modelos `.tflite` generados y su formato.
2. **La medición:** tamaño, memoria, latencia y precisión de cada variante, con las dos APIs de LiteRT (`Interpreter` y `CompiledModel`).
3. **La ejecución en vivo:** clasificación en tiempo real con **OpenCV** y la cámara, cambiando de modelo con una tecla.

## Arquitectura

```text
Google Colab
│
└── entrenar_rps.ipynb
    ├── Entrena una CNN con Keras
    ├── Convierte a .tflite (TFLiteConverter)
    │   ├── rps_fp32.tflite
    │   ├── rps_fp16.tflite
    │   ├── rps_int8_dyn.tflite
    │   └── rps_int8_full.tflite
    └── Exporta el set de prueba (372 imágenes)
            │
            │  artefactos_rps.zip
            ▼
Laptop Linux (Host)
│
├── evaluar.py
│   └── tamaño + RAM + latencia + precisión
│
└── litert_rps.py
    ├── OpenCV → cámara /dev/video0
    └── LiteRT → Interpreter o CompiledModel (CPU)
```

## Requisitos

Se necesita:

- Linux x86-64 (probado en Ubuntu 22.04 y 24.04)
- Python 3.10 o superior
- Una cámara web (opcional: existe un modo con imagen fija)
- Una cuenta de Google, únicamente para generar los modelos en Colab

Las dependencias de Python son `ai-edge-litert`, NumPy y OpenCV. El script `setup.sh` las instala en un entorno virtual.

## Estructura de archivos

```text
demostracion/
├── Demostracion.md
├── entrenar_rps.ipynb
├── requirements.txt
├── setup.sh
├── verificar_entorno.py
├── rps_common.py
├── evaluar.py
├── litert_rps.py
├── modelos/
│   ├── rps_fp32.tflite
│   ├── rps_fp16.tflite
│   ├── rps_int8_dyn.tflite
│   ├── rps_int8_full.tflite
│   ├── labels.txt
│   └── manifest.json
└── datos/
    ├── README.md
    └── test/
        ├── Piedra/
        ├── Papel/
        └── Tijera/
```

### Importante

La carpeta `modelos/` y la carpeta `datos/` **no se crean con `setup.sh`**: se generan con el cuaderno `entrenar_rps.ipynb` (ver la siguiente sección).

Los archivos `resultados/`, `docs/entorno.md` y `test_hand.jpg` se crean durante la preparación y la ejecución.

## Preparación previa

Esta preparación se realiza una sola vez y no forma parte de la ejecución en clase.

### Generar los modelos en Colab

1. Abrir `entrenar_rps.ipynb` en Google Colab.
2. Seleccionar **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU** (opcional, acelera el entrenamiento).
3. Seleccionar **Entorno de ejecución → Ejecutar todas**.

Al finalizar se descarga el archivo `artefactos_rps.zip`.

La primera celda del cuaderno instala `importlib-resources`:

```text
%pip install -q importlib-resources ai-edge-litert==2.3.0
```

Esta instalación es necesaria porque `tensorflow_datasets` utiliza `importlib_resources` y Colab no lo incluye. Sin ella, `tfds.load` falla con `ModuleNotFoundError`.

### Copiar los artefactos

Descomprimir el archivo dentro de la carpeta `demostracion/`:

```bash
cd ~/mini-taller-liteRT/demostracion
unzip ~/Descargas/artefactos_rps.zip -d .
```

La dirección mostrada es un ejemplo, puede cambiarla a conveniencia.

### Preparar la laptop

```bash
chmod +x setup.sh
./setup.sh
```

El script realiza lo siguiente:

1. Verifica que Python sea 3.10 o superior.
2. Crea el entorno virtual `.venv`.
3. Instala las dependencias de `requirements.txt`.
4. Valida que cada modelo sea un `.tflite` real.
5. Comprueba el set de prueba, la cámara y Vulkan.

Al terminar no debe aparecer ninguna marca `[✗]`. Las marcas `[!]` son avisos y no impiden ejecutar la demo.

Luego activar el entorno y registrar las características del equipo:

```bash
source .venv/bin/activate
python verificar_entorno.py
```

Este último comando crea `docs/entorno.md` con las versiones instaladas y con las capacidades de LiteRT del equipo (por ejemplo, si la GPU está disponible).

### Crear la imagen de respaldo (opcional)

Si la cámara llegara a fallar, la demo puede continuar con una imagen fija. Para crearla:

```bash
python litert_rps.py
```

Mostrar una mano con un gesto claro y presionar `s`. Se guarda el archivo `test_hand.jpg`. Luego salir con `q`.

---

# Ejecución de la demostración en clase

Esta sección contiene únicamente los pasos que deben ejecutarse durante la demostración en vivo.

La generación de los modelos en Colab y la preparación de la laptop se realizan previamente.

La demostración utiliza una sola terminal, abierta en la carpeta `demostracion/` y con el entorno virtual activado:

```bash
cd ~/mini-taller-liteRT/demostracion
source .venv/bin/activate
```

Se recomienda aumentar el tamaño de la fuente de la terminal para que el público pueda leerla.

---

## 1. Revisar los modelos generados

```bash
ls -lh modelos/
```

Salida esperada:

```text
-rw-r--r-- 1 usuario usuario   20 ... labels.txt
-rw-r--r-- 1 usuario usuario 1.5K ... manifest.json
-rw-r--r-- 1 usuario usuario 2.0M ... rps_fp16.tflite
-rw-r--r-- 1 usuario usuario 4.0M ... rps_fp32.tflite
-rw-r--r-- 1 usuario usuario 1.1M ... rps_int8_dyn.tflite
-rw-r--r-- 1 usuario usuario 1.1M ... rps_int8_full.tflite
```

Los cuatro archivos provienen **del mismo modelo entrenado**. Solamente cambia la forma de representar los números:

```text
FP32  →  4.0 MiB   (100 %)   32 bits por peso
FP16  →  2.0 MiB   ( 50 %)   16 bits por peso
INT8  →  1.0 MiB   ( 25 %)    8 bits por peso
```

---

## 2. Verificar el formato del archivo

Un archivo `.tflite` es un FlatBuffer. Los bytes 4 al 7 contienen el identificador `TFL3`:

```bash
head -c 8 modelos/rps_fp32.tflite | od -c
```

Salida esperada:

```text
0000000 034  \0  \0  \0   T   F   L   3
0000010
```

Esta es la misma verificación que realiza `setup.sh`. Si un archivo no contiene `TFL3`, no es un modelo real (normalmente es una página de error guardada por una descarga fallida).

---

## 3. Revisar el manifiesto

```bash
python -m json.tool modelos/manifest.json | head -30
```

Salida esperada (los valores pueden variar):

```text
{
    "generado": "...",
    "versiones": {
        "tensorflow": "...",
        "tensorflow_datasets": "...",
        "ai_edge_litert": "2.3.0",
        "numpy": "..."
    },
    "dataset": {
        "nombre": "rock_paper_scissors",
        "licencia": "CC BY 2.0 (Laurence Moroney)",
        "imagenes_prueba": {
            "Piedra": 124,
            "Papel": 124,
            "Tijera": 124
        }
    },
    ...
```

El manifiesto registra con qué versiones se generó cada modelo y la precisión que obtuvo en Colab. En el siguiente paso se comprueba esa precisión **en este equipo**.

---

## 4. Medir los modelos en la laptop

```bash
python evaluar.py
```

El script prueba cada modelo con las dos APIs de LiteRT. Cada combinación se ejecuta en un proceso independiente, para que la memoria de un modelo no afecte la medición de otro. La ejecución tarda menos de un minuto.

Salida esperada (los valores dependen del equipo):

```text
=== Evaluación LiteRT ===
CPU: ... | ... hilos | Python ... | ai-edge-litert 2.3.0
Modelos: rps_fp32.tflite, rps_fp16.tflite, rps_int8_dyn.tflite, rps_int8_full.tflite
Etiquetas: ['Piedra', 'Papel', 'Tijera'] | Datos: .../datos/test
  [1/8] rps_fp32        interpreter CPU  ... p50 ... ms | precisión ...%
  [2/8] rps_fp32        compiled    CPU  ... p50 ... ms | precisión ...%
  ...

Modelo         API                Tamaño   vs FP32 RAM modelo Carga   invoke p50 p90    Precisión Coincide FP32
rps_fp32       Interpreter/CPU    4.0 MiB  100%    ...        ...     ...        ...    ...       100.0%
rps_fp32       CompiledModel/CPU  4.0 MiB  100%    ...        ...     ...        ...    ...       100.0%
rps_fp16       Interpreter/CPU    2.0 MiB  50%     ...        ...     ...        ...    ...       ...
...
rps_int8_full  CompiledModel/CPU  1.0 MiB  25%     ...        ...     ...        ...    ...       ...

Reporte: resultados/resultados.md | Datos: resultados/resultados.csv
```

Cómo interpretar cada columna:

- **Tamaño** y **vs FP32** → tamaño del archivo. FP16 ocupa cerca del 50 % e INT8 cerca del 25 %.
- **RAM modelo** → memoria que agrega cada modelo al proceso. Baja **menos** que el tamaño del archivo, porque también incluye activaciones, búferes y estructuras internas del runtime.
- **Carga** → tiempo para crear el motor (leer el modelo y reservar memoria).
- **invoke p50** y **p90** → latencia de **solamente** la inferencia, sin captura de cámara ni preprocesamiento. Un p90 cercano al p50 indica una latencia estable.
- **Precisión** → aciertos sobre las 372 imágenes de prueba, con el mismo preprocesamiento que usa la cámara en vivo.
- **Coincide FP32** → porcentaje de imágenes en que la variante toma la misma decisión que FP32. Permite medir el efecto de la cuantización.

Observaciones esperadas:

```text
INT8 es más rápido que FP32 en CPU
FP16 no mejora la latencia en CPU (su ventaja está en la GPU)
Interpreter y CompiledModel producen la misma precisión
CompiledModel suele usar más RAM que Interpreter
```

---

## 5. Probar la aceleración por GPU (opcional)

```bash
python evaluar.py --gpu --apis compiled --modelos modelos/rps_fp16.tflite
```

`CompiledModel` permite elegir el acelerador sin modificar el resto del código. En Linux, la GPU se utiliza mediante WebGPU sobre Vulkan.

Si el equipo tiene un driver Vulkan funcional, aparece una fila adicional `CompiledModel/GPU` que puede compararse con la de CPU.

Si no lo tiene, aparece:

```text
[No disponible: rps_fp16]
  GPU no disponible: LiteRT usa WebGPU sobre Vulkan en Linux y no encontró un adaptador utilizable ...
```

Este resultado no es una falla de la demo. La aceleración depende del hardware y del driver disponibles, y la ejecución continúa en CPU.

---

## 6. Iniciar la clasificación en vivo

```bash
python litert_rps.py
```

El programa carga los cuatro modelos y muestra:

```text
=== LiteRT - Piedra, Papel o Tijera ===
  rps_fp32: 4085 KiB | entrada 150x150x3 float32 | salida 3 clases float32 | CompiledModel/CPU
  rps_fp16: 2047 KiB | entrada 150x150x3 float32 | salida 3 clases float32 | CompiledModel/CPU
  rps_int8_dyn: 1036 KiB | entrada 150x150x3 float32 | salida 3 clases float32 | CompiledModel/CPU
  rps_int8_full: 1039 KiB | entrada 150x150x3 float32 | salida 3 clases float32 | CompiledModel/CPU
Etiquetas: ['Piedra', 'Papel', 'Tijera']

Controles: m = modelo | f = cámara/imagen | s = guardar imagen | q/ESC = salir
```

Se abre una ventana con la imagen de la cámara. La demo inicia con el modelo `rps_int8_full`.

Mostrar a la cámara una mano haciendo piedra, papel y tijera, preferiblemente sobre un fondo liso y con buena iluminación frontal.

La ventana muestra tres líneas de información:

```text
Papel  96%                                       ← predicción y confianza
rps_int8_full  1.0 MiB  |  CompiledModel/CPU     ← modelo, tamaño y API
preproc 1.0 ms | invoke 0.9 ms | 30 FPS          ← tiempos (promedio de 30 cuadros)
```

La confianza que aparece en pantalla es la probabilidad de la clase ganadora. **No** es la precisión del modelo: la precisión se midió en el paso 4.

El procesamiento de cada cuadro es:

```text
Cámara (cuadro BGR)
      |
      v
Recorte cuadrado central
      |
      v
BGR → RGB + redimensionado a 150x150
      |
      v
Normalización (0-1)
      |
      v
LiteRT: invoke()
      |
      v
Probabilidades → clase ganadora
      |
      v
Texto sobre la imagen (OpenCV)
```

---

## 7. Cambiar de modelo en vivo

Con la ventana seleccionada, presionar:

```text
m
```

Cada pulsación pasa al siguiente modelo:

```text
FP32 → FP16 → INT8 dinámico → INT8 completo → FP32 → ...
```

En la terminal se muestra el modelo activo:

```text
Modelo: rps_fp32: 4085 KiB | entrada 150x150x3 float32 | salida 3 clases float32 | CompiledModel/CPU
```

Con la misma escena, las predicciones son prácticamente iguales. Lo que cambia es el tamaño del modelo y el tiempo de `invoke` que aparece en la tercera línea.

---

## 8. Ejecutar con la API clásica (opcional)

Presionar `q` para salir y ejecutar:

```bash
python litert_rps.py --api interpreter
```

La aplicación es la misma. Solamente cambia el motor de inferencia: la segunda línea de la ventana muestra `Interpreter/CPU`.

---

# Cierre de la demostración

## 9. Salir del programa

Con la ventana seleccionada:

```text
q
```

También puede utilizarse `ESC`. Al salir, OpenCV libera la cámara y cierra la ventana.

---

## 10. Desactivar el entorno virtual

```bash
deactivate
```

---

# Funcionamiento

La demo está dividida en tres archivos de Python:

- `rps_common.py` → código compartido: validación del modelo, preprocesamiento y la clase `Motor`.
- `evaluar.py` → mide tamaño, memoria, latencia y precisión de cada combinación modelo/API.
- `litert_rps.py` → captura la cámara, ejecuta el modelo y dibuja el resultado.

El código compartido garantiza que la precisión medida en `evaluar.py` corresponde exactamente al mismo preprocesamiento que se utiliza en la cámara en vivo.

## Las dos APIs de LiteRT

La clase `Motor` ofrece los mismos tres pasos para ambas APIs:

```python
motor = Motor("modelos/rps_int8_full.tflite", api="compiled")   # o api="interpreter"
motor.cargar_entrada(x)       # copiar el tensor de entrada
motor.ejecutar()              # SOLO la inferencia
probs = motor.leer_salida()   # probabilidades
```

| Paso | `Interpreter` | `CompiledModel` |
|---|---|---|
| Crear | `Interpreter(...)` + `allocate_tensors()` | `CompiledModel.from_file(...)` |
| Entrada | `set_tensor()` | `buffer.write()` |
| Ejecutar | `invoke()` | `run_by_index()` |
| Salida | `get_tensor()` | `buffer.read()` |
| Acelerador | CPU | CPU o GPU |

### Limitación de CompiledModel en Python

El `TensorBuffer` de `CompiledModel` admite `float32`, `int32` e `int8`, pero **no `uint8`**.

Por esa razón, el modelo `rps_int8_full` conserva la entrada y la salida en `float32`. Si se utiliza un modelo con entrada `uint8` (por ejemplo, un modelo cuantizado de Teachable Machine), `litert_rps.py` utiliza automáticamente `Interpreter` para ese modelo y lo indica en pantalla.

## Las cuatro variantes del modelo

- **`rps_fp32`** → modelo original, sin compresión. Es la referencia.
- **`rps_fp16`** → pesos en 16 bits.
- **`rps_int8_dyn`** → pesos en 8 bits. Las activaciones se cuantizan durante la ejecución, sin calibración.
- **`rps_int8_full`** → pesos y activaciones en 8 bits, calibrados con 200 imágenes representativas.

## Preprocesamiento

El tamaño de entrada se lee del modelo, por lo que el código no necesita modificarse con otra resolución. Para modelos `float32` existen dos normalizaciones:

- `--norm 0-1` → divide entre 255. Es la utilizada por los modelos de este repositorio.
- `--norm -1-1` → divide entre 127.5 y resta 1. Es la utilizada por los modelos de Teachable Machine en punto flotante.

---

# Controles

Durante la ejecución de `litert_rps.py`:

- **M** → cambiar al siguiente modelo.
- **F** → alternar entre la cámara y la imagen de respaldo.
- **S** → guardar el cuadro actual como imagen de respaldo (`test_hand.jpg`).
- **Q** → salir del programa.
- **ESC** → salir del programa.

# Opciones de línea de comandos

## `litert_rps.py`

- `--api compiled|interpreter` → API de LiteRT. Por defecto, `compiled`.
- `--gpu` → intenta usar la GPU. Si no está disponible, continúa en CPU.
- `--model ruta.tflite` → utiliza un solo modelo y desactiva la tecla `M`.
- `--camera N` → utiliza `/dev/videoN`. Por defecto, `0`.
- `--image ruta.jpg` → imagen de respaldo. Por defecto, `test_hand.jpg`.
- `--norm 0-1|-1-1` → normalización para modelos `float32`.
- `--hilos N` → hilos de CPU. Por defecto, `4`.
- `--no-mirror` → desactiva el efecto espejo.
- `--autoprueba N` → procesa N cuadros por modelo sin abrir ventana y finaliza.
- `--verbose` → muestra los registros internos de LiteRT.

Por ejemplo, para comprobar que todo funciona sin abrir la ventana:

```bash
python litert_rps.py --autoprueba 30
```

## `evaluar.py`

- `--modelos` → carpetas o archivos `.tflite`. Por defecto, `modelos/`.
- `--apis interpreter compiled` → APIs a medir. Por defecto, ambas.
- `--gpu` → agrega `CompiledModel` en GPU.
- `--rapido` → versión corta: 90 imágenes y 100 repeticiones.
- `--repeticiones N` → repeticiones de `invoke` para la latencia. Por defecto, `200`.
- `--max-imagenes N` → limita las imágenes de precisión. Por defecto, todas.
- `--hilos N` → hilos de CPU. Por defecto, `4`.
- `--norm 0-1|-1-1` → normalización para modelos `float32`.
- `--salida carpeta` → carpeta del reporte. Por defecto, `resultados/`.
- `--verbose` → muestra los registros internos de LiteRT.

# Cámara

El programa utiliza por defecto `/dev/video0`. Para comprobar las cámaras disponibles:

```bash
ls /dev/video*
```

Por ejemplo:

```text
/dev/video0
/dev/video1
```

Si la cámara que se desea utilizar es `/dev/video1`:

```bash
python litert_rps.py --camera 1
```

Si no se detecta ninguna cámara, o si falla durante la ejecución, el programa pasa automáticamente a la imagen de respaldo `test_hand.jpg`. La tecla `F` permite alternar entre ambas fuentes.

# Solución de problemas

### El modelo no es válido

Si aparece:

```text
'modelos/rps_fp32.tflite' NO es un modelo .tflite válido
```

o:

```text
The model is not a valid Flatbuffer buffer
```

el archivo no es un modelo real. Casi siempre es una página HTML guardada por una descarga fallida. Comprobar el contenido:

```bash
ls -l modelos/
head -c 100 modelos/rps_fp32.tflite
```

Un modelo real pesa entre 1 y 4 MiB. Si el archivo empieza con `<?xml` o `<html`, volver a descomprimir `artefactos_rps.zip` o ejecutar de nuevo el cuaderno.

### No hay modelos

Si aparece:

```text
[ERROR] No hay modelos en '/ruta/a/demostracion/modelos'.
```

comprobar que `artefactos_rps.zip` se descomprimió dentro de la carpeta `demostracion/` y que existe `modelos/`:

```bash
ls modelos/
```

### Falla `tfds.load` en Colab

Si aparece:

```text
ModuleNotFoundError: No module named 'importlib_resources'
```

ejecutar primero la celda `%pip install` del inicio del cuaderno. No ejecutar el entrenamiento con `!python archivo.py`: el comando `%pip` solamente funciona dentro de un cuaderno.

### La GPU no está disponible

Si aparece:

```text
GPU no disponible: LiteRT usa WebGPU sobre Vulkan en Linux y no encontró un adaptador utilizable
```

comprobar el driver de Vulkan:

```bash
sudo apt install vulkan-tools
vulkaninfo --summary
```

Si no aparece ningún dispositivo, la GPU no puede utilizarse en este equipo y la demo continúa en CPU. Para ver el registro completo, agregar `--verbose`.

### CompiledModel no admite uint8

Si aparece:

```text
CompiledModel (Python) no admite entradas uint8; use --api interpreter para este modelo.
```

el modelo tiene entrada `uint8`. En `litert_rps.py` el cambio de API es automático. En `evaluar.py`, utilizar:

```bash
python evaluar.py --apis interpreter
```

### Error `externally-managed-environment`

Aparece cuando se ejecuta `pip` fuera del entorno virtual. Ejecutar `./setup.sh` y activar el entorno:

```bash
source .venv/bin/activate
```

### No se puede acceder a la cámara

Si aparece:

```text
[AVISO] No se pudo abrir /dev/video0.
```

comprobar que existe:

```bash
ls /dev/video*
```

También comprobar que ninguna otra aplicación esté utilizando la cámara, como un navegador, Zoom u otra aplicación de videollamadas.

Si la cámara existe pero no se puede abrir, comprobar que el usuario pertenece al grupo `video`:

```bash
groups
sudo usermod -aG video $USER
```

Después de agregar el grupo, cerrar sesión y volver a iniciarla.

### La ventana no se abre

Si aparece un error relacionado con `xcb`, `wayland` o `Qt`, ejecutar:

```bash
QT_QPA_PLATFORM=xcb python litert_rps.py
```

Si persiste, comprobar el funcionamiento sin ventana:

```bash
python litert_rps.py --autoprueba 30
```

### Las latencias son inestables

Si el p90 es mucho mayor que el p50:

- Conectar la laptop a la corriente. En batería, la CPU reduce su frecuencia.
- Cerrar navegadores y otras aplicaciones pesadas.
- Aumentar las repeticiones con `--repeticiones 500`.

### Las predicciones en vivo son incorrectas

- Utilizar un fondo liso y buena iluminación frontal. El dataset de entrenamiento está formado por manos generadas por computadora sobre fondo blanco, por lo que una cámara real puede producir resultados menos estables.
- Si se utiliza un modelo de Teachable Machine en punto flotante, ejecutar con `--norm -1-1`.
- Comprobar que el orden de `modelos/labels.txt` coincide con las salidas del modelo. El programa avisa si la cantidad de etiquetas no coincide.

### Mensajes `RDRND` o `CPU random generator`

Son avisos del firmware de algunos procesadores AMD. No afectan el funcionamiento de LiteRT.

# Resumen

Esta demo muestra el flujo de trabajo de LiteRT en un equipo Linux:

- `TFLiteConverter` → convierte un modelo de Keras al formato `.tflite`.
- Cuantización **FP16** e **INT8** → reduce el tamaño del modelo. En INT8 completo se calibra con un dataset representativo.
- `Interpreter` → API clásica de LiteRT para ejecutar modelos en CPU.
- `CompiledModel` → API de LiteRT 2.x. Compila el modelo para el acelerador elegido (CPU o GPU).
- `TensorBuffer` → búferes de entrada y salida gestionados por LiteRT.
- Identificador **`TFL3`** → permite comprobar que un archivo es un modelo `.tflite` válido.
- Latencia **p50 / p90** → mide la inferencia de forma estable, sin incluir la captura ni el preprocesamiento.
- `cv.VideoCapture()` → captura de video desde la cámara.
- `cv.cvtColor()` y `cv.resize()` → preparan cada cuadro para el modelo.
- `cv.imshow()` y `cv.waitKey()` → muestran el resultado y detectan las teclas de control.

El resultado es una aplicación que mide y ejecuta en tiempo real el mismo modelo en cuatro representaciones distintas, y que permite compararlas con la misma escena y el mismo código.