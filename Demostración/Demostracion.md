# Piedra, Papel o Tijera: Clasificación en Tiempo Real con LiteRT

Este demo utiliza la API oficial de **LiteRT** (`ai-edge-litert`) y **OpenCV** para capturar video en tiempo real desde una cámara web, preprocesar los fotogramas, ejecutar la inferencia con un modelo de visión por computadora (*Rock, Paper, Scissors*) y mostrar en pantalla la clase detectada junto con su nivel de confianza, la latencia de inferencia y los FPS.

El programa incluye un **modo de contingencia**: con la tecla **`f`** se alterna entre la cámara en vivo y una imagen estática, y con la tecla **`s`** se guarda el fotograma actual como imagen de respaldo. Si la cámara falla durante la demostración, el programa pasa automáticamente a la imagen estática.

> **Nota importante sobre el modelo.** No existe un modelo `.tflite` oficial de Piedra/Papel/Tijera descargable por URL directa. El modelo debe **generarse** (entrenándolo en Google Colab con el script incluido, o con Teachable Machine) y copiarse a la carpeta del proyecto. Una versión anterior de esta guía descargaba el modelo con `curl` desde una URL inexistente, lo que dejaba una página de error guardada como `rps_model.tflite` y provocaba el error `The model is not a valid Flatbuffer buffer`.

---

## Flujo de Trabajo Completo con LiteRT

```text
 ENTRENAR            CONVERTIR            VALIDAR              EJECUTAR (LiteRT)
 Colab / Teachable → .tflite (FlatBuffer) → cabecera "TFL3" →  Interpreter → allocate_tensors
 Machine             + labels.txt                              → set_tensor → invoke → get_tensor
```

1. **Entrenar** un clasificador de imágenes (Keras en Colab o Teachable Machine).
2. **Convertir** el modelo al formato `.tflite` (con cuantización opcional).
3. **Validar** que el archivo sea un FlatBuffer TFLite real.
4. **Ejecutar** la inferencia en el equipo embebido/local con LiteRT.

---

## Requisitos del Sistema

* Linux / Ubuntu 22.04 LTS o superior (compatible con distribuciones basadas en Debian)
* Python 3.9 o superior
* Una cámara web (accesible en Linux mediante `/dev/video0`)
* Herramientas base del sistema: `python3-venv` y `python3-pip`
* Una cuenta de Google (para Colab) **o** acceso a [Teachable Machine](https://teachablemachine.withgoogle.com) para generar el modelo

---

## Estructura de Archivos

```text
Demostracion/
├── setup.sh                 # Prepara el entorno y valida el modelo
├── litert_rps.py            # Programa principal de la demo
├── entrenar_rps_colab.py    # (Opcional) Entrenamiento y exportación en Colab
├── rps_model.tflite         # Modelo generado por usted (ver sección siguiente)
├── labels.txt               # Etiquetas, una por línea (viene con el modelo)
├── test_hand.jpg            # Imagen de contingencia (se crea con la tecla 's')
└── .venv/                   # Entorno virtual (lo crea setup.sh)
```

---

## Paso 1: Obtener el Modelo (`rps_model.tflite` y `labels.txt`)

### Opción A: Entrenar en Google Colab (recomendada para explicar el flujo completo)

1. Abra [Google Colab](https://colab.research.google.com) y cree un cuaderno nuevo.
2. Suba `entrenar_rps_colab.py` (o pegue su contenido en una celda) y ejecútelo con `!python entrenar_rps_colab.py`.
3. El script entrena una CNN pequeña con el dataset `rock_paper_scissors`, la convierte a TFLite, verifica la precisión del `.tflite` y descarga `rps_model.tflite` y `labels.txt`.
4. Copie ambos archivos a la carpeta `Demostracion/`.

El modelo generado recibe imágenes de **150×150×3** en `float32` normalizadas a `[0, 1]` y devuelve 3 probabilidades en el orden **Piedra, Papel, Tijera**.

> El dataset es sintético (manos generadas por computadora sobre fondo claro). Con una cámara real funciona mejor con **fondo liso y buena iluminación**.

### Opción B: Teachable Machine (más fiable en vivo)

1. Cree un proyecto **Image Project → Standard image model** con tres clases: `Piedra`, `Papel`, `Tijera`.
2. Grabe muestras con su propia cámara, en el fondo y la luz en que hará la demostración, y entrene el modelo.
3. Use **Export Model → TensorFlow Lite** y descargue el paquete (contiene `model.tflite` y `labels.txt`).
4. Renombre `model.tflite` a `rps_model.tflite` y copie ambos archivos a `Demostracion/`.
5. Si exportó el modelo en **punto flotante** (float32), ejecute la demo con `--norm -1-1`. Si exportó el **cuantizado** (uint8), no necesita ningún parámetro adicional.

El formato de `labels.txt` de Teachable Machine (`0 Piedra`) es interpretado correctamente por el programa.

---

## Paso 2: Preparación Automatizada (`setup.sh`)

Debido a la restricción **PEP 668** en las distribuciones recientes de Ubuntu/Debian, que impide instalar paquetes globalmente con `pip`, las dependencias se instalan dentro de un entorno virtual (`.venv`).

El script **no descarga el modelo**: instala las dependencias, **valida** que `rps_model.tflite` sea un archivo TFLite real (la cabecera FlatBuffer debe contener `TFL3`) y, si encuentra un archivo inválido, lo aparta como `rps_model.tflite.invalido` para que el problema sea evidente.

Cree el archivo `setup.sh` con el siguiente contenido:

```bash
#!/usr/bin/env bash
# Prepara el entorno de la demo LiteRT. NO descarga el modelo a ciegas:
# valida que rps_model.tflite sea un FlatBuffer TFLite real.

VENV_DIR=".venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

is_valid_tflite() {
    # Un .tflite válido tiene el identificador "TFL3" en los bytes 4-7
    [ -s "$1" ] && [ "$(head -c 8 "$1" 2>/dev/null | tail -c 4 2>/dev/null)" = "TFL3" ]
}

echo "======================================================"
echo "  Configuración de entorno - LiteRT"
echo "======================================================"

# 1. Paquetes del sistema
if ! dpkg -s python3-venv >/dev/null 2>&1 || ! dpkg -s python3-pip >/dev/null 2>&1; then
    echo "[!] Instalando python3-venv y python3-pip..."
    sudo apt update && sudo apt install -y python3-venv python3-pip || exit 1
fi

# 2. Entorno virtual
if [ ! -d "$VENV_DIR" ]; then
    echo "[+] Creando entorno virtual '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR" || exit 1
else
    echo "[✓] Entorno virtual detectado."
fi
"$PIP_BIN" install --upgrade pip >/dev/null 2>&1

# 3. Dependencias de Python (Pillow ya no es necesario)
echo "[+] Verificando dependencias de Python..."
declare -A PACKAGES=(
    ["ai-edge-litert"]="ai_edge_litert"
    ["opencv-python"]="cv2"
    ["numpy"]="numpy"
)
for pkg in "${!PACKAGES[@]}"; do
    if ! "$PYTHON_BIN" -c "import ${PACKAGES[$pkg]}" >/dev/null 2>&1; then
        echo "   [-] Instalando $pkg..."
        "$PIP_BIN" install "$pkg" || exit 1
    else
        echo "   [✓] $pkg"
    fi
done

# 4. Validación del modelo
echo "[+] Verificando recursos..."
if is_valid_tflite rps_model.tflite; then
    echo "   [✓] rps_model.tflite es un modelo TFLite válido."
else
    if [ -e rps_model.tflite ]; then
        echo "   [✗] rps_model.tflite existe pero NO es un modelo válido:"
        echo "       $(head -c 120 rps_model.tflite | tr '\n' ' ')"
        echo "       Se renombra a rps_model.tflite.invalido"
        mv -f rps_model.tflite rps_model.tflite.invalido
    fi
    echo "   [!] Falta rps_model.tflite. Genérelo con entrenar_rps_colab.py (Google Colab)"
    echo "       o expórtelo desde Teachable Machine, y cópielo a esta carpeta."
fi

if [ -f labels.txt ]; then
    echo "   [✓] labels.txt ($(wc -l < labels.txt) líneas):"
    sed 's/^/       /' labels.txt | head -5
else
    echo "   [!] labels.txt no existe (se usará Piedra/Papel/Tijera por defecto)."
fi

[ -f test_hand.jpg ] || echo "   [i] test_hand.jpg: se crea con la tecla 's' durante la demo."

echo "======================================================"
echo "  Siguiente paso: source .venv/bin/activate && python litert_rps.py"
echo "======================================================"
```

---

## Paso 3: Código Principal de la Demostración (`litert_rps.py`)

Guarde el siguiente código en un archivo llamado `litert_rps.py`:

```python
#!/usr/bin/env python3
"""
Piedra, Papel o Tijera - Clasificación en tiempo real con LiteRT + OpenCV.

Flujo LiteRT: validar modelo -> Interpreter -> allocate_tensors -> leer metadatos
-> preprocesar -> set_tensor -> invoke -> get_tensor -> interpretar salida.

Teclas:  q / ESC = salir | f = alternar cámara / imagen de contingencia
         s = guardar el fotograma actual como imagen de contingencia
"""
import argparse
import os
import sys
import time

import cv2 as cv
import numpy as np
from ai_edge_litert.interpreter import Interpreter

DEFAULT_LABELS = ["Piedra", "Papel", "Tijera"]
WINDOW = "LiteRT - Piedra, Papel o Tijera"


# --------------------------------------------------------------------------
# Validación de archivos
# --------------------------------------------------------------------------
def validate_tflite(path):
    """Un .tflite válido es un FlatBuffer con el identificador 'TFL3' en los bytes 4-8."""
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] No existe el modelo '{path}'. Consulte la sección de modelo en la guía.")
    with open(path, "rb") as f:
        head = f.read(512)
    if os.path.getsize(path) < 1024 or head[4:8] != b"TFL3":
        preview = head[:150].decode("utf-8", errors="replace").replace("\n", " ")
        sys.exit(
            f"[ERROR] '{path}' NO es un modelo .tflite válido "
            f"({os.path.getsize(path)} bytes).\n"
            f"        Contenido inicial: {preview!r}\n"
            "        Probablemente la descarga falló y se guardó una página de error.\n"
            "        Borre el archivo y obtenga un modelo real (ver guía)."
        )


def load_labels(path):
    """Admite una etiqueta por línea, o el formato de Teachable Machine ('0 Nombre')."""
    if not os.path.isfile(path):
        return DEFAULT_LABELS
    labels = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                line = parts[1]
            labels.append(line)
    return labels or DEFAULT_LABELS


# --------------------------------------------------------------------------
# Preprocesamiento y postprocesamiento
# --------------------------------------------------------------------------
def center_square_crop(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s]


def preprocess(frame, width, height, dtype, norm):
    """Recorte cuadrado central -> BGR a RGB -> resize -> lote -> tipo de dato del tensor."""
    rgb = cv.cvtColor(center_square_crop(frame), cv.COLOR_BGR2RGB)
    resized = cv.resize(rgb, (width, height), interpolation=cv.INTER_AREA)
    batch = np.expand_dims(resized, axis=0)

    if dtype == np.float32:
        x = batch.astype(np.float32)
        return x / 255.0 if norm == "0-1" else (x / 127.5) - 1.0
    if dtype == np.int8:
        return (batch.astype(np.int16) - 128).astype(np.int8)
    return batch.astype(np.uint8)  # modelos cuantizados uint8: píxeles crudos 0-255


def read_probabilities(interpreter, out_detail):
    """Obtiene probabilidades, des-cuantizando y aplicando softmax si hace falta."""
    raw = interpreter.get_tensor(out_detail["index"])
    probs = np.squeeze(raw).astype(np.float32)

    scale, zero_point = out_detail["quantization"]
    if raw.dtype in (np.uint8, np.int8) and scale:
        probs = (probs - zero_point) * scale

    if probs.min() < 0.0 or probs.max() > 1.0001 or abs(float(probs.sum()) - 1.0) > 0.05:
        e = np.exp(probs - probs.max())  # la salida eran logits
        probs = e / e.sum()
    return probs


# --------------------------------------------------------------------------
# Captura de video
# --------------------------------------------------------------------------
def open_camera(index):
    cap = cv.VideoCapture(index, cv.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def draw_overlay(img, lines, footer):
    cv.rectangle(img, (0, 0), (img.shape[1], 12 + 30 * len(lines)), (0, 0, 0), -1)
    for i, (text, color) in enumerate(lines):
        cv.putText(img, text, (15, 30 + 30 * i), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv.putText(img, footer, (15, img.shape[0] - 15), cv.FONT_HERSHEY_SIMPLEX,
               0.5, (220, 220, 220), 1)


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default="rps_model.tflite")
    p.add_argument("--labels", default="labels.txt")
    p.add_argument("--image", default="test_hand.jpg", help="imagen de contingencia")
    p.add_argument("--camera", type=int, default=0, help="índice N de /dev/videoN")
    p.add_argument("--norm", choices=["0-1", "-1-1"], default="0-1",
                   help="normalización para modelos float32 (Keras: 0-1, Teachable Machine: -1-1)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-mirror", action="store_true", help="no voltear la cámara como espejo")
    return p.parse_args()


def main():
    args = parse_args()
    print("=== Inicializando LiteRT y OpenCV ===")

    # PASO 1: validar e instanciar el intérprete
    validate_tflite(args.model)
    try:
        interpreter = Interpreter(model_path=args.model, num_threads=args.threads)
    except Exception as e:
        sys.exit(f"[ERROR] No se pudo cargar el modelo: {e}")

    # PASO 2: reservar memoria para los tensores
    interpreter.allocate_tensors()

    # PASO 3: inspeccionar entrada/salida
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    _, height, width, channels = inp["shape"]
    dtype = inp["dtype"]
    labels = load_labels(args.labels)

    print(f"Modelo        : {args.model}")
    print(f"Entrada       : {width}x{height}x{channels}  tipo={np.dtype(dtype).name}")
    print(f"Salida        : forma={list(out['shape'])}  tipo={np.dtype(out['dtype']).name}")
    print(f"Etiquetas     : {labels}")
    if len(labels) != out["shape"][-1]:
        print(f"[AVISO] El modelo tiene {out['shape'][-1]} clases pero labels.txt "
              f"tiene {len(labels)} etiquetas.")

    # Inferencia de calentamiento (la primera llamada siempre es más lenta)
    interpreter.set_tensor(inp["index"], np.zeros(inp["shape"], dtype=dtype))
    interpreter.invoke()

    # PASO 4: fuentes de video
    cap = open_camera(args.camera)
    static_img = cv.imread(args.image) if os.path.isfile(args.image) else None
    use_camera = cap is not None
    if cap is None:
        print(f"[AVISO] No se pudo abrir /dev/video{args.camera}.")
    if static_img is None:
        print(f"[AVISO] No hay imagen de contingencia '{args.image}' "
              "(use la tecla 's' con la cámara para crearla).")
    if not use_camera and static_img is None:
        sys.exit("[ERROR] No hay cámara ni imagen de contingencia disponibles.")

    print("\nControles: q/ESC = salir | f = cámara/imagen | s = guardar imagen de contingencia\n")

    prev = time.perf_counter()
    while True:
        # --- Adquisición ---
        if use_camera:
            ok, frame = cap.read()
            if not ok:
                print("[AVISO] Falló la lectura de la cámara.")
                if static_img is None:
                    break
                use_camera = False
                continue
            if not args.no_mirror:
                frame = cv.flip(frame, 1)
            source = f"Fuente: camara /dev/video{args.camera}"
        else:
            frame = static_img.copy()
            source = f"Fuente: imagen fija ({args.image}) - modo contingencia"

        raw_frame = frame.copy()  # sin texto, para la tecla 's'

        # --- PASOS 5-8: preprocesar, inferir, leer resultados ---
        t0 = time.perf_counter()
        x = preprocess(frame, width, height, dtype, args.norm)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        probs = read_probabilities(interpreter, out)
        infer_ms = (time.perf_counter() - t0) * 1000

        idx = int(np.argmax(probs))
        name = labels[idx] if idx < len(labels) else f"Clase {idx}"

        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        # --- PASO 9: visualización ---
        draw_overlay(
            frame,
            [(f"Prediccion: {name} ({probs[idx] * 100:.1f}%)", (0, 255, 0)),
             (f"Inferencia: {infer_ms:.1f} ms | {fps:.0f} FPS", (0, 255, 255))],
            source,
        )
        cv.imshow(WINDOW, frame)

        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            if not use_camera and cap is None:
                print("[AVISO] No hay cámara disponible.")
            elif use_camera and static_img is None:
                print("[AVISO] No hay imagen de contingencia; presione 's' primero.")
            else:
                use_camera = not use_camera
                print("Modo:", "cámara" if use_camera else "imagen fija")
        elif key == ord("s") and use_camera:
            cv.imwrite(args.image, raw_frame)
            static_img = raw_frame
            print(f"Imagen de contingencia guardada en '{args.image}'")

    if cap is not None:
        cap.release()
    cv.destroyAllWindows()
    print("Demostración finalizada.")


if __name__ == "__main__":
    main()
```

---

## Paso 4: Guía de Ejecución

1. Ingrese al directorio de la demostración:

```bash
cd ~/Desktop/Taller_embebidos/LiteRT/Demostracion
```

2. Verifique que `rps_model.tflite` y `labels.txt` (Paso 1) estén en la carpeta y dé permisos de ejecución a `setup.sh`:

```bash
chmod +x setup.sh
```

3. Prepare el entorno (instala librerías y valida el modelo):

```bash
./setup.sh
```

4. Active el entorno virtual:

```bash
source .venv/bin/activate
```

5. Inicie la demostración:

```bash
python litert_rps.py
```

Al iniciar correctamente, el programa imprimirá las especificaciones del modelo cargado y abrirá la ventana de visualización. Ejemplo con el modelo de Colab (los valores dependen del modelo usado):

```text
=== Inicializando LiteRT y OpenCV ===
Modelo        : rps_model.tflite
Entrada       : 150x150x3  tipo=float32
Salida        : forma=[1, 3]  tipo=float32
Etiquetas     : ['Piedra', 'Papel', 'Tijera']

Controles: q/ESC = salir | f = cámara/imagen | s = guardar imagen de contingencia
```

6. **Antes de la presentación**: con la cámara en vivo, muestre una mano y pulse **`s`**. Esto crea `test_hand.jpg`, la imagen de contingencia real para su iluminación y fondo.

### Opciones de línea de comandos

| Opción | Descripción | Valor por defecto |
|---|---|---|
| `--model` | Ruta del modelo `.tflite` | `rps_model.tflite` |
| `--labels` | Archivo de etiquetas | `labels.txt` |
| `--image` | Imagen de contingencia | `test_hand.jpg` |
| `--camera N` | Usa `/dev/videoN` | `0` |
| `--norm {0-1,-1-1}` | Normalización para modelos `float32` | `0-1` |
| `--threads N` | Hilos de CPU para el intérprete | `4` |
| `--no-mirror` | Desactiva el efecto espejo de la cámara | espejo activado |

Ejemplo: `python litert_rps.py --camera 1 --norm -1-1`

---

## Funcionamiento Detallado de la API de LiteRT

El programa implementa el ciclo de inferencia de LiteRT con los siguientes componentes:

1. **`Interpreter(model_path, num_threads)`**: carga el grafo del modelo desde el archivo `.tflite` (un FlatBuffer). Falla con *"not a valid Flatbuffer buffer"* si el archivo no es un modelo real.
2. **`allocate_tensors()`**: reserva la memoria para los tensores intermedios, de entrada y de salida.
3. **`get_input_details()` / `get_output_details()`**: leen en tiempo de ejecución la forma (alto, ancho, canales), el tipo de dato (`float32`, `uint8` o `int8`) y los parámetros de cuantización de cada tensor.
4. **`preprocess()`**: recorta un cuadrado central del fotograma (para no deformar la mano), convierte BGR a RGB, redimensiona al tamaño del tensor y ajusta el tipo de dato: `float32` se normaliza a `[0, 1]` (o `[-1, 1]` con `--norm -1-1`) y los modelos cuantizados reciben los píxeles directamente.
5. **`set_tensor()`**: copia los datos preprocesados al tensor de entrada.
6. **`invoke()`**: ejecuta la red neuronal. La primera llamada se hace con datos vacíos como *calentamiento*, ya que siempre es más lenta.
7. **`get_tensor()`**: recupera la salida. Si el modelo es cuantizado se des-cuantiza con `scale` y `zero_point`, y si la salida son *logits* se aplica softmax para obtener probabilidades.

La latencia mostrada en pantalla mide únicamente el preprocesamiento, la inferencia y la lectura de resultados; no incluye la captura de la cámara ni el dibujado de la ventana.

---

## Controles de la Aplicación

* **F** → Alterna entre la cámara en vivo y la imagen de contingencia.
* **S** → (Con la cámara activa) guarda el fotograma actual como imagen de contingencia.
* **Q** o **ESC** → Finalizan la ejecución y liberan la cámara.

---

## Diagnóstico y Solución de Problemas

### `The model is not a valid Flatbuffer buffer`
El archivo `rps_model.tflite` no es un modelo TFLite válido; casi siempre es una página de error HTML/XML guardada por una descarga fallida. Compruébelo con:

```bash
ls -l rps_model.tflite        # un modelo real pesa decenas o cientos de KB
head -c 300 rps_model.tflite  # si empieza con "<?xml" o "<html", es una página de error
```

Elimine el archivo y genere un modelo real siguiendo el Paso 1. El programa y `setup.sh` ya validan la cabecera `TFL3` y muestran el contenido inicial del archivo cuando no es válido.

### Mensajes `WARNING: CPU random generator seem to be failing` / `RDRND`
Son avisos de la CPU o el firmware (conocidos en algunos procesadores AMD). No afectan a LiteRT y pueden ignorarse.

### Error `externally-managed-environment` (PEP 668)
Ocurre al ejecutar `pip install` fuera del entorno virtual. Ejecute `./setup.sh` y active el entorno con `source .venv/bin/activate`.

### Cámara no detectada
Si no se puede abrir la cámara, el programa pasa automáticamente al modo de contingencia (si existe `test_hand.jpg`). Para ver los dispositivos de video disponibles:

```bash
ls /dev/video*
```

Si su cámara es `/dev/video1`, no hace falta editar el código:

```bash
python litert_rps.py --camera 1
```

Si hay permisos insuficientes, verifique que su usuario pertenezca al grupo `video` (`groups`) y, de no ser así, ejecute `sudo usermod -aG video $USER` y vuelva a iniciar sesión.

### La ventana no abre (error de Qt / `xcb`)
En algunas sesiones Wayland de Ubuntu, OpenCV no encuentra el plugin gráfico. Pruebe:

```bash
QT_QPA_PLATFORM=xcb python litert_rps.py
```

### Las predicciones son incorrectas o inestables
* Use un **fondo liso** y buena iluminación; el dataset de Colab es sintético y es sensible al fondo.
* Si el modelo es `float32` de Teachable Machine, ejecute con `--norm -1-1`.
* Para mejores resultados en vivo, entrene con Teachable Machine usando su propia cámara y escenario.
* Verifique que el orden de `labels.txt` coincida con el orden de las clases del modelo. El programa avisa si el número de etiquetas no coincide con el número de salidas.

### Mismatch de dimensiones en el tensor
El tamaño de entrada se lee dinámicamente de `input_details[0]['shape']`, por lo que no es necesario cambiar el código al usar otro modelo (por ejemplo, Teachable Machine usa 224×224).

---

## Script de Entrenamiento y Exportación (`entrenar_rps_colab.py`)

Código completo del script de la Opción A (ejecutar en Google Colab):

```python
# Entrena un clasificador Piedra/Papel/Tijera y lo exporta a .tflite (LiteRT).
# Ejecutar en Google Colab (GPU opcional): pegar en una celda o subir y correr
#   !python entrenar_rps_colab.py
# Salida: rps_model.tflite y labels.txt  (entrada 150x150x3 float32 en [0,1], salida softmax de 3 clases)

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

IMG, BATCH, EPOCHS = 150, 64, 12

(ds_train, ds_test), info = tfds.load(
    "rock_paper_scissors", split=["train", "test"], as_supervised=True, with_info=True)
print("Orden de clases en el dataset:", info.features["label"].names)  # rock, paper, scissors


def prep_eval(img, lbl):
    img = tf.image.resize(img, (IMG, IMG))
    return tf.cast(img, tf.float32) / 255.0, lbl


def prep_train(img, lbl):
    img = tf.image.random_flip_left_right(img)
    crop = tf.random.uniform([], 220, 301, dtype=tf.int32)            # zoom aleatorio
    img = tf.image.random_crop(img, tf.stack([crop, crop, 3]))
    img = tf.cast(tf.image.resize(img, (IMG, IMG)), tf.float32) / 255.0
    img = tf.image.random_brightness(img, 0.25)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    return tf.clip_by_value(img, 0.0, 1.0), lbl


AUTOTUNE = tf.data.AUTOTUNE
train = ds_train.shuffle(1000).map(prep_train, AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)
test = ds_test.map(prep_eval, AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)

L = tf.keras.layers
model = tf.keras.Sequential([
    L.Input((IMG, IMG, 3)),
    L.Conv2D(32, 3, activation="relu"), L.MaxPooling2D(),
    L.Conv2D(64, 3, activation="relu"), L.MaxPooling2D(),
    L.Conv2D(128, 3, activation="relu"), L.MaxPooling2D(),
    L.Conv2D(128, 3, activation="relu"), L.MaxPooling2D(),
    L.Flatten(), L.Dropout(0.5),
    L.Dense(128, activation="relu"),
    L.Dense(3, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, epochs=EPOCHS, validation_data=test)

# --- Conversión a TFLite (cuantización dinámica: entrada/salida siguen siendo float32) ---
model.export("rps_saved_model")
converter = tf.lite.TFLiteConverter.from_saved_model("rps_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
assert tflite_bytes[4:8] == b"TFL3", "El archivo generado no es un FlatBuffer TFLite"

with open("rps_model.tflite", "wb") as f:
    f.write(tflite_bytes)
with open("labels.txt", "w", encoding="utf-8") as f:
    f.write("Piedra\nPapel\nTijera\n")   # mismo orden que el dataset: rock, paper, scissors

# --- Verificación rápida del .tflite con el intérprete ---
interp = tf.lite.Interpreter(model_content=tflite_bytes)
interp.allocate_tensors()
i, o = interp.get_input_details()[0], interp.get_output_details()[0]
print("Entrada:", i["shape"], i["dtype"].__name__, "| Salida:", o["shape"])

correct = total = 0
for x, y in test.unbatch().take(300):
    interp.set_tensor(i["index"], x[None].numpy())
    interp.invoke()
    correct += int(np.argmax(interp.get_tensor(o["index"])) == int(y))
    total += 1
print(f"Precisión del .tflite en {total} imágenes de prueba: {correct / total:.1%}")

try:
    from google.colab import files
    files.download("rps_model.tflite")
    files.download("labels.txt")
except ImportError:
    print("Copie rps_model.tflite y labels.txt a la carpeta de la demo.")
```

---

## Resumen del Entregable (Demo Técnica)

Esta demostración muestra el flujo completo de trabajo con LiteRT en sistemas embebidos:

* **Entrenamiento y conversión** de un modelo de clasificación a `.tflite` (Colab o Teachable Machine).
* **Validación** del artefacto del modelo antes de cargarlo.
* **Inferencia en CPU** con LiteRT (`ai-edge-litert`), con soporte para modelos `float32` y cuantizados.
* **Preprocesamiento y visualización** eficientes del video con OpenCV, midiendo latencia y FPS.
* **Modo de contingencia** (*fallback*) para demostraciones en vivo, con captura de la imagen de respaldo mediante la tecla `s`.