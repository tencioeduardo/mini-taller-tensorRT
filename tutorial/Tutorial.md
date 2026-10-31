# Del Hub al borde: descargar, cuantizar y medir un modelo con LiteRT

Tutorial guiado del mini-taller **"Despliegue de aprendizaje automático en sistemas embebidos: el caso de LiteRT"**.

En la demostración se mostró el flujo completo de LiteRT con un modelo entrenado por nosotros. En este tutorial usted recorrerá con más detalle las etapas de **optimización y evaluación**, partiendo de un modelo ya convertido que se descarga del **LiteRT Community Hub** en Hugging Face:

1. **Descargar** MobileNet-v2 (clasificador de 1000 clases de ImageNet) en dos versiones: precisión completa (FP32) y cuantizada estática (INT8).
2. **Inspeccionarlo** para conocer su entrada, su salida y su cuantización antes de usarlo.
3. **Ejecutarlo** con la API moderna `CompiledModel` sobre una foto tomada por usted.
4. **Cuantizarlo usted mismo** con LiteRT Quantizer (cuantización dinámica).
5. **Medir** latencia y memoria de las tres variantes en su propio equipo.
6. **Evaluar** cuánta precisión se pierde con cada cuantización.
7. **Resumir** todo en una tabla comparativa medida por usted.

> **Entregable.** Un PDF con los resultados que la consola muestra en cada paso. No necesita copiar la pantalla a mano: cada paso guarda automáticamente su salida en `resultados/paso_XX.txt` (ver la sección *Preparación del PDF de entrega*). Los pasos deben ejecutarse **en orden y uno por uno**, leyendo la explicación de cada uno.

---

## Flujo de Trabajo

```text
 PASO 0      PASO 1          PASO 2         PASO 3          PASO 4          PASO 5-6        PASO 7
 Entorno  →  Descargar    →  Inspeccionar → Ejecutar     →  Cuantizar    →  Medir        →  Resumir
 (.venv)     Hub: FP32,      firmas,        CompiledModel   litert-         latencia,       tabla
             INT8 +          tensores,      + foto propia   quantizer       RAM y           final
             100 imágenes    cuantización                   (dinámica)      precisión
```

### Relación con la presentación

| Paso | Concepto de la presentación |
|---|---|
| 1 | LiteRT Community Hub en Hugging Face (diap. 10). El modelo fue convertido desde PyTorch Vision (diap. 9). |
| 2 | Formato `.tflite` / FlatBuffers (diap. 5) y fórmula de la cuantización (diap. 7). |
| 3 | API `CompiledModel`: compila una vez y gestiona los búferes (diap. 5). |
| 4 | Cuantización post-entrenamiento, PTQ (diap. 7 y 8). |
| 5 | Aceleración en CPU con XNNPACK (diap. 6) y evaluación de latencia y RAM (diap. 11). |
| 6 | Validación de la precisión tras cuantizar (diap. 8 y 11). |
| 7 | "Valida siempre con mediciones propias" (diap. 11). |

---

## Requisitos

**Hardware**
* Computadora con Linux de 64 bits: procesador x86-64 (Intel/AMD) o ARM64 (por ejemplo, Raspberry Pi 4/5 con sistema operativo de 64 bits).
* Unos 400 MB libres en disco (el entorno virtual ocupa ~300 MB; modelos e imágenes, ~35 MB).
* Un teléfono o cámara para tomar **una foto** (paso 3).

**Software**
* Linux (probado en Ubuntu 22.04 y 24.04; funciona en otras distribuciones).
* **Python 3.10 a 3.14** (verifique con `python3 --version`).
* Paquetes del sistema `python3-venv` y `python3-pip` (el paso 0 los instala en Ubuntu/Debian si faltan).
* Conexión a Internet **solo en los pasos 0 y 1**.

**Tiempo estimado:** 40 a 60 minutos.

**Versiones fijas de las bibliotecas** (las instala el paso 0; no las cambie, porque las API de este ecosistema cambian con frecuencia):

| Paquete | Versión | Uso |
|---|---|---|
| `ai-edge-litert` | 2.3.0 | Runtime LiteRT: `CompiledModel` e `Interpreter` |
| `litert-quantizer` | 0.11.0 | Cuantización post-entrenamiento (antes llamado `ai-edge-quantizer`) |
| `huggingface_hub` | 2.4.0 | Descarga de modelos del Hub |
| `numpy`, `pillow` | las compatibles con su Python | Tensores e imágenes |

---

## Estructura de Archivos

```text
Tutorial/
├── setup.sh               # Paso 0: entorno virtual y reporte del equipo
├── comun.py               # Funciones compartidas por todos los pasos
├── 01_descargar.py        # Paso 1: modelos del Hub, etiquetas e imágenes
├── 02_inspeccionar.py     # Paso 2: inspección de los modelos .tflite
├── 03_inferencia.py       # Paso 3: inferencia con CompiledModel sobre su foto
├── 04_cuantizar.sh        # Paso 4: cuantización dinámica con litert-quantizer
├── 05_benchmark.py        # Paso 5: latencia y RAM
├── 06_precision.py        # Paso 6: precisión de cada variante
├── 07_resumen.py          # Paso 7: tabla comparativa final
├── mi_foto.jpg            # Su foto (paso 3)
│
├── .venv/                 # (se crea en el paso 0)
├── modelos/               # (se crea en el paso 1; el modelo dinámico, en el paso 4)
├── imagenes/              # (se crea en el paso 1) lote/ con 100 imágenes y verdad.csv
└── resultados/            # (se crea en el paso 0) paso_00.txt ... paso_07.txt  ← para el PDF
```

Cree una carpeta de trabajo y, dentro de ella, cada archivo con el contenido que se indica en este documento:

```bash
mkdir -p ~/Tutorial_LiteRT
cd ~/Tutorial_LiteRT
```

> En todo el tutorial, los comandos se ejecutan **dentro de esta carpeta**.

---

## Paso 0: Preparación del Entorno (`setup.sh`)

Las distribuciones recientes de Ubuntu/Debian aplican la restricción **PEP 668**, que impide instalar paquetes con `pip` fuera de un entorno virtual. Por eso todo se instala en `.venv`. El script además **fija las versiones** de las bibliotecas y genera un reporte de su equipo (CPU, núcleos, RAM e instrucciones SIMD), que es el contexto necesario para interpretar las mediciones de los pasos 5 y 6.

Cree el archivo `setup.sh`:

```bash
#!/usr/bin/env bash
# =============================================================================
# PASO 0 - Preparación del entorno del tutorial de LiteRT
#   1. Verifica la versión de Python (LiteRT admite 3.10 a 3.14).
#   2. Crea el entorno virtual .venv e instala versiones fijas de los paquetes.
#   3. Genera el reporte del equipo en resultados/paso_00.txt.
# =============================================================================

VENV_DIR=".venv"
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# Versiones probadas para este tutorial (no cambiarlas)
PAQUETES=(
    "ai-edge-litert==2.3.0"      # runtime LiteRT (CompiledModel e Interpreter)
    "litert-quantizer==0.11.0"   # cuantización post-entrenamiento (antes ai-edge-quantizer)
    "huggingface_hub==2.4.0"     # descarga de modelos del LiteRT Community Hub
    "pillow"                     # lectura de imágenes
    "numpy"                      # tensores
)

echo "======================================================"
echo "  PASO 0: preparación del entorno - Tutorial LiteRT"
echo "======================================================"

# ---- 1. Python del sistema -------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "[✗] No se encontró python3. Instálelo con el gestor de paquetes de su distribución."
    exit 1
fi
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
PYMIN=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$(python3 -c 'import sys; print(sys.version_info[0])')" != "3" ] || [ "$PYMIN" -lt 10 ] || [ "$PYMIN" -gt 14 ]; then
    echo "[✗] Python $PYVER no es compatible. LiteRT 2.3.0 requiere Python 3.10 a 3.14."
    exit 1
fi
echo "[✓] Python $PYVER compatible."

# ---- 2. Soporte para entornos virtuales ------------------------------------
if ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then
    if command -v apt >/dev/null 2>&1; then
        echo "[!] Instalando python3-venv y python3-pip (se pedirá su contraseña)..."
        sudo apt update && sudo apt install -y python3-venv python3-pip || exit 1
    else
        echo "[✗] Falta el módulo venv de Python. Instálelo con el gestor de paquetes"
        echo "    de su distribución (por ejemplo: dnf install python3-pip, pacman -S python-pip)."
        exit 1
    fi
fi

# ---- 3. Entorno virtual -----------------------------------------------------
if [ ! -x "$PY" ]; then
    echo "[+] Creando entorno virtual '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR" || { echo "[✗] No se pudo crear el entorno virtual."; exit 1; }
else
    echo "[✓] Entorno virtual detectado."
fi
"$PIP" install -q --upgrade pip >/dev/null 2>&1

# ---- 4. Paquetes ------------------------------------------------------------
echo "[+] Instalando paquetes (puede tardar unos minutos la primera vez)..."
if ! "$PIP" install -q "${PAQUETES[@]}"; then
    echo "[✗] Falló la instalación. Revise su conexión a Internet y vuelva a ejecutar ./setup.sh"
    exit 1
fi
if ! "$PY" -c "import ai_edge_litert.compiled_model, litert_quantizer, huggingface_hub, PIL, numpy" >/dev/null 2>&1; then
    echo "[✗] Los paquetes se instalaron, pero no se pueden importar. Vea 'Solución de problemas'."
    exit 1
fi
echo "[✓] Paquetes instalados e importables."

# ---- 5. Reporte del equipo (evidencia para el PDF) -------------------------
mkdir -p resultados
{
    echo "======================================================================"
    echo "  PASO 0: Reporte del entorno"
    echo "  $(date '+%Y-%m-%d %H:%M:%S') | equipo: $(hostname)"
    echo "======================================================================"
    echo
    echo "--- Sistema ---------------------------------------------------------"
    echo "Sistema operativo : $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
    echo "Kernel            : $(uname -r)  ($(uname -m))"
    echo "CPU               : $(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
    echo "Núcleos lógicos   : $(nproc)"
    echo "RAM total         : $(free -h | awk '/^Mem:/{print $2}')"
    # Instrucciones SIMD que aprovecha XNNPACK (x86: SSE/AVX, ARM: NEON = asimd)
    FLAGS=$(grep -m1 -E '^(flags|Features)' /proc/cpuinfo 2>/dev/null | tr ' ' '\n' \
            | grep -xE 'sse4_2|avx|avx2|fma|avx512f|avx512_vnni|avx_vnni|asimd|asimddp' | tr '\n' ' ')
    echo "SIMD (XNNPACK)    : ${FLAGS:-no detectado}"
    echo
    echo "--- Python y paquetes -----------------------------------------------"
    echo "Python            : $("$PY" --version 2>&1)"
    for p in ai-edge-litert litert-quantizer huggingface_hub numpy pillow; do
        printf "%-18s: %s\n" "$p" "$("$PIP" show "$p" 2>/dev/null | sed -n 's/^Version: //p')"
    done
    echo
    echo "[✓] Paso 0 completado: entorno listo."
} | tee resultados/paso_00.txt

echo
echo "[i] Salida guardada en resultados/paso_00.txt"
echo "======================================================"
echo "  Siguiente paso:"
echo "    source .venv/bin/activate"
echo "    python 01_descargar.py"
echo "======================================================"
```

Dé permisos de ejecución al script y ejecute el paso 0:

```bash
chmod +x setup.sh
./setup.sh
```

**Salida esperada** (los datos del sistema serán los de su equipo):

```text
======================================================================
  PASO 0: Reporte del entorno
  2026-10-31 09:41:33 | equipo: vm
======================================================================

--- Sistema ---------------------------------------------------------
Sistema operativo : Ubuntu 24.04.5 LTS
Kernel            : 6.18.44-fc-v185  (x86_64)
CPU               : Intel(R) Xeon(R) Processor @ 2.10GHz
Núcleos lógicos   : 2
RAM total         : 7.8Gi
SIMD (XNNPACK)    : fma sse4_2 avx avx2 avx512f avx_vnni avx512_vnni 

--- Python y paquetes -----------------------------------------------
Python            : Python 3.13.18
ai-edge-litert    : 2.3.0
litert-quantizer  : 0.11.0
huggingface_hub   : 2.4.0
numpy             : 2.5.4
pillow            : 13.0.0

[✓] Paso 0 completado: entorno listo.
```

Active el entorno virtual. **Debe hacerlo cada vez que abra una terminal nueva** para continuar el tutorial:

```bash
source .venv/bin/activate
```

El indicador de la terminal mostrará `(.venv)` al inicio.

**Qué observar:** la línea `SIMD (XNNPACK)` lista las instrucciones vectoriales de su procesador (AVX2, AVX-512, NEON/`asimd`). XNNPACK, el acelerador de CPU de LiteRT, las usa para ejecutar varias operaciones en una sola instrucción.

---

## Archivo de apoyo: `comun.py`

Todos los pasos en Python importan este módulo. Contiene:

* **Rutas** de modelos, imágenes y resultados.
* **`Registro`**: imprime un encabezado (paso, fecha, equipo) y copia todo lo que aparece en pantalla a `resultados/paso_XX.txt`.
* **`redirigir_logs_nativos`**: LiteRT escribe mensajes internos desde C++ (`INFO: ...`) que no se pueden apagar con variables de entorno. Se envían a `resultados/litert_nativo.log` para que la consola quede legible; los errores de Python siguen apareciendo en pantalla.
* **`validar_tflite`**: comprueba la cabecera `TFL3` del FlatBuffer (la lección aprendida en la demostración).
* **`preprocesar`**: convierte una imagen al tensor que espera MobileNet-v2 (se explica en el paso 3).
* **`Clasificador`**: empaqueta el ciclo de `CompiledModel` que usted escribirá a mano en el paso 3, y además cuantiza la entrada y des-cuantiza la salida cuando el modelo es INT8. Los pasos 5 y 6 lo usan para no repetir código.

Cree el archivo `comun.py`:

```python
"""
comun.py - Funciones compartidas por los pasos del tutorial de LiteRT.

Contiene:
  * Rutas y nombres de archivo usados en todos los pasos.
  * Registro: copia todo lo que se imprime en pantalla a resultados/paso_XX.txt.
  * capturar_logs_nativos: aparta los mensajes internos (en C++) de LiteRT.
  * validar_tflite: comprueba que un archivo sea un FlatBuffer .tflite real.
  * preprocesar: convierte una imagen al tensor que espera MobileNet-v2.
  * Clasificador: envuelve CompiledModel (lo mismo que se hace a mano en el paso 3).
"""
import contextlib
import datetime
import json
import os
import platform
import socket
import sys
import tempfile
import time

import numpy as np

# --------------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------------
DIR_MODELOS = "modelos"
DIR_RESULTADOS = "resultados"
DIR_LOTE = os.path.join("imagenes", "lote")
ARCHIVO_VERDAD = os.path.join("imagenes", "verdad.csv")
ARCHIVO_ETIQUETAS = os.path.join(DIR_MODELOS, "imagenet-1k-id2label.json")

MODELO_FP32 = os.path.join(DIR_MODELOS, "mobilenet_v2.tflite")
MODELO_DINAMICO = os.path.join(DIR_MODELOS, "mobilenet_v2_dynamic_wi8_afp32.tflite")
MODELO_INT8 = os.path.join(DIR_MODELOS, "mobilenet_v2_int8_channelwise.tflite")

# (nombre corto, descripción, ruta) en el orden en que se comparan
VARIANTES = [
    ("FP32", "precisión completa (Hub)", MODELO_FP32),
    ("DIN-W8", "dinámica wi8/afp32 (propia, paso 4)", MODELO_DINAMICO),
    ("INT8", "estática wi8/ai8 (Hub)", MODELO_INT8),
]

# Normalización de ImageNet usada por los modelos de PyTorch Vision
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIACION = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------
# Salida en pantalla y en archivo
# --------------------------------------------------------------------------
class _Tee:
    """Escribe a la vez en la terminal y en un archivo."""

    def __init__(self, terminal, archivo):
        self.terminal, self.archivo = terminal, archivo

    def write(self, texto):
        self.terminal.write(texto)
        self.archivo.write(texto)

    def flush(self):
        self.terminal.flush()
        self.archivo.flush()


_logs_redirigidos = False


def redirigir_logs_nativos():
    """
    LiteRT escribe mensajes INFO/WARNING desde C++ directamente en el
    descriptor 2 (stderr), y no se pueden apagar con variables de entorno.
    Aquí se envían a resultados/litert_nativo.log, mientras que los errores
    de Python se siguen mostrando en la terminal.
    """
    global _logs_redirigidos
    if _logs_redirigidos:
        return
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    sys.stderr.flush()
    terminal = os.dup(2)
    sys.stderr = os.fdopen(terminal, "w", buffering=1, encoding="utf-8")
    log = open(os.path.join(DIR_RESULTADOS, "litert_nativo.log"), "a", encoding="utf-8")
    log.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv)}\n")
    log.flush()
    os.dup2(log.fileno(), 2)
    _logs_redirigidos = True


@contextlib.contextmanager
def Registro(paso, titulo, activo=True):
    """Imprime un encabezado y guarda toda la salida en resultados/paso_XX.txt."""
    redirigir_logs_nativos()
    if not activo:
        yield
        return
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    ruta = os.path.join(DIR_RESULTADOS, f"paso_{paso:02d}.txt")
    with open(ruta, "w", encoding="utf-8") as archivo:
        original = sys.stdout
        sys.stdout = _Tee(original, archivo)
        try:
            ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("=" * 70)
            print(f"  PASO {paso}: {titulo}")
            print(f"  {ahora} | equipo: {socket.gethostname()} | "
                  f"Python {platform.python_version()}")
            print("=" * 70)
            yield
        finally:
            sys.stdout = original
            print(f"\n[i] Salida guardada en {ruta}")


def seccion(titulo):
    print(f"\n--- {titulo} " + "-" * max(0, 64 - len(titulo)))


def ok(mensaje):
    print(f"[✓] {mensaje}")


def fallar(mensaje):
    """Termina el programa con un mensaje de error claro."""
    print(f"\n[✗] ERROR: {mensaje}")
    sys.exit(1)


def tamano_mb(ruta):
    return os.path.getsize(ruta) / 1e6


# --------------------------------------------------------------------------
# Mensajes internos de LiteRT
# --------------------------------------------------------------------------
@contextlib.contextmanager
def capturar_logs_nativos(destino):
    """
    Captura los mensajes nativos de LiteRT de un bloque de código: los desvía
    a un archivo temporal y, al salir, deja su texto en la lista `destino`
    (y una copia en resultados/litert_nativo.log si está activo).
    """
    copia = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(copia, 2)
            os.close(copia)
            tmp.seek(0)
            texto = tmp.read().decode("utf-8", errors="replace")
            destino.append(texto)
            if _logs_redirigidos:
                os.write(2, texto.encode("utf-8"))


# --------------------------------------------------------------------------
# Validación de archivos
# --------------------------------------------------------------------------
def validar_tflite(ruta):
    """Un .tflite válido es un FlatBuffer con el identificador 'TFL3' en los bytes 4-8."""
    if not os.path.isfile(ruta):
        return False, "el archivo no existe"
    with open(ruta, "rb") as f:
        cabecera = f.read(512)
    if cabecera[4:8] != b"TFL3":
        vista = cabecera[:80].decode("utf-8", errors="replace").replace("\n", " ")
        return False, f"no tiene la cabecera TFL3 (inicia con: {vista!r})"
    return True, f"cabecera {cabecera[4:8].decode()} correcta"


def exigir_modelo(ruta, paso_que_lo_crea):
    valido, detalle = validar_tflite(ruta)
    if not valido:
        fallar(f"'{ruta}': {detalle}.\n"
               f"    Ejecute primero el paso {paso_que_lo_crea}.")


# --------------------------------------------------------------------------
# Preprocesamiento (igual al de la tarjeta del modelo en Hugging Face)
# --------------------------------------------------------------------------
def preprocesar(ruta_imagen):
    """
    Imagen -> tensor float32 de forma (1, 3, 224, 224), formato NCHW.
      1. Redimensiona el lado corto a 232 px (manteniendo la proporción).
      2. Recorta el cuadrado central de 224 x 224.
      3. Escala a [0, 1] y normaliza con la media y desviación de ImageNet.
      4. Reordena de HWC (alto, ancho, canales) a CHW, como en PyTorch.
    """
    from PIL import Image

    img = Image.open(ruta_imagen).convert("RGB")
    w, h = img.size
    s = 232
    if w < h:
        img = img.resize((s, int(h * s / w)), Image.BILINEAR)
    else:
        img = img.resize((int(w * s / h), s), Image.BILINEAR)
    izq = int(round((img.size[0] - 224) / 2.0))
    arr = int(round((img.size[1] - 224) / 2.0))
    img = img.crop((izq, arr, izq + 224, arr + 224))

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEDIA) / DESVIACION
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def cargar_etiquetas():
    if not os.path.isfile(ARCHIVO_ETIQUETAS):
        fallar(f"No existe '{ARCHIVO_ETIQUETAS}'. Ejecute primero el paso 1.")
    with open(ARCHIVO_ETIQUETAS, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


# --------------------------------------------------------------------------
# Parámetros de cuantización de entrada/salida
# --------------------------------------------------------------------------
def leer_tensores_es(ruta):
    """
    Devuelve los detalles de los tensores de entrada y salida, incluida la
    cuantización (escala, punto cero). En Python, CompiledModel informa forma
    y tipo pero no la escala ni el punto cero, por eso se leen con Interpreter.
    """
    from ai_edge_litert.interpreter import Interpreter

    logs = []
    with capturar_logs_nativos(logs):
        interprete = Interpreter(model_path=ruta)
    return interprete.get_input_details()[0], interprete.get_output_details()[0]


# --------------------------------------------------------------------------
# Clasificador: empaqueta el ciclo de CompiledModel del paso 3
# --------------------------------------------------------------------------
class Clasificador:
    """
    Carga un .tflite con CompiledModel y ejecuta inferencias.
    Si el modelo es INT8 estático, cuantiza la entrada y des-cuantiza la
    salida con:  real = escala * (entero - punto_cero)
    """

    def __init__(self, ruta, hilos=1):
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator
        from ai_edge_litert.options import CpuOptions, Options

        entrada, salida = leer_tensores_es(ruta)
        self.tipo_in, self.tipo_out = entrada["dtype"], salida["dtype"]
        self.q_in, self.q_out = entrada["quantization"], salida["quantization"]
        self.n_clases = int(salida["shape"][-1])

        opciones = Options(hardware_accelerators=HardwareAccelerator.CPU,
                           cpu_options=CpuOptions(num_threads=hilos))
        self.logs = []
        t0 = time.perf_counter()
        with capturar_logs_nativos(self.logs):
            self.modelo = CompiledModel.from_file(ruta, options=opciones)
        self.ms_compilacion = (time.perf_counter() - t0) * 1000
        self.entradas = self.modelo.create_input_buffers(0)
        self.salidas = self.modelo.create_output_buffers(0)

    def _a_tipo_entrada(self, x):
        if self.tipo_in == np.float32:
            return x
        escala, cero = self.q_in
        info = np.iinfo(self.tipo_in)
        q = np.round(x / escala) + cero
        return np.clip(q, info.min, info.max).astype(self.tipo_in)

    def logits(self, x):
        """x: tensor float32 (1, 3, 224, 224). Devuelve 1000 logits float32."""
        self.entradas[0].write(self._a_tipo_entrada(x))
        self.modelo.run_by_index(0, self.entradas, self.salidas)
        y = self.salidas[0].read(self.n_clases, self.tipo_out)
        if self.tipo_out == np.float32:
            return y.astype(np.float32)
        escala, cero = self.q_out
        return (y.astype(np.float32) - cero) * escala

    def probabilidades(self, x):
        return softmax(self.logits(x))

    @property
    def totalmente_acelerado(self):
        return self.modelo.is_fully_accelerated()
```

---

## Paso 1: Descargar el Modelo del Hub (`01_descargar.py`)

El **LiteRT Community Hub** es la organización `litert-community` de Hugging Face, donde Google AI Edge y la comunidad publican modelos ya convertidos a `.tflite`. Usaremos el repositorio [`litert-community/MobileNet-v2`](https://huggingface.co/litert-community/MobileNet-v2), que contiene el mismo modelo en varias versiones:

| Archivo | Descripción | Tamaño |
|---|---|---|
| `mobilenet_v2.tflite` | Precisión completa (FP32) | 14 MB |
| `mobilenet_v2_dynamic_wi8_afp32.tflite` | Cuantización dinámica: pesos INT8, activaciones FP32 | 3.9 MB |
| `mobilenet_v2_int8_channelwise.tflite` | Cuantización estática INT8 (pesos y activaciones), calibrada | 4.0 MB |
| `mobilenet_v2_Google_Tensor_G5_apply_plugin.tflite` | Compilado por adelantado (AOT) para la NPU de Google Tensor G5 | 8.4 MB |

Descargaremos la **FP32** y la **INT8 estática**. La versión dinámica **la generará usted** en el paso 4 y la comparará con las demás. La INT8 estática requiere un proceso de calibración con datos representativos (diapositiva 8): aquí la usamos ya calibrada por los autores del Hub.

Según su tarjeta, el modelo **se convirtió desde un checkpoint de PyTorch Vision** (el camino de la diapositiva 9). Esto tiene una consecuencia práctica que veremos en los pasos 2 y 3: la entrada conserva el formato de PyTorch.

Este paso también descarga:
* Las **etiquetas** de las 1000 clases de ImageNet (`imagenet-1k-id2label.json`, del dataset `huggingface/label-files`).
* Un **lote de 100 imágenes etiquetadas** de ImageNet (una cada 10 clases), del repositorio público `EliSchwartz/imagenet-sample-images` en GitHub. Se usan en el paso 6 para medir la precisión.

Dos decisiones de diseño a destacar:
* Se fija la **revisión** (commit) del repositorio del Hub. Si los autores actualizan el modelo, usted seguirá descargando exactamente los mismos archivos que el resto del grupo: los resultados son reproducibles.
* Cada `.tflite` descargado se **valida** (cabecera `TFL3`) antes de usarlo.

Cree el archivo `01_descargar.py`:

```python
#!/usr/bin/env python3
"""
PASO 1 - Descargar el modelo del LiteRT Community Hub y los datos de prueba.

  * Modelos (Hugging Face, organización litert-community, repositorio MobileNet-v2):
      - mobilenet_v2.tflite                   -> precisión completa (FP32)
      - mobilenet_v2_int8_channelwise.tflite  -> cuantización estática INT8
  * Etiquetas de ImageNet (Hugging Face, dataset huggingface/label-files).
  * Lote de 100 imágenes etiquetadas de ImageNet (GitHub), usado en el paso 6.

Es el único paso que necesita Internet. Si se interrumpe, puede volver a
ejecutarse: lo ya descargado y válido no se descarga de nuevo.
"""
import csv
import os
import time
import urllib.request

from comun import (ARCHIVO_ETIQUETAS, ARCHIVO_VERDAD, DIR_LOTE, DIR_MODELOS,
                   MODELO_FP32, MODELO_INT8, Registro, fallar, ok, seccion,
                   tamano_mb, validar_tflite)

REPO_MODELO = "litert-community/MobileNet-v2"
# Se fija la versión (commit) del repositorio para que todos descarguen
# exactamente los mismos archivos, aunque el repositorio cambie después.
REVISION = "48e55de4765732640b9702c520999e4bf1a1f50f"

URL_IMAGENES = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"

# Una imagen de muestra por cada 10 clases de ImageNet (clases 0, 10, 20, ..., 990).
# El nombre empieza con el identificador de la clase (synset) de ImageNet.
LOTE = """
n01440764_tench n01530575_brambling n01601694_water_ouzel
n01641577_bullfrog n01682714_American_chameleon
n01698640_American_alligator n01740131_night_snake n01770081_harvestman
n01795545_black_grouse n01820546_lorikeet n01860187_black_swan
n01924916_flatworm n01980166_fiddler_crab n02007558_flamingo
n02027492_red-backed_sandpiper n02077923_sea_lion n02088094_Afghan_hound
n02090721_Irish_wolfhound n02093428_American_Staffordshire_terrier
n02095889_Sealyham_terrier n02097474_Tibetan_terrier
n02100236_German_short-haired_pointer n02102480_Sussex_spaniel
n02105855_Shetland_sheepdog n02107908_Appenzeller n02110185_Siberian_husky
n02112137_chow n02114548_white_wolf n02120505_grey_fox n02128925_jaguar
n02165105_tiger_beetle n02219486_ant n02268853_damselfly
n02325366_wood_rabbit n02391049_zebra n02417914_ibex n02444819_otter
n02484975_guenon n02493509_titi n02526121_eel n02669723_academic_gown
n02727426_apiary n02787622_banjo n02802426_basketball n02823428_beer_bottle
n02860847_bobsled n02894605_breakwater n02948072_candle
n02977058_cash_machine n03000247_chain_mail n03042490_cliff_dwelling
n03095699_container_ship n03131574_crib n03196217_digital_clock
n03240683_drilling_platform n03297495_espresso_maker
n03379051_football_helmet n03424325_gasmask n03457902_greenhouse
n03485407_hand-held_computer n03532672_hook n03595614_jersey
n03642806_laptop n03680355_Loafer n03717622_manhole_cover
n03759954_microphone n03776460_mobile_home n03791053_motor_scooter
n03825788_nipple n03868242_oxcart n03887697_paper_towel
n03908714_pencil_sharpener n03937543_pill_bottle n03967562_plow
n03995372_power_drill n04033995_quilt n04070727_refrigerator
n04120489_running_shoe n04147183_schooner n04204238_shopping_basket
n04243546_slot n04264628_space_bar n04310018_steam_locomotive
n04336792_stretcher n04367480_swab n04399382_teddy n04443257_tobacco_shop
n04482393_tricycle n04509417_unicycle n04540053_volleyball
n04562935_water_tower n04597913_wooden_spoon n06874185_traffic_light
n07684084_French_loaf n07716906_spaghetti_squash n07747607_orange
n07836838_chocolate_sauce n09193705_alp n09472597_volcano n12768682_buckeye
""".split()


def descargar_hub(repo, archivo, repo_type=None, revision=None):
    """Descarga un archivo de Hugging Face dentro de la carpeta modelos/."""
    from huggingface_hub import hf_hub_download

    destino = os.path.join(DIR_MODELOS, archivo)
    if os.path.isfile(destino) and (not archivo.endswith(".tflite")
                                    or validar_tflite(destino)[0]):
        print(f"    (ya existe) {destino}")
        return destino
    t0 = time.perf_counter()
    try:
        ruta = hf_hub_download(repo, archivo, repo_type=repo_type,
                               revision=revision, local_dir=DIR_MODELOS)
    except Exception as e:  # sin red, repositorio movido, etc.
        fallar(f"No se pudo descargar '{archivo}' de '{repo}':\n    {e}")
    print(f"    descargado {ruta} en {time.perf_counter() - t0:.1f} s")
    return ruta


def es_jpeg_valido(ruta):
    from PIL import Image
    try:
        with Image.open(ruta) as img:
            img.verify()
        return True
    except Exception:
        return False


def descargar_imagen(nombre):
    destino = os.path.join(DIR_LOTE, nombre)
    if os.path.isfile(destino) and es_jpeg_valido(destino):
        return "existe"
    for intento in range(3):
        try:
            urllib.request.urlretrieve(URL_IMAGENES + nombre, destino)
            if es_jpeg_valido(destino):
                return "nueva"
        except Exception:
            time.sleep(1 + intento)
    if os.path.exists(destino):
        os.remove(destino)
    return "error"


def main():
    os.makedirs(DIR_MODELOS, exist_ok=True)
    os.makedirs(DIR_LOTE, exist_ok=True)

    seccion("1.1 Modelos del LiteRT Community Hub")
    print(f"Repositorio: https://huggingface.co/{REPO_MODELO}  (revisión {REVISION[:7]})")
    for ruta_local in (MODELO_FP32, MODELO_INT8):
        descargar_hub(REPO_MODELO, os.path.basename(ruta_local), revision=REVISION)

    seccion("1.2 Validación de los archivos .tflite")
    for ruta in (MODELO_FP32, MODELO_INT8):
        valido, detalle = validar_tflite(ruta)
        if not valido:
            fallar(f"{ruta}: {detalle}. Borre el archivo y repita el paso 1.")
        print(f"[✓] {os.path.basename(ruta):40s} {tamano_mb(ruta):6.2f} MB  {detalle}")

    seccion("1.3 Etiquetas de las 1000 clases de ImageNet")
    descargar_hub("huggingface/label-files", os.path.basename(ARCHIVO_ETIQUETAS),
                  repo_type="dataset")
    from comun import cargar_etiquetas
    etiquetas = cargar_etiquetas()
    if len(etiquetas) != 1000:
        fallar(f"Se esperaban 1000 etiquetas y hay {len(etiquetas)}.")
    ok(f"{len(etiquetas)} etiquetas.")
    print(f"    Ejemplos: 0 = '{etiquetas[0][:30]}' | 999 = '{etiquetas[999][:30]}'")

    seccion("1.4 Lote de 100 imágenes etiquetadas (para el paso 6)")
    print(f"Fuente: {URL_IMAGENES}")
    if len(LOTE) != 100:
        fallar("La lista LOTE debe tener exactamente 100 nombres.")
    conteo = {"existe": 0, "nueva": 0, "error": 0}
    fallidas = []
    for i, base in enumerate(LOTE, start=1):
        estado = descargar_imagen(base + ".JPEG")
        conteo[estado] += 1
        if estado == "error":
            fallidas.append(base)
        if i % 20 == 0:
            print(f"    {i:3d}/100 imágenes procesadas")
    # Clase verdadera: la imagen k del lote es de la clase 10*k de ImageNet
    with open(ARCHIVO_VERDAD, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["archivo", "clase"])
        for k, base in enumerate(LOTE):
            escritor.writerow([base + ".JPEG", 10 * k])
    print(f"    nuevas: {conteo['nueva']} | ya existían: {conteo['existe']} | "
          f"con error: {conteo['error']}")
    if fallidas:
        fallar(f"No se pudieron descargar {len(fallidas)} imágenes "
               f"(p. ej. {fallidas[0]}). Vuelva a ejecutar el paso 1.")
    ok(f"100 imágenes en {DIR_LOTE}/ y clases verdaderas en {ARCHIVO_VERDAD}")

    seccion("Resumen")
    ok("Paso 1 completado: modelos, etiquetas e imágenes listos.")
    print("    A partir de aquí ya no se necesita Internet.")


if __name__ == "__main__":
    with Registro(1, "Descarga del modelo (Hub) y de los datos de prueba"):
        main()
```

Ejecute el paso 1 (con el entorno activo):

```bash
python 01_descargar.py
```

**Salida esperada:**

```text
======================================================================
  PASO 1: Descarga del modelo (Hub) y de los datos de prueba
  2026-10-31 09:45:57 | equipo: vm | Python 3.13.18
======================================================================

--- 1.1 Modelos del LiteRT Community Hub ----------------------------
Repositorio: https://huggingface.co/litert-community/MobileNet-v2  (revisión 48e55de)
    descargado modelos/mobilenet_v2.tflite en X.X s
    descargado modelos/mobilenet_v2_int8_channelwise.tflite en X.X s

--- 1.2 Validación de los archivos .tflite --------------------------
[✓] mobilenet_v2.tflite                       14.02 MB  cabecera TFL3 correcta
[✓] mobilenet_v2_int8_channelwise.tflite       4.05 MB  cabecera TFL3 correcta

--- 1.3 Etiquetas de las 1000 clases de ImageNet --------------------
    descargado modelos/imagenet-1k-id2label.json en X.X s
[✓] 1000 etiquetas.
    Ejemplos: 0 = 'tench, Tinca tinca' | 999 = 'toilet tissue, toilet paper, ba'

--- 1.4 Lote de 100 imágenes etiquetadas (para el paso 6) -----------
Fuente: https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/
     20/100 imágenes procesadas
     40/100 imágenes procesadas
     60/100 imágenes procesadas
     80/100 imágenes procesadas
    100/100 imágenes procesadas
    nuevas: 100 | ya existían: 0 | con error: 0
[✓] 100 imágenes en imagenes/lote/ y clases verdaderas en imagenes/verdad.csv

--- Resumen ---------------------------------------------------------
[✓] Paso 1 completado: modelos, etiquetas e imágenes listos.
    A partir de aquí ya no se necesita Internet.
```

Si se interrumpe por un problema de red, vuelva a ejecutarlo: lo que ya se descargó correctamente no se descarga de nuevo.

---

## Paso 2: Inspeccionar los Modelos (`02_inspeccionar.py`)

Antes de usar un modelo descargado hay que responder tres preguntas: **qué forma y tipo tiene su entrada, qué devuelve su salida y si está cuantizado**. Equivocarse en cualquiera de ellas no produce un error: produce predicciones incorrectas en silencio.

El script usa dos API de LiteRT:
* `Interpreter` (clásica, la de la demostración): permite leer la **escala y el punto cero** de cada tensor, que la API `CompiledModel` de Python aún no expone, y listar los **operadores** del grafo.
* `CompiledModel` (moderna): informa las **firmas** del modelo, el tamaño de los búferes y si el grafo quedó completamente acelerado.

También cuenta los **segmentos XNNPACK**: al preparar el modelo para la CPU, LiteRT entrega a XNNPACK las partes del grafo que este sabe acelerar. Si todo el grafo es compatible, queda en **1 segmento**; si no, el grafo se parte y las porciones no compatibles corren con kernels genéricos (más lentos). Esta es la *ejecución heterogénea* de la diapositiva 6, observada en la práctica.

Cree el archivo `02_inspeccionar.py`:

```python
#!/usr/bin/env python3
"""
PASO 2 - Inspeccionar un modelo .tflite antes de usarlo.

Para cada modelo muestra:
  * Tamaño y cabecera del FlatBuffer.
  * Firmas (signatures): nombres de entrada y salida con los que se invoca.
  * Tensores de entrada y salida: forma, tipo de dato y parámetros de cuantización.
  * Operadores del grafo y tipos de dato de todos sus tensores.
  * En cuántos segmentos queda repartido el grafo al delegarlo a XNNPACK (CPU).

Uso:
  python 02_inspeccionar.py                      # FP32 e INT8 del Hub
  python 02_inspeccionar.py modelos/otro.tflite  # cualquier modelo
"""
import argparse
import os
from collections import Counter

import numpy as np

from comun import (MODELO_FP32, MODELO_INT8, Registro, capturar_logs_nativos,
                   fallar, ok, seccion, tamano_mb, validar_tflite)


def describir_tensor(det):
    escala, cero = det["quantization"]
    cuant = (f"escala={escala:.6f}, punto_cero={cero}" if escala
             else "sin cuantizar")
    forma = "x".join(str(d) for d in det["shape"])
    return f"{det['name']}\n        forma=({forma})  tipo={np.dtype(det['dtype']).name}  [{cuant}]"


def inspeccionar(ruta):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.interpreter import Interpreter

    seccion(f"Modelo: {os.path.basename(ruta)}")
    valido, detalle = validar_tflite(ruta)
    if not valido:
        fallar(f"{ruta}: {detalle}")
    print(f"Tamaño en disco : {tamano_mb(ruta):.2f} MB ({os.path.getsize(ruta):,} bytes)")
    print(f"FlatBuffer      : {detalle}")

    logs = []
    with capturar_logs_nativos(logs):
        # 1) Grafo tal como está en el archivo -> tensores y su cuantización
        sin_delegar = Interpreter(model_path=ruta)
        # 2) Modelo preparado para CPU: allocate_tensors reparte el grafo en XNNPACK
        delegado = Interpreter(model_path=ruta)
        delegado.allocate_tensors()
        # 3) API moderna: CompiledModel
        compilado = CompiledModel.from_file(ruta)

    firmas = compilado.get_signature_list()
    print(f"Firmas          : {firmas}")

    entrada = sin_delegar.get_input_details()[0]
    salida = sin_delegar.get_output_details()[0]
    print(f"Entrada         : {describir_tensor(entrada)}")
    print(f"Salida          : {describir_tensor(salida)}")
    req = compilado.get_input_buffer_requirements(0, 0)
    print(f"Búfer de entrada: {req['buffer_size']:,} bytes (reservado por CompiledModel)")

    ops = [o for o in delegado._get_ops_details() if o["op_name"] != "DELEGATE"]
    segmentos = sum(1 for o in delegado._get_ops_details() if o["op_name"] == "DELEGATE")
    conteo_ops = Counter(o["op_name"] for o in ops)
    print(f"Operadores      : {len(ops)} en total")
    for nombre, n in conteo_ops.most_common():
        print(f"        {nombre:20s} {n:3d}")

    tipos = Counter(np.dtype(t["dtype"]).name for t in sin_delegar.get_tensor_details())
    print("Tensores por tipo de dato: " + ", ".join(f"{k}={v}" for k, v in tipos.most_common()))
    print(f"Segmentos XNNPACK: {segmentos}  (1 = todo el grafo corre en un solo bloque acelerado)")
    print(f"CompiledModel totalmente acelerado: {compilado.is_fully_accelerated()}")
    return {"entrada": entrada, "salida": salida}


def main(rutas):
    resultados = {r: inspeccionar(r) for r in rutas}
    if len(rutas) > 1:
        seccion("Comparación de entradas")
        for ruta, info in resultados.items():
            e = info["entrada"]
            print(f"  {os.path.basename(ruta):40s} entrada {np.dtype(e['dtype']).name:7s} "
                  f"forma {[int(d) for d in e['shape']]}")
    print()
    ok("Inspección completada.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inspecciona modelos .tflite")
    p.add_argument("modelos", nargs="*", default=[MODELO_FP32, MODELO_INT8])
    p.add_argument("--sin-registro", action="store_true",
                   help="no guardar en resultados/paso_02.txt (lo usa el paso 4)")
    args = p.parse_args()
    with Registro(2, "Inspección de los modelos", activo=not args.sin_registro):
        main(args.modelos)
```

Ejecute el paso 2:

```bash
python 02_inspeccionar.py
```

**Salida esperada:**

```text
======================================================================
  PASO 2: Inspección de los modelos
  2026-10-31 09:42:30 | equipo: vm | Python 3.13.18
======================================================================

--- Modelo: mobilenet_v2.tflite -------------------------------------
Tamaño en disco : 14.02 MB (14,020,256 bytes)
FlatBuffer      : cabecera TFL3 correcta
Firmas          : {'serving_default': {'inputs': ['args_0'], 'outputs': ['output_0']}}
Entrada         : serving_default_args_0:0
        forma=(1x3x224x224)  tipo=float32  [sin cuantizar]
Salida          : StatefulPartitionedCall:0
        forma=(1x1000)  tipo=float32  [sin cuantizar]
Búfer de entrada: 602,112 bytes (reservado por CompiledModel)
Operadores      : 70 en total
        CONV_2D               35
        DEPTHWISE_CONV_2D     17
        ADD                   10
        PAD                    5
        TRANSPOSE              1
        SUM                    1
        FULLY_CONNECTED        1
Tensores por tipo de dato: float32=177, int32=3
Segmentos XNNPACK: 1  (1 = todo el grafo corre en un solo bloque acelerado)
CompiledModel totalmente acelerado: True

--- Modelo: mobilenet_v2_int8_channelwise.tflite --------------------
Tamaño en disco : 4.05 MB (4,049,232 bytes)
FlatBuffer      : cabecera TFL3 correcta
Firmas          : {'serving_default': {'inputs': ['args_0'], 'outputs': ['output_0']}}
Entrada         : serving_default_args_0:0
        forma=(1x3x224x224)  tipo=int8  [escala=0.017755, punto_cero=-14]
Salida          : StatefulPartitionedCall:0
        forma=(1x1000)  tipo=int8  [escala=0.103258, punto_cero=-49]
Búfer de entrada: 150,528 bytes (reservado por CompiledModel)
Operadores      : 70 en total
        CONV_2D               35
        DEPTHWISE_CONV_2D     17
        ADD                   10
        PAD                    5
        TRANSPOSE              1
        SUM                    1
        FULLY_CONNECTED        1
Tensores por tipo de dato: int8=124, int32=56
Segmentos XNNPACK: 1  (1 = todo el grafo corre en un solo bloque acelerado)
CompiledModel totalmente acelerado: True

--- Comparación de entradas -----------------------------------------
  mobilenet_v2.tflite                      entrada float32 forma [1, 3, 224, 224]
  mobilenet_v2_int8_channelwise.tflite     entrada int8    forma [1, 3, 224, 224]

[✓] Inspección completada.
```

**Qué observar:**
* **Forma `1x3x224x224`:** es formato **NCHW** (lote, canales, alto, ancho), el de PyTorch. El modelo de la demostración, hecho en Keras, usaba **NHWC** (`1x150x150x3`). Darle una imagen en el orden equivocado es el error más común con modelos descargados.
* **El modelo INT8 recibe y devuelve enteros** (`int8`), con su **escala** y **punto cero**. Por eso la imagen se debe cuantizar antes de entrar y la salida se debe des-cuantizar (lo hace `Clasificador` en `comun.py`).
* **Búfer de entrada:** 602 112 bytes en FP32 frente a 150 528 bytes en INT8: un byte por valor en lugar de cuatro.
* **Mismos 70 operadores** en ambas versiones: la cuantización cambia los **tipos de dato**, no la arquitectura.
* Ambos modelos quedan en **1 segmento XNNPACK** (totalmente acelerados).

---

## Paso 3: Primera Inferencia con `CompiledModel` (`03_inferencia.py`)

### 3.1 Tome su foto

MobileNet-v2 reconoce las 1000 clases de ImageNet: objetos cotidianos, animales, alimentos y vehículos. Tome con su teléfono una foto de **un objeto** que esté en esa lista, por ejemplo:

* `laptop`, `computer keyboard`, `mouse`, `remote control`, `cellular telephone`
* `coffee mug`, `water bottle`, `banana`, `orange`, `running shoe`, `backpack`, `sunglasses`

Recomendaciones: un solo objeto, **centrado y ocupando buena parte de la imagen** (el modelo usa el recorte cuadrado central), buena iluminación y fondo poco cargado. Copie la foto a la carpeta del tutorial con el nombre `mi_foto.jpg` (también se acepta PNG).

> Su foto hace que su resultado sea único. Si el modelo se equivoca, no lo oculte: forma parte del análisis.

### 3.2 El preprocesamiento

El modelo espera exactamente el mismo preprocesamiento con el que se entrenó en PyTorch Vision (función `preprocesar` de `comun.py`, igual a la de la tarjeta del modelo):

1. Redimensionar el lado corto a 232 px y recortar el **cuadrado central de 224×224**.
2. Escalar los píxeles a [0, 1] y **normalizar** con la media `[0.485, 0.456, 0.406]` y la desviación `[0.229, 0.224, 0.225]` de ImageNet (un valor por canal R, G, B).
3. Reordenar de **HWC a CHW** y agregar la dimensión del lote → `(1, 3, 224, 224)`.

### 3.3 El ciclo de `CompiledModel`

```text
CompiledModel.from_file()                ← carga y compila el grafo para el hardware (una sola vez)
create_input_buffers / create_output_buffers   ← el runtime reserva la memoria de E/S
entradas[0].write(x)                     ← copia el tensor preprocesado al búfer
run_by_index(0, entradas, salidas)       ← ejecuta la firma 0 (inferencia)
salidas[0].read(n, np.float32)           ← lee el resultado del búfer de salida
```

Compárelo con la demostración, que usaba `Interpreter`: `allocate_tensors → set_tensor → invoke → get_tensor`. Con `CompiledModel` la selección del acelerador y la gestión de los búferes las hace el runtime.

Cree el archivo `03_inferencia.py`:

```python
#!/usr/bin/env python3
"""
PASO 3 - Primera inferencia con la API CompiledModel de LiteRT.

Se clasifica una foto tomada por usted con el modelo FP32, escribiendo a mano
cada etapa del ciclo de CompiledModel:

  from_file -> create_input_buffers / create_output_buffers
            -> write -> run_by_index -> read -> softmax -> Top-5

Uso:
  python 03_inferencia.py --imagen mi_foto.jpg
"""
import argparse
import os
import re
import time

import numpy as np

from comun import (MODELO_FP32, Registro, cargar_etiquetas, capturar_logs_nativos,
                   exigir_modelo, fallar, ok, preprocesar, seccion, softmax)


def resumir_logs(texto):
    """Conserva solo las líneas de los registros nativos que explican el hardware."""
    claves = ("RegisterAccelerator", "registered", "could not be loaded",
              "XNNPACK", "Flatbuffer model")
    vistas = set()
    for linea in texto.splitlines():
        if any(c in linea for c in claves):
            linea = linea.split("] ", 1)[-1] if "] " in linea else linea
            linea = re.sub(r"^INFO: |ptr=0x[0-9a-fA-F]+, ", "", linea)
            linea = linea.split("(/")[0].strip()      # recorta rutas largas
            if linea not in vistas:
                vistas.add(linea)
                print(f"    {linea}")


def main(ruta_imagen):
    exigir_modelo(MODELO_FP32, 1)
    if not os.path.isfile(ruta_imagen):
        fallar(f"No existe la imagen '{ruta_imagen}'. Tome una foto (ver tutorial) "
               "y pase su ruta con --imagen.")
    etiquetas = cargar_etiquetas()

    from ai_edge_litert.compiled_model import CompiledModel

    # ---------------------------------------------------------------- 3.1
    seccion("3.1 Crear el CompiledModel (compilación para el hardware)")
    logs = []
    t0 = time.perf_counter()
    with capturar_logs_nativos(logs):
        modelo = CompiledModel.from_file(MODELO_FP32)   # por defecto: CPU
    ms_compilar = (time.perf_counter() - t0) * 1000
    print(f"Modelo: {MODELO_FP32}")
    print(f"Tiempo de carga y compilación: {ms_compilar:.1f} ms (se hace UNA sola vez)")
    print("Mensajes del runtime (aceleradores que LiteRT intentó registrar):")
    resumir_logs(logs[0])
    print(f"¿Todo el grafo quedó en el acelerador?: {modelo.is_fully_accelerated()}")

    # ---------------------------------------------------------------- 3.2
    seccion("3.2 Firmas y búferes de entrada/salida")
    firmas = modelo.get_signature_list()
    nombre_firma = list(firmas)[0]
    print(f"Firmas disponibles: {firmas}")
    for nombre, det in modelo.get_input_tensor_details(nombre_firma).items():
        print(f"Entrada '{nombre}': tensor {det['name']}, tipo {det['dtype']}, forma {det['shape']}")
    entradas = modelo.create_input_buffers(0)     # 0 = primera firma
    salidas = modelo.create_output_buffers(0)
    req_in = modelo.get_input_buffer_requirements(0, 0)
    req_out = modelo.get_output_buffer_requirements(0, 0)
    n_salidas = req_out["buffer_size"] // np.dtype(np.float32).itemsize
    print(f"Búfer de entrada : {req_in['buffer_size']:,} bytes "
          f"(= 1x3x224x224 valores float32 de 4 bytes)")
    print(f"Búfer de salida  : {req_out['buffer_size']:,} bytes (= {n_salidas} valores float32)")

    # ---------------------------------------------------------------- 3.3
    seccion("3.3 Preprocesamiento de la imagen")
    from PIL import Image
    with Image.open(ruta_imagen) as img:
        print(f"Imagen original: {ruta_imagen}  ({img.size[0]}x{img.size[1]} px, {img.mode})")
    x = preprocesar(ruta_imagen)
    print(f"Tensor de entrada: forma={x.shape}  tipo={x.dtype}")
    print(f"Valores normalizados: mín={x.min():.2f}  máx={x.max():.2f}  media={x.mean():.3f}")
    print("Formato NCHW: (lote=1, canales=3, alto=224, ancho=224), como en PyTorch.")

    # ---------------------------------------------------------------- 3.4
    seccion("3.4 Inferencia: write -> run_by_index -> read")
    tiempos = []
    for _ in range(3):
        t0 = time.perf_counter()
        entradas[0].write(x)
        modelo.run_by_index(0, entradas, salidas)
        logits = salidas[0].read(n_salidas, np.float32)
        tiempos.append((time.perf_counter() - t0) * 1000)
    print(f"Tiempo de inferencia: 1.ª = {tiempos[0]:.1f} ms, 2.ª = {tiempos[1]:.1f} ms, "
          f"3.ª = {tiempos[2]:.1f} ms")
    print(f"Salida cruda (logits): {n_salidas} valores, rango [{logits.min():.2f}, "
          f"{logits.max():.2f}], suma {logits.sum():.2f}")
    print("Como la suma no es 1, la salida son logits: se aplica softmax.")

    # ---------------------------------------------------------------- 3.5
    seccion("3.5 Resultado: Top-5")
    probs = softmax(logits)
    for pos, clase in enumerate(np.argsort(probs)[::-1][:5], start=1):
        barra = "#" * int(round(probs[clase] * 30))
        print(f"  {pos}. [{clase:3d}] {etiquetas[clase][:38]:38s} {probs[clase] * 100:6.2f} % {barra}")
    print()
    ok("Paso 3 completado: inferencia con CompiledModel.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagen", required=True, help="ruta de su foto (JPG o PNG)")
    args = p.parse_args()
    with Registro(3, "Inferencia con CompiledModel sobre una foto propia"):
        main(args.imagen)
```

Ejecute el paso 3 indicando su foto:

```bash
python 03_inferencia.py --imagen mi_foto.jpg
```

**Salida esperada** (con una foto de una laptop):

```text
======================================================================
  PASO 3: Inferencia con CompiledModel sobre una foto propia
  2026-10-31 09:42:30 | equipo: vm | Python 3.13.18
======================================================================

--- 3.1 Crear el CompiledModel (compilación para el hardware) -------
Modelo: modelos/mobilenet_v2.tflite
Tiempo de carga y compilación: 10.0 ms (se hace UNA sola vez)
Mensajes del runtime (aceleradores que LiteRT intentó registrar):
    RegisterAccelerator: name=NpuAccelerator
    NPU accelerator registered.
    RegisterAccelerator: name=GPU WebGPU
    Dynamically loaded GPU accelerator(libLiteRtWebGpuAccelerator.so) registered.
    RegisterAccelerator: name=CpuAccelerator
    XNNPACK CPU accelerator registered.
    NPU accelerator could not be loaded and registered: kLiteRtStatusErrorNotFound.
    Flatbuffer model initialized directly from incoming litert model.
    Created TensorFlow Lite XNNPACK delegate for CPU.
¿Todo el grafo quedó en el acelerador?: True

--- 3.2 Firmas y búferes de entrada/salida --------------------------
Firmas disponibles: {'serving_default': {'inputs': ['args_0'], 'outputs': ['output_0']}}
Entrada 'args_0': tensor serving_default_args_0:0, tipo float32, forma [1, 3, 224, 224]
Búfer de entrada : 602,112 bytes (= 1x3x224x224 valores float32 de 4 bytes)
Búfer de salida  : 4,000 bytes (= 1000 valores float32)

--- 3.3 Preprocesamiento de la imagen -------------------------------
Imagen original: mi_foto.jpg  (500x354 px, RGB)
Tensor de entrada: forma=(1, 3, 224, 224)  tipo=float32
Valores normalizados: mín=-2.12  máx=2.64  media=-0.414
Formato NCHW: (lote=1, canales=3, alto=224, ancho=224), como en PyTorch.

--- 3.4 Inferencia: write -> run_by_index -> read -------------------
Tiempo de inferencia: 1.ª = 13.0 ms, 2.ª = 9.0 ms, 3.ª = 8.7 ms
Salida cruda (logits): 1000 valores, rango [-6.99, 9.56], suma 0.18
Como la suma no es 1, la salida son logits: se aplica softmax.

--- 3.5 Resultado: Top-5 --------------------------------------------
  1. [620] laptop, laptop computer                 25.21 % ########
  2. [424] barbershop                              24.21 % #######
  3. [423] barber chair                             5.56 % ##
  4. [681] notebook, notebook computer              4.93 % #
  5. [797] sleeping bag                             3.63 % #

[✓] Paso 3 completado: inferencia con CompiledModel.
```

**Qué observar:**
* **Mensajes del runtime:** LiteRT intenta registrar NPU, GPU y CPU. En una computadora común no hay NPU compatible (`could not be loaded`) y la inferencia queda en la CPU con **XNNPACK**. Es la selección de backend de la diapositiva 6.
* **La compilación ocurre una sola vez**; luego cada inferencia reutiliza los búferes.
* **La primera inferencia suele ser más lenta** que las siguientes (calentamiento de cachés y memoria). Por eso el paso 5 descarta las primeras ejecuciones.
* **La salida son logits**, no probabilidades: hay que aplicar *softmax*. Esto depende de cómo se exportó cada modelo (el de la demostración ya traía softmax).

---

## Paso 4: Cuantización Dinámica con LiteRT Quantizer (`04_cuantizar.sh`)

**LiteRT Quantizer** (paquete `litert-quantizer`, antes llamado *AI Edge Quantizer*) es la herramienta oficial de cuantización post-entrenamiento (PTQ) de LiteRT. Recibe un modelo `.tflite` FP32 y una **receta**, y produce el modelo cuantizado. El nombre de cada receta describe qué se cuantiza:

| Receta | Pesos | Activaciones | ¿Calibración? |
|---|---|---|---|
| `weight_only_wi8_afp32` | INT8 (se des-cuantizan al cargar) | FP32 | No |
| **`dynamic_wi8_afp32`** (este paso) | INT8 | FP32 (se cuantizan "al vuelo" en algunas operaciones) | No |
| `static_wi8_ai8` (la INT8 del Hub) | INT8 | INT8 | **Sí**, con datos representativos |

La cuantización dinámica es la más sencilla: basta un comando, sin datos de calibración. El script ejecuta la herramienta, compara tamaños y vuelve a usar `02_inspeccionar.py` para examinar el modelo resultante.

Cree el archivo `04_cuantizar.sh`:

```bash
#!/usr/bin/env bash
# =============================================================================
# PASO 4 - Cuantización dinámica propia con LiteRT Quantizer (línea de comandos)
#
# Receta dynamic_wi8_afp32:
#   wi8   -> pesos (weights) en INT8
#   afp32 -> activaciones en FP32 (se calculan en flotante durante la inferencia)
# No requiere datos de calibración: solo el modelo FP32.
# =============================================================================

ENTRADA="modelos/mobilenet_v2.tflite"
SALIDA="modelos/mobilenet_v2_dynamic_wi8_afp32.tflite"
RECETA="dynamic_wi8_afp32"
REGISTRO="resultados/paso_04.txt"

# ---- Verificaciones previas ------------------------------------------------
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[✗] El entorno virtual no está activo. Ejecute: source .venv/bin/activate"
    exit 1
fi
if ! command -v litert-quantizer >/dev/null 2>&1; then
    echo "[✗] No se encontró el comando litert-quantizer. Vuelva a ejecutar ./setup.sh"
    exit 1
fi
if [ ! -f "$ENTRADA" ]; then
    echo "[✗] No existe $ENTRADA. Ejecute primero el paso 1 (python 01_descargar.py)."
    exit 1
fi
# Un .tflite válido tiene el identificador "TFL3" en los bytes 4-7
if [ "$(head -c 8 "$ENTRADA" | tail -c 4)" != "TFL3" ]; then
    echo "[✗] $ENTRADA no es un modelo .tflite válido. Bórrelo y repita el paso 1."
    exit 1
fi
mkdir -p resultados

{
    echo "======================================================================"
    echo "  PASO 4: Cuantización dinámica con litert-quantizer"
    echo "  $(date '+%Y-%m-%d %H:%M:%S') | equipo: $(hostname)"
    echo "======================================================================"
    echo
    echo "--- 4.1 Comando ejecutado -------------------------------------------"
    echo "litert-quantizer --model_file=$ENTRADA \\"
    echo "                 --recipe=$RECETA --output_dir=modelos --overwrite_outputs"
    echo
    echo "--- 4.2 Salida de la herramienta ------------------------------------"
    rm -f "$SALIDA"
    INICIO=$(date +%s%N)
    litert-quantizer --model_file="$ENTRADA" --recipe="$RECETA" \
                     --output_dir=modelos --overwrite_outputs 2>&1 \
        | grep -vE '^(INFO|WARNING): \[|^└|Applying Transformations'
    FIN=$(date +%s%N)
    if [ ! -f "$SALIDA" ]; then
        echo "[✗] La herramienta no generó $SALIDA. Revise los mensajes anteriores."
        exit 1
    fi
    echo "Tiempo total del comando (incluye arranque de Python): $(( (FIN - INICIO) / 1000000 )) ms"
    echo
    echo "--- 4.3 Comparación de tamaños --------------------------------------"
    python - "$ENTRADA" "$SALIDA" <<'EOF'
import os, sys
a, b = (os.path.getsize(p) for p in sys.argv[1:3])
print(f"  Original FP32        : {a/1e6:6.2f} MB  ({a:,} bytes)")
print(f"  Dinámica wi8/afp32   : {b/1e6:6.2f} MB  ({b:,} bytes)")
print(f"  Reducción            : {100*(1-b/a):.1f} %  ({a/b:.2f}x más pequeño)")
EOF
    echo
    echo "--- 4.4 Inspección del modelo cuantizado ----------------------------"
    python 02_inspeccionar.py --sin-registro "$SALIDA"
    echo
    echo "[✓] Paso 4 completado: modelo cuantizado creado."
} 2>&1 | tee "$REGISTRO"

echo
echo "[i] Salida guardada en $REGISTRO"
```

Dele permisos de ejecución y ejecútelo:

```bash
chmod +x 04_cuantizar.sh
./04_cuantizar.sh
```

**Salida esperada:**

```text
======================================================================
  PASO 4: Cuantización dinámica con litert-quantizer
  2026-10-31 09:42:30 | equipo: vm
======================================================================

--- 4.1 Comando ejecutado -------------------------------------------
litert-quantizer --model_file=modelos/mobilenet_v2.tflite \
                 --recipe=dynamic_wi8_afp32 --output_dir=modelos --overwrite_outputs

--- 4.2 Salida de la herramienta ------------------------------------
Model name: modelos/mobilenet_v2.tflite
Original model size: 13.37 MiB
Quantized model size: 3.65 MiB
Quantization Ratio: 0.27 (3.7x smaller)
Total time: 53.31 ms
Wrote quantized model to modelos/mobilenet_v2_dynamic_wi8_afp32.tflite.
Tiempo total del comando (incluye arranque de Python): 420 ms

--- 4.3 Comparación de tamaños --------------------------------------
  Original FP32        :  14.02 MB  (14,020,256 bytes)
  Dinámica wi8/afp32   :   3.83 MB  (3,828,944 bytes)
  Reducción            : 72.7 %  (3.66x más pequeño)

--- 4.4 Inspección del modelo cuantizado ----------------------------

--- Modelo: mobilenet_v2_dynamic_wi8_afp32.tflite -------------------
Tamaño en disco : 3.83 MB (3,828,944 bytes)
FlatBuffer      : cabecera TFL3 correcta
Firmas          : {'serving_default': {'inputs': ['args_0'], 'outputs': ['output_0']}}
Entrada         : serving_default_args_0:0
        forma=(1x3x224x224)  tipo=float32  [sin cuantizar]
Salida          : StatefulPartitionedCall:0
        forma=(1x1000)  tipo=float32  [sin cuantizar]
Búfer de entrada: 602,112 bytes (reservado por CompiledModel)
Operadores      : 70 en total
        CONV_2D               35
        DEPTHWISE_CONV_2D     17
        ADD                   10
        PAD                    5
        TRANSPOSE              1
        SUM                    1
        FULLY_CONNECTED        1
Tensores por tipo de dato: float32=124, int8=53, int32=3
Segmentos XNNPACK: 18  (1 = todo el grafo corre en un solo bloque acelerado)
CompiledModel totalmente acelerado: False

[✓] Inspección completada.

[✓] Paso 4 completado: modelo cuantizado creado.
```

**Qué observar:**
* El archivo se reduce **~73 %** (de 14.0 a 3.8 MB): los pesos pasan de 4 bytes a 1 byte. La herramienta reporta los tamaños en **MiB** (1 MiB = 1 048 576 bytes) y el script en **MB** (1 MB = 1 000 000 bytes); por eso 13.37 MiB = 14.02 MB.
* La entrada y la salida **siguen siendo `float32`**: solo cambiaron los pesos (aparecen tensores `int8`).
* **¡El grafo quedó partido en 18 segmentos XNNPACK y no está totalmente acelerado!** Guarde esta observación: la explicará el paso 5.

---

## Paso 5: Latencia y Memoria (`05_benchmark.py`)

Ahora se mide, **en su equipo**, cuánto tarda y cuánta memoria usa cada variante. El script sigue las buenas prácticas de *benchmarking*:

* **Calentamiento:** las primeras 10 inferencias se descartan.
* **Muchas repeticiones** (100) y se reporta la **mediana** (menos sensible a interrupciones del sistema que el promedio), el **mínimo** y el **percentil 90** (el "peor caso habitual", importante en tiempo real).
* Se prueba con **1 hilo** y con varios hilos (hasta 4), configurando `CpuOptions(num_threads=...)` de `CompiledModel`.
* La **RAM** se mide en un proceso nuevo para cada modelo, leyendo el pico de memoria residente (`VmHWM` en `/proc/self/status`) antes y después de cargarlo y ejecutarlo.

Cree el archivo `05_benchmark.py`:

```python
#!/usr/bin/env python3
"""
PASO 5 - Medir latencia y memoria de las tres variantes en SU equipo.

Para cada variante (FP32, dinámica propia, INT8 del Hub) y cada número de hilos:
  * Tiempo de carga y compilación del CompiledModel.
  * Latencia de inferencia: se descartan las primeras ejecuciones (calentamiento)
    y se reportan mínimo, mediana y percentil 90 de N repeticiones.
  * Si todo el grafo quedó en el acelerador de CPU (XNNPACK).
  * RAM adicional en ejecución, medida en un proceso aparte para cada modelo.

Uso:
  python 05_benchmark.py                     # 100 repeticiones
  python 05_benchmark.py --repeticiones 300  # medición más estable
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from comun import (DIR_LOTE, DIR_RESULTADOS, VARIANTES,
                   Clasificador, Registro, exigir_modelo, fallar, ok,
                   preprocesar, seccion, tamano_mb)


def entrada_de_prueba():
    """Usa la primera imagen del lote (el contenido no cambia la latencia)."""
    ruta = os.path.join(DIR_LOTE, "n01440764_tench.JPEG")
    if not os.path.isfile(ruta):
        fallar(f"No existe '{ruta}'. Ejecute primero el paso 1.")
    return preprocesar(ruta)


def medir_ram(ruta):
    """Ejecuta este mismo script en un proceso nuevo para medir la RAM de un solo modelo."""
    salida = subprocess.run([sys.executable, __file__, "--medir-ram", ruta],
                            capture_output=True, text=True)
    for linea in salida.stdout.splitlines():
        if linea.startswith("{"):
            return json.loads(linea)
    fallar(f"No se pudo medir la RAM de {ruta}:\n{salida.stderr[-500:]}")


def pico_ram_mb():
    """Pico de memoria residente del proceso (VmHWM en /proc/self/status)."""
    with open("/proc/self/status", encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("VmHWM:"):
                return int(linea.split()[1]) / 1024          # kB -> MB
    return 0.0


def proceso_medir_ram(ruta):
    """(Proceso hijo) Pico de RAM antes y después de cargar el modelo y ejecutarlo."""
    x = entrada_de_prueba()
    base = pico_ram_mb()
    clf = Clasificador(ruta, hilos=1)
    for _ in range(5):
        clf.logits(x)
    pico = pico_ram_mb()
    print(json.dumps({"base_mb": base, "pico_mb": pico, "modelo_mb": pico - base}))


def medir_latencia(ruta, hilos, x, calentamiento, repeticiones):
    clf = Clasificador(ruta, hilos=hilos)
    for _ in range(calentamiento):
        clf.logits(x)
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        clf.logits(x)
        tiempos.append((time.perf_counter() - t0) * 1000)
    t = np.array(tiempos)
    return {"hilos": hilos, "compilacion_ms": clf.ms_compilacion,
            "min_ms": float(t.min()), "mediana_ms": float(np.median(t)),
            "p90_ms": float(np.percentile(t, 90)),
            "acelerado": bool(clf.totalmente_acelerado)}


def main(args):
    for _, _, ruta in VARIANTES:
        exigir_modelo(ruta, 4 if "dynamic" in ruta else 1)
    x = entrada_de_prueba()
    n_cpu = os.cpu_count() or 1
    lista_hilos = sorted({1, min(4, n_cpu)})

    seccion("Configuración")
    print(f"Núcleos lógicos: {n_cpu} | hilos a probar: {lista_hilos}")
    print(f"Calentamiento: {args.calentamiento} ejecuciones | medición: {args.repeticiones} ejecuciones")
    print("Consejo: cierre otros programas y conecte el cargador si es una laptop.")

    resultados = {}
    for corto, descripcion, ruta in VARIANTES:
        seccion(f"{corto}: {descripcion}")
        print(f"Archivo: {ruta} ({tamano_mb(ruta):.2f} MB)")
        filas = []
        for hilos in lista_hilos:
            r = medir_latencia(ruta, hilos, x, args.calentamiento, args.repeticiones)
            filas.append(r)
            print(f"  hilos={hilos}: compilación {r['compilacion_ms']:6.1f} ms | "
                  f"latencia mín {r['min_ms']:6.2f}  mediana {r['mediana_ms']:6.2f}  "
                  f"p90 {r['p90_ms']:6.2f} ms")
        ram = medir_ram(ruta)
        print(f"  Totalmente acelerado en XNNPACK: {'sí' if filas[0]['acelerado'] else 'NO'}")
        print(f"  RAM: proceso base {ram['base_mb']:.0f} MB -> pico {ram['pico_mb']:.0f} MB "
              f"(el modelo agrega {ram['modelo_mb']:.1f} MB)")
        resultados[corto] = {"descripcion": descripcion, "archivo": ruta,
                             "tamano_mb": tamano_mb(ruta), "latencias": filas,
                             "ram_modelo_mb": ram["modelo_mb"]}

    seccion("Tabla de latencia (mediana, ms)")
    print(f"  {'Variante':8s} " + " ".join(f"{f'{h} hilo(s)':>12s}" for h in lista_hilos)
          + f" {'vs FP32 (1 hilo)':>18s}")
    ref = resultados["FP32"]["latencias"][0]["mediana_ms"]
    for corto, r in resultados.items():
        meds = [f["mediana_ms"] for f in r["latencias"]]
        rel = ref / meds[0]
        if corto == "FP32":
            texto_rel = "referencia"
        elif rel >= 1:
            texto_rel = f"{rel:.2f}x más rápido"
        else:
            texto_rel = f"{1 / rel:.2f}x más lento"
        print(f"  {corto:8s} " + " ".join(f"{m:12.2f}" for m in meds) + f" {texto_rel:>18s}")

    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    with open(os.path.join(DIR_RESULTADOS, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump({"n_cpu": n_cpu, "repeticiones": args.repeticiones,
                   "variantes": resultados}, f, indent=2)
    print()
    ok("Paso 5 completado: mediciones guardadas en resultados/benchmark.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeticiones", type=int, default=100)
    p.add_argument("--calentamiento", type=int, default=10)
    p.add_argument("--medir-ram", metavar="MODELO", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.medir_ram:
        proceso_medir_ram(args.medir_ram)
    else:
        with Registro(5, "Latencia y memoria de las tres variantes"):
            main(args)
```

Ejecute el paso 5 (tarda menos de un minuto; no use la computadora mientras corre):

```bash
python 05_benchmark.py
```

**Salida esperada** (en un equipo de prueba con 2 núcleos; sus tiempos serán distintos):

```text
======================================================================
  PASO 5: Latencia y memoria de las tres variantes
  2026-10-31 09:42:31 | equipo: vm | Python 3.13.18
======================================================================

--- Configuración ---------------------------------------------------
Núcleos lógicos: 2 | hilos a probar: [1, 2]
Calentamiento: 10 ejecuciones | medición: 100 ejecuciones
Consejo: cierre otros programas y conecte el cargador si es una laptop.

--- FP32: precisión completa (Hub) ----------------------------------
Archivo: modelos/mobilenet_v2.tflite (14.02 MB)
  hilos=1: compilación   10.7 ms | latencia mín   6.59  mediana   9.93  p90  11.71 ms
  hilos=2: compilación    3.3 ms | latencia mín   3.93  mediana   4.53  p90   4.72 ms
  Totalmente acelerado en XNNPACK: sí
  RAM: proceso base 44 MB -> pico 105 MB (el modelo agrega 61.0 MB)

--- DIN-W8: dinámica wi8/afp32 (propia, paso 4) ---------------------
Archivo: modelos/mobilenet_v2_dynamic_wi8_afp32.tflite (3.83 MB)
  hilos=1: compilación    3.9 ms | latencia mín  13.59  mediana  15.20  p90  21.45 ms
  hilos=2: compilación    3.0 ms | latencia mín  13.10  mediana  14.55  p90  17.10 ms
  Totalmente acelerado en XNNPACK: NO
  RAM: proceso base 44 MB -> pico 93 MB (el modelo agrega 48.9 MB)

--- INT8: estática wi8/ai8 (Hub) ------------------------------------
Archivo: modelos/mobilenet_v2_int8_channelwise.tflite (4.05 MB)
  hilos=1: compilación    3.7 ms | latencia mín   4.53  mediana   4.98  p90   5.93 ms
  hilos=2: compilación    3.6 ms | latencia mín   3.08  mediana   3.74  p90   4.01 ms
  Totalmente acelerado en XNNPACK: sí
  RAM: proceso base 44 MB -> pico 82 MB (el modelo agrega 38.4 MB)

--- Tabla de latencia (mediana, ms) ---------------------------------
  Variante    1 hilo(s)    2 hilo(s)   vs FP32 (1 hilo)
  FP32             9.93         4.53         referencia
  DIN-W8          15.20        14.55    1.53x más lento
  INT8             4.98         3.74   1.99x más rápido

[✓] Paso 5 completado: mediciones guardadas en resultados/benchmark.json
```

**Qué observar:**
* **INT8 estática es la más rápida:** la aritmética entera y el menor tráfico de memoria aceleran la inferencia, y todo el grafo corre en XNNPACK.
* **La dinámica es más pequeña pero MÁS LENTA que FP32.** No es un error de medición. Con activaciones en FP32, parte del trabajo de cuantizar y des-cuantizar se repite en cada inferencia, y el grafo partido (18 segmentos, paso 4) obliga a ejecutar porciones con kernels genéricos y a pasar datos entre segmentos. La propia documentación de LiteRT Quantizer advierte que esta receta puede no mejorar la latencia. Este resultado es la razón de la frase de la diapositiva 11: **mida siempre en su hardware**.
* **RAM:** el archivo INT8 es ~3.5 veces más pequeño, pero la RAM que agrega el modelo baja bastante menos. Es la aclaración de la diapositiva 7: en ejecución también ocupan memoria las activaciones, los búferes y el propio runtime.
* **Hilos:** más hilos reducen la latencia de los modelos totalmente acelerados; en la variante dinámica ayudan poco.

---

## Paso 6: Precisión de Cada Variante (`06_precision.py`)

Reducir el tamaño no sirve de nada si el modelo deja de acertar. Este paso clasifica el lote de 100 imágenes con las tres variantes y calcula:

* **Top-1:** porcentaje de imágenes en las que la clase más probable es la correcta.
* **Top-5:** porcentaje en las que la clase correcta está entre las 5 más probables.
* **Concordancia con FP32:** porcentaje de imágenes en las que la variante predice **lo mismo que el modelo original**. Es la métrica clave al validar una cuantización, porque **no requiere etiquetas**: en un proyecto real se puede medir con cualquier conjunto de imágenes del entorno donde se desplegará el sistema.

Cree el archivo `06_precision.py`:

```python
#!/usr/bin/env python3
"""
PASO 6 - Medir cuánta precisión se pierde al cuantizar.

Clasifica el lote de 100 imágenes etiquetadas con las tres variantes y reporta:
  * Precisión Top-1 y Top-5 frente a la clase verdadera.
  * Concordancia con FP32: en cuántas imágenes la variante predice lo mismo
    que el modelo original (no necesita etiquetas; es la métrica clave al
    validar una cuantización).
  * Las imágenes en las que cada variante cambió su predicción.

Uso:
  python 06_precision.py
"""
import csv
import json
import os

import numpy as np

from comun import (ARCHIVO_VERDAD, DIR_LOTE, DIR_RESULTADOS, VARIANTES,
                   Clasificador, Registro, cargar_etiquetas, exigir_modelo,
                   fallar, ok, preprocesar, seccion)


def cargar_lote():
    if not os.path.isfile(ARCHIVO_VERDAD):
        fallar(f"No existe '{ARCHIVO_VERDAD}'. Ejecute primero el paso 1.")
    with open(ARCHIVO_VERDAD, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    faltan = [r["archivo"] for r in filas
              if not os.path.isfile(os.path.join(DIR_LOTE, r["archivo"]))]
    if faltan:
        fallar(f"Faltan {len(faltan)} imágenes del lote. Ejecute de nuevo el paso 1.")
    return [(r["archivo"], int(r["clase"])) for r in filas]


def main():
    for _, _, ruta in VARIANTES:
        exigir_modelo(ruta, 4 if "dynamic" in ruta else 1)
    etiquetas = cargar_etiquetas()
    lote = cargar_lote()

    seccion("Preprocesamiento del lote")
    tensores = [preprocesar(os.path.join(DIR_LOTE, a)) for a, _ in lote]
    verdad = np.array([c for _, c in lote])
    print(f"{len(tensores)} imágenes listas (1x3x224x224 float32 cada una).")
    print("Nota: son imágenes de muestra de ImageNet y algunas pueden haberse usado en el")
    print("entrenamiento; la precisión absoluta puede ser optimista. Lo importante aquí es")
    print("la COMPARACIÓN entre variantes, que usan exactamente las mismas imágenes.")

    hilos = min(4, os.cpu_count() or 1)
    resultados, top1_fp32, prob_fp32 = {}, None, None
    for corto, descripcion, ruta in VARIANTES:
        seccion(f"{corto}: {descripcion}")
        clf = Clasificador(ruta, hilos=hilos)
        probs = np.stack([clf.probabilidades(x) for x in tensores])
        top5 = np.argsort(probs, axis=1)[:, ::-1][:, :5]
        top1 = top5[:, 0]
        acierto1 = float(np.mean(top1 == verdad))
        acierto5 = float(np.mean([v in fila for v, fila in zip(verdad, top5)]))
        if top1_fp32 is None:
            top1_fp32, prob_fp32 = top1, probs
        concordancia = float(np.mean(top1 == top1_fp32))
        dif_conf = float(np.mean(np.abs(probs[np.arange(len(top1)), top1_fp32]
                                        - prob_fp32[np.arange(len(top1)), top1_fp32])))
        print(f"  Top-1: {acierto1 * 100:5.1f} %   Top-5: {acierto5 * 100:5.1f} %   "
              f"Concordancia con FP32: {concordancia * 100:5.1f} %")
        print(f"  Diferencia media de confianza en la clase elegida por FP32: "
              f"{dif_conf * 100:.2f} puntos porcentuales")
        cambios = np.flatnonzero(top1 != top1_fp32)
        if corto != "FP32":
            if len(cambios) == 0:
                print("  No cambió ninguna predicción respecto a FP32.")
            else:
                print(f"  Predicciones distintas a FP32 ({len(cambios)}), primeras 5:")
                for i in cambios[:5]:
                    print(f"    {lote[i][0][:30]:30s} FP32: {etiquetas[top1_fp32[i]][:18]:18s} "
                          f"-> {etiquetas[top1[i]][:18]}")
        resultados[corto] = {"descripcion": descripcion, "top1": acierto1,
                             "top5": acierto5, "concordancia_fp32": concordancia,
                             "cambios": int(len(cambios))}

    seccion("Tabla de precisión")
    print(f"  {'Variante':8s} {'Top-1':>8s} {'Top-5':>8s} {'Concordancia FP32':>19s}")
    for corto, r in resultados.items():
        print(f"  {corto:8s} {r['top1'] * 100:7.1f}% {r['top5'] * 100:7.1f}% "
              f"{r['concordancia_fp32'] * 100:18.1f}%")

    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    with open(os.path.join(DIR_RESULTADOS, "precision.json"), "w", encoding="utf-8") as f:
        json.dump({"imagenes": len(lote), "variantes": resultados}, f, indent=2)
    print()
    ok("Paso 6 completado: resultados guardados en resultados/precision.json")


if __name__ == "__main__":
    with Registro(6, "Precisión de las tres variantes"):
        main()
```

Ejecute el paso 6:

```bash
python 06_precision.py
```

**Salida esperada** (ejemplo):

```text
======================================================================
  PASO 6: Precisión de las tres variantes
  2026-10-31 09:42:38 | equipo: vm | Python 3.13.18
======================================================================

--- Preprocesamiento del lote ---------------------------------------
100 imágenes listas (1x3x224x224 float32 cada una).
Nota: son imágenes de muestra de ImageNet y algunas pueden haberse usado en el
entrenamiento; la precisión absoluta puede ser optimista. Lo importante aquí es
la COMPARACIÓN entre variantes, que usan exactamente las mismas imágenes.

--- FP32: precisión completa (Hub) ----------------------------------
  Top-1:  81.0 %   Top-5:  97.0 %   Concordancia con FP32: 100.0 %
  Diferencia media de confianza en la clase elegida por FP32: 0.00 puntos porcentuales

--- DIN-W8: dinámica wi8/afp32 (propia, paso 4) ---------------------
  Top-1:  79.0 %   Top-5:  97.0 %   Concordancia con FP32:  97.0 %
  Diferencia media de confianza en la clase elegida por FP32: 2.55 puntos porcentuales
  Predicciones distintas a FP32 (3), primeras 5:
    ... (archivo de la imagen, predicción de FP32 -> predicción de la variante)

--- INT8: estática wi8/ai8 (Hub) ------------------------------------
  Top-1:  74.0 %   Top-5:  97.0 %   Concordancia con FP32:  87.0 %
  Diferencia media de confianza en la clase elegida por FP32: 12.45 puntos porcentuales
  Predicciones distintas a FP32 (13), primeras 5:
    ... (archivo de la imagen, predicción de FP32 -> predicción de la variante)

--- Tabla de precisión ----------------------------------------------
  Variante    Top-1    Top-5   Concordancia FP32
  FP32        81.0%    97.0%              100.0%
  DIN-W8      79.0%    97.0%               97.0%
  INT8        74.0%    97.0%               87.0%

[✓] Paso 6 completado: resultados guardados en resultados/precision.json
```

**Qué observar:**
* La **dinámica** casi no cambia las predicciones: los cálculos se hacen en FP32 y solo los pesos perdieron resolución.
* La **INT8 estática** se aleja más de FP32, porque también las activaciones se redondean a 256 niveles. Aun así, el Top-5 suele mantenerse: cuando la predicción cambia, normalmente es hacia una clase parecida (por ejemplo, entre dos especies de rana, o entre caballito del diablo y libélula).
* Si la pérdida de precisión fuera inaceptable para una aplicación, el siguiente paso sería **QAT** o **precisión mixta** (diapositiva 8).

---

## Paso 7: Resumen Comparativo (`07_resumen.py`)

El último paso reúne los resultados de los pasos 5 y 6 en una sola tabla —la versión **medida en su equipo** de la tabla conceptual de la diapositiva 11— y verifica que existan todos los registros para el PDF.

Cree el archivo `07_resumen.py`:

```python
#!/usr/bin/env python3
"""
PASO 7 - Resumen final: la tabla comparativa medida en SU equipo.

Reúne los resultados de los pasos 4, 5 y 6 en una sola tabla (la versión
medida de la diapositiva "Evaluación: tamaño, latencia y precisión") y
verifica que existan todos los registros que van en el PDF de entrega.

Uso:
  python 07_resumen.py
"""
import json
import os

from comun import DIR_RESULTADOS, Registro, fallar, ok, seccion


def leer_json(nombre, paso):
    ruta = os.path.join(DIR_RESULTADOS, nombre)
    if not os.path.isfile(ruta):
        fallar(f"No existe '{ruta}'. Ejecute primero el paso {paso}.")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def modelo_cpu():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linea in f:
                if linea.lower().startswith(("model name", "hardware")):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return "desconocido"


def main():
    bench = leer_json("benchmark.json", 5)
    prec = leer_json("precision.json", 6)
    vb, vp = bench["variantes"], prec["variantes"]
    hilos = [f["hilos"] for f in vb["FP32"]["latencias"]]
    h_max = hilos[-1]

    seccion("Equipo")
    print(f"CPU: {modelo_cpu()} | núcleos lógicos: {bench['n_cpu']}")

    seccion("Tabla comparativa final")
    col_hmax = f"Lat {h_max}h"
    enc = (f"  {'Variante':8s} {'Archivo':>8s} {'Reduc.':>7s} {'Lat 1h':>8s} {col_hmax:>8s} "
           f"{'RAM+':>7s} {'Top-1':>6s} {'Concord.':>8s} {'XNNPACK':>8s}")
    print(enc)
    print("  " + "-" * (len(enc) - 2))
    tam_ref = vb["FP32"]["tamano_mb"]
    for corto in vb:
        b, p = vb[corto], vp[corto]
        lat = [f["mediana_ms"] for f in b["latencias"]]
        print(f"  {corto:8s} {b['tamano_mb']:5.2f} MB {100 * (1 - b['tamano_mb'] / tam_ref):6.1f}% "
              f"{lat[0]:8.2f} {lat[-1]:8.2f} {b['ram_modelo_mb']:4.0f} MB "
              f"{p['top1'] * 100:5.0f}% {p['concordancia_fp32'] * 100:7.0f}% "
              f"{'completo' if b['latencias'][0]['acelerado'] else 'parcial':>8s}")
    print(f"\n  Archivo = tamaño en disco. Lat 1h / {col_hmax} = latencia mediana en ms con 1 y"
          f" {h_max} hilos.")
    print("  RAM+ = memoria que agrega el modelo cargado y ejecutándose.")
    print(f"  Top-1 y Concord. (concordancia con FP32) sobre {prec['imagenes']} imágenes del lote.")

    seccion("Observaciones calculadas a partir de sus mediciones")
    lat1 = {k: v["latencias"][0]["mediana_ms"] for k, v in vb.items()}
    for corto in ("DIN-W8", "INT8"):
        r_tam = tam_ref / vb[corto]["tamano_mb"]
        r_ram = vb["FP32"]["ram_modelo_mb"] / max(vb[corto]["ram_modelo_mb"], 1e-6)
        r_lat = lat1["FP32"] / lat1[corto]
        vel = (f"{r_lat:.2f}x más rápida" if r_lat >= 1 else f"{1 / r_lat:.2f}x más lenta")
        print(f"  * {corto} frente a FP32: archivo {r_tam:.1f}x más pequeño, "
              f"RAM extra {r_ram:.1f}x menor,")
        print(f"    inferencia {vel} (1 hilo), concordancia "
              f"{vp[corto]['concordancia_fp32'] * 100:.0f} %, "
              f"grafo {'completo' if vb[corto]['latencias'][0]['acelerado'] else 'partido'} en XNNPACK.")
    if h_max > 1:
        acel = lat1["FP32"] / vb["FP32"]["latencias"][-1]["mediana_ms"]
        print(f"  * Pasar de 1 a {h_max} hilos aceleró FP32 {acel:.2f}x.")

    seccion("Registros para el PDF de entrega")
    faltantes = 0
    for paso in range(8):
        ruta = os.path.join(DIR_RESULTADOS, f"paso_{paso:02d}.txt")
        existe = os.path.isfile(ruta) or paso == 7
        faltantes += not existe
        print(f"  [{'✓' if existe else '✗'}] {ruta}")
    print()
    if faltantes:
        print(f"[!] Faltan {faltantes} registros: ejecute los pasos marcados con ✗.")
    else:
        ok("Paso 7 completado: todos los registros están listos para el PDF.")


if __name__ == "__main__":
    with Registro(7, "Resumen comparativo"):
        main()
```

Ejecute el paso 7:

```bash
python 07_resumen.py
```

**Salida esperada** (ejemplo):

```text
======================================================================
  PASO 7: Resumen comparativo
  2026-10-31 09:42:41 | equipo: vm | Python 3.13.18
======================================================================

--- Equipo ----------------------------------------------------------
CPU: Intel(R) Xeon(R) Processor @ 2.10GHz | núcleos lógicos: 2

--- Tabla comparativa final -----------------------------------------
  Variante  Archivo  Reduc.   Lat 1h   Lat 2h    RAM+  Top-1 Concord.  XNNPACK
  ----------------------------------------------------------------------------
  FP32     14.02 MB    0.0%     9.93     4.53   61 MB    81%     100% completo
  DIN-W8    3.83 MB   72.7%    15.20    14.55   49 MB    79%      97%  parcial
  INT8      4.05 MB   71.1%     4.98     3.74   38 MB    74%      87% completo

  Archivo = tamaño en disco. Lat 1h / Lat 2h = latencia mediana en ms con 1 y 2 hilos.
  RAM+ = memoria que agrega el modelo cargado y ejecutándose.
  Top-1 y Concord. (concordancia con FP32) sobre 100 imágenes del lote.

--- Observaciones calculadas a partir de sus mediciones -------------
  * DIN-W8 frente a FP32: archivo 3.7x más pequeño, RAM extra 1.2x menor,
    inferencia 1.53x más lenta (1 hilo), concordancia 97 %, grafo partido en XNNPACK.
  * INT8 frente a FP32: archivo 3.5x más pequeño, RAM extra 1.6x menor,
    inferencia 1.99x más rápida (1 hilo), concordancia 87 %, grafo completo en XNNPACK.
  * Pasar de 1 a 2 hilos aceleró FP32 2.19x.

--- Registros para el PDF de entrega --------------------------------
  [✓] resultados/paso_00.txt
  [✓] resultados/paso_01.txt
  [✓] resultados/paso_02.txt
  [✓] resultados/paso_03.txt
  [✓] resultados/paso_04.txt
  [✓] resultados/paso_05.txt
  [✓] resultados/paso_06.txt
  [✓] resultados/paso_07.txt

[✓] Paso 7 completado: todos los registros están listos para el PDF.
```

### Conclusión del tutorial

No existe una variante "mejor" en general; existe la mejor **para un hardware y una aplicación**:

* **FP32:** máxima precisión; el mayor tamaño y uso de memoria.
* **Dinámica (pesos INT8):** la más fácil de obtener (un comando) y reduce el archivo casi 4 veces, pero en este modelo y en CPU no acelera; incluso puede volverlo más lento.
* **INT8 estática:** la más rápida y ligera, ideal para CPU y NPU embebidas; requiere calibración y pierde algo de precisión, que hay que validar.

---

## Preparación del PDF de Entrega

El PDF debe contener la salida de consola de los **pasos 0 a 7**, en orden. Ya está guardada en la carpeta `resultados/`.

1. Verifique que el paso 7 marque todos los registros con `[✓]`.
2. Una todos los registros en un solo archivo de texto:

```bash
cat resultados/paso_0*.txt > resultados/entrega.txt
```

3. Abra `resultados/entrega.txt` en LibreOffice Writer (o en su editor de texto), seleccione todo el texto y aplique una **fuente monoespaciada** (por ejemplo, *DejaVu Sans Mono* o *Liberation Mono*) de **9 pt**, con márgenes de 2 cm. Así las tablas quedan alineadas: ninguna línea supera los 90 caracteres.
4. Exporte a PDF (*Archivo → Exportar a PDF*, o *Imprimir → Imprimir a archivo*).

> Si repite un paso, su registro se reemplaza por el de la última ejecución. Repita el paso 7 al final para que el resumen use los datos más recientes.

---

## Funcionamiento Detallado de las API de LiteRT Usadas

| Llamada | Paso | Qué hace |
|---|---|---|
| `hf_hub_download(repo, archivo, revision=...)` | 1 | Descarga un archivo de un repositorio de Hugging Face, en una revisión (commit) fija. |
| `Interpreter(model_path)` | 2 | API clásica. Carga el FlatBuffer; `get_input_details()` y `get_output_details()` devuelven forma, tipo y `quantization = (escala, punto_cero)`. |
| `Interpreter.allocate_tensors()` | 2 | Reserva memoria y aplica el delegado XNNPACK, que se queda con los segmentos del grafo que puede acelerar. |
| `CompiledModel.from_file(ruta, options=...)` | 3, 5, 6 | API moderna. Carga y compila el modelo para el acelerador elegido (`HardwareAccelerator.CPU` por defecto). `CpuOptions(num_threads=N)` fija los hilos. |
| `get_signature_list()` | 2, 3 | Firmas del modelo: nombres de entradas y salidas (`args_0` → `output_0`). |
| `create_input_buffers(0)` / `create_output_buffers(0)` | 3 | Crea los `TensorBuffer` de E/S de la firma 0, con el tamaño exacto que necesita el modelo. |
| `buffer.write(array)` / `buffer.read(n, dtype)` | 3 | Copia un arreglo de NumPy al búfer y lee el resultado (admite `float32`, `int32` e `int8`). |
| `run_by_index(0, entradas, salidas)` | 3 | Ejecuta la inferencia de la firma 0. |
| `is_fully_accelerated()` | 2, 3, 5 | Indica si todo el grafo quedó en el acelerador. |
| `litert-quantizer --model_file ... --recipe ...` | 4 | Cuantiza un `.tflite` con una receta predefinida y escribe el modelo resultante. |

---

## Diagnóstico y Solución de Problemas

### `error: externally-managed-environment` (PEP 668)
Ocurre al usar `pip install` fuera del entorno virtual. No use `pip` directamente: ejecute `./setup.sh` y luego `source .venv/bin/activate`.

### `ModuleNotFoundError: No module named 'ai_edge_litert'` (u otro paquete)
El entorno virtual no está activo. Ejecute `source .venv/bin/activate` y verifique que el indicador muestre `(.venv)`. Debe hacerse en cada terminal nueva.

### `Permission denied` al ejecutar `./setup.sh` o `./04_cuantizar.sh`
Faltan permisos de ejecución: `chmod +x setup.sh 04_cuantizar.sh`.

### `Python X.Y no es compatible`
`ai-edge-litert` 2.3.0 publica paquetes para Python 3.10 a 3.14. En Ubuntu 20.04 (Python 3.8) instale una versión más reciente de Python, por ejemplo con el gestor `uv` o con el PPA *deadsnakes*.

### `ERROR: Could not find a version that satisfies the requirement ai-edge-litert==2.3.0`
Su sistema no es de 64 bits (x86-64 o ARM64), o su versión de Python no es compatible. En una Raspberry Pi, use el sistema operativo de **64 bits**: no hay paquetes para ARM de 32 bits.

### El paso 1 no puede descargar de Hugging Face
Verifique su conexión. Si su red bloquea Hugging Face, descargue los archivos desde un navegador y guárdelos en la carpeta `modelos/` con estos nombres exactos:

* `https://huggingface.co/litert-community/MobileNet-v2/resolve/48e55de4765732640b9702c520999e4bf1a1f50f/mobilenet_v2.tflite`
* `https://huggingface.co/litert-community/MobileNet-v2/resolve/48e55de4765732640b9702c520999e4bf1a1f50f/mobilenet_v2_int8_channelwise.tflite`
* `https://huggingface.co/datasets/huggingface/label-files/resolve/main/imagenet-1k-id2label.json`

Luego ejecute de nuevo `python 01_descargar.py`: detectará los archivos y solo descargará lo que falte.

### `no tiene la cabecera TFL3` / `The model is not a valid Flatbuffer buffer`
El archivo no es un modelo, casi siempre una página de error guardada por una descarga fallida (el problema que tuvo la primera versión de la demostración). Bórrelo y repita el paso 1:

```bash
head -c 200 modelos/mobilenet_v2.tflite   # si empieza con "<html" o "<?xml", no es un modelo
rm modelos/mobilenet_v2.tflite
python 01_descargar.py
```

### `No se pudieron descargar N imágenes`
Fallo temporal de la red con GitHub. Ejecute de nuevo el paso 1.

### `cannot identify image file 'mi_foto.jpg'`
La foto está en un formato que Pillow no lee, típicamente **HEIC** (iPhone). Conviértala a JPG con cualquier editor de imágenes, o configure la cámara en *Ajustes → Cámara → Formatos → Más compatible*.

### Las predicciones del paso 3 no tienen sentido
* El objeto quizá no pertenece a las 1000 clases de ImageNet; pruebe con uno de la lista sugerida.
* El objeto debe estar **centrado** y ocupar buena parte de la foto (se usa el recorte central).
* Mejore la iluminación y use un fondo sencillo.

### Muchos mensajes `INFO:` o `WARNING:` sobre NPU, GPU u OpenVINO
Son normales: LiteRT intenta cargar todos los aceleradores y avisa de los que no encuentra. Los scripts los guardan en `resultados/litert_nativo.log` para no ensuciar la consola.

### Las latencias del paso 5 varían mucho entre ejecuciones
* Cierre otros programas (en especial el navegador) y conecte el cargador si usa una laptop: en batería, la CPU reduce su frecuencia.
* Repita la medición con más muestras: `python 05_benchmark.py --repeticiones 300`.
* Lo importante son las **proporciones** entre variantes, que se mantienen aunque cambien los valores absolutos.

### La versión dinámica es más lenta que FP32
No es un error. Se explica en el paso 5.

### `El entorno virtual no está activo` al ejecutar `./04_cuantizar.sh`
Ejecute `source .venv/bin/activate` y repita el paso 4.

---

## Referencias

* Google AI Edge. *LiteRT CompiledModel Python API*. https://ai.google.dev/edge/litert/next/python
* Google AI Edge. *Model optimization (LiteRT)*. https://developers.google.com/edge/litert/quantization/model_optimization
* Google AI Edge. *LiteRT Quantizer* (`litert-quantizer`). https://pypi.org/project/litert-quantizer
* Google AI Edge. *AI Edge Quantizer*, repositorio. https://github.com/google-ai-edge/ai-edge-quantizer
* LiteRT Community. *MobileNet-v2* (tarjeta del modelo y archivos). https://huggingface.co/litert-community/MobileNet-v2
* Hugging Face. *huggingface/label-files* (etiquetas de ImageNet-1k). https://huggingface.co/datasets/huggingface/label-files
* E. Schwartz. *imagenet-sample-images* (una imagen de muestra por clase de ImageNet). https://github.com/EliSchwartz/imagenet-sample-images
* M. Sandler, A. Howard, M. Zhu, A. Zhmoginov y L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks", *CVPR*, 2018.
* J. Deng *et al.*, "ImageNet: A large-scale hierarchical image database", *CVPR*, 2009.
