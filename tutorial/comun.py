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
