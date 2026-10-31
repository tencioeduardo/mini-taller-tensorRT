#!/usr/bin/env python3
"""
rps_common.py - Funciones compartidas por la demo en vivo (litert_rps.py) y por
el script de evaluación (evaluar.py).

Tener un único módulo garantiza que la precisión medida en evaluar.py corresponde
exactamente al mismo preprocesamiento que se usa con la cámara en vivo.

Contenido:
    validar_tflite()     Verifica que un archivo sea un FlatBuffer TFLite real.
    cargar_etiquetas()   Lee labels.txt (una etiqueta por línea o formato Teachable Machine).
    preprocesar()        Fotograma BGR de OpenCV -> tensor de entrada del modelo.
    a_probabilidades()   Salida cruda del modelo -> vector de probabilidades.
    Motor                Envoltura con la misma interfaz para las dos APIs de LiteRT:
                         Interpreter (clásica) y CompiledModel (LiteRT 2.x).
"""
import contextlib
import os
import sys

import cv2 as cv
import numpy as np

ETIQUETAS_POR_DEFECTO = ["Piedra", "Papel", "Tijera"]

# Tamaño mínimo razonable de un modelo real. Una página de error HTML guardada como
# .tflite suele pesar menos que esto.
TAMANO_MINIMO_BYTES = 1024


# --------------------------------------------------------------------------
# Silenciar los registros nativos de LiteRT
# --------------------------------------------------------------------------
@contextlib.contextmanager
def silenciar_stderr_nativo(activo=True):
    """
    LiteRT escribe sus mensajes INFO/WARNING desde C++ directamente al descriptor 2
    (stderr), por lo que no se pueden ocultar con el módulo `logging` de Python.
    Durante la demo esos mensajes ensucian la consola, así que se redirige el
    descriptor 2 a /dev/null mientras se crea el motor.

    Si ocurre un error, el mensaje de Python (la excepción) sí se muestra, porque
    se lanza después de restaurar stderr. Use --verbose en los scripts para ver
    todos los registros nativos (útil para diagnosticar la GPU).
    """
    if not activo:
        yield
        return
    sys.stderr.flush()
    copia = os.dup(2)
    nulo = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nulo, 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(copia, 2)
        os.close(copia)
        os.close(nulo)


def silenciar_al_salir():
    """
    Al terminar el programa, LiteRT imprime mensajes al destruir sus aceleradores.
    Esta función redirige stderr a /dev/null justo antes de salir para que la
    consola de la demo termine limpia. Llamarla solo después del último mensaje.
    """
    sys.stderr.flush()
    nulo = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nulo, 2)


def mensaje_error_gpu(exc):
    """Traduce el error nativo de la GPU a un mensaje comprensible."""
    return ("GPU no disponible: LiteRT usa WebGPU sobre Vulkan en Linux y no encontró "
            "un adaptador utilizable (driver Vulkan ausente o no compatible). "
            f"Detalle: {str(exc).splitlines()[0][:80]} ... (use --verbose para ver el registro completo)")


# --------------------------------------------------------------------------
# Validación de archivos
# --------------------------------------------------------------------------
def validar_tflite(ruta):
    """
    Un .tflite es un FlatBuffer. Los FlatBuffers con identificador de archivo
    guardan 4 bytes de "file identifier" en las posiciones 4 a 7; para TFLite
    ese identificador es b"TFL3". Si no aparece, el archivo no es un modelo
    (típicamente es una página HTML/XML de una descarga fallida).

    Lanza ValueError con un diagnóstico en lugar de terminar el programa, para
    que cada script decida cómo reaccionar.
    """
    if not os.path.isfile(ruta):
        raise ValueError(f"No existe el modelo '{ruta}'.")
    tamano = os.path.getsize(ruta)
    with open(ruta, "rb") as f:
        cabecera = f.read(512)
    if tamano < TAMANO_MINIMO_BYTES or cabecera[4:8] != b"TFL3":
        vista = cabecera[:120].decode("utf-8", errors="replace").replace("\n", " ")
        raise ValueError(
            f"'{ruta}' NO es un modelo .tflite válido ({tamano} bytes).\n"
            f"  Contenido inicial: {vista!r}\n"
            "  Si empieza con '<html' o '<?xml', es una página de error guardada por una descarga fallida."
        )
    return tamano


def cargar_etiquetas(ruta):
    """
    Admite dos formatos:
      - Una etiqueta por línea:            Piedra
      - Formato de Teachable Machine:      0 Piedra
    Si el archivo no existe, devuelve las etiquetas por defecto.
    """
    if not ruta or not os.path.isfile(ruta):
        return list(ETIQUETAS_POR_DEFECTO)
    etiquetas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            partes = linea.split(maxsplit=1)
            if len(partes) == 2 and partes[0].isdigit():
                linea = partes[1]
            etiquetas.append(linea)
    return etiquetas or list(ETIQUETAS_POR_DEFECTO)


# --------------------------------------------------------------------------
# Preprocesamiento
# --------------------------------------------------------------------------
def recorte_cuadrado_central(imagen):
    """Recorta el cuadrado central para no deformar la mano al redimensionar."""
    h, w = imagen.shape[:2]
    lado = min(h, w)
    y0, x0 = (h - lado) // 2, (w - lado) // 2
    return imagen[y0:y0 + lado, x0:x0 + lado]


def preprocesar(imagen_bgr, ancho, alto, dtype, norm="0-1"):
    """
    Convierte un fotograma de OpenCV (BGR, uint8, HxWx3) al tensor que espera el modelo.

    Pasos:
      1. Recorte cuadrado central.
      2. BGR -> RGB (OpenCV entrega BGR; los modelos se entrenan con RGB).
      3. Redimensionado al tamaño de entrada del modelo (INTER_AREA es la
         interpolación recomendada para reducir imágenes).
      4. Se agrega la dimensión de lote: (1, alto, ancho, 3).
      5. Conversión al tipo de dato del tensor de entrada:
         - float32: se normaliza según `norm`:
               "0-1"  -> x / 255            (modelo de Colab de este repositorio)
               "-1-1" -> x / 127.5 - 1      (Teachable Machine en punto flotante)
         - uint8:   píxeles crudos 0..255.
         - int8:    píxeles desplazados a -128..127.

    Nota sobre las entradas cuantizadas (hallazgo H6 de la revisión): para uint8 e
    int8 se usa la convención estándar de los modelos de imagen, en la que el rango
    entero completo representa el rango flotante con el que se entrenó el modelo.
    Bajo esa convención, cuantizar con los parámetros (escala, punto cero) del
    tensor da exactamente los mismos enteros que pasar los píxeles crudos, sin
    depender de que --norm sea correcto. Por eso aquí no se aplican esos parámetros;
    el Motor sí los reporta (ver Motor.rango_entrada) para poder verificarlo.
    """
    rgb = cv.cvtColor(recorte_cuadrado_central(imagen_bgr), cv.COLOR_BGR2RGB)
    redim = cv.resize(rgb, (ancho, alto), interpolation=cv.INTER_AREA)
    lote = np.expand_dims(redim, axis=0)

    if dtype == np.float32:
        x = lote.astype(np.float32)
        return x / 255.0 if norm == "0-1" else (x / 127.5) - 1.0
    if dtype == np.int8:
        return (lote.astype(np.int16) - 128).astype(np.int8)
    return lote.astype(np.uint8)


# --------------------------------------------------------------------------
# Postprocesamiento
# --------------------------------------------------------------------------
def a_probabilidades(salida_cruda, cuantizacion=(0.0, 0)):
    """
    Convierte la salida del modelo en probabilidades:
      1. Si la salida es entera (uint8/int8), se des-cuantiza:
            valor_real = escala * (entero - punto_cero)
         (la misma fórmula de la diapositiva 7).
      2. Si los valores no parecen probabilidades (negativos, mayores que 1 o
         que no suman ~1), se interpretan como logits y se aplica softmax.
    """
    crudo = np.asarray(salida_cruda)
    probs = np.squeeze(crudo).astype(np.float32)

    escala, punto_cero = cuantizacion
    if crudo.dtype in (np.uint8, np.int8) and escala:
        probs = (probs - punto_cero) * escala

    if probs.min() < 0.0 or probs.max() > 1.0001 or abs(float(probs.sum()) - 1.0) > 0.05:
        e = np.exp(probs - probs.max())
        probs = e / e.sum()
    return probs


# --------------------------------------------------------------------------
# Motor de inferencia: misma interfaz para Interpreter y CompiledModel
# --------------------------------------------------------------------------
class Motor:
    """
    Envuelve las dos APIs de LiteRT detrás de los mismos tres pasos, para poder
    medirlas por separado y compararlas con el mismo código:

        motor.cargar_entrada(x)   # copiar el tensor de entrada
        motor.ejecutar()          # SOLO la inferencia (lo que se cronometra)
        motor.leer_salida()       # obtener probabilidades

    API "interpreter" (clásica, diapositiva 5):
        Interpreter -> allocate_tensors -> set_tensor -> invoke -> get_tensor.
        Usa la CPU (XNNPACK).

    API "compiled" (CompiledModel, LiteRT 2.x, diapositiva 5):
        CompiledModel.from_file compila el grafo una vez para el acelerador
        elegido y crea los búferes de E/S (TensorBuffer).
        Permite pedir GPU (diapositiva 6). En Linux la GPU usa WebGPU sobre
        Vulkan, por lo que requiere un driver Vulkan funcional.

    Limitación conocida de CompiledModel en Python: TensorBuffer admite float32,
    int32 e int8, pero no uint8. Los modelos con entrada uint8 (por ejemplo, el
    cuantizado de Teachable Machine) solo funcionan con la API "interpreter".
    """

    APIS = ("interpreter", "compiled")
    _TIPOS_COMPILED = (np.float32, np.int32, np.int8)

    def __init__(self, ruta, api="compiled", hilos=4, acelerador="cpu", verbose=False):
        if api not in self.APIS:
            raise ValueError(f"API desconocida '{api}'. Opciones: {self.APIS}")
        if api == "interpreter" and acelerador != "cpu":
            raise ValueError("La API Interpreter de Python solo ejecuta en CPU; use --api compiled para GPU.")

        self.ruta = ruta
        self.nombre = os.path.splitext(os.path.basename(ruta))[0]
        self.tamano_bytes = validar_tflite(ruta)
        self.api = api
        self.acelerador = acelerador

        # Toda la creación va silenciada: LiteRT registra aceleradores y crea su
        # entorno en varios momentos (import, compilación, creación de búferes).
        # Si algo falla, la excepción se muestra igual (stderr ya está restaurado).
        with silenciar_stderr_nativo(not verbose):
            if api == "interpreter":
                self._crear_interpreter(ruta, hilos)
            else:
                self._crear_compiled(ruta, hilos, acelerador)

    # --- creación según la API ---------------------------------------------
    def _crear_interpreter(self, ruta, hilos):
        from ai_edge_litert.interpreter import Interpreter

        self._interp = Interpreter(model_path=ruta, num_threads=hilos)
        # Reserva la memoria de todos los tensores (entrada, intermedios, salida).
        self._interp.allocate_tensors()
        det_in = self._interp.get_input_details()[0]
        det_out = self._interp.get_output_details()[0]
        self._idx_in, self._idx_out = det_in["index"], det_out["index"]
        self._guardar_metadatos(det_in["shape"], det_in["dtype"], det_in["quantization"],
                                det_out["shape"], det_out["dtype"], det_out["quantization"])
        self.totalmente_acelerado = None

    def _crear_compiled(self, ruta, hilos, acelerador):
        # El import crea el entorno de LiteRT y registra los aceleradores
        # disponibles (CPU/XNNPACK, GPU/WebGPU, NPU).
        from ai_edge_litert.compiled_model import (
            CompiledModel, CpuOptions, HardwareAccelerator, Options)

        if acelerador == "gpu":
            # GPU con CPU como respaldo para las operaciones que la GPU no soporte.
            aceleradores = HardwareAccelerator.GPU | HardwareAccelerator.CPU
        else:
            aceleradores = HardwareAccelerator.CPU
        opciones = Options(hardware_accelerators=aceleradores,
                           cpu_options=CpuOptions(num_threads=hilos))
        # Compila el grafo UNA vez para el acelerador elegido (diapositiva 5).
        self._modelo = CompiledModel.from_file(ruta, options=opciones)

        # Metadatos leídos del propio CompiledModel (firma 0, primera entrada/salida).
        firma = self._modelo.get_signature_by_index(0)
        det_in = self._modelo.get_input_tensor_details(firma["key"])[firma["inputs"][0]]
        det_out = self._modelo.get_output_tensor_details(firma["key"])[firma["outputs"][0]]
        dtype_in = np.dtype(det_in["dtype"]).type
        dtype_out = np.dtype(det_out["dtype"]).type
        if dtype_in not in self._TIPOS_COMPILED:
            raise TypeError(
                f"CompiledModel (Python) no admite entradas {np.dtype(dtype_in).name}; "
                "use --api interpreter para este modelo."
            )

        # CompiledModel no expone los parámetros de cuantización. Solo si algún
        # extremo es entero se abren con un Interpreter temporal (no ocurre con
        # los modelos de este repositorio, que tienen E/S en float32).
        cuant_in = cuant_out = (0.0, 0)
        if dtype_in != np.float32 or dtype_out != np.float32:
            from ai_edge_litert.interpreter import Interpreter
            tmp = Interpreter(model_path=ruta)
            cuant_in = tmp.get_input_details()[0]["quantization"]
            cuant_out = tmp.get_output_details()[0]["quantization"]
            del tmp

        self._guardar_metadatos(det_in["shape"], dtype_in, cuant_in,
                                det_out["shape"], dtype_out, cuant_out)
        # Búferes de E/S gestionados por LiteRT (TensorBuffer).
        self._buf_in = self._modelo.create_input_buffers(0)
        self._buf_out = self._modelo.create_output_buffers(0)
        # La primera escritura en un TensorBuffer inicializa el entorno de LiteRT de
        # forma diferida (y emite registros); se hace aquí, dentro del bloque silenciado.
        self._buf_in[0].write(np.zeros(self.forma_entrada, dtype=self.dtype_entrada))
        self.totalmente_acelerado = self._modelo.is_fully_accelerated()

    def _guardar_metadatos(self, forma_in, dtype_in, cuant_in, forma_out, dtype_out, cuant_out):
        self.forma_entrada = tuple(int(v) for v in forma_in)
        _, self.alto, self.ancho, self.canales = self.forma_entrada
        self.dtype_entrada = np.dtype(dtype_in).type
        self.cuant_entrada = tuple(cuant_in)
        self.dtype_salida = np.dtype(dtype_out).type
        self.cuant_salida = tuple(cuant_out)
        self.num_clases = int(forma_out[-1])

    # --- descripción --------------------------------------------------------
    @property
    def etiqueta_api(self):
        if self.api == "interpreter":
            return "Interpreter/CPU"
        return f"CompiledModel/{self.acelerador.upper()}"

    @property
    def rango_entrada(self):
        """Rango flotante que representan las entradas cuantizadas (None si es float)."""
        escala, punto_cero = self.cuant_entrada
        if not escala:
            return None
        info = np.iinfo(self.dtype_entrada)
        return (escala * (info.min - punto_cero), escala * (info.max - punto_cero))

    def describir(self):
        rango = self.rango_entrada
        txt_rango = f"  rango_real=[{rango[0]:.3f}, {rango[1]:.3f}]" if rango else ""
        return (f"{self.nombre}: {self.tamano_bytes / 1024:.0f} KiB | "
                f"entrada {self.ancho}x{self.alto}x{self.canales} {np.dtype(self.dtype_entrada).name}{txt_rango} | "
                f"salida {self.num_clases} clases {np.dtype(self.dtype_salida).name} | {self.etiqueta_api}")

    # --- ciclo de inferencia ------------------------------------------------
    def cargar_entrada(self, x):
        if self.api == "interpreter":
            self._interp.set_tensor(self._idx_in, x)
        else:
            self._buf_in[0].write(np.ascontiguousarray(x))

    def ejecutar(self):
        if self.api == "interpreter":
            self._interp.invoke()
        else:
            self._modelo.run_by_index(0, self._buf_in, self._buf_out)

    def leer_salida(self):
        if self.api == "interpreter":
            crudo = self._interp.get_tensor(self._idx_out)
        else:
            crudo = self._buf_out[0].read(self.num_clases, self.dtype_salida)
        return a_probabilidades(crudo, self.cuant_salida)

    def predecir(self, x):
        self.cargar_entrada(x)
        self.ejecutar()
        return self.leer_salida()

    def calentar(self, repeticiones=3):
        """La primera inferencia es más lenta (reserva de memoria, empaquetado de pesos)."""
        x = np.zeros(self.forma_entrada, dtype=self.dtype_entrada)
        for _ in range(repeticiones):
            self.predecir(x)


def crear_motor(ruta, api="compiled", hilos=4, acelerador="cpu", verbose=False):
    """
    Crea un Motor sin lanzar excepciones: devuelve (motor, None) o (None, mensaje).

    Si la creación falla (por ejemplo, GPU sin driver Vulkan o modelo uint8 con
    CompiledModel), los objetos nativos del intento fallido se liberan dentro del
    bloque silenciado, para que sus registros de destrucción no ensucien la consola.
    """
    import gc

    with silenciar_stderr_nativo(not verbose):
        try:
            return Motor(ruta, api=api, hilos=hilos, acelerador=acelerador, verbose=verbose), None
        except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
            mensaje = mensaje_error_gpu(exc) if acelerador == "gpu" else str(exc)
            exc = None  # suelta la referencia al traceback y a los objetos nativos
            gc.collect()
            return None, mensaje
