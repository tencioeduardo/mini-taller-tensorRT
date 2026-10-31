#!/usr/bin/env python3
"""
litert_rps.py - Acto 3 de la demostración: clasificación en vivo con LiteRT + OpenCV.

Flujo por fotograma:
    cámara (o imagen fija) -> preprocesar -> cargar_entrada -> ejecutar -> leer_salida -> dibujar

Todos los modelos de modelos/ se cargan al inicio, para cambiar entre ellos en vivo
con la tecla 'm' y comparar FP32 contra INT8 sobre la misma escena.

Teclas:
    m        cambiar al siguiente modelo (FP32 -> FP16 -> INT8 dinámico -> INT8 completo)
    f        alternar entre la cámara y la imagen de contingencia
    s        guardar el fotograma actual como imagen de contingencia (solo con cámara)
    q / ESC  salir

Ejemplos:
    python litert_rps.py                         # CompiledModel en CPU, empieza con INT8 completo
    python litert_rps.py --api interpreter       # API clásica
    python litert_rps.py --gpu                   # intenta GPU; si no hay driver, sigue en CPU
    python litert_rps.py --camera 1              # usa /dev/video1
    python litert_rps.py --autoprueba 60         # verificación sin ventana (checklist previo)
"""
import argparse
import collections
import glob
import os
import sys
import time

import cv2 as cv
import numpy as np

import rps_common as rc

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
VENTANA = "LiteRT - Piedra, Papel o Tijera"
ORDEN_PREFERIDO = ["rps_fp32", "rps_fp16", "rps_int8_dyn", "rps_int8_full"]
MODELO_INICIAL = "rps_int8_full"
VENTANA_PROMEDIO = 30           # fotogramas para el promedio móvil de latencia y FPS


# --------------------------------------------------------------------------
# Carga de modelos
# --------------------------------------------------------------------------
def descubrir_modelos(args):
    """Lista los .tflite a cargar: --model explícito, o todos los de --modelos."""
    if args.model:
        return [args.model]
    rutas = glob.glob(os.path.join(args.modelos, "*.tflite"))
    if not rutas:
        # Compatibilidad con la versión anterior del repositorio (un solo modelo en la raíz).
        heredado = os.path.join(DIR_BASE, "rps_model.tflite")
        if os.path.isfile(heredado):
            print(f"[AVISO] '{args.modelos}' está vacío; se usa el modelo heredado rps_model.tflite.")
            return [heredado]
        sys.exit(f"[ERROR] No hay modelos en '{args.modelos}'. Ejecute el cuaderno entrenar_rps.ipynb "
                 "y descomprima artefactos_rps.zip dentro de la carpeta demostracion/.")

    def clave(r):
        n = os.path.splitext(os.path.basename(r))[0]
        return (ORDEN_PREFERIDO.index(n) if n in ORDEN_PREFERIDO else len(ORDEN_PREFERIDO), n)
    return sorted(rutas, key=clave)


def cargar_motores(rutas, args):
    """
    Crea un Motor por modelo con la API pedida. Si la GPU no está disponible se
    informa una sola vez y se continúa en CPU; si un modelo no es compatible con
    CompiledModel (entrada uint8), ese modelo usa Interpreter.
    """
    acelerador = "gpu" if args.gpu else "cpu"
    if args.gpu and args.api != "compiled":
        print("[AVISO] La GPU solo está disponible con --api compiled; se usa CPU.")
        acelerador = "cpu"

    motores = []
    for ruta in rutas:
        motor, error = rc.crear_motor(ruta, api=args.api, hilos=args.hilos,
                                      acelerador=acelerador, verbose=args.verbose)
        if motor is None and acelerador == "gpu":
            print(f"[AVISO] {error}\n        Se continúa en CPU.")
            acelerador = "cpu"
            motor, error = rc.crear_motor(ruta, api=args.api, hilos=args.hilos, verbose=args.verbose)
        if motor is None and args.api == "compiled":
            print(f"[AVISO] {os.path.basename(ruta)}: {error}\n        Se usa la API Interpreter para este modelo.")
            motor, error = rc.crear_motor(ruta, api="interpreter", hilos=args.hilos, verbose=args.verbose)
        if motor is None:
            print(f"[AVISO] Se omite {os.path.basename(ruta)}: {error}")
            continue
        motor.calentar()
        motores.append(motor)
    if not motores:
        sys.exit("[ERROR] No se pudo cargar ningún modelo.")
    return motores


# --------------------------------------------------------------------------
# Captura y dibujo
# --------------------------------------------------------------------------
def abrir_camara(indice):
    # Oculta los avisos internos de OpenCV si la cámara no existe; el programa ya
    # informa el problema con su propio mensaje.
    try:
        cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    cap = cv.VideoCapture(indice, cv.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def dibujar(img, lineas, pie):
    """Banda negra superior con las líneas de texto y un pie de página."""
    alto_banda = 14 + 28 * len(lineas)
    cv.rectangle(img, (0, 0), (img.shape[1], alto_banda), (0, 0, 0), -1)
    for i, (texto, color, escala) in enumerate(lineas):
        cv.putText(img, texto, (12, 28 + 28 * i), cv.FONT_HERSHEY_SIMPLEX, escala, color, 2)
    cv.rectangle(img, (0, img.shape[0] - 30), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv.putText(img, pie, (12, img.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


# --------------------------------------------------------------------------
# Inferencia de un fotograma, con tiempos separados
# --------------------------------------------------------------------------
def inferir(motor, frame, norm):
    """Devuelve (probabilidades, ms de preprocesamiento, ms de invoke)."""
    t0 = time.perf_counter()
    x = rc.preprocesar(frame, motor.ancho, motor.alto, motor.dtype_entrada, norm)
    motor.cargar_entrada(x)
    t1 = time.perf_counter()
    motor.ejecutar()                       # SOLO la inferencia
    t2 = time.perf_counter()
    probs = motor.leer_salida()
    return probs, (t1 - t0) * 1000, (t2 - t1) * 1000


# --------------------------------------------------------------------------
# Modo sin ventana: verificación previa a la demo
# --------------------------------------------------------------------------
def autoprueba(motores, etiquetas, args, cap, imagen_fija):
    """
    Procesa N fotogramas con cada modelo sin abrir ventana. Confirma que la fuente
    de video, los modelos y el preprocesamiento funcionan juntos antes de la demo.
    """
    fuente = "cámara" if cap is not None else f"imagen fija ({os.path.basename(args.image)})"
    print(f"\n=== Autoprueba: {args.autoprueba} fotogramas por modelo | fuente: {fuente} ===")
    for motor in motores:
        pre, inv, preds = [], [], []
        for _ in range(args.autoprueba):
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    sys.exit("[ERROR] Falló la lectura de la cámara durante la autoprueba.")
            else:
                frame = imagen_fija.copy()
            probs, t_pre, t_inv = inferir(motor, frame, args.norm)
            pre.append(t_pre); inv.append(t_inv); preds.append(int(np.argmax(probs)))
        clase = collections.Counter(preds).most_common(1)[0][0]
        nombre = etiquetas[clase] if clase < len(etiquetas) else f"Clase {clase}"
        print(f"  {motor.nombre:<14} {motor.etiqueta_api:<18} preproc {np.mean(pre):5.2f} ms | "
              f"invoke p50 {np.median(inv):5.2f} ms | predicción más frecuente: {nombre}")
    print("Autoprueba completada.")


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="usar solo este .tflite (desactiva la tecla 'm')")
    p.add_argument("--modelos", default=os.path.join(DIR_BASE, "modelos"),
                   help="carpeta con los modelos para la tecla 'm'")
    p.add_argument("--labels", default=os.path.join(DIR_BASE, "modelos", "labels.txt"))
    p.add_argument("--image", default=os.path.join(DIR_BASE, "test_hand.jpg"), help="imagen de contingencia")
    p.add_argument("--camera", type=int, default=0, help="índice N de /dev/videoN")
    p.add_argument("--api", choices=["compiled", "interpreter"], default="compiled",
                   help="API de LiteRT: CompiledModel (LiteRT 2.x) o Interpreter (clásica)")
    p.add_argument("--gpu", action="store_true", help="intentar GPU (WebGPU/Vulkan); si falla, CPU")
    p.add_argument("--norm", choices=["0-1", "-1-1"], default="0-1",
                   help="normalización para modelos float32 (Colab: 0-1, Teachable Machine: -1-1)")
    p.add_argument("--hilos", type=int, default=4, help="hilos de CPU para la inferencia")
    p.add_argument("--no-mirror", action="store_true", help="no voltear la cámara como espejo")
    p.add_argument("--autoprueba", type=int, default=0, metavar="N",
                   help="procesar N fotogramas por modelo sin ventana y salir")
    p.add_argument("--verbose", action="store_true", help="mostrar los registros nativos de LiteRT")
    return p.parse_args()


def main():
    args = parse_args()
    print("=== LiteRT - Piedra, Papel o Tijera ===")

    # PASO 1: cargar los modelos (validación TFL3 + creación del motor con la API elegida)
    motores = cargar_motores(descubrir_modelos(args), args)
    etiquetas = rc.cargar_etiquetas(args.labels if os.path.isfile(args.labels)
                                    else os.path.join(DIR_BASE, "labels.txt"))
    for m in motores:
        print("  " + m.describir())
        if m.num_clases != len(etiquetas):
            print(f"  [AVISO] {m.nombre} tiene {m.num_clases} salidas pero hay {len(etiquetas)} etiquetas.")
    print(f"Etiquetas: {etiquetas}")
    actual = next((i for i, m in enumerate(motores) if m.nombre == MODELO_INICIAL), 0)

    # PASO 2: fuentes de video
    cap = abrir_camara(args.camera)
    imagen_fija = cv.imread(args.image) if os.path.isfile(args.image) else None
    if cap is None:
        print(f"[AVISO] No se pudo abrir /dev/video{args.camera}.")
    if imagen_fija is None:
        print(f"[AVISO] No hay imagen de contingencia '{os.path.basename(args.image)}' "
              "(con la cámara activa, pulse 's' para crearla).")
    if cap is None and imagen_fija is None:
        sys.exit("[ERROR] No hay cámara ni imagen de contingencia disponibles.")

    if args.autoprueba:
        autoprueba(motores, etiquetas, args, cap, imagen_fija)
        if cap is not None:
            cap.release()
        rc.silenciar_al_salir()
        return

    usar_camara = cap is not None
    print("\nControles: m = modelo | f = cámara/imagen | s = guardar imagen | q/ESC = salir\n")

    hist_pre = collections.deque(maxlen=VENTANA_PROMEDIO)
    hist_inv = collections.deque(maxlen=VENTANA_PROMEDIO)
    hist_cuadro = collections.deque(maxlen=VENTANA_PROMEDIO)
    previo = time.perf_counter()

    while True:
        # --- Adquisición ---
        if usar_camara:
            ok, frame = cap.read()
            if not ok:
                print("[AVISO] Falló la lectura de la cámara; se pasa a la imagen de contingencia.")
                if imagen_fija is None:
                    break
                usar_camara = False
                continue
            if not args.no_mirror:
                frame = cv.flip(frame, 1)
            fuente = f"camara /dev/video{args.camera}"
        else:
            frame = imagen_fija.copy()
            fuente = "imagen fija (contingencia)"
        frame_limpio = frame.copy()          # sin texto, para la tecla 's'

        # --- Inferencia ---
        motor = motores[actual]
        probs, t_pre, t_inv = inferir(motor, frame, args.norm)
        hist_pre.append(t_pre)
        hist_inv.append(t_inv)
        ahora = time.perf_counter()
        hist_cuadro.append(ahora - previo)
        previo = ahora

        idx = int(np.argmax(probs))
        nombre = etiquetas[idx] if idx < len(etiquetas) else f"Clase {idx}"
        fps = len(hist_cuadro) / max(sum(hist_cuadro), 1e-6)

        # --- Visualización ---
        dibujar(frame, [
            (f"{nombre}  {probs[idx] * 100:.0f}%", (0, 255, 0), 0.85),
            (f"{motor.nombre}  {motor.tamano_bytes / 1048576:.1f} MiB  |  {motor.etiqueta_api}",
             (255, 255, 255), 0.55),
            (f"preproc {np.mean(hist_pre):.1f} ms | invoke {np.mean(hist_inv):.2f} ms | {fps:.0f} FPS",
             (0, 255, 255), 0.55),
        ], f"{fuente}   |   m: modelo  f: fuente  s: guardar  q: salir")
        cv.imshow(VENTANA, frame)

        # --- Teclado ---
        tecla = cv.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("m"):
            if len(motores) > 1:
                actual = (actual + 1) % len(motores)
                hist_pre.clear(); hist_inv.clear()
                print(f"Modelo: {motores[actual].describir()}")
            else:
                print("[AVISO] Solo hay un modelo cargado.")
        elif tecla == ord("f"):
            if not usar_camara and cap is None:
                print("[AVISO] No hay cámara disponible.")
            elif usar_camara and imagen_fija is None:
                print("[AVISO] No hay imagen de contingencia; pulse 's' primero.")
            else:
                usar_camara = not usar_camara
                print("Fuente:", "cámara" if usar_camara else "imagen fija")
        elif tecla == ord("s") and usar_camara:
            cv.imwrite(args.image, frame_limpio)
            imagen_fija = frame_limpio
            print(f"Imagen de contingencia guardada en '{os.path.basename(args.image)}'")

    if cap is not None:
        cap.release()
    cv.destroyAllWindows()
    print("Demostración finalizada.")
    rc.silenciar_al_salir()


if __name__ == "__main__":
    main()
