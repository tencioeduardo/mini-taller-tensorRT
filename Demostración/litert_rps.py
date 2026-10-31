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