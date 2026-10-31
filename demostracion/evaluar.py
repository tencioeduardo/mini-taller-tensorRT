#!/usr/bin/env python3
"""
evaluar.py - Acto 2 de la demostración: medir en la laptop lo que la teoría prometió.

Para cada combinación de modelo (FP32, FP16, INT8 dinámico, INT8 completo) y API de
LiteRT (Interpreter, CompiledModel y, opcionalmente, CompiledModel en GPU) mide:

  Tamaño       Bytes del archivo .tflite                               (diapositiva 7)
  RAM modelo   Memoria (RSS) que agrega el modelo al proceso, sin       (diapositivas 2 y 7)
               contar el costo fijo de las bibliotecas de LiteRT
  Carga        Tiempo de crear el motor (leer, compilar, reservar)     (diapositiva 5)
  invoke p50   Latencia mediana de SOLO la inferencia                  (diapositiva 11)
  invoke p90   Latencia del percentil 90 (estabilidad)                 (diapositiva 11)
  Precisión    Aciertos sobre el set de prueba, con el MISMO            (diapositiva 13)
               preprocesamiento que la demo en vivo (rps_common.py)
  Coincide     % de predicciones iguales a FP32: aísla el efecto        (diapositiva 8)
  FP32         de la cuantización

Cada combinación se ejecuta en un SUBPROCESO independiente. Así la medición de RAM
de un modelo no se contamina con la de otro (la memoria liberada no siempre vuelve
al sistema operativo dentro de un mismo proceso).

Uso típico:
    python evaluar.py                     # todos los modelos de modelos/, ambas APIs
    python evaluar.py --gpu               # agrega CompiledModel en GPU (si hay driver)
    python evaluar.py --rapido            # versión corta para la demo en vivo
    python evaluar.py --modelos modelos/rps_fp32.tflite modelos/rps_int8_full.tflite

Resultados: tabla en consola + resultados/resultados.md + resultados/resultados.csv
"""
import argparse
import csv
import datetime
import glob
import json
import os
import platform
import subprocess
import sys
import time

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
ORDEN_PREFERIDO = ["rps_fp32", "rps_fp16", "rps_int8_dyn", "rps_int8_full"]
EXTENSIONES_IMG = (".jpg", ".jpeg", ".png", ".bmp")


# ==========================================================================
# Utilidades de memoria (Linux: /proc/self/status)
# ==========================================================================
def leer_memoria_kib(campo):
    """VmRSS = memoria residente actual; VmHWM = pico de memoria residente."""
    with open("/proc/self/status", encoding="ascii") as f:
        for linea in f:
            if linea.startswith(campo + ":"):
                return int(linea.split()[1])
    return 0


# ==========================================================================
# Trabajador: mide UNA combinación modelo/API/acelerador y devuelve JSON
# ==========================================================================
def listar_imagenes(dir_datos, etiquetas):
    """
    Espera datos/test/<Etiqueta>/*.jpg. El índice de clase de cada imagen es la
    posición de su carpeta en labels.txt, así el mismo set sirve para cualquier
    modelo con esas etiquetas (por ejemplo, uno de Teachable Machine).
    """
    muestras = []
    for indice, etiqueta in enumerate(etiquetas):
        carpeta = os.path.join(dir_datos, etiqueta)
        if not os.path.isdir(carpeta):
            continue
        for nombre in sorted(os.listdir(carpeta)):
            if nombre.lower().endswith(EXTENSIONES_IMG):
                muestras.append((os.path.join(carpeta, nombre), indice))
    return muestras


def percentil(valores, p):
    import numpy as np
    return float(np.percentile(valores, p)) if valores else float("nan")


def trabajador(cfg):
    """Se ejecuta en un subproceso. Imprime una única línea JSON en stdout."""
    import cv2 as cv
    import numpy as np
    import rps_common as rc

    resultado = {"modelo": os.path.splitext(os.path.basename(cfg["modelo"]))[0],
                 "api": cfg["api"], "acelerador": cfg["acelerador"]}

    # Línea base de memoria: el proceso ya tiene Python, NumPy y OpenCV cargados.
    rss_base = leer_memoria_kib("VmRSS")
    # Costo fijo del runtime: cargar las bibliotecas nativas de LiteRT, antes de
    # abrir cualquier modelo. Se separa para que "RAM modelo" refleje solo lo que
    # agrega cada .tflite (pesos, activaciones, búferes y, en CompiledModel, el
    # entorno con el registro de aceleradores).
    import importlib
    with rc.silenciar_stderr_nativo(not cfg["verbose"]):
        for modulo in ("ai_edge_litert.interpreter", "ai_edge_litert.compiled_model"):
            importlib.import_module(modulo)
    rss_runtime = leer_memoria_kib("VmRSS")

    t0 = time.perf_counter()
    motor, error = rc.crear_motor(cfg["modelo"], api=cfg["api"], hilos=cfg["hilos"],
                                  acelerador=cfg["acelerador"], verbose=cfg["verbose"])
    if motor is None:
        resultado["error"] = error
        print(json.dumps(resultado))
        return
    motor.calentar(5)                       # incluye la carga en el tiempo y la RAM
    carga_ms = (time.perf_counter() - t0) * 1000
    rss_motor = leer_memoria_kib("VmRSS")

    etiquetas = rc.cargar_etiquetas(cfg["labels"])
    muestras = listar_imagenes(cfg["datos"], etiquetas)
    if cfg["max_imagenes"]:
        # Submuestreo uniforme para conservar el balance entre clases.
        paso = max(1, len(muestras) // cfg["max_imagenes"])
        muestras = muestras[::paso][:cfg["max_imagenes"]]
    if not muestras:
        resultado["error"] = f"No hay imágenes en '{cfg['datos']}' para las etiquetas {etiquetas}."
        print(json.dumps(resultado))
        return

    # --- Latencia: SOLO ejecutar(), con una imagen real ya cargada en la entrada ---
    x = rc.preprocesar(cv.imread(muestras[0][0]), motor.ancho, motor.alto,
                       motor.dtype_entrada, cfg["norm"])
    motor.cargar_entrada(x)
    for _ in range(20):                      # calentamiento adicional (cachés, frecuencia de CPU)
        motor.ejecutar()
    tiempos = []
    for _ in range(cfg["repeticiones"]):
        t = time.perf_counter()
        motor.ejecutar()
        tiempos.append((time.perf_counter() - t) * 1000)

    # --- Precisión: preprocesamiento idéntico al de la cámara en vivo ---
    predicciones, reales, tiempos_pre = [], [], []
    for ruta, clase in muestras:
        imagen = cv.imread(ruta)
        t = time.perf_counter()
        x = rc.preprocesar(imagen, motor.ancho, motor.alto, motor.dtype_entrada, cfg["norm"])
        tiempos_pre.append((time.perf_counter() - t) * 1000)
        probs = motor.predecir(x)
        predicciones.append(int(np.argmax(probs)))
        reales.append(clase)

    n = len(etiquetas)
    confusion = [[0] * n for _ in range(n)]
    for r, p in zip(reales, predicciones):
        if r < n and p < n:
            confusion[r][p] += 1

    resultado.update({
        "etiqueta_api": motor.etiqueta_api,
        "tamano_bytes": motor.tamano_bytes,
        "ram_runtime_kib": rss_runtime - rss_base,
        "ram_modelo_kib": rss_motor - rss_runtime,
        "ram_pico_kib": leer_memoria_kib("VmHWM"),
        "carga_ms": carga_ms,
        "invoke_p50_ms": percentil(tiempos, 50),
        "invoke_p90_ms": percentil(tiempos, 90),
        "invoke_media_ms": float(np.mean(tiempos)),
        "preproceso_media_ms": float(np.mean(tiempos_pre)),
        "precision": float(np.mean(np.array(predicciones) == np.array(reales))),
        "imagenes": len(muestras),
        "predicciones": predicciones,
        "confusion": confusion,
        "totalmente_acelerado": motor.totalmente_acelerado,
    })
    print(json.dumps(resultado))


# ==========================================================================
# Coordinador: lanza los subprocesos y arma el reporte
# ==========================================================================
def descubrir_modelos(entradas):
    rutas = []
    for e in entradas:
        rutas += sorted(glob.glob(os.path.join(e, "*.tflite"))) if os.path.isdir(e) else [e]

    def clave(r):
        nombre = os.path.splitext(os.path.basename(r))[0]
        return (ORDEN_PREFERIDO.index(nombre) if nombre in ORDEN_PREFERIDO else len(ORDEN_PREFERIDO), nombre)
    return sorted(dict.fromkeys(rutas), key=clave)


def ejecutar_combinacion(cfg):
    """Lanza el trabajador en un proceso nuevo y devuelve su JSON."""
    cmd = [sys.executable, os.path.abspath(__file__), "--_trabajador", json.dumps(cfg)]
    salida = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, cwd=DIR_BASE,
                            stderr=None if cfg["verbose"] else subprocess.DEVNULL)
    lineas = [l for l in salida.stdout.strip().splitlines() if l.startswith("{")]
    if not lineas:
        return {"modelo": os.path.basename(cfg["modelo"]), "api": cfg["api"],
                "acelerador": cfg["acelerador"],
                "error": f"el subproceso terminó sin resultado (código {salida.returncode}); use --verbose"}
    return json.loads(lineas[-1])


def info_equipo():
    cpu = platform.processor() or "desconocido"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linea in f:
                if linea.startswith("model name"):
                    cpu = linea.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import importlib.metadata as md
        version_litert = md.version("ai-edge-litert")
    except Exception:  # noqa: BLE001
        version_litert = "desconocida"
    return {"fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "cpu": cpu, "nucleos": os.cpu_count(),
            "sistema": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "ai_edge_litert": version_litert}


def fmt_kib(kib):
    return f"{kib / 1024:.1f} MiB" if kib >= 1024 else f"{kib:.0f} KiB"


def armar_filas(resultados):
    """Agrega 'vs FP32' y 'Coincide FP32' comparando con FP32 en la misma API."""
    referencia = {}
    for r in resultados:
        if r["modelo"] == "rps_fp32" and "error" not in r:
            referencia[(r["api"], r["acelerador"])] = r
    filas = []
    for r in resultados:
        fila = dict(r)
        ref = referencia.get((r["api"], r["acelerador"])) or next(iter(referencia.values()), None)
        if "error" not in r and ref:
            fila["vs_fp32"] = r["tamano_bytes"] / ref["tamano_bytes"]
            iguales = sum(a == b for a, b in zip(r["predicciones"], ref["predicciones"]))
            fila["coincide_fp32"] = iguales / max(1, len(ref["predicciones"]))
        filas.append(fila)
    return filas


COLUMNAS = [("Modelo", 14), ("API", 18), ("Tamaño", 9), ("vs FP32", 8), ("RAM modelo", 11),
            ("Carga", 9), ("invoke p50", 11), ("p90", 9), ("Precisión", 10), ("Coincide FP32", 13)]


def celdas(f):
    if "error" in f:
        api = "CompiledModel" if f["api"] == "compiled" else "Interpreter"
        return [f["modelo"], f"{api}/{f['acelerador'].upper()}", "—", "—", "—", "—",
                "—", "—", "no disp.", "—"]
    return [f["modelo"], f["etiqueta_api"], fmt_kib(f["tamano_bytes"] / 1024),
            f"{f['vs_fp32']:.0%}" if "vs_fp32" in f else "—",
            fmt_kib(f["ram_modelo_kib"]), f"{f['carga_ms']:.0f} ms",
            f"{f['invoke_p50_ms']:.2f} ms", f"{f['invoke_p90_ms']:.2f} ms",
            f"{f['precision']:.1%}",
            f"{f['coincide_fp32']:.1%}" if "coincide_fp32" in f else "—"]


def imprimir_tabla(filas):
    encabezado = "".join(n.ljust(a) for n, a in COLUMNAS)
    print("\n" + encabezado)
    print("-" * len(encabezado))
    for f in filas:
        print("".join(str(c).ljust(a) for c, (_, a) in zip(celdas(f), COLUMNAS)))
    # Cada causa de error se explica una sola vez (p. ej., la GPU falla igual para todos los modelos).
    causas = {}
    for f in filas:
        if "error" in f:
            causas.setdefault(f["error"], []).append(f["modelo"])
    for causa, modelos in causas.items():
        print(f"\n[No disponible: {', '.join(modelos)}]\n  {causa}")


def guardar(filas, equipo, args, etiquetas, dir_salida):
    os.makedirs(dir_salida, exist_ok=True)

    # --- CSV: datos completos para análisis posterior ---
    campos = ["modelo", "api", "acelerador", "tamano_bytes", "vs_fp32", "ram_runtime_kib", "ram_modelo_kib", "ram_pico_kib",
              "carga_ms", "invoke_p50_ms", "invoke_p90_ms", "invoke_media_ms", "preproceso_media_ms",
              "precision", "coincide_fp32", "imagenes", "error"]
    ruta_csv = os.path.join(dir_salida, "resultados.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)

    # --- Markdown: reporte legible y versionable ---
    md = ["# Resultados de evaluación LiteRT\n",
          "| Equipo | |", "|---|---|",
          f"| Fecha | {equipo['fecha']} |", f"| CPU | {equipo['cpu']} ({equipo['nucleos']} hilos lógicos) |",
          f"| Sistema | {equipo['sistema']} |", f"| Python | {equipo['python']} |",
          f"| ai-edge-litert | {equipo['ai_edge_litert']} |",
          f"| Configuración | {args.hilos} hilos, {args.repeticiones} repeticiones de invoke, "
          f"normalización {args.norm} |", ""]
    md.append("## Tabla comparativa\n")
    md.append("| " + " | ".join(n for n, _ in COLUMNAS) + " |")
    md.append("|" + "---|" * len(COLUMNAS))
    for f in filas:
        md.append("| " + " | ".join(str(c) for c in celdas(f)) + " |")
    causas = {}
    for f in filas:
        if "error" in f:
            causas.setdefault(f["error"], []).append(f"`{f['modelo']}` ({f['api']}/{f['acelerador']})")
    if causas:
        md.append("\n**Combinaciones no disponibles:**\n")
        md += [f"- {', '.join(quienes)}: {causa}" for causa, quienes in causas.items()]

    md.append("\n## Matrices de confusión (filas = clase real, columnas = predicción)\n")
    vistos = set()
    for f in filas:
        if "error" in f or f["modelo"] in vistos:
            continue
        vistos.add(f["modelo"])
        md.append(f"**{f['modelo']}** ({f['etiqueta_api']}, {f['imagenes']} imágenes)\n")
        md.append("| | " + " | ".join(etiquetas) + " |")
        md.append("|---|" + "---|" * len(etiquetas))
        for nombre, fila in zip(etiquetas, f["confusion"]):
            md.append(f"| {nombre} | " + " | ".join(str(v) for v in fila) + " |")
        md.append("")

    md.append("## Cómo leer estos resultados\n")
    runtime = sorted(f["ram_runtime_kib"] for f in filas if "error" not in f)
    if runtime:
        md.append(f"- **Costo fijo del runtime:** cargar las bibliotecas de LiteRT agrega "
                  f"~{fmt_kib(runtime[len(runtime) // 2])} al proceso antes de abrir cualquier modelo; "
                  "no se incluye en la columna RAM modelo.")
    md.append("- **Tamaño vs RAM modelo:** la cuantización reduce el archivo hasta ~25 %, pero la RAM "
              "baja menos: también incluye activaciones, búferes y estructuras del runtime "
              "(aclaración de la diapositiva 7).")
    md.append("- **Interpreter vs CompiledModel:** CompiledModel suele usar más RAM porque, al crear su "
              "entorno, registra todos los aceleradores disponibles (CPU, GPU y NPU), aunque ejecute en CPU. "
              "A cambio, gestiona los búferes y permite cambiar de acelerador sin cambiar el código.")
    md.append("- **invoke p50/p90:** solo la inferencia, sin captura ni preprocesamiento. "
              "p90 cercano a p50 indica latencia estable.")
    md.append("- **Coincide FP32:** porcentaje de imágenes en que la variante toma la misma decisión que "
              "FP32. Aísla el efecto de la cuantización del error propio del modelo.")
    md.append("- **Precisión:** medida con el mismo preprocesamiento de la cámara en vivo "
              "(`rps_common.preprocesar`), sobre imágenes que el modelo no vio al entrenar.")
    ruta_md = os.path.join(dir_salida, "resultados.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return ruta_md, ruta_csv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelos", nargs="+", default=[os.path.join(DIR_BASE, "modelos")],
                   help="carpetas o archivos .tflite (por defecto: modelos/)")
    p.add_argument("--datos", default=os.path.join(DIR_BASE, "datos", "test"))
    p.add_argument("--labels", default=os.path.join(DIR_BASE, "modelos", "labels.txt"))
    p.add_argument("--apis", nargs="+", choices=["interpreter", "compiled"],
                   default=["interpreter", "compiled"])
    p.add_argument("--gpu", action="store_true", help="agrega CompiledModel en GPU (WebGPU/Vulkan)")
    p.add_argument("--hilos", type=int, default=4)
    p.add_argument("--repeticiones", type=int, default=200, help="repeticiones de invoke para la latencia")
    p.add_argument("--max-imagenes", type=int, default=0, help="limita las imágenes de precisión (0 = todas)")
    p.add_argument("--rapido", action="store_true",
                   help="versión corta para la demo: 90 imágenes, 100 repeticiones")
    p.add_argument("--norm", choices=["0-1", "-1-1"], default="0-1")
    p.add_argument("--salida", default=os.path.join(DIR_BASE, "resultados"))
    p.add_argument("--verbose", action="store_true", help="muestra los registros nativos de LiteRT")
    p.add_argument("--_trabajador", help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args._trabajador:
        sys.path.insert(0, DIR_BASE)
        trabajador(json.loads(args._trabajador))
        return

    if args.rapido:
        args.max_imagenes = args.max_imagenes or 90
        args.repeticiones = min(args.repeticiones, 100)

    sys.path.insert(0, DIR_BASE)
    import rps_common as rc

    modelos = descubrir_modelos(args.modelos)
    if not modelos:
        sys.exit("[ERROR] No se encontraron modelos .tflite. Ejecute el cuaderno entrenar_rps.ipynb "
                 "y descomprima artefactos_rps.zip dentro de esta carpeta.")
    if not os.path.isdir(args.datos):
        sys.exit(f"[ERROR] No existe la carpeta de prueba '{args.datos}'.")
    etiquetas = rc.cargar_etiquetas(args.labels)

    combinaciones = [(a, "cpu") for a in args.apis]
    if args.gpu:
        combinaciones.append(("compiled", "gpu"))

    equipo = info_equipo()
    print("=== Evaluación LiteRT ===")
    print(f"CPU: {equipo['cpu']} | {equipo['nucleos']} hilos | Python {equipo['python']} | "
          f"ai-edge-litert {equipo['ai_edge_litert']}")
    print(f"Modelos: {', '.join(os.path.basename(m) for m in modelos)}")
    print(f"Etiquetas: {etiquetas} | Datos: {args.datos}")

    resultados = []
    total = len(modelos) * len(combinaciones)
    for i, ruta in enumerate(modelos):
        for j, (api, acel) in enumerate(combinaciones):
            n = i * len(combinaciones) + j + 1
            nombre = os.path.splitext(os.path.basename(ruta))[0]
            print(f"  [{n}/{total}] {nombre:<14} {api:<11} {acel.upper():<4} ...", end=" ", flush=True)
            r = ejecutar_combinacion({
                "modelo": os.path.abspath(ruta), "api": api, "acelerador": acel, "hilos": args.hilos,
                "labels": os.path.abspath(args.labels), "datos": os.path.abspath(args.datos),
                "repeticiones": args.repeticiones, "max_imagenes": args.max_imagenes,
                "norm": args.norm, "verbose": args.verbose})
            print("no disponible" if "error" in r else
                  f"p50 {r['invoke_p50_ms']:.2f} ms | precisión {r['precision']:.1%}")
            resultados.append(r)

    filas = armar_filas(resultados)
    imprimir_tabla(filas)
    ruta_md, ruta_csv = guardar(filas, equipo, args, etiquetas, args.salida)
    print(f"\nReporte: {os.path.relpath(ruta_md)} | Datos: {os.path.relpath(ruta_csv)}")


if __name__ == "__main__":
    main()
