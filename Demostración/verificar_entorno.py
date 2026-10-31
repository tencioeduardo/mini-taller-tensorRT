#!/usr/bin/env python3
"""
verificar_entorno.py - Fase 0 de la ruta de trazabilidad, ejecutada en la laptop de la demo.

Registra en docs/entorno.md el equipo y las versiones con que se ejecuta la demostración,
y comprueba las capacidades de LiteRT que el guion necesita:

  - Versiones de Python, ai-edge-litert, NumPy y OpenCV.
  - CPU, núcleos y memoria (contexto para interpretar las latencias).
  - Que la API CompiledModel funcione en CPU (diapositiva 5).
  - Si la GPU está disponible para CompiledModel (diapositiva 6).
  - Qué cámaras existen en /dev/video*.

Uso:  python verificar_entorno.py   (dentro del entorno .venv)
El archivo generado se versiona en el repositorio como evidencia (hallazgos H4 y G2).
"""
import datetime
import gc
import glob
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR_BASE)


def version(paquete):
    try:
        return md.version(paquete)
    except md.PackageNotFoundError:
        return "no instalado"


def leer_cpu():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linea in f:
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconocida"


def leer_ram_gib():
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for linea in f:
                if linea.startswith("MemTotal:"):
                    return int(linea.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("nan")


def leer_distribucion():
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            datos = dict(l.strip().split("=", 1) for l in f if "=" in l)
        return datos.get("PRETTY_NAME", "").strip('"')
    except OSError:
        return platform.platform()


def probar_vulkan():
    if not shutil.which("vulkaninfo"):
        return "vulkaninfo no instalado (sudo apt install vulkan-tools)"
    try:
        r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"error al ejecutar vulkaninfo: {e}"
    gpus = [l.split("=", 1)[1].strip() for l in r.stdout.splitlines() if "deviceName" in l]
    return ", ".join(gpus) if gpus else f"sin dispositivos (código {r.returncode})"


def main():
    import rps_common as rc

    print("=== Verificación del entorno de la demostración ===")
    filas = {
        "Fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Distribución": leer_distribucion(),
        "Kernel": platform.release(),
        "CPU": f"{leer_cpu()} ({os.cpu_count()} hilos lógicos)",
        "RAM total": f"{leer_ram_gib():.1f} GiB",
        "Python": platform.python_version(),
        "ai-edge-litert": version("ai-edge-litert"),
        "numpy": version("numpy"),
        "opencv-python": version("opencv-python"),
    }

    modelos = sorted(glob.glob(os.path.join(DIR_BASE, "modelos", "*.tflite")),
                     key=lambda r: (os.path.basename(r) != "rps_fp32.tflite", r))
    pruebas = {}
    if not modelos:
        pruebas["Modelos"] = "no hay modelos en modelos/ (no se pudo probar LiteRT)"
    else:
        ruta = modelos[0]
        pruebas["Modelo de prueba"] = os.path.basename(ruta)
        for api, acel in (("interpreter", "cpu"), ("compiled", "cpu"), ("compiled", "gpu")):
            clave = f"{'CompiledModel' if api == 'compiled' else 'Interpreter'} en {acel.upper()}"
            # Silenciado completo (creación, uso y destrucción del motor): LiteRT
            # emite registros nativos incluso al liberar sus aceleradores.
            with rc.silenciar_stderr_nativo():
                motor, error = rc.crear_motor(ruta, api=api, acelerador=acel)
                if motor is not None:
                    motor.calentar()
                    acelerado = motor.totalmente_acelerado
                    del motor
                gc.collect()
            if error:
                # Solo la explicación, sin el detalle nativo truncado.
                pruebas[clave] = "NO disponible: " + error.split(" Detalle:")[0].replace("GPU no disponible: ", "")
            else:
                extra = f" (totalmente acelerado: {'sí' if acelerado else 'no'})" if acel == "gpu" else ""
                pruebas[clave] = "funciona" + extra
    pruebas["Vulkan"] = probar_vulkan()
    camaras = sorted(glob.glob("/dev/video*"))
    pruebas["Cámaras"] = ", ".join(camaras) if camaras else "ninguna detectada"

    for k, v in {**filas, **pruebas}.items():
        print(f"  {k:<22} {v}")

    os.makedirs(os.path.join(DIR_BASE, "docs"), exist_ok=True)
    ruta_md = os.path.join(DIR_BASE, "docs", "entorno.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("# Entorno de la demostración\n\n")
        f.write("Generado por `verificar_entorno.py` en la laptop de la presentación. "
                "Evidencia de cierre de los hallazgos H4 (versiones) y G2 (GPU) de la ruta de trazabilidad.\n\n")
        f.write("## Equipo y versiones\n\n| Elemento | Valor |\n|---|---|\n")
        f.writelines(f"| {k} | {v} |\n" for k, v in filas.items())
        f.write("\n## Capacidades de LiteRT comprobadas\n\n| Prueba | Resultado |\n|---|---|\n")
        f.writelines(f"| {k} | {v} |\n" for k, v in pruebas.items())
        f.write("\nLas versiones exactas de todas las dependencias están en `requirements.lock.txt` "
                "(generado por `setup.sh`).\n")
    print(f"\nRegistrado en {os.path.relpath(ruta_md)}")
    rc.silenciar_al_salir()


if __name__ == "__main__":
    main()
