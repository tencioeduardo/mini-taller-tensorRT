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
