#!/usr/bin/env bash
# =============================================================================
# setup.sh - Prepara y verifica el entorno de la demostración LiteRT.
#
#   1. Verifica Python (3.10 o superior) y los paquetes del sistema.
#   2. Crea el entorno virtual .venv (requerido por PEP 668 en Ubuntu/Debian).
#   3. Instala requirements.txt y guarda las versiones exactas en requirements.lock.txt.
#   4. Valida los modelos (cabecera FlatBuffer "TFL3"), las etiquetas y el set de prueba.
#   5. Revisa la cámara y la disponibilidad de Vulkan (necesario para la GPU).
#
# NO descarga modelos: se generan con entrenar_rps.ipynb (Google Colab).
# Uso:  chmod +x setup.sh && ./setup.sh
# =============================================================================

cd "$(dirname "$0")" || exit 1

VENV_DIR=".venv"
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
ERRORES=0

ok()    { echo "   [✓] $*"; }
aviso() { echo "   [!] $*"; }
falla() { echo "   [✗] $*"; ERRORES=$((ERRORES + 1)); }

is_valid_tflite() {
    # Un .tflite es un FlatBuffer con el identificador "TFL3" en los bytes 4-7.
    [ -s "$1" ] && [ "$(head -c 8 "$1" 2>/dev/null | tail -c 4 2>/dev/null)" = "TFL3" ]
}

echo "======================================================"
echo "  Configuración del entorno - Demostración LiteRT"
echo "======================================================"

# --- 1. Python y paquetes del sistema ----------------------------------------
echo "[1/5] Python y paquetes del sistema"
if ! command -v python3 >/dev/null 2>&1; then
    falla "python3 no está instalado."; exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    falla "Se requiere Python 3.10 o superior (encontrado: $(python3 --version 2>&1))."; exit 1
fi
ok "$(python3 --version)"
if command -v dpkg >/dev/null 2>&1 && ! dpkg -s python3-venv >/dev/null 2>&1; then
    aviso "Instalando python3-venv (requiere sudo)..."
    sudo apt update && sudo apt install -y python3-venv python3-pip || exit 1
fi

# --- 2. Entorno virtual -------------------------------------------------------
echo "[2/5] Entorno virtual"
if [ ! -x "$PY" ]; then
    python3 -m venv "$VENV_DIR" || { falla "No se pudo crear $VENV_DIR"; exit 1; }
    ok "Creado $VENV_DIR"
else
    ok "Detectado $VENV_DIR"
fi

# --- 3. Dependencias ----------------------------------------------------------
echo "[3/5] Dependencias de Python (requirements.txt)"
"$PIP" install -q --upgrade pip >/dev/null 2>&1
if "$PIP" install -q -r requirements.txt; then
    "$PIP" freeze > requirements.lock.txt
    ok "Instaladas. Versiones exactas en requirements.lock.txt:"
    grep -iE '^(ai-edge-litert|numpy|opencv-python)==' requirements.lock.txt | sed 's/^/       /'
else
    falla "Falló la instalación de dependencias."; exit 1
fi
if "$PY" -c "from ai_edge_litert.compiled_model import CompiledModel" >/dev/null 2>&1; then
    ok "API CompiledModel disponible"
else
    falla "No se pudo importar ai_edge_litert.compiled_model"
fi

# --- 4. Modelos, etiquetas y set de prueba ------------------------------------
echo "[4/5] Modelos y datos"
shopt -s nullglob
MODELOS=(modelos/*.tflite)
if [ ${#MODELOS[@]} -eq 0 ]; then
    falla "No hay modelos en modelos/. Ejecute entrenar_rps.ipynb en Colab y descomprima"
    echo "       artefactos_rps.zip dentro de esta carpeta."
fi
for m in "${MODELOS[@]}"; do
    if is_valid_tflite "$m"; then
        ok "$(basename "$m") ($(du -h "$m" | cut -f1))"
    else
        falla "$(basename "$m") NO es un FlatBuffer TFLite válido. Inicio del archivo:"
        echo "       $(head -c 100 "$m" | tr '\n' ' ')"
    fi
done
if [ -f modelos/labels.txt ]; then
    ok "labels.txt: $(tr '\n' ' ' < modelos/labels.txt)"
else
    aviso "modelos/labels.txt no existe (se usará Piedra/Papel/Tijera)."
fi
[ -f modelos/manifest.json ] && ok "manifest.json presente" || aviso "manifest.json no existe."
if [ -d datos/test ]; then
    for d in datos/test/*/; do
        ok "datos/test/$(basename "$d"): $(find "$d" -maxdepth 1 -type f | wc -l) imágenes"
    done
else
    falla "No existe datos/test/ (necesario para evaluar.py)."
fi

# --- 5. Cámara y GPU ------------------------------------------------------------
echo "[5/5] Cámara y GPU"
CAMARAS=(/dev/video*)
if [ ${#CAMARAS[@]} -gt 0 ]; then
    ok "Dispositivos de video: ${CAMARAS[*]}"
    if ! id -nG | grep -qw video; then
        aviso "Su usuario no está en el grupo 'video'. Si la cámara no abre:"
        echo "       sudo usermod -aG video \$USER  (y vuelva a iniciar sesión)"
    fi
else
    aviso "No se detectó cámara. La demo usará la imagen de contingencia test_hand.jpg."
fi
[ -f test_hand.jpg ] && ok "test_hand.jpg presente" || aviso "test_hand.jpg no existe: créela con la tecla 's' durante un ensayo."
if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >/dev/null 2>&1; then
    ok "Vulkan disponible: la GPU puede probarse con --gpu"
else
    aviso "Vulkan no detectado (vulkaninfo): la GPU probablemente no estará disponible."
    echo "       Para comprobarlo: sudo apt install vulkan-tools && vulkaninfo --summary"
fi

echo "======================================================"
if [ "$ERRORES" -gt 0 ]; then
    echo "  Hay $ERRORES problema(s) marcados con [✗]. Revíselos antes de la demo."
else
    echo "  Entorno listo. Siguientes pasos:"
    echo "    source .venv/bin/activate"
    echo "    python verificar_entorno.py      # registra el equipo en docs/entorno.md"
    echo "    python evaluar.py                # Acto 2"
    echo "    python litert_rps.py             # Acto 3"
fi
echo "======================================================"
exit "$ERRORES"
