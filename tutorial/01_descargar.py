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
