# flake8: noqa
# ruff: noqa
# Script de PROGRAMA: se ejecuta con exec() en un espacio de nombres compartido con los scripts de su familia (familias/champion/)
# (mediciones.ejecucion.ejecutar_familia). No es un módulo importable. MANTENIDO A MANO: este archivo es la fuente de verdad.
#
# PORTADA LITERAL del notebook original champion_minimayorista_brechas_mensual.ipynb (fuera de este paquete): todas sus celdas de código, en orden, marcadas con
# "celda N del notebook" (un traceback con número de línea de este archivo se ubica en la celda de arriba). Adaptaciones, y solo estas:
#   1) el mes de cierre sale de MES_CIERRE_OVERRIDE (lo inyecta el motor);
#   2) ids, ejecutar_grilla, _correr_semilla y el bootstrap/distribución de brechas salen de módulos compartidos (eran IDÉNTICOS a los de surtido);
#   3) ejecutar_grilla consulta BigQuery por defecto: los snapshots con hash de SQL los maneja mediciones.insumos.
# El matching congelado usa el mecanismo propio del notebook (df_matches.csv + meta_pares_congelados.json en outputs/): no necesita estado_matching.json.

from mediciones.parametros import (  # parámetros metodológicos comunes (los valores del notebook de Champion eran idénticos)
    CALIPER_REL,
    CALIPER_ULTIMO,
    CORR_UMBRAL,
    MARGEN_ERROR_OBJETIVO,
    MAX_SEMILLAS_PROBAR_EQUIVALENCIA,
    NIVELES_IC,
    NIVEL_MAX,
    N_BOOT,
    N_SEMILLAS_DEFAULT,
    OUTLIER_K,
    OUTLIER_METODO,
    OUTLIER_NIVEL,
    SEED_BOOT,
    USAR_LOG,
    VENTANA_MESES,
)

# ══════════════ celda 2 del notebook ══════════════
import os

# ── Config de paralelismo/threads — DEBE ir ANTES de importar numpy/pandas ──
# Máquina objetivo: Debian 11, Python 3.10, 32 vCPU / 128GB RAM.
# El pipeline de matching paraleliza las 50 semillas con joblib (backend='threads', ver
# celda del pipeline): pairwise_distances/linear_sum_assignment liberan el GIL, así que
# varios hilos corren en paralelo real sin el costo de serializar los DataFrames grandes.
# Limitamos los hilos internos de BLAS/OpenMP para que no compitan entre sí por los mismos
# núcleos que ya está repartiendo joblib (evita over-subscription).
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from dateutil.relativedelta import relativedelta
from sklearn.neighbors import BallTree

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('mode.copy_on_write', True)   # pandas 2.x: menos copias implícitas -> menos RAM/tiempo

# ── N_JOBS para las etapas paralelizables (matching por semilla, ver celda del pipeline) ──
N_JOBS = max(1, os.cpu_count() - 2)   # deja 2 núcleos libres para el SO / Jupyter
print(f"[PERFORMANCE] CPUs detectados: {os.cpu_count()} | N_JOBS configurado para paralelismo: {N_JOBS}")

# ══════════════ celda 3 del notebook ══════════════
# =============================================================================
# CACHÉ LOCAL DE DESCARGAS — evita volver a golpear BigQuery si el CSV ya existe
# =============================================================================
CACHE_DIR = 'data_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Columnas que se auto-parsean como fecha al leer desde caché (por nombre, no por posición)

# Columnas de código de cliente -> SIEMPRE se normalizan a string. Sin esto, una
# tabla puede volver de BigQuery/CSV como int64 y otra como object (por ceros a la
# izquierda, nulos, etc.), y cualquier merge('cod_cliente') truena con
# "ValueError: You are trying to merge on object and int64 columns".
# OJO: el filtro exige que empiece con 'cod_' (no solo que contenga 'cliente'),
# porque hay columnas como 'flg_cliente_roma' (booleana) que NO son un código
# y se romperían si se casteara a string.
# Las funciones de ids y la lectura de BigQuery son idénticas a las de surtido: viven en módulos compartidos (mediciones.metodologia.ids, mediciones.grilla).
from mediciones.metodologia.ids import es_columna_id_cliente as _es_columna_id_cliente, normalizar_ids as _normalizar_ids
from mediciones.grilla import ejecutar_grilla as _grilla


def ejecutar_grilla(query: str, nombre_cache: str, project_id: str = "acpe-dev-uc-sandbox-aa",
                     verbose: bool = True, forzar_descarga: bool = True):   # el motor usa mediciones.insumos (snapshots con hash de SQL); caché CSV del notebook desactivado
    """Consulta BigQuery con caché CSV local (ver mediciones.grilla). Lo propio de Champion: caché sin sufijo; lee los ids como texto; NO convierte fechas ni Decimal en la ruta de BigQuery."""
    return _grilla(query, nombre_cache, cache_dir=CACHE_DIR, project_id=project_id, verbose=verbose, forzar_descarga=forzar_descarga,
                   sufijo_cache=None, ids_como_texto_en_cache=True, fechas_en_consulta=False, decimales_en_consulta=False)

# ══════════════ celda 4 del notebook ══════════════
# =============================================================================
# PARÁMETROS GLOBALES — Champion Minimayorista
# =============================================================================
COSECHA_ANCLA = pd.Timestamp('2026-03-01')   # cosecha única agrupada (mes ancla)
# ── EJECUCIÓN MENSUAL: este es el ÚNICO parámetro que se mueve cada mes ───────────
MES_CIERRE    = pd.Timestamp(globals().get('MES_CIERRE_OVERRIDE', '2026-09-01'))   # último mes de sell-out incluido (SETIEMBRE)
MAX_PERIODO   = MES_CIERRE                   # alias: el resto del notebook sigue leyendo MAX_PERIODO
PERIODO_LIMITE_EXCL = MAX_PERIODO + pd.DateOffset(months=1)   # se filtra periodo < este
CATEGORIAS = ['Galletas', 'Cereales']        # SOLO estas dos (sin Inka Chips / Papas Fritas)

# Meses calendario medidos desde el ancla hasta MES_CIERRE (mar..set -> 7)
MESES_POST = ((MAX_PERIODO.year - COSECHA_ANCLA.year) * 12
              + (MAX_PERIODO.month - COSECHA_ANCLA.month) + 1)

ARCHIVO_COSECHAS  = "1. Base de clientes desagregada por mes.xlsx"
ARCHIVO_SEGMENTOS = "Base Minimayorista vf.xlsx"

# ── BASE LEGACY de clientes (lista vigente del programa, setiembre) ──────────────
# Sus segmentos (Oro/Plata/Oro +) NO se usan en el matching (taxonomía distinta a la de
# ARCHIVO_SEGMENTOS). Se usa para conciliar quién es cliente de acción.
ARCHIVO_LEGACY = "BD_Clientes_Perfectos_Snack_-_Setiembre.xlsx"
# [legacy_archivo:inicio] El nombre trae el mes (y a veces un error de tipeo: 'Perdectos'): se toma el archivo del MES DE CIERRE; si no hay, el más reciente.
import glob as _glob
_MESES_ES = {'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
             'setiembre': 9, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12}


def _mes_del_nombre(_nombre):
    _n = os.path.basename(_nombre).lower()
    return next((_m for _k, _m in _MESES_ES.items() if _k in _n), None)


_cands_leg = sorted(_glob.glob('BD_Clientes_*Snack*.xlsx'), key=os.path.getmtime)
_del_mes_leg = [_c for _c in _cands_leg if _mes_del_nombre(_c) == MES_CIERRE.month]
if _del_mes_leg:
    ARCHIVO_LEGACY = _del_mes_leg[-1]
elif _cands_leg:
    ARCHIVO_LEGACY = _cands_leg[-1]
    print(f"[LEGACY] no hay archivo del mes {MES_CIERRE:%Y-%m} en el nombre: se usa el más reciente ({ARCHIVO_LEGACY})")
print(f"[LEGACY] archivo de clientes: {ARCHIVO_LEGACY}")
# [legacy_archivo:fin]
# Cambios de código de cliente (acumulativo, una fila por cambio): ver mediciones/metodologia/equivalencias.py
ARCHIVO_EQUIVALENCIAS = "equivalencias_codigos.csv"
ARCHIVO_SEGMENTOS_MANUAL = "segmentos_manual.csv"     # opcional: segmentos de clientes que la base de minimayoristas no trae (se completa a mano, acumulativo)
# Mes de entrada que se le asigna a un cliente de la legacy que NO aparece en ninguna hoja
# mensual del Excel de cosechas (por defecto: entra como incremento del mes de cierre).
LEGACY_ENTRADA_SIN_MES = MES_CIERRE

# ── EJECUCIÓN INCREMENTAL: pares congelados ──────────────────────────────────────
RUTA_SALIDA            = os.path.join('outputs', 'Champion Minimayorista - Medición')
RUTA_PARES_CONGELADOS  = os.path.join(RUTA_SALIDA, 'df_matches.csv')        # lo escribe la celda de guardado
RUTA_META_PARES        = os.path.join(RUTA_SALIDA, 'meta_pares_congelados.json')
FORZAR_RECALCULO_TOTAL = False    # True = ignora los pares congelados y rehace TODO el matching
# Si un control de un par congelado entra después al programa (queda contaminado):
#   'alertar'       -> solo avisa (el par se mantiene)
#   'descartar_par' -> saca el par; su cliente de acción vuelve a buscar pareja este mes
POLITICA_CONTROL_CONTAMINADO = 'alertar'
MODO_INCREMENTAL = os.path.exists(RUTA_PARES_CONGELADOS) and not FORZAR_RECALCULO_TOTAL

# Guardia: una hoja de cosecha tiene cientos de clientes. Si una hoja mapeada trae miles,
# probablemente NO es una cosecha (ej. una lista de elegibles) y marcarla como acción
# arruinaría el control. Se corta con un error explícito en vez de seguir en silencio.
MAX_CLIENTES_HOJA_COSECHA = 1500

# Clientes con cosecha ANTERIOR al ancla (Ene/Feb en el Excel de cosechas) recibieron
# el piloto antes -> son controles contaminados. True = se sacan del pool de control.
# (En 1.ipynb NO se sacaban: al filtrar cosecha >= marzo quedaban como 'control elegible'.)
EXCLUIR_PREVIAS_DE_CONTROL = True

# En 1.ipynb 'ffvv_impulso' se calculaba pero el filtro (cosecha notna | impulso) no
# excluía a nadie (la cosecha siempre era notna). False = mismo comportamiento efectivo.
# True = el control solo puede venir de clientes con fuerza de venta de impulso.
SOLO_CONTROL_IMPULSO = False

print(f"Ancla: {COSECHA_ANCLA.date()} | corte (MES_CIERRE): {MAX_PERIODO.date()} | meses medidos: {MESES_POST} | "
      f"modo: {'INCREMENTAL (pares congelados)' if MODO_INCREMENTAL else 'COMPLETO (matching de todos)'}")

# ══════════════ celda 6 del notebook ══════════════
# =============================================================================
# CARGA BIGQUERY (con caché local, ver ejecutar_grilla)
# =============================================================================
_SUF = f"_{MES_CIERRE:%Y%m}"   # sufijo de caché: al avanzar el mes se vuelve a descargar

def _z10(s):
    """Código de cliente DEX: string de 10 dígitos con ceros a la izquierda (mismo formato
    que ya usaba 1.ipynb para el Excel). Evita perder los ceros al leer el CSV de caché."""
    return s.astype(str).str.strip().str.zfill(10)

sql_sell_out = """
SELECT a.periodo, a.des_zona_venta_alicorp, a.des_oficina_venta_alicorp,
 a.cod_grupo_precio_dex, a.des_negocio, a.cod_cliente_dex,
 a.nom_cliente_alicorp_actual, a.des_categoria,
 SUM(a.mnt_venta_real) AS mnt_venta_real, SUM(a.num_peso_real) AS num_peso_real
FROM `acpe-prod-dlk-shrd-gld.delivery_reporte_comercial.cr_venta_sell_out_actual` a
WHERE a.periodo >= '2024-01-01' AND a.des_negocio = 'Core'
 AND a.des_categoria IN ('Galletas','Cereales') AND a.cod_grupo_precio_dex = 'MA'
GROUP BY ALL;
"""
sell_out = ejecutar_grilla(sql_sell_out, nombre_cache='champion_sell_out' + _SUF)

sql_terr = """
SELECT DISTINCT v.id_interlocutor_origen cod_cliente_dex, id_fuerza_venta_origen
FROM `acpe-prod-dlk-shrd-gld.gld_cliente.dex_interlocutor_territorio` v
LEFT JOIN `acpe-prod-dlk-shrd-gld.gld_cliente.dex_personal_territorio` t
  ON v.id_territorio_origen=t.id_territorio_origen
WHERE v.id_territorio IS NOT NULL
"""
terr = ejecutar_grilla(sql_terr, nombre_cache='champion_terr' + _SUF)

# Coordenadas + fecha de creación (solo clientes con coordenadas válidas): se usa para la
# homologación de códigos por geolocalización (igual que 1.ipynb).
sql_cus = """
SELECT id_interlocutor_origen cod_cliente, fec_creacion,
 val_coordenada_x, val_coordenada_y
FROM `acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente`
WHERE cod_boca_salida IN ('MA') AND val_coordenada_x IS NOT NULL
 AND val_coordenada_x != "0" AND val_coordenada_y IS NOT NULL AND val_coordenada_y != "0"
"""
cus_df = ejecutar_grilla(sql_cus, nombre_cache='champion_cus' + _SUF)

# Fecha de creación de TODOS los clientes MA (sin filtro de coordenadas). La metodología
# nueva la usa para distinguir 'no existía el año pasado' (nuevo, volumen puro) de
# 'reactivado'. Con cus_df (filtrado por coordenadas) los clientes sin coordenadas
# quedarían con creación desconocida y se clasificarían mal como 'reactivado'.
sql_fec = """
SELECT id_interlocutor_origen cod_cliente, MIN(fec_creacion) fec_creacion
FROM `acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente`
WHERE cod_boca_salida IN ('MA')
GROUP BY 1
"""
fec_df = ejecutar_grilla(sql_fec, nombre_cache='champion_fec_creacion' + _SUF)

# Normalizar códigos a 10 dígitos en TODAS las tablas (mismo formato que el Excel)
sell_out['cod_cliente_dex'] = _z10(sell_out['cod_cliente_dex'])
terr['cod_cliente_dex']     = _z10(terr['cod_cliente_dex'])
cus_df['cod_cliente']       = _z10(cus_df['cod_cliente'])
fec_df['cod_cliente']       = _z10(fec_df['cod_cliente'])

cus_df['fec_creacion_mes'] = pd.to_datetime(cus_df['fec_creacion']).dt.to_period('M').dt.to_timestamp()
fec_df['fec_creacion']     = pd.to_datetime(fec_df['fec_creacion'])
print(f"sell_out: {len(sell_out):,} filas | {sell_out.cod_cliente_dex.nunique():,} clientes")

# ══════════════ celda 7 del notebook ══════════════
# =============================================================================
# EXCELS LOCALES: cosechas (base de clientes por mes) y segmentos
# =============================================================================
hojas = pd.read_excel(ARCHIVO_COSECHAS, sheet_name=None)
df_cos = pd.concat([h.assign(mes=n) for n, h in hojas.items()], ignore_index=True)
df_cos['Cliente DEX Cod'] = df_cos['Cliente DEX Cod'].astype(str).str.zfill(10)
df_cos = df_cos.rename(columns={'Cliente DEX Cod': 'cod_cliente_dex', 'mes': 'cosecha_mes'})

# 'Julio' y 'Agosto' agregados al mapa: el corte ahora llega a agosto. Si esas hojas no
# existen en el Excel no pasa nada; si existen, sus clientes entran como altas de ese mes
# (antes, al no estar en el mapa, quedaban como control elegible).
mapa_cosecha = {"Enero": "2026-01-01", "Febrero": "2026-02-01", "Marzo": "2026-03-01",
                "Abril": "2026-04-01", "Mayo": "2026-05-01", "Junio": "2026-06-01",
                "Julio": "2026-07-01", "Agosto": "2026-08-01",
                "Septiembre": "2026-09-01", "Setiembre": "2026-09-01", "Octubre": "2026-10-01",
                "Noviembre": "2026-11-01", "Diciembre": "2026-12-01"}
df_cos['cosecha_mes_ajustada'] = pd.to_datetime(df_cos['cosecha_mes'].map(mapa_cosecha), errors='coerce')

print("Hojas del Excel de cosechas y clientes por hoja:")
_n_hoja = df_cos.groupby('cosecha_mes', dropna=False)['cod_cliente_dex'].nunique()
print(_n_hoja.to_string())
print("(las hojas pueden ser ACUMULADAS: cada mes lista a los que ya estaban; por eso la entrada "
      "real del cliente se toma como el MÍNIMO mes en el que aparece)")
_grandes = [h for h in _n_hoja.index if h in mapa_cosecha and _n_hoja[h] > MAX_CLIENTES_HOJA_COSECHA]
if _grandes:
    raise ValueError(f"Hojas de cosecha con más de {MAX_CLIENTES_HOJA_COSECHA:,} clientes: {_grandes}. "
                     f"¿Son una lista de elegibles y no una cosecha? Revisar el Excel o el mapa_cosecha.")
_no = df_cos.loc[df_cos['cosecha_mes_ajustada'].isna(), 'cosecha_mes'].unique()
if len(_no):
    print(f"\n[AVISO] Hojas SIN mapa de cosecha (se tratan como control elegible, NO como acción): {list(_no)}")

seg = pd.read_excel(ARCHIVO_SEGMENTOS)
seg['customer_id'] = seg['customer_id'].astype(str).str.zfill(10)

# ── BASE LEGACY de clientes ──────────────────────────────────────────────────────
# [legacy_lectura:inicio]
legacy = pd.read_excel(ARCHIVO_LEGACY)
legacy.columns = [str(c).strip() for c in legacy.columns]          # ej. 'Código antiguo ' trae espacio final
if 'cod_cliente_dex' not in legacy.columns and 'Cliente DEX Cod' in legacy.columns:
    legacy = legacy.rename(columns={'Cliente DEX Cod': 'cod_cliente_dex'})     # formato de la base de octubre (la misma columna del Excel de cosechas)
legacy['cod_cliente_dex'] = _z10(legacy['cod_cliente_dex'])
if 'Código antiguo' in legacy.columns:
    _ant = legacy['Código antiguo'].dropna().astype('int64').astype(str).str.zfill(10)
    legacy['cod_antiguo'] = _ant.reindex(legacy.index)
else:
    legacy['cod_antiguo'] = np.nan
_dup_leg = int(legacy.duplicated('cod_cliente_dex').sum())
legacy = legacy.drop_duplicates('cod_cliente_dex')
print(f"\n[LEGACY] {len(legacy):,} clientes únicos ({_dup_leg} duplicados eliminados) | "
      f"con código antiguo: {int(legacy['cod_antiguo'].notna().sum())}")
# [legacy_lectura:fin]

# ══════════════ celda 8 del notebook ══════════════
# =============================================================================
# SELL-OUT: tipos, nombre de distribuidora, fecha de creación
# =============================================================================
sell_out['mnt_venta_real'] = sell_out['mnt_venta_real'].astype(float)
sell_out['num_peso_real']  = sell_out['num_peso_real'].astype(float)
sell_out['periodo'] = pd.to_datetime(sell_out['periodo']).dt.to_period('M').dt.to_timestamp()
sell_out['nom_cliente_alicorp_actual'] = np.where(
    sell_out['nom_cliente_alicorp_actual'] == 'PACIFICO DEX SOCIEDAD ANONIMA CERRADA',
    'PACIFICO DEX SAC', sell_out['nom_cliente_alicorp_actual'])

sell_out_fec = sell_out.merge(
    cus_df[['cod_cliente', 'fec_creacion_mes', 'val_coordenada_x', 'val_coordenada_y']],
    left_on='cod_cliente_dex', right_on='cod_cliente', how='left').drop(columns='cod_cliente')
sell_out_fec['clientes_nuevos_2026'] = (
    sell_out_fec.fec_creacion_mes >= pd.Timestamp('2026-01-01')).astype(bool)

# ══════════════ celda 9 del notebook ══════════════
# Homologacion por geolocalizacion (clientes a 0m -> codigo mas antiguo)
coords = (sell_out_fec[['cod_cliente_dex','val_coordenada_x','val_coordenada_y','clientes_nuevos_2026']]
          .drop_duplicates('cod_cliente_dex').copy())
for c in ['val_coordenada_x','val_coordenada_y']:
    coords[c] = pd.to_numeric(coords[c], errors='coerce')
coords = coords.dropna(subset=['val_coordenada_x','val_coordenada_y'])
nuevos = coords[coords['clientes_nuevos_2026']].copy()
no_nuevos = coords[~coords['clientes_nuevos_2026']].copy()
if len(nuevos) and len(no_nuevos):
    tree = BallTree(np.radians(no_nuevos[['val_coordenada_y','val_coordenada_x']].values), metric='haversine')
    d, idx = tree.query(np.radians(nuevos[['val_coordenada_y','val_coordenada_x']].values), k=1)
    nuevos['distancia_m'] = d[:,0]*6_371_000
    nuevos['cod_cliente_dex_vecino'] = no_nuevos['cod_cliente_dex'].values[idx[:,0]]
    pares_0m = nuevos[nuevos['distancia_m']==0][['cod_cliente_dex','cod_cliente_dex_vecino']].drop_duplicates()
else:
    pares_0m = pd.DataFrame(columns=['cod_cliente_dex','cod_cliente_dex_vecino'])

fecha_cliente = (sell_out_fec[['cod_cliente_dex','fec_creacion_mes']]
                 .drop_duplicates('cod_cliente_dex').set_index('cod_cliente_dex')['fec_creacion_mes'])
parent = {}
def find(x):
    parent.setdefault(x,x)
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[ra]=rb
for _,r in pares_0m.iterrows(): union(r['cod_cliente_dex'], r['cod_cliente_dex_vecino'])
grupos={}
for cod in set(pares_0m['cod_cliente_dex'])|set(pares_0m['cod_cliente_dex_vecino']):
    grupos.setdefault(find(cod),set()).add(cod)
mapa_homologacion={}
for miembros in grupos.values():
    fv={c:fecha_cliente.get(c,pd.NaT) for c in miembros}; fv={c:f for c,f in fv.items() if pd.notna(f)}
    if not fv: continue
    antiguo=min(fv,key=fv.get)
    for c in miembros:
        if c!=antiguo: mapa_homologacion[c]=antiguo
# Códigos antiguos informados en la base legacy: el historial puede estar partido entre
# código nuevo y antiguo. Se unifican al más antiguo (misma convención que la geolocalización).
_n_ant = 0
for _, _r in legacy.dropna(subset=['cod_antiguo']).iterrows():
    _nuevo, _ant = _r['cod_cliente_dex'], _r['cod_antiguo']
    if _nuevo != _ant and _nuevo not in mapa_homologacion:
        mapa_homologacion[_nuevo] = mapa_homologacion.get(_ant, _ant)
        _n_ant += 1
print(f"Homologados adicionales por 'Código antiguo' de la legacy: {_n_ant}")
# Equivalencias de códigos informadas en la tabla acumulativa (cambios de código posteriores): misma convención, al más antiguo.
from mediciones.metodologia.equivalencias import leer_equivalencias
_n_eq = 0
for _nuevo, _ant in leer_equivalencias(ARCHIVO_EQUIVALENCIAS).items():
    if _nuevo not in mapa_homologacion:
        mapa_homologacion[_nuevo] = mapa_homologacion.get(_ant, _ant)
        _n_eq += 1
print(f"Homologados adicionales por {ARCHIVO_EQUIVALENCIAS}: {_n_eq}")
sell_out_fec['cod_cliente_dex'] = sell_out_fec['cod_cliente_dex'].replace(mapa_homologacion)
print(f"Clientes homologados: {len(mapa_homologacion)}")

# [segmentos_manual:inicio] La base de minimayoristas no trae a los clientes nuevos. `segmentos_manual.csv` (customer_id, Segmento) completa SOLO a quienes la base
# no trae: si la base ya tiene al cliente, manda la base. Se puede indicar el código nuevo o el antiguo (se unifica igual que el resto).
_n_man = 0
if os.path.exists(ARCHIVO_SEGMENTOS_MANUAL):
    _man = pd.read_csv(ARCHIVO_SEGMENTOS_MANUAL, dtype=str)
    _man.columns = [str(c).strip() for c in _man.columns]
    _falta_man = [c for c in ('customer_id', 'Segmento') if c not in _man.columns]
    if _falta_man:
        raise ValueError(f"{ARCHIVO_SEGMENTOS_MANUAL}: faltan las columnas {_falta_man} (se esperan customer_id, Segmento y opcional observacion)")
    _man = _man.dropna(subset=['customer_id', 'Segmento'])
    _man['customer_id'] = _z10(_man['customer_id']).replace(mapa_homologacion)
    _man['Segmento'] = _man['Segmento'].astype(str).str.strip()
    _desconocidos = sorted(set(_man['Segmento']) - set(seg['Segmento'].dropna().astype(str)))
    if _desconocidos:
        print(f"  [AVISO] {ARCHIVO_SEGMENTOS_MANUAL}: segmentos que no existen en la base ({_desconocidos}); revisa que estén bien escritos "
              f"(los de la base son {sorted(seg['Segmento'].dropna().astype(str).unique())})")
    _ya_en_base = _man['customer_id'].isin(seg['customer_id'])
    seg = pd.concat([seg, _man.loc[~_ya_en_base, ['customer_id', 'Segmento']]], ignore_index=True)
    _n_man = int((~_ya_en_base).sum())
    print(f"Segmentos completados desde {ARCHIVO_SEGMENTOS_MANUAL}: {_n_man} (ignorados {int(_ya_en_base.sum())} que la base ya trae)")
# [segmentos_manual:fin]

# ══════════════ celda 10 del notebook ══════════════
# =============================================================================
# COSECHA ÚNICA AGRUPADA + BASE LIMPIA (formato que espera el pipeline del surtido)
# =============================================================================
# 1) Homologar códigos del Excel de cosechas (mismo mapa de la celda de homologación)
df_cos['cod_cliente_dex'] = df_cos['cod_cliente_dex'].replace(mapa_homologacion)

# 2) Entrada REAL de cada cliente de acción = primer mes de cosecha >= ancla.
#    (cosecha agrupada: TODOS comparten el ancla 2026-03-01, pero cada uno conserva
#    su mes real de entrada -> se usa para el conteo acumulativo mes a mes.)
df_trat = df_cos[(df_cos['cosecha_mes_ajustada'] >= COSECHA_ANCLA) &
                 (df_cos['cosecha_mes_ajustada'] <= MES_CIERRE)]   # hojas posteriores al cierre no cuentan
df_cosecha = (df_trat.groupby('cod_cliente_dex', as_index=False)
                     .agg(min_periodo_cosecha=('cosecha_mes_ajustada', 'min')))

# 3) Clientes con cosecha PREVIA al ancla (contaminados como control)
ids_tratados = set(df_cosecha['cod_cliente_dex'])
ids_previos = set(df_cos.loc[df_cos['cosecha_mes_ajustada'] < COSECHA_ANCLA, 'cod_cliente_dex']) - ids_tratados
print(f"Clientes con cosecha previa al ancla (Ene/Feb) que NO están en el piloto desde marzo: {len(ids_previos):,}")

# 3b) CONCILIACIÓN con la base LEGACY (lista vigente de clientes del programa)
legacy['cod_original'] = legacy['cod_cliente_dex']                  # código tal como viene en la lista (para la conciliación mensual)
legacy['cod_cliente_dex'] = legacy['cod_cliente_dex'].replace(mapa_homologacion)
ids_leg = set(legacy['cod_cliente_dex'])
solo_leg_previos = (ids_leg - ids_tratados) & ids_previos          # entraron en Ene/Feb: siguen fuera de la acción
solo_leg         = (ids_leg - ids_tratados) - ids_previos          # en la legacy y sin mes de entrada en las hojas
solo_hojas       = ids_tratados - ids_leg                          # en las hojas (>= ancla) pero NO en la legacy
# [conciliacion_mensual:inicio] Qué cambió en la lista de clientes respecto al mes anterior (solo informativo; no altera el cálculo).
# Un cambio de código no cuenta como alta ni como baja: se unifica al código antiguo antes de comparar.
try:
    _mes_ant = MES_CIERRE - pd.DateOffset(months=1)
    _arch_ant = [_c for _c in _cands_leg if _mes_del_nombre(_c) == _mes_ant.month and _c != ARCHIVO_LEGACY]
    _nom_leg = legacy.set_index('cod_cliente_dex')['Cliente DEX'].to_dict() if 'Cliente DEX' in legacy.columns else {}
    _filas_conc = []
    _altas_reales, _bajas_reales, _cambios_nuevos = set(), set(), pd.DataFrame()
    if _arch_ant:
        _prev = pd.read_excel(_arch_ant[-1])
        _prev.columns = [str(_c).strip() for _c in _prev.columns]
        if 'cod_cliente_dex' not in _prev.columns and 'Cliente DEX Cod' in _prev.columns:
            _prev = _prev.rename(columns={'Cliente DEX Cod': 'cod_cliente_dex'})
        _prev['cod_cliente_dex'] = _z10(_prev['cod_cliente_dex'])
        _prev_raw = set(_prev['cod_cliente_dex'])
        _prev['id_h'] = _prev['cod_cliente_dex'].replace(mapa_homologacion)
        _altas_reales, _bajas_reales = ids_leg - set(_prev['id_h']), set(_prev['id_h']) - ids_leg
        _cambios_nuevos = legacy[legacy['cod_antiguo'].notna() & ~legacy['cod_original'].isin(_prev_raw)]
        _nom_prev = _prev.set_index('id_h')['Cliente DEX'].to_dict() if 'Cliente DEX' in _prev.columns else {}
        _filas_conc += [{'tipo': 'alta', 'cod_cliente_dex': _c, 'nombre': _nom_leg.get(_c, ''), 'detalle': 'está en la lista de este mes y no en la del mes anterior'}
                        for _c in sorted(_altas_reales)]
        _filas_conc += [{'tipo': 'baja', 'cod_cliente_dex': _c, 'nombre': _nom_prev.get(_c, ''), 'detalle': 'estaba en la lista del mes anterior y ya no'}
                        for _c in sorted(_bajas_reales)]
        _filas_conc += [{'tipo': 'cambio_de_codigo', 'cod_cliente_dex': _r['cod_cliente_dex'], 'nombre': _r.get('Cliente DEX', ''),
                         'detalle': f"código nuevo {_r['cod_original']} -> se une al antiguo {_r['cod_antiguo']}"} for _, _r in _cambios_nuevos.iterrows()]
    _sin_seg = sorted(set(legacy['cod_cliente_dex']) - set(seg['customer_id']))
    _filas_conc += [{'tipo': 'sin_segmento', 'cod_cliente_dex': _c, 'nombre': _nom_leg.get(_c, ''),
                     'detalle': ('alta de este mes' if _c in _altas_reales else 'ya estaba en la lista') + ': agrégalo a ' + ARCHIVO_SEGMENTOS_MANUAL}
                    for _c in _sin_seg]
    print(f"\n[CONCILIACIÓN MENSUAL DE CLIENTES] {MES_CIERRE:%Y-%m} vs {_mes_ant:%Y-%m}  ({os.path.basename(ARCHIVO_LEGACY)})")
    if _arch_ant:
        print(f"  Altas: {len(_altas_reales)} | Bajas: {len(_bajas_reales)} | Códigos antiguos nuevos este mes: {len(_cambios_nuevos)}")
    else:
        print(f"  (no hay archivo de clientes de {_mes_ant:%Y-%m} en la carpeta: no se puede comparar altas y bajas)")
    print(f"  Sin segmento (quedarán 'Sin Asignar'): {len(_sin_seg)}" + (f" -> completa {ARCHIVO_SEGMENTOS_MANUAL}" if _sin_seg else ""))
    if _filas_conc:
        os.makedirs('conciliacion', exist_ok=True)
        _ruta_conc = os.path.join('conciliacion', f"clientes_{MES_CIERRE:%Y%m}.csv")
        pd.DataFrame(_filas_conc).to_csv(_ruta_conc, index=False, encoding='utf-8-sig')
        print(f"  Detalle en {_ruta_conc}")
except Exception as _e:
    print(f"  [AVISO] no se pudo armar la conciliación mensual de clientes ({type(_e).__name__}: {_e}); el cálculo sigue")
# [conciliacion_mensual:fin]
print("\n[CONCILIACIÓN LEGACY vs HOJAS MENSUALES]")
print(f"  En ambas                                  : {len(ids_leg & ids_tratados):,}")
print(f"  Solo en legacy (sin mes de entrada)       : {len(solo_leg):,}  -> entrada = {LEGACY_ENTRADA_SIN_MES.date()}")
print(f"  Solo en legacy con cosecha PREVIA al ancla: {len(solo_leg_previos):,}  -> se mantienen fuera de la acción")
print(f"  Solo en hojas (>= ancla), no están en legacy: {len(solo_hojas):,}  -> se mantienen como acción (revisar)")
if len(solo_hojas):
    print("  [AVISO] clientes en hojas y no en la legacy: puede ser baja del programa o código no homologado. "
          f"Muestra: {sorted(solo_hojas)[:10]}")
# [solo_leg_entrada:inicio]
# Entrada previa de cada cliente: lo ya PUBLICADO (cubre a todos) y, de respaldo, el estado congelado (pares y sin pareja).
if 'ENTRADAS_PUBLICADAS' not in globals():
    print("  [AVISO] el runner no pasó ENTRADAS_PUBLICADAS: runner.py y frontera.py están DESACTUALIZADOS respecto a champion.py (se actualizan juntos). "
          "Sin eso, las altas que solo vienen en la lista de clientes pueden perder sus meses anteriores y aparecer como deriva.")
_publicadas = {_a: pd.Timestamp(_f) for _a, _f in (globals().get('ENTRADAS_PUBLICADAS') or {}).items()}
_entradas_prev = dict(_publicadas)
if MODO_INCREMENTAL:
    for _ruta, _col in ((RUTA_PARES_CONGELADOS, 'min_per_ac'), (os.path.join(RUTA_SALIDA, 'df_sin_pareja.csv'), 'min_periodo_cosecha')):
        if os.path.exists(_ruta):
            _d = pd.read_csv(_ruta, dtype={'cod_accion': str})
            if _col in _d.columns:
                for _a, _f in zip(_d['cod_accion'], pd.to_datetime(_d[_col], errors='coerce')):
                    if pd.notna(_f):
                        _entradas_prev.setdefault(_a, _f)
if len(solo_leg):
    # Las altas que solo vienen en la legacy (no tienen hoja mensual) entran en el mes de cierre. Pero si el cliente YA tiene entrada (publicada, o en el
    # estado congelado), la conserva: reasignarlo al mes de cierre le quitaría sus meses anteriores y lo sacaría de sus pares.
    _ids_leg = sorted(solo_leg)
    _entrada_leg = [min(_entradas_prev.get(_c, LEGACY_ENTRADA_SIN_MES), LEGACY_ENTRADA_SIN_MES) for _c in _ids_leg]
    _n_cons = sum(1 for _c in _ids_leg if _c in _entradas_prev)
    df_cosecha = pd.concat([df_cosecha, pd.DataFrame({'cod_cliente_dex': _ids_leg, 'min_periodo_cosecha': _entrada_leg})], ignore_index=True)
    ids_tratados = ids_tratados | solo_leg
    print(f"  -> se agregaron {len(solo_leg):,} clientes de la legacy: {len(solo_leg) - _n_cons:,} nuevos con entrada {LEGACY_ENTRADA_SIN_MES.date()} "
          f"y {_n_cons:,} que ya tenían entrada (publicada o congelada) y la conservan")
# BAJAS que vinieron solo por la lista: un cliente ya PUBLICADO como acción que dejó de estar en la lista y en las hojas sigue siendo acción (igual que las bajas de las hojas):
# si no, desaparece del universo y sus filas publicadas ya no se recalculan. Se ignoran los que entraron después del cierre y los que hoy se unifican a otro código.
_bajas_pub = {_c: _f for _c, _f in _publicadas.items()
              if _f <= MES_CIERRE and _c not in ids_tratados and mapa_homologacion.get(_c, _c) not in ids_tratados and _c not in ids_previos}
if _bajas_pub:
    df_cosecha = pd.concat([df_cosecha, pd.DataFrame({'cod_cliente_dex': sorted(_bajas_pub), 'min_periodo_cosecha': [_bajas_pub[_c] for _c in sorted(_bajas_pub)]})],
                           ignore_index=True)
    ids_tratados = ids_tratados | set(_bajas_pub)
    print(f"  [BAJAS] {len(_bajas_pub):,} clientes publicados que ya no figuran en la lista ni en las hojas se mantienen como acción con su entrada publicada: "
          f"{sorted(_bajas_pub)[:10]}")
# [solo_leg_entrada:fin]

sell_out_tag = (sell_out_fec
    .merge(df_cosecha, on='cod_cliente_dex', how='left')
    .merge(seg.rename(columns={'customer_id': 'cod_cliente_dex'})[['cod_cliente_dex', 'Segmento']]
              .drop_duplicates('cod_cliente_dex'), on='cod_cliente_dex', how='left'))

# cosecha_agrupada = ancla para TODO cliente de acción; NaT para control
sell_out_tag['cosecha_agrupada'] = sell_out_tag['min_periodo_cosecha'].where(
    sell_out_tag['min_periodo_cosecha'].isna(), COSECHA_ANCLA)
sell_out_tag['Segmento'] = sell_out_tag['Segmento'].fillna('Sin Asignar')

terr_impu = terr[terr.id_fuerza_venta_origen.isin(['IMPU', 'MAYOR', 'MA_IMP'])]
sell_out_tag['ffvv_impulso'] = sell_out_tag.cod_cliente_dex.isin(terr_impu.cod_cliente_dex.unique()).astype(bool)

# 4) Filtros del pool de control (opcionales, ver parámetros)
es_accion = sell_out_tag['cosecha_agrupada'].notna()
if EXCLUIR_PREVIAS_DE_CONTROL and ids_previos:
    n_prev = sell_out_tag.loc[sell_out_tag.cod_cliente_dex.isin(ids_previos), 'cod_cliente_dex'].nunique()
    sell_out_tag = sell_out_tag[~sell_out_tag.cod_cliente_dex.isin(ids_previos)]
    print(f"[CONTROL] {n_prev:,} clientes con cosecha previa fuera del pool de control")
    es_accion = sell_out_tag['cosecha_agrupada'].notna()
if SOLO_CONTROL_IMPULSO:
    sell_out_tag = sell_out_tag[es_accion | sell_out_tag['ffvv_impulso']]
    print("[CONTROL] pool restringido a fuerza de venta de impulso")

# 5) Corte al MES_CIERRE y compras positivas (igual criterio que 1.ipynb)
sell_out_tag = sell_out_tag[(sell_out_tag['periodo'] < PERIODO_LIMITE_EXCL) &
                            (sell_out_tag['mnt_venta_real'] > 0) & (sell_out_tag['num_peso_real'] > 0)]

# 6) Base limpia: 1 fila por cliente × categoría × periodo (atributos = fila de mayor peso)
_orden = sell_out_tag.sort_values('num_peso_real', ascending=False)
base_seg_limpio = (_orden.groupby(['cod_cliente_dex', 'des_categoria', 'periodo'], as_index=False)
    .agg(peso=('num_peso_real', 'sum'), monto=('mnt_venta_real', 'sum'),
         des_oficina=('des_oficina_venta_alicorp', 'first'),
         nom_compania=('nom_cliente_alicorp_actual', 'first'),
         des_segmento_transaccional=('Segmento', 'first'),
         des_zona=('des_zona_venta_alicorp', 'first'),
         min_periodo_cosecha=('min_periodo_cosecha', 'first'),
         cosecha_agrupada=('cosecha_agrupada', 'first'),
         ffvv_impulso=('ffvv_impulso', 'first'))
    .rename(columns={'cod_cliente_dex': 'cod_cliente'}))
base_seg_limpio['periodo'] = pd.to_datetime(base_seg_limpio['periodo'])
base_seg_limpio['min_periodo_cosecha'] = pd.to_datetime(base_seg_limpio['min_periodo_cosecha'])
base_seg_limpio['cosecha_agrupada'] = pd.to_datetime(base_seg_limpio['cosecha_agrupada'])
del _orden

print(f"\nbase_seg_limpio: {len(base_seg_limpio):,} filas | {base_seg_limpio.cod_cliente.nunique():,} clientes")
print(f"Periodos: {base_seg_limpio.periodo.min().date()} .. {base_seg_limpio.periodo.max().date()}")

# ══════════════ celda 11 del notebook ══════════════
# =============================================================================
# CHEQUEOS DE LA BASE (acción por mes real de entrada, duplicados)
# =============================================================================
acc = base_seg_limpio.dropna(subset=['cosecha_agrupada'])
print("Clientes de ACCIÓN con ventas en Galletas/Cereales, por mes real de entrada:")
print(acc.groupby('min_periodo_cosecha')['cod_cliente'].nunique().to_string())
print(f"Total acción: {acc.cod_cliente.nunique():,} | control candidato: "
      f"{base_seg_limpio.cod_cliente.nunique() - acc.cod_cliente.nunique():,}")
print("\nAcción por categoría:")
print(acc.groupby('des_categoria')['cod_cliente'].nunique().to_string())

dups = base_seg_limpio.duplicated(['periodo', 'cod_cliente', 'des_categoria']).sum()
assert dups == 0, f"Hay {dups} filas duplicadas cliente×categoría×periodo"
assert set(base_seg_limpio.des_categoria.unique()) <= set(CATEGORIAS), "Categoría fuera de Galletas/Cereales"
assert base_seg_limpio.periodo.max() <= MAX_PERIODO, "Hay periodos posteriores al corte"

# ══════════════ celda 12 del notebook ══════════════
# =============================================================================
# EMBUDO ACUMULADO GA vs GC, por categoría y mes (lámina "Embudo acumulado de implementación")
# =============================================================================
# GA (acción, acumulado) = clientes cuyo mes REAL de entrada <= mes.
# GC (control, disponible) = universo elegible de la categoría MENOS lo ya consumido por GA
#   acumulado (misma definición del PPT del surtido: GA + GC = universo, constante).
#   - universo   = clientes MA con ventas en la categoría (2024 -> corte), incluye a los de acción
#   - variante impulso = mismo universo restringido a fuerza de venta de impulso
# Además: ratio GC/GA (¿el control alcanza para sustentar el matching?).
meses_emb = pd.date_range(COSECHA_ANCLA, MAX_PERIODO, freq='MS')
_ent_excel = df_cosecha.set_index('cod_cliente_dex')['min_periodo_cosecha']   # TODOS los tratados (con o sin ventas)

filas_emb = []
for cat in CATEGORIAS:
    _b = base_seg_limpio[base_seg_limpio.des_categoria == cat]
    cli_c = _b.groupby('cod_cliente').agg(entrada=('min_periodo_cosecha', 'first'),
                                          impulso=('ffvv_impulso', 'first'))
    universo, universo_imp = len(cli_c), int(cli_c['impulso'].sum())
    prev_ga = 0
    for m in meses_emb:
        es_ga = cli_c['entrada'] <= m
        ga = int(es_ga.sum())
        ga_imp = int((es_ga & cli_c['impulso']).sum())
        filas_emb.append({
            'categoria': cat, 'mes': m, 'mes_label': m.strftime('%b-%y'),
            'GA_acumulado': ga, 'GA_nuevos_del_mes': ga - prev_ga,
            'GA_excel_total_acumulado': int((_ent_excel <= m).sum()),
            'GC_disponible': universo - ga,
            'GC_disponible_impulso': universo_imp - ga_imp,
            'universo_categoria': universo, 'universo_impulso': universo_imp,
            'pct_GA_del_universo': round(100 * ga / universo, 1) if universo else np.nan,
            'ratio_GC_GA': round((universo - ga) / ga, 2) if ga else np.nan})
        prev_ga = ga
df_embudo_acum = pd.DataFrame(filas_emb)

print("EMBUDO ACUMULADO GA vs GC (control disponible = universo - GA acumulado)")
for cat in CATEGORIAS:
    t = df_embudo_acum[df_embudo_acum.categoria == cat].set_index('mes_label')
    print(f"\n{cat}  (universo {int(t.universo_categoria.iloc[0]):,} | impulso {int(t.universo_impulso.iloc[0]):,})")
    print(t[['GA_acumulado', 'GA_nuevos_del_mes', 'GC_disponible', 'GC_disponible_impulso', 'ratio_GC_GA']].T.to_string())

# ══════════════ celda 13 del notebook ══════════════
# =============================================================================
# SEGMENTO DE TODOS LOS CLIENTES IMPLEMENTADOS — referencia (sin filtro de ventas)
# =============================================================================
# A diferencia de "Distribución por segmento (pool limpio)" -- que ya excluye clientes
# sin compra, sin recurrencia >=6 meses o con outliers -- esto es el universo COMPLETO
# de clientes que recibieron el piloto según el Excel de cosechas (df_cosecha), sin
# ningún filtro de ventas ni de categoría. Responde directamente: "de los X clientes
# implementados a este mes, ¿cuántos son Premium / Plus / Alto / Base?".
_impl = df_cosecha.merge(
    seg.rename(columns={'customer_id': 'cod_cliente_dex'})[['cod_cliente_dex', 'Segmento']].drop_duplicates('cod_cliente_dex'),
    on='cod_cliente_dex', how='left')
_impl['Segmento'] = _impl['Segmento'].fillna('Sin Asignar')

filas_impl_seg = []
for m in pd.date_range(COSECHA_ANCLA, MAX_PERIODO, freq='MS'):
    sub = _impl[_impl.min_periodo_cosecha <= m]
    cnt = sub['Segmento'].value_counts()
    pct = (100 * cnt / cnt.sum()).round(1)
    for seg_ in cnt.index:
        filas_impl_seg.append({'mes': m, 'mes_label': m.strftime('%b-%y'), 'segmento': seg_,
                               'n_clientes': int(cnt[seg_]), 'pct': float(pct[seg_])})
df_dist_seg_implementados = pd.DataFrame(filas_impl_seg)

print("SEGMENTO DE TODOS LOS CLIENTES IMPLEMENTADOS (referencia, independiente de ventas/categoría)")
piv_n = df_dist_seg_implementados.pivot_table(index='segmento', columns='mes_label', values='n_clientes', fill_value=0)
piv_p = df_dist_seg_implementados.pivot_table(index='segmento', columns='mes_label', values='pct', fill_value=0.0)
orden_mes = pd.date_range(COSECHA_ANCLA, MAX_PERIODO, freq='MS').strftime('%b-%y')
piv_n, piv_p = piv_n.reindex(columns=orden_mes), piv_p.reindex(columns=orden_mes)
tabla = piv_n.astype(int).astype(str) + ' (' + piv_p.round(1).astype(str) + '%)'
print(tabla.to_string())
print(f"\nTotal implementados a {orden_mes[-1]}: {int(piv_n[orden_mes[-1]].sum()):,} clientes")

# Chequeo: el total de este mes debe cuadrar con GA_excel_total_acumulado del embudo
_tot_chk = int(piv_n[orden_mes[-1]].sum())
assert _tot_chk == int(df_embudo_acum[df_embudo_acum.mes == MAX_PERIODO].GA_excel_total_acumulado.iloc[0]), \
    "el total de clientes implementados no cuadra con el embudo acumulado"
print("OK  total implementados cuadra con GA_excel_total_acumulado del embudo")

# ══════════════ celda 16 del notebook ══════════════
# =============================================================================
# TAMAÑO DE MUESTRA MÍNIMO — reemplaza los umbrales fijos (n_min=30, MIN_CT_OFICINA=30)
# que no tenían relación con el tamaño real de la población. Fórmula estándar de
# muestreo para una proporción (p=0.5, caso conservador) CON corrección por
# población finita, porque acá la "población" (AC_limpio, CT_limpio, etc.) es
# conocida y relativamente chica -> ignorar la corrección sobreestima el n necesario.
# =============================================================================
from mediciones.metodologia.muestra import (CONFIANZA_OBJETIVO, Z_CONF, n_minimo_muestra, margen_error_real, n_minimo_requerido)   # compartido con la familia de cosechas (antes copiado aquí, idéntico)

# ══════════════ celda 17 del notebook ══════════════
# =============================================================================
# PROYECTO ROMA — PIPELINE DE MATCHING v4 (corregido)
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
from sklearn.metrics import pairwise_distances
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import json, os

pd.set_option('display.width', 250)

# ─────────────────────────────────────────────────────────────────────────────
# TEST FORMAL DE TENDENCIAS PARALELAS (pre-cosecha) — ANCOVA CLASICO, por celda
# cosecha×categoría. Version 2: a pedido del usuario, se reemplaza el F-test de
# Wald sobre el modelo clusterizado (trayectoria mes a mes completa) por un
# ANCOVA clasico de 2 puntos, SIN ajuste por cliente -- decision 2026-09-17.
#
# Diseño: el pre-periodo (12 meses) se parte en 2 mitades. 'pre' = promedio
# log1p de la PRIMERA mitad (linea base, mas lejos de la cosecha). 'post' =
# promedio log1p de la SEGUNDA mitad (mas cerca de la cosecha). Cada cliente
# (AC o CT) aporta UNA sola fila (pre, post) -> a diferencia de la version
# anterior (12 filas por cliente, correlacionadas entre si), aqui no hace
# falta clusterizar: cada observacion es independiente. Eso SI habilita usar
# la tabla ANCOVA clasica (statsmodels.stats.anova.anova_lm) sin el problema
# de p-valor artificialmente bajo que tenia con datos repetidos por cliente.
#
# Se ajusta post ~ grupo * pre (ANCOVA con interaccion) y se prueba el termino
# de interaccion grupo:pre con la tabla ANCOVA de Tipo II.
# H0 (supuesto de homogeneidad de pendientes de ANCOVA): no hay interaccion
# grupo:pre -> el crecimiento de linea-base a segunda-mitad es igual para AC
# y CT -> tendencias paralelas.
#
# *** Trade-off del nuevo diseño (documentado, no oculto): resumir 12 meses en
# 2 puntos pierde informacion de la forma de la trayectoria mes a mes (un
# quiebre puntual a mitad de periodo puede no reflejarse en el promedio de
# cada mitad). A cambio, el modelo es mas simple y evita clusterizar. ***
#
# *** Sigue pendiente confirmar con negocio: UMBRAL_PARALELISMO=0.10 (mismo
# umbral que tenia el test viejo, pre-TOST) reintroduce a proposito la
# critica original -- con mas clientes, mas poder para rechazar por
# diferencias sin relevancia de negocio. ***
# ─────────────────────────────────────────────────────────────────────────────
UMBRAL_PARALELISMO = 0.10   # no rechazar H0 con holgura -> exigimos p > 0.10, no solo > 0.05

def test_ancova_paralelismo(cod_accion, cod_control, ac_df, ct_df, cols_m):
    """
    ANCOVA clasico -- un cliente = una fila (sin ajuste por cliente, no hace
    falta clusterizar). Resume el pre-periodo en 2 puntos: 'pre' = promedio
    log1p de la PRIMERA mitad de meses (linea base), 'post' = promedio log1p
    de la SEGUNDA mitad (mas cerca de la cosecha). Ajusta post ~ grupo*pre y
    prueba la interaccion grupo:pre con la tabla ANCOVA clasica (Tipo II) --
    H0: no hay interaccion -> el crecimiento de linea-base a segunda-mitad es
    igual entre AC y CT -> tendencias paralelas.
    Devuelve (f_stat, p_valor, ok).
    """
    n = len(cols_m)
    mitad = n // 2
    cols_pre, cols_post = cols_m[:mitad], cols_m[mitad:]

    filas = []
    for cid in cod_accion:
        pre = np.log1p(ac_df.loc[cid, cols_pre].to_numpy(float)).mean()
        post = np.log1p(ac_df.loc[cid, cols_post].to_numpy(float)).mean()
        filas.append((f"AC_{cid}", 1, pre, post))
    for cid in cod_control:
        pre = np.log1p(ct_df.loc[cid, cols_pre].to_numpy(float)).mean()
        post = np.log1p(ct_df.loc[cid, cols_post].to_numpy(float)).mean()
        filas.append((f"CT_{cid}", 0, pre, post))
    dfl = pd.DataFrame(filas, columns=['cliente', 'grupo', 'pre', 'post'])
    try:
        mod = smf.ols('post ~ grupo * pre', data=dfl).fit()   # OLS clasico, SIN cluster
        tabla = anova_lm(mod, typ=2)                          # tabla ANCOVA clasica (valida: 1 fila/cliente)
        f_stat = float(tabla.loc['grupo:pre', 'F'])
        p_valor = float(tabla.loc['grupo:pre', 'PR(>F)'])
        ok = bool(pd.notna(p_valor) and p_valor > UMBRAL_PARALELISMO)
        return f_stat, p_valor, ok
    except Exception:
        return np.nan, np.nan, False


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# HELPERS PARA LA EJECUCIÓN INCREMENTAL (evaluación sobre el conjunto ACUMULADO de pares)
# ─────────────────────────────────────────────────────────────────────────────
def metricas_trayectoria_pares(cod_a, cod_c, ac_df, ct_df, cols_m):
    """Mismas métricas de comparabilidad que _correr_semilla (corr de variaciones, ratio_vol,
    pendiente relativa de la brecha), pero sobre un conjunto de pares dado."""
    t_ac = ac_df.loc[cod_a, cols_m].mean().to_numpy()
    t_ct = ct_df.loc[cod_c, cols_m].mean().to_numpy()
    d_ac, d_ct = np.diff(t_ac), np.diff(t_ct)
    corr  = (np.corrcoef(d_ac, d_ct)[0, 1] if d_ac.std() > 0 and d_ct.std() > 0 else np.nan)
    ratio = d_ac.std() / d_ct.std() if d_ct.std() > 0 else np.inf
    brecha = t_ac - t_ct
    pend_rel = (linregress(np.arange(len(brecha)), brecha).slope / t_ac.mean()
                if t_ac.mean() > 0 else np.inf)
    ok = bool(pd.notna(corr) and corr >= CORR_UMBRAL and RATIO_MIN <= ratio <= RATIO_MAX)
    return {'corr': round(corr, 3) if pd.notna(corr) else np.nan, 'ratio_vol': round(ratio, 3),
            'pend_rel': round(pend_rel, 5), 'valida': ok, '_tac': t_ac, '_tct': t_ct}


def evaluar_celda_acumulada(pares_cong, pares_nuevos, ac_all, ct_all, cols_m, traj=None):
    """Comparabilidad (corr/ratio/pendiente + ANCOVA) sobre TODOS los pares de la celda:
    congelados + nuevos del mes. Los pares cuyo cliente ya no está en la población limpia
    (p. ej. cambió un umbral de outlier) se excluyen de la evaluación y se avisa.
    'traj' = trayectorias de TODOS los clientes de la ventana (sin filtros de limpieza): así un par congelado
    se sigue evaluando aunque el cliente haya caído de la población limpia por un umbral que se mueve con el pool."""
    ac_df = traj if traj is not None else ac_all
    ct_df = traj if traj is not None else ct_all
    cod_a = list(pares_cong['cod_accion']) + (list(pares_nuevos['cod_accion']) if pares_nuevos is not None and len(pares_nuevos) else [])
    cod_c = list(pares_cong['cod_control']) + (list(pares_nuevos['cod_control']) if pares_nuevos is not None and len(pares_nuevos) else [])
    mask = [(a in ac_df.index) and (c in ct_df.index) for a, c in zip(cod_a, cod_c)]
    if not all(mask):
        print(f"  [AVISO] {len(mask) - sum(mask)} par(es) acumulados ya no están en la población limpia "
              f"-> se excluyen solo de la evaluación de comparabilidad")
    cod_a = [a for a, m in zip(cod_a, mask) if m]
    cod_c = [c for c, m in zip(cod_c, mask) if m]
    assert len(cod_a) > 0, "evaluar_celda_acumulada: sin pares para evaluar"
    ev = metricas_trayectoria_pares(cod_a, cod_c, ac_df, ct_df, cols_m)
    f_stat, p_val, ok = test_ancova_paralelismo(cod_a, cod_c, ac_df, ct_df, cols_m)
    ev.update({'pares': len(cod_a), 'controles_unicos': len(set(cod_c)),
               'max_reuso_observado': int(pd.Series(cod_c).value_counts().max()),
               'n_muestra': len(set(cod_a)),
               'cobertura_pct': round(100 * len(set(cod_a)) / len(ac_all), 1),
               'ancova_f': f_stat, 'ancova_p': p_val, 'paralelismo_ok': ok, 'cod_a': cod_a})
    return ev


def _mejor_acumulado(ev, p_nuevos, pend, ac_all):
    """Arma el dict 'best' con la misma forma que usa el resto del pipeline, pero describiendo el
    conjunto ACUMULADO. '_p' / '_pend' siguen siendo SOLO lo nuevo de este mes."""
    return {'seed': 1, 'n_muestra': ev['n_muestra'], 'pares': ev['pares'],
            'cobertura_pct': ev['cobertura_pct'], 'corr': ev['corr'], 'ratio_vol': ev['ratio_vol'],
            'pend_rel': ev['pend_rel'], 'valida': ev['valida'],
            'controles_unicos': ev['controles_unicos'], 'max_reuso_observado': ev['max_reuso_observado'],
            '_p': p_nuevos, '_pend': pend, '_tac': ev['_tac'], '_tct': ev['_tct'],
            '_ac': ac_all.loc[ac_all.index.intersection(list(dict.fromkeys(ev['cod_a'])))]}


def _fila_resumen(cosecha, cat, ac_all, ct_all, best, min_meses_celda, nivel_estrato,
                  ancova_f, ancova_p, paralelismo_ok, n_val, semillas_probadas, modo, pares_nuevos):
    return {
        'cosecha': cosecha, 'grupo': GRUPO_COSECHA.get(cosecha, '?'), 'categoria': cat,
        'AC_limpio': len(ac_all), 'CT_limpio': len(ct_all),
        'AC_muestra': best['n_muestra'], 'seed': best['seed'], 'pares': best['pares'],
        'controles_unicos': best.get('controles_unicos', best['pares']),
        'max_reuso_observado': best.get('max_reuso_observado', 1),
        'min_meses_usado': min_meses_celda,
        'cobertura_pct': best['cobertura_pct'], 'corr': best['corr'],
        'ratio_vol': best['ratio_vol'], 'pend_rel': best['pend_rel'],
        'valida': best['valida'], 'nivel_estrato': nivel_estrato,
        'ancova_f': round(ancova_f, 4) if pd.notna(ancova_f) else np.nan,
        'ancova_pval': round(ancova_p, 4) if pd.notna(ancova_p) else np.nan,
        'umbral_paralelismo': UMBRAL_PARALELISMO,
        'semillas_validas': n_val, 'semillas_probadas_paralelismo': semillas_probadas,
        'paralelismo_ok': paralelismo_ok,
        'modo_ejecucion': modo, 'pares_nuevos_mes': pares_nuevos}


# ─────────────────────────────────────────────────────────────────────────────
# MATCHING PARA UNA SEMILLA — extraído a función para poder paralelizar (joblib)
# Cada llamada es independiente entre sí (misma ac/ct, distinta submuestra por seed)
# -> ideal para ejecutar en paralelo entre los N_JOBS núcleos disponibles.
# ─────────────────────────────────────────────────────────────────────────────
from mediciones.nucleo.matching import _correr_semilla   # idéntica a la de surtido


# ─────────────────────────────────────────────────────────────────────────────
# PASO 0 — PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────
# Cosecha ÚNICA agrupada (ancla marzo). Los clientes que entran en abr/may/jun comparten
# el ancla pero conservan su mes real de entrada (min_periodo_cosecha) -> conteo acumulativo.
COSECHAS   = [COSECHA_ANCLA]
CATEGORIAS = ['Galletas', 'Cereales']

# Etiquetas de negocio para las cosechas agrupadas (mismo mapeo de la celda 22)
GRUPO_COSECHA = {COSECHA_ANCLA: 'CHAMPION MM (Mar + altas Abr-Jun, acumulativo)'}

# MIN_MESES por categoría — PARAMETRIZABLE (dict). Corrida de comparación con
# Cereales=6 (igual que Galletas), para aislar el efecto de esa relajación del
# resto de cambios. El valor acordado con el negocio (Cereales=4) queda
# documentado en las instrucciones del proyecto para cuando se reactive.
MIN_MESES_POR_CATEGORIA = {'Galletas': 8, 'Cereales': 4}
MIN_MESES_DEFAULT = 6
EXCLUIR_NUEVOS = True

DROP_FILAS_NEG  = True
EXIGIR_POSITIVO = True


# Muestreo — depende SOLO de la población de acción (FIX 2)
MUESTREAR     = True
FRAC_MUESTREO = 0.60
ESTRATOS_FULL = ['min_periodo_cosecha', 'des_oficina',
                 'nom_compania', 'des_segmento_transaccional']
MIN_POR_CELDA = 5             # mediana de celda bajo esto -> baja nivel de cascada

# Matching — sin min_periodo_cosecha (los CT no lo tienen)
MATCH_FULL     = ['des_oficina', 'nom_compania', 'des_segmento_transaccional']

W_NIVEL  = 0.5                # cercanía (default)
W_FORMA  = 0.5                # paralelismo (default)

# ── Cascada de candidatas para el test de paralelismo (ANOVA) ───────────────
# Antes limitaba cuántas semillas se testeaban formalmente con TOST (más caro
# que el resto de la selección). El ANCOVA sobre 2 puntos por cliente
# tiene un costo similar -> se mantiene el mismo tope por razones de tiempo
# de cómputo, no por necesidad estadística del nuevo test.

RATIO_MIN, RATIO_MAX = 0.70, 1.40

# ── Overrides puntuales — REACTIVADOS: G1-Cereales y G2-Galletas fallan
# 'valida' por ratio_vol fuera de rango (0.568 y 1.706 vs [0.70,1.40]) pese a
# tener corr razonable (0.769 y 0.758) -- mismo síntoma que Junio-Galletas ya
# documentado en las instrucciones del proyecto (ratio_vol fuera de rango con
# población balanceada). Se les da más peso a la forma de la trayectoria
# (W_FORMA=0.7, en vez de matchear casi solo por nivel/estrato) y más semillas
# (150 en vez de 50) para que la búsqueda tenga más chances de encontrar un
# control cuya volatilidad pre-cosecha se parezca más a la de acción.
# Champion: sin overrides de entrada. Se activan por celda solo si el embudo/valida los pide
# (ej. ratio_vol fuera de rango con población balanceada -> W_FORMA=0.7 y 150 semillas).
PESOS_OVERRIDE = {}
N_SEMILLAS_OVERRIDE = {}

# ── Reuso de control 1:N — DESACTIVADO en esta corrida de comparación ──
# 1:1 estricto en TODAS las cosechas (comportamiento original de v6), para aislar
# el efecto del resto de cambios sobre el resultado. El reuso 1:N en Grupo 01
# (justificado por escasez real de control, 96-98% del CT ya en uso en 1:1) queda
# documentado en las instrucciones del proyecto para cuando se reactive.
MAX_REUSO_CONTROL_POR_COSECHA = {}
MAX_REUSO_CONTROL_DEFAULT = 1        # 1:1 estricto (comportamiento original)

COL_TRATADO    = 'cosecha_agrupada'
GRAFICAR       = True

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — ROL A NIVEL CLIENTE
# ─────────────────────────────────────────────────────────────────────────────
df = base_seg_limpio.copy()
df['periodo']          = pd.to_datetime(df['periodo'])
df['cosecha_agrupada'] = pd.to_datetime(df['cosecha_agrupada'])
if 'min_periodo_cosecha' in df.columns:
    df['min_periodo_cosecha'] = pd.to_datetime(df['min_periodo_cosecha'])

cli_cosecha  = df.dropna(subset=[COL_TRATADO]).groupby('cod_cliente')[COL_TRATADO].min()
TRATADOS_ALL = set(cli_cosecha.index)

# FIX 1: "nuevo" = primera transacción REAL en toda la base
cli_primera_compra = df.groupby('cod_cliente')['periodo'].min()
print(f"[COBERTURA] La base arranca en {df['periodo'].min().date()}. "
      f"Un cliente que ya compraba antes de esa fecha se marcará como nuevo por error.")

cli_min_per = (df.dropna(subset=['min_periodo_cosecha'])
                 .groupby('cod_cliente')['min_periodo_cosecha'].min()
               if 'min_periodo_cosecha' in df.columns
               else pd.Series(dtype='datetime64[ns]'))

print(f"Clientes en base : {df['cod_cliente'].nunique():,}")
print(f"Acciones (todas) : {len(TRATADOS_ALL):,}")
print(f"Candidatos espejo: {df['cod_cliente'].nunique() - len(TRATADOS_ALL):,}")

ct_ids = set(df['cod_cliente'].unique()) - TRATADOS_ALL
ct_con_min = len(set(cli_min_per.index) & ct_ids)
print(f"\n[CHEQUEO] Controles con min_periodo_cosecha: {ct_con_min:,} de {len(ct_ids):,}")
print("  -> es atributo de ACCIÓN: sirve como ESTRATO, no como llave de matching."
      if ct_con_min < len(ct_ids) * 0.5 else
      "  -> los CT sí lo tienen: puedes moverlo a MATCH_FULL.")

print("\n[ESTRUCTURA] min_periodo_cosecha por cosecha_agrupada:")
print(df.dropna(subset=[COL_TRATADO]).groupby(COL_TRATADO)['min_periodo_cosecha']
        .agg(n_valores='nunique',
             valores=lambda s: [str(x.date()) for x in sorted(s.dropna().unique())])
        .to_string())

# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN MENSUAL — PARES CONGELADOS
# ─────────────────────────────────────────────────────────────────────────────
_meta_actual = {'ancla': str(COSECHA_ANCLA.date()), 'ventana_meses': VENTANA_MESES,
                'min_meses': MIN_MESES_POR_CATEGORIA, 'min_meses_default': MIN_MESES_DEFAULT,
                'excluir_nuevos': EXCLUIR_NUEVOS, 'match_full': MATCH_FULL,
                'caliper_rel': CALIPER_REL, 'caliper_ultimo': CALIPER_ULTIMO, 'nivel_max': NIVEL_MAX,
                'outlier': [OUTLIER_METODO, OUTLIER_K, OUTLIER_NIVEL], 'w_nivel': W_NIVEL,
                'w_forma': W_FORMA, 'usar_log': USAR_LOG, 'categorias': CATEGORIAS}
_meta_actual = json.loads(json.dumps(_meta_actual))     # normaliza tipos para comparar con el JSON guardado

df_congelados = pd.DataFrame()
df_sin_cong   = pd.DataFrame()
if MODO_INCREMENTAL:
    _prev = pd.read_csv(RUTA_PARES_CONGELADOS, dtype={'cod_accion': str, 'cod_control': str},
                        parse_dates=['cosecha', 'min_per_ac'])
    if os.path.exists(RUTA_META_PARES):
        _meta_prev = json.load(open(RUTA_META_PARES, encoding='utf-8'))
        _dif = {k: (_meta_prev[k], v) for k, v in _meta_actual.items() if k in _meta_prev and _meta_prev[k] != v}
        if _dif:
            raise ValueError("Los parámetros del matching cambiaron respecto a los pares congelados "
                             f"(anterior, actual): {_dif}. Mezclar criterios distintos en un mismo acumulado no es "
                             "comparable -> revertir el parámetro o correr con FORZAR_RECALCULO_TOTAL=True.")
    else:
        print("[AVISO] no hay meta_pares_congelados.json: no se pudo verificar que los parámetros del "
              "matching sean los de la corrida que generó df_matches.csv (se guarda al final de esta corrida).")
    _entrada_prev = _prev['min_per_ac'].fillna(_prev['cosecha'])
    # Congelado = par cuyo cliente de acción entró ANTES del mes de cierre. Los que entran en MES_CIERRE
    # se vuelven a emparejar -> correr dos veces el mismo mes da el mismo resultado.
    df_congelados = _prev[_entrada_prev < MES_CIERRE].copy()
    print(f"\n[CONGELADOS] {len(df_congelados):,} pares de {len(_prev):,} en {RUTA_PARES_CONGELADOS} "
          f"(entrada < {MES_CIERRE.date()}); el resto se re-empareja este mes")
    _ruta_sp = os.path.join(RUTA_SALIDA, 'df_sin_pareja.csv')
    if os.path.exists(_ruta_sp):
        _sp = pd.read_csv(_ruta_sp, dtype={'cod_accion': str}, parse_dates=['cosecha', 'min_periodo_cosecha'])
        df_sin_cong = _sp[_sp['min_periodo_cosecha'].fillna(_sp['cosecha']) < MES_CIERRE].copy()
    _acc_fuera = df_congelados[~df_congelados['cod_accion'].isin(TRATADOS_ALL)]
    if len(_acc_fuera):
        print(f"  [AVISO] {len(_acc_fuera):,} pares congelados cuyo cliente de acción ya no figura como tratado "
              f"(¿baja de la base?). Se mantienen en el acumulado.")
    _cont = df_congelados[df_congelados['cod_control'].isin(TRATADOS_ALL)]
    if len(_cont):
        print(f"  [ALERTA] {len(_cont):,} pares congelados cuyo CONTROL entró después al programa (contaminado):")
        print(_cont.groupby(['des_categoria']).size().rename('pares').to_string())
        if POLITICA_CONTROL_CONTAMINADO == 'descartar_par':
            df_congelados = df_congelados.drop(index=_cont.index)
            print("  -> POLITICA 'descartar_par': se sacan esos pares y su acción vuelve a buscar pareja.")
        else:
            print("  -> POLITICA 'alertar': el par se mantiene (el DiD de esos controles queda sesgado hacia 0 "
                  "desde su mes de entrada). Cambiar POLITICA_CONTROL_CONTAMINADO si se quiere descartarlos.")

# Cascada de matching (FIX 4)
NIVELES = [[len(MATCH_FULL) - k + 1, MATCH_FULL[:k], CALIPER_REL]
           for k in range(len(MATCH_FULL), 0, -1)][:NIVEL_MAX]
NIVELES[-1][2] = CALIPER_ULTIMO

# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
matches_out, sinpar_out, funnel_out, resumen_out, semillas_out, pools_out, viabilidad_out = [], [], [], [], [], {}, []

for cosecha in COSECHAS:
    for cat in CATEGORIAS:

        ini   = cosecha - relativedelta(months=VENTANA_MESES)
        fin   = cosecha - relativedelta(months=1)
        meses = pd.date_range(ini, fin, freq='MS')

        print("\n" + "=" * 110)
        print(f"COSECHA {cosecha.date()} [{GRUPO_COSECHA.get(cosecha, '?')}]  |  {cat}  |  "
              f"ventana {ini.date()} .. {fin.date()}")
        print("=" * 110)

        min_meses_celda = MIN_MESES_POR_CATEGORIA.get(cat, MIN_MESES_DEFAULT)
        w_nivel_celda = PESOS_OVERRIDE.get((cosecha, cat), {}).get('W_NIVEL', W_NIVEL)
        w_forma_celda = PESOS_OVERRIDE.get((cosecha, cat), {}).get('W_FORMA', W_FORMA)
        n_semillas_celda = N_SEMILLAS_OVERRIDE.get((cosecha, cat), N_SEMILLAS_DEFAULT)
        semillas_celda = list(range(1, n_semillas_celda + 1))
        max_reuso_celda = MAX_REUSO_CONTROL_POR_COSECHA.get(cosecha, MAX_REUSO_CONTROL_DEFAULT)
        if (cosecha, cat) in PESOS_OVERRIDE or (cosecha, cat) in N_SEMILLAS_OVERRIDE:
            print(f"  [OVERRIDE] W_NIVEL={w_nivel_celda} W_FORMA={w_forma_celda} "
                  f"N_SEMILLAS={n_semillas_celda} (celda con override específico)")
        if max_reuso_celda > 1:
            print(f"  [REUSO CONTROL] max_reuso_celda={max_reuso_celda} "
                  f"(un control puede matchear hasta {max_reuso_celda} veces en esta cosecha)")

        w = df[(df['periodo'] >= ini) & (df['periodo'] <= fin) &
               (df['des_categoria'] == cat)].copy()
        if w.empty:
            print("  [SKIP] sin datos"); continue

        n_neg = (w['peso'] < 0).sum()
        if DROP_FILAS_NEG and n_neg:
            print(f"  [NOTA] {n_neg:,} filas peso<0 eliminadas "
                  f"({w.loc[w.peso < 0, 'peso'].sum():,.1f} kg). Compra BRUTA "
                  f"-> misma regla en el post-período.")
            w = w[w['peso'] >= 0]

        wp  = w[w['peso'] > 0]
        agg = (w.groupby('cod_cliente').agg(peso_12m=('peso', 'sum'))
                 .join(wp.groupby('cod_cliente').agg(n_meses=('periodo', 'nunique')))
                 .fillna({'n_meses': 0}))

        tray = (w.pivot_table(index='cod_cliente', columns='periodo',
                              values='peso', aggfunc='sum')
                  .reindex(columns=meses).fillna(0.0).clip(lower=0))
        cols_m = [f"m{i:02d}" for i in range(len(meses))]
        tray.columns = cols_m

        attrs = (w.sort_values('periodo').groupby('cod_cliente').tail(1)
                   .set_index('cod_cliente')[['des_oficina', 'nom_compania',
                                              'des_segmento_transaccional']])

        pool = agg.join(tray).join(attrs)
        pool['min_periodo_cosecha'] = cli_min_per.reindex(pool.index)

        es_ac = pool.index.map(lambda i: cli_cosecha.get(i, pd.NaT) == cosecha)
        pool['rol'] = np.where(es_ac, 'AC',
                        np.where(pool.index.isin(TRATADOS_ALL), 'EXCL', 'CT'))

        # ── EMBUDO ───────────────────────────────────────────────────────────
        fn = []
        _snap = lambda e, p: fn.append({'etapa': e,
                                        'AC': int((p.rol == 'AC').sum()),
                                        'CT': int((p.rol == 'CT').sum())})
        _snap('0. Con datos en ventana', pool)
        pool = pool[pool['rol'] != 'EXCL'];  _snap('1. Sin acción de otra cosecha', pool)
        if EXCLUIR_NUEVOS:
            nuevos = pool['n_meses'] == 0    # no compró en los 12 meses pre-cosecha
            pool = pool[~nuevos]
        _snap('2. Sin clientes nuevos', pool)
        if EXIGIR_POSITIVO:
            pool = pool[pool['peso_12m'] > 0]
        _snap('3. Con volumen > 0', pool)
        pool = pool[pool['n_meses'] >= min_meses_celda]
        _snap(f'4. Recurrencia >= {min_meses_celda}', pool)

        if OUTLIER_METODO != 'ninguno' and len(pool):
            v = np.log1p(pool['peso_12m']) if OUTLIER_METODO == 'iqr_log' else pool['peso_12m']
            tmp = pool[OUTLIER_NIVEL].copy(); tmp['_v'] = v.values
            if OUTLIER_METODO in ('iqr_log', 'iqr_raw'):
                q1 = tmp.groupby(OUTLIER_NIVEL)['_v'].transform(lambda s: s.quantile(.25))
                q3 = tmp.groupby(OUTLIER_NIVEL)['_v'].transform(lambda s: s.quantile(.75))
                lo, hi = q1 - OUTLIER_K * (q3 - q1), q3 + OUTLIER_K * (q3 - q1)
            else:
                hi = tmp.groupby(OUTLIER_NIVEL)['_v'].transform(lambda s: s.quantile(.99))
                lo = pd.Series(-np.inf, index=tmp.index)
            out = (tmp['_v'] < lo) | (tmp['_v'] > hi)
            print("\n  Tasa de outlier por oficina (pool combinado AC+CT, en log):")
            print(pd.DataFrame({
                'n_pool'   : pool.groupby('des_oficina').size(),
                'n_outlier': out.groupby(pool['des_oficina']).sum(),
                'pct'      : (out.groupby(pool['des_oficina']).mean() * 100).round(1),
                'lim_sup_kg': (np.expm1(hi.groupby(pool['des_oficina']).first())
                               if OUTLIER_METODO == 'iqr_log'
                               else hi.groupby(pool['des_oficina']).first()).round(1),
            }).to_string().replace('\n', '\n  '))
            pool = pool[~out]
        _snap('5. Sin outliers', pool)

        fdf = pd.DataFrame(fn)
        fdf['AC_pct'] = (100 * fdf.AC / max(1, fdf.AC.iloc[0])).round(1)
        fdf['CT_pct'] = (100 * fdf.CT / max(1, fdf.CT.iloc[0])).round(1)
        print("\n  EMBUDO DE LIMPIEZA")
        print(fdf.to_string(index=False).replace('\n', '\n  '))
        for r in fn:
            funnel_out.append({'cosecha': cosecha, 'categoria': cat, **r})

        ac = pool[pool.rol == 'AC'].copy()
        ct = pool[pool.rol == 'CT'].copy()
        ac_all, ct_all = ac, ct          # población limpia COMPLETA de la celda (acumulada)
        if len(ac_all) == 0 or len(ct_all) == 0:
            print("  [SKIP] celda sin acción o sin control"); continue

        viab = pd.DataFrame({'AC': ac_all.groupby('des_oficina').size(),
                             'CT': ct_all.groupby('des_oficina').size()}).fillna(0).astype(int)
        viab['ratio_CT_AC'] = (viab.CT / viab.AC.replace(0, np.nan)).round(3)
        # Viabilidad = MIN_CT_OFICINA=30 fijo, IGUAL que v6 (piso fijo, no dinámico).
        # n_min_requerido (fórmula con corrección por población finita) se deja como
        # columna INFORMATIVA adicional -- no decide 'viable', solo da contexto extra
        # de qué tan lejos/cerca está esa oficina del estándar 5%/95% del proyecto.
        MIN_CT_OFICINA = 30
        viab['n_min_requerido'] = [n_minimo_requerido(a, c_) for a, c_ in zip(viab.AC, viab.CT)]
        viab['viable'] = viab.CT >= MIN_CT_OFICINA
        print("\n  VIABILIDAD POR OFICINA")
        print(viab.to_string().replace('\n', '\n  '))
        for ofi, r in viab[~viab.viable].iterrows():
            print(f"  [NO VIABLE] {ofi}: {r.CT} controles para {r.AC:,} acciones. "
                  f"La muestra sí se toma; el problema saldrá en cobertura de matching.")

        # Guardamos la viabilidad de TODAS las oficinas (viables y no viables) —
        # esto es lo que sustenta, más adelante, si se puede reportar la brecha
        # partida por oficina o si hay que decir explícitamente "sin control suficiente".
        for ofi, r in viab.iterrows():
            viabilidad_out.append({
                'cosecha': cosecha, 'grupo': GRUPO_COSECHA.get(cosecha, '?'), 'categoria': cat,
                'des_oficina': ofi,
                'AC': int(r.AC), 'CT': int(r.CT), 'n_min_requerido': int(r.n_min_requerido),
                'ratio_CT_AC': float(r.ratio_CT_AC) if pd.notna(r.ratio_CT_AC) else np.nan,
                'viable': bool(r.viable)})

        # ── INCREMENTAL: solo se empareja lo que no está en un par congelado ─────────────
        cong_cc = (df_congelados[(df_congelados['cosecha'] == cosecha) & (df_congelados['des_categoria'] == cat)]
                   if MODO_INCREMENTAL and len(df_congelados) else pd.DataFrame())
        es_inc_celda = len(cong_cc) > 0
        if es_inc_celda:
            # Solo se empareja a quienes ENTRAN en MES_CIERRE. Quien entró antes y no tiene par (porque no cayó
            # en la muestra o no encontró control) NO se re-intenta: es parte del histórico congelado.
            _es_nuevo = ac_all['min_periodo_cosecha'] >= MES_CIERRE
            ac = ac_all[_es_nuevo & ~ac_all.index.isin(cong_cc['cod_accion'])]
            ct = ct_all[~ct_all.index.isin(cong_cc['cod_control'])]
            print(f"  [INCREMENTAL] {len(cong_cc):,} pares congelados | AC que entran en {MES_CIERRE.date()} "
                  f"(a emparejar): {len(ac):,} | CT libres (no usados en pares congelados): {len(ct):,}")
            if len(ac) == 0 or len(ct) == 0:
                print("  [INCREMENTAL] nada que emparejar este mes -> se mantienen los pares congelados "
                      "y se re-evalúa la comparabilidad del acumulado")
                for a_id in ac.index:       # si hay acción pero no control libre, queda sin pareja
                    sinpar_out.append({
                        'cosecha': cosecha, 'des_categoria': cat, 'cod_accion': a_id,
                        'des_oficina': ac.at[a_id, 'des_oficina'], 'nom_compania': ac.at[a_id, 'nom_compania'],
                        'des_segmento_transaccional': ac.at[a_id, 'des_segmento_transaccional'],
                        'min_periodo_cosecha': ac.at[a_id, 'min_periodo_cosecha'],
                        'peso_12m': ac.at[a_id, 'peso_12m'], 'n_meses': ac.at[a_id, 'n_meses']})
                ev = evaluar_celda_acumulada(cong_cc, None, ac_all, ct_all, cols_m, traj=tray)
                best = _mejor_acumulado(ev, pd.DataFrame(), list(ac.index), ac_all)
                resumen_out.append(_fila_resumen(cosecha, cat, ac_all, ct_all, best, min_meses_celda,
                                                 np.nan, ev['ancova_f'], ev['ancova_p'], ev['paralelismo_ok'],
                                                 0, 0, 'incremental', 0))
                print(f"  [ACUMULADO] pares={best['pares']:,} corr={best['corr']} ratio_vol={best['ratio_vol']} "
                      f"valida={best['valida']} | ANCOVA p={ev['ancova_p']:.4f} paralelismo_ok={ev['paralelismo_ok']}")
                pools_out[(cosecha, cat)] = {'ac': ac_all, 'ct': ct_all, 'meses': meses,
                                             'ac_s': best['_ac'], 't_ac': best['_tac'],
                                             't_ct': best['_tct'], 'best': best}
                continue

        # ── CASCADA DE ESTRATOS (FIX 5) ──────────────────────────────────────
        estratos_usar, nivel_estrato = ESTRATOS_FULL[:], 1
        for k in range(len(ESTRATOS_FULL), 0, -1):
            cand = ESTRATOS_FULL[:k]
            tam = ac.groupby([ac[c].astype(str) for c in cand]).size()
            estratos_usar = cand
            nivel_estrato = len(ESTRATOS_FULL) - k + 1
            if len(tam) and tam.median() >= MIN_POR_CELDA:
                break
        tam = ac.groupby([ac[c].astype(str) for c in estratos_usar]).size()
        print(f"\n  ESTRATOS (nivel {nivel_estrato}): {' × '.join(estratos_usar)}")
        print(f"    celdas={len(tam)} | mediana={tam.median():.0f} | "
              f"celdas<{MIN_POR_CELDA}: {(tam < MIN_POR_CELDA).sum()}")

        # ── ESPACIO DE MATCHING ──────────────────────────────────────────────
        f_log = (lambda M: np.log1p(M)) if USAR_LOG else (lambda M: M)
        Lac, Lct = f_log(ac[cols_m].to_numpy(float)), f_log(ct[cols_m].to_numpy(float))
        Fac, Fct = np.diff(Lac, axis=1), np.diff(Lct, axis=1)
        Lac_df, Lct_df = pd.DataFrame(Lac, index=ac.index), pd.DataFrame(Lct, index=ct.index)
        Fac_df, Fct_df = pd.DataFrame(Fac, index=ac.index), pd.DataFrame(Fct, index=ct.index)

        _r = np.random.default_rng(0)
        sa = _r.choice(len(ac), min(600, len(ac)), replace=False)
        sc = _r.choice(len(ct), min(600, len(ct)), replace=False)
        MED_N = float(np.median(pairwise_distances(Lac[sa], Lct[sc]))) or 1.0
        MED_F = float(np.median(pairwise_distances(Fac[sa], Fct[sc]))) or 1.0

        # ── LOOP DE SEMILLAS — PARALELIZADO CON JOBLIB (N_JOBS núcleos) ────────
        # Cada semilla es independiente -> se reparte entre los núcleos disponibles.
        # backend='threads' (no procesos): pairwise_distances/linear_sum_assignment son
        # C/Fortran y liberan el GIL, y así evitamos el costo de volver a serializar
        # ac/ct/Lac_df/... (varios miles de filas) en cada una de las 50 semillas.
        # Medido: ~15x más rápido que 'processes' para este tamaño de pool.
        # ── PRECOMPUTO por celda (cosecha x categoria) — una vez, no una vez por semilla ──
        # ac_por_oficina: el agrupamiento por oficina no depende del seed -> se arma
        # una vez y se comparte entre threads (memoria compartida, solo lectura, seguro).
        # ka_full/kc_full: la clave de estrato de cada cliente (ej. "oficina|compania|
        # segmento") tampoco depende del seed -> se calcula una vez por combinacion de
        # columnas usada en NIVELES, en vez de recalcularla con astype+join en cada una
        # de las N semillas (para celdas 1:1, el lado control se repetia identico).
        ac_por_oficina_pre = dict(list(ac.groupby('des_oficina')))
        claves_unicas = []
        for _, claves_n, _ in NIVELES:
            if claves_n not in claves_unicas:
                claves_unicas.append(claves_n)
        ka_full_pre = {tuple(cl): ac[cl].astype(str).agg('|'.join, axis=1) for cl in claves_unicas}
        kc_full_pre = {tuple(cl): ct[cl].astype(str).agg('|'.join, axis=1) for cl in claves_unicas}

        print(f"\n  ITERANDO {n_semillas_celda} SEMILLAS  (paralelo, N_JOBS={N_JOBS}, backend=threads)")

        resultados_semilla = Parallel(n_jobs=N_JOBS, prefer='threads')(
            delayed(_correr_semilla)(
                seed, ac, ct, cols_m, estratos_usar, Lac_df, Lct_df, Fac_df, Fct_df,
                MED_N, MED_F, NIVELES, (MUESTREAR and not es_inc_celda), FRAC_MUESTREO,
                w_nivel_celda, w_forma_celda, CORR_UMBRAL, RATIO_MIN, RATIO_MAX,
                max_reuso=max_reuso_celda,
                ac_por_oficina=ac_por_oficina_pre,
                ka_full_por_claves=ka_full_pre, kc_full_por_claves=kc_full_pre)
            for seed in ([1] if es_inc_celda else semillas_celda))   # incremental: sin muestreo -> 1 pasada
        resultados_semilla = [r for r in resultados_semilla if r is not None]

        if not resultados_semilla:
            if es_inc_celda:
                print("  [INCREMENTAL] ningún cliente nuevo encontró pareja -> quedan sin pareja; "
                      "se mantiene el acumulado congelado")
                resultados_semilla = [{'seed': 1, 'n_muestra': len(ac), 'pares': 0, 'cobertura_pct': 0.0,
                                       'corr': np.nan, 'ratio_vol': np.nan, 'pend_rel': np.nan,
                                       'dist_med': np.nan, 'valida': False, 'controles_unicos': 0,
                                       'max_reuso_observado': 0,
                                       '_p': pd.DataFrame(columns=['nivel_match', 'ronda_reuso', 'cod_accion',
                                                                    'cod_control', 'dist_total', 'dist_nivel',
                                                                    'dist_forma', 'peso_pob']),
                                       '_ac': ac, '_pend': list(ac.index), '_tac': None, '_tct': None}]
            else:
                print("  [SKIP] ninguna semilla produjo pares"); continue

        rs_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                              for r in resultados_semilla])
        print(rs_df.to_string(index=False).replace('\n', '\n  '))
        for r in rs_df.to_dict('records'):
            semillas_out.append({'cosecha': cosecha, 'categoria': cat, **r})

        val = [r for r in resultados_semilla if r['valida']]

        # ── BÚSQUEDA DE LA MEJOR SEMILLA QUE TAMBIÉN SEA PARALELA (ANCOVA) ──────
        # AJUSTE (Champion): el ANCOVA se corre SIEMPRE, incluso si NINGUNA semilla
        # cumple corr/ratio_vol (val vacío) -- antes, en ese caso, el ANCOVA nunca se
        # calculaba y paralelismo_ok quedaba en False "por defecto", no por resultado
        # de un test. Ahora se buscan candidatas en 'val' si hay, y si no, en TODAS las
        # semillas (resultados_semilla), siempre ordenadas por pend_rel (mejor proxy
        # primero) y probadas en cascada hasta encontrar la primera paralela.
        # 'valida' (corr/ratio_vol) queda igual de informativa que antes -- solo deja
        # de decidir si el ANCOVA se corre o no.
        if es_inc_celda:
            # Comparabilidad sobre el ACUMULADO (congelados + nuevos), no sobre el incremento:
            # con pocos clientes nuevos el ANCOVA / corr / ratio_vol no tendrían poder ni sentido.
            _best_inc = resultados_semilla[0]
            ev = evaluar_celda_acumulada(cong_cc, _best_inc['_p'], ac_all, ct_all, cols_m, traj=tray)
            ancova_f, ancova_p, paralelismo_ok = ev['ancova_f'], ev['ancova_p'], ev['paralelismo_ok']
            semillas_probadas = 0
            best = _mejor_acumulado(ev, _best_inc['_p'], _best_inc['_pend'], ac_all)
            print(f"\n  >> INCREMENTAL | pares nuevos={len(_best_inc['_p']):,} | sin pareja={len(_best_inc['_pend']):,} | "
                  f"ACUMULADO: pares={best['pares']:,} corr={best['corr']} ratio={best['ratio_vol']} "
                  f"pend_rel={best['pend_rel']} cobertura={best['cobertura_pct']}% "
                  f"| ANCOVA F={ancova_f:.3f}, p={ancova_p:.4f} -> "
                  f"{'PARALELA' if paralelismo_ok else 'NO PARALELA'} (umbral>{UMBRAL_PARALELISMO})")
        else:
            ancova_f = ancova_p = np.nan
            paralelismo_ok = False
            semillas_probadas = 0
            best = None

            candidatas = sorted(val if val else resultados_semilla, key=lambda r: abs(r['pend_rel']))
            tag_val = "" if val else " [ninguna semilla es 'valida' por corr/ratio_vol -- ANCOVA corrido igual]"
            for cand in candidatas[:MAX_SEMILLAS_PROBAR_EQUIVALENCIA]:
                semillas_probadas += 1
                f_stat, p_valor, ok = test_ancova_paralelismo(
                    cand['_p'].cod_accion, cand['_p'].cod_control, ac, ct, cols_m)
                if ok:
                    best = cand
                    ancova_f, ancova_p, paralelismo_ok = f_stat, p_valor, ok
                    print(f"\n  >> SEMILLA {best['seed']} | corr={best['corr']} "
                          f"ratio={best['ratio_vol']} pend_rel={best['pend_rel']} "
                          f"pares={best['pares']:,} cobertura={best['cobertura_pct']}% "
                          f"| PARALELA tras probar {semillas_probadas} semilla(s) "
                          f"(F={f_stat:.3f}, p={p_valor:.4f}, umbral>{UMBRAL_PARALELISMO})" + tag_val)
                    break
            if best is None:
                # Ninguna de las candidatas probadas fue paralela -> nos quedamos con la
                # mejor por pend_rel (igual que antes), pero documentado como NO paralela,
                # no se reporta como si hubiera pasado.
                best = candidatas[0]
                ancova_f, ancova_p, paralelismo_ok = test_ancova_paralelismo(
                    best['_p'].cod_accion, best['_p'].cod_control, ac, ct, cols_m)
                print(f"\n  >> SEMILLA {best['seed']} | corr={best['corr']} "
                      f"ratio={best['ratio_vol']} pend_rel={best['pend_rel']} "
                      f"pares={best['pares']:,} cobertura={best['cobertura_pct']}% "
                      f"| [NO PARALELA] ninguna de las {semillas_probadas} semilla(s) "
                      f"probadas (de {len(candidatas)} candidatas) superó el umbral "
                      f"p>{UMBRAL_PARALELISMO} -> se documenta, no se descarta la celda." + tag_val)

        if best.get('controles_unicos', best['pares']) < best['pares']:
            print(f"  [REUSO CONTROL EN USO] {best['pares']:,} pares usan solo "
                  f"{best['controles_unicos']:,} controles únicos (máx. reuso observado: "
                  f"{best['max_reuso_observado']}x) -> clusterizar SE por cod_control en el DiD.")

        for r in best['_p'].to_dict('records'):
            a_id, c_id = r['cod_accion'], r['cod_control']
            matches_out.append({
                'cosecha': cosecha, 'des_categoria': cat, 'seed': best['seed'],
                'celda_valida': best['valida'], **r,
                'ofi_ac': ac.at[a_id, 'des_oficina'],  'ofi_ct': ct.at[c_id, 'des_oficina'],
                'comp_ac': ac.at[a_id, 'nom_compania'], 'comp_ct': ct.at[c_id, 'nom_compania'],
                'seg_ac': ac.at[a_id, 'des_segmento_transaccional'],
                'seg_ct': ct.at[c_id, 'des_segmento_transaccional'],
                'min_per_ac': ac.at[a_id, 'min_periodo_cosecha'],
                'peso_ac': ac.at[a_id, 'peso_12m'],   'peso_ct': ct.at[c_id, 'peso_12m'],
                'nmeses_ac': ac.at[a_id, 'n_meses'],  'nmeses_ct': ct.at[c_id, 'n_meses']})

        for a_id in best['_pend']:
            sinpar_out.append({
                'cosecha': cosecha, 'des_categoria': cat, 'cod_accion': a_id,
                'des_oficina': ac.at[a_id, 'des_oficina'],
                'nom_compania': ac.at[a_id, 'nom_compania'],
                'des_segmento_transaccional': ac.at[a_id, 'des_segmento_transaccional'],
                'min_periodo_cosecha': ac.at[a_id, 'min_periodo_cosecha'],
                'peso_12m': ac.at[a_id, 'peso_12m'], 'n_meses': ac.at[a_id, 'n_meses']})

        # ── RESULTADO DE PARALELISMO (ANOVA) sobre la semilla finalmente elegida ──
        # Ya se calculó dentro de la búsqueda de arriba (ancova_f/ancova_p/ok) -- no
        # se vuelve a correr la regresión, solo se deja registrado para el resumen.
        print(f"\n  [TEST ANCOVA PARALELISMO] H0 (homogeneidad de pendientes): no hay interacción grupo:pre | "
              f"F={ancova_f:.3f}, p={ancova_p:.4f}" if pd.notna(ancova_p) else "  [TEST ANCOVA] no calculable")
        print(f"    -> {'OK: no se rechaza H0, tendencias paralelas' if paralelismo_ok else '[ALERTA] se rechaza H0: tendencias NO paralelas, revisar celda'} "
              f"(umbral p>{UMBRAL_PARALELISMO}, semillas válidas probadas: {semillas_probadas} de {len(val)})")

        resumen_out.append(_fila_resumen(cosecha, cat, ac_all, ct_all, best, min_meses_celda, nivel_estrato,
                                         ancova_f, ancova_p, paralelismo_ok, len(val), semillas_probadas,
                                         'incremental' if es_inc_celda else 'completo', len(best['_p'])))

        pools_out[(cosecha, cat)] = {'ac': ac_all, 'ct': ct_all, 'meses': meses,
                                     'ac_s': best['_ac'], 't_ac': best['_tac'],
                                     't_ct': best['_tct'], 'best': best}

df_matches     = pd.DataFrame(matches_out)
df_sin_pareja  = pd.DataFrame(sinpar_out)
df_funnel      = pd.DataFrame(funnel_out)
df_resumen     = pd.DataFrame(resumen_out)
# AJUSTE (Champion): reportable = SOLO paralelismo_ok (test ANCOVA formal, p > umbral).
# 'valida' (corr >= 0.55 y ratio_vol en 0.70-1.40) queda como columna INFORMATIVA en
# df_resumen, ya no es un gate para reportable. Motivo: en Galletas 'valida' es False
# por diferencia de ESCALA entre acción y control (ver distribución por segmento más
# abajo), no por falta de paralelismo -- y el criterio pedido es el ANCOVA únicamente.
# Advertencia que se mantiene igual: una celda con corr/ratio_vol fuera de rango puede
# pasar paralelismo_ok con un coeficiente inestable si acción y control difieren mucho
# en escala (ver 'Brecha ANCOVA' más abajo) -- por eso 'valida' se deja visible, no se
# borra, aunque ya no bloquee.
df_resumen['reportable'] = df_resumen['paralelismo_ok']

# ── INCREMENTAL: df_matches ACUMULADO = pares congelados + pares nuevos del mes ──────
if MODO_INCREMENTAL and len(df_congelados):
    df_matches = pd.concat([df_congelados, df_matches], ignore_index=True)
    _vmap = {(r.cosecha, r.categoria): r.valida for r in df_resumen.itertuples()}
    df_matches['celda_valida'] = [_vmap.get((c, k), v) for c, k, v in
                                  zip(df_matches['cosecha'], df_matches['des_categoria'], df_matches['celda_valida'])]
    assert not df_matches.duplicated(['cosecha', 'des_categoria', 'cod_accion']).any(), \
        "un cliente de acción quedó en más de un par dentro de la misma celda"
    if not MAX_REUSO_CONTROL_POR_COSECHA and MAX_REUSO_CONTROL_DEFAULT == 1:
        assert not df_matches.duplicated(['cosecha', 'des_categoria', 'cod_control']).any(), \
            "un control quedó en más de un par con reuso 1:1 -> revisar la lógica de congelados"
    if len(df_sin_cong):       # sin pareja de meses anteriores + los de este mes (sin los que ya tienen par)
        df_sin_pareja = pd.concat([df_sin_cong, df_sin_pareja], ignore_index=True)
        _con_par = set(zip(df_matches['des_categoria'], df_matches['cod_accion']))
        df_sin_pareja = df_sin_pareja[[(k, a) not in _con_par for k, a in
                                       zip(df_sin_pareja['des_categoria'], df_sin_pareja['cod_accion'])]]
        df_sin_pareja = df_sin_pareja.drop_duplicates(['cosecha', 'des_categoria', 'cod_accion']).reset_index(drop=True)
    print(f"\n[INCREMENTAL] df_matches acumulado: {len(df_matches):,} pares "
          f"({len(df_congelados):,} congelados + {len(df_matches) - len(df_congelados):,} nuevos)")
df_semillas    = pd.DataFrame(semillas_out)
df_viabilidad  = pd.DataFrame(viabilidad_out)

print("\n" + "=" * 110); print("VIABILIDAD POR OFICINA — TODAS LAS CELDAS"); print("=" * 110)
print(df_viabilidad.to_string(index=False))
print(f"\nCeldas cosecha×categoría×oficina SIN control suficiente para {MARGEN_ERROR_OBJETIVO*100:.0f}% "
      f"de margen de error: {(~df_viabilidad.viable).sum():,} de {len(df_viabilidad):,}")

print("\n" + "=" * 110); print("RESUMEN GENERAL"); print("=" * 110)
print(df_resumen.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# TABLAS DE COMPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────
comp_pob_mue, comp_ac_ct = [], []

for (cosecha, cat), inf in pools_out.items():
    ac_pob, ac_mue, ct_pob = inf['ac'], inf['ac_s'], inf['ct']
    print("\n" + "=" * 110)
    print(f"COMPARACIONES — {cosecha.date()} | {cat}")
    print("=" * 110)

    # (A) Población vs Muestra
    for col in ESTRATOS_FULL:
        p = ac_pob[col].astype(str).value_counts(normalize=True).mul(100)
        m = ac_mue[col].astype(str).value_counts(normalize=True).mul(100)
        t = pd.DataFrame({'pob_%': p.round(1), 'muestra_%': m.round(1)}).fillna(0.0)
        t['dif_pp']    = (t['muestra_%'] - t['pob_%']).round(1)
        t['n_pob']     = ac_pob[col].astype(str).value_counts()
        t['n_muestra'] = ac_mue[col].astype(str).value_counts()
        t = t.fillna(0).sort_values('n_pob', ascending=False)
        t.index.name = col
        print(f"\n  [A] POBLACIÓN vs MUESTRA — {col}")
        print(t.to_string().replace('\n', '\n  '))
        for idx, r in t.iterrows():
            if abs(r['dif_pp']) > 5:
                print(f"     [ALERTA] {idx}: desvío {r['dif_pp']}pp")
        comp_pob_mue.append(t.assign(cosecha=cosecha, categoria=cat, variable=col))

    # (B) Acción vs Control (post-limpieza, pre-matching)
    for col in MATCH_FULL:
        a = ac_pob[col].astype(str).value_counts(normalize=True).mul(100)
        c = ct_pob[col].astype(str).value_counts(normalize=True).mul(100)
        t = pd.DataFrame({'accion_%': a.round(1), 'control_%': c.round(1)}).fillna(0.0)
        t['dif_pp']    = (t['control_%'] - t['accion_%']).round(1)
        t['n_accion']  = ac_pob[col].astype(str).value_counts()
        t['n_control'] = ct_pob[col].astype(str).value_counts()
        t = t.fillna(0).sort_values('n_accion', ascending=False)
        t.index.name = col
        print(f"\n  [B] ACCIÓN vs CONTROL — {col}")
        print(t.to_string().replace('\n', '\n  '))
        comp_ac_ct.append(t.assign(cosecha=cosecha, categoria=cat, variable=col))

df_comp_pob_mue = pd.concat(comp_pob_mue) if comp_pob_mue else pd.DataFrame()
df_comp_ac_ct   = pd.concat(comp_ac_ct)   if comp_ac_ct   else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO: trayectoria 12M por cosecha_agrupada
# ─────────────────────────────────────────────────────────────────────────────
if GRAFICAR and pools_out:
    for cosecha in COSECHAS:
        cells = [k for k in pools_out if k[0] == cosecha]
        if not cells:
            continue
        fig, axes = plt.subplots(1, len(cells), figsize=(7 * len(cells), 4.6), squeeze=False)
        for axi, key in enumerate(cells):
            inf, ax = pools_out[key], axes[0][axi]
            x = [m.strftime('%b-%y') for m in inf['meses']]
            ax.plot(x, inf['t_ac'], marker='o', lw=2.2, color='#1f3864', label='Acción')
            ax.plot(x, inf['t_ct'], marker='s', lw=2.2, ls='--', color='#c00000', label='Control')
            ax.fill_between(x, inf['t_ac'], inf['t_ct'], alpha=0.12, color='grey')
            r = inf['best']
            ax.set_title(f"{key[0].date()} | {key[1]}  "
                         f"[{'VÁLIDA' if r['valida'] else 'INCOMPARABLE'}]\n"
                         f"seed={r['seed']} corr={r['corr']} ratio={r['ratio_vol']} "
                         f"n={r['pares']:,} cob={r['cobertura_pct']}%",
                         fontsize=9.5, color='#1f3864' if r['valida'] else '#c00000')
            ax.set_ylabel('Peso promedio (kg)'); ax.legend(fontsize=8)
            ax.grid(alpha=0.25); ax.tick_params(axis='x', rotation=45, labelsize=8)
        fig.suptitle(f"Trayectoria pre-cosecha 12M — cosecha_agrupada {cosecha.date()}",
                     fontsize=12, weight='bold')
        plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNÓSTICO
# ─────────────────────────────────────────────────────────────────────────────
if len(df_matches):
    if len(df_semillas):     # en una corrida incremental sin nuevos no hay semillas que comparar
        print("\nESTABILIDAD ENTRE SEMILLAS")
        print(df_semillas.groupby(['cosecha', 'categoria'])
              .agg(n_validas=('valida', 'sum'), pares_min=('pares', 'min'),
                   pares_max=('pares', 'max'), corr_min=('corr', 'min'),
                   corr_max=('corr', 'max'), cob_min=('cobertura_pct', 'min'),
                   cob_max=('cobertura_pct', 'max')).to_string())

    bal = []
    for (c, k), g in df_matches.groupby(['cosecha', 'des_categoria']):
        for v in ['peso', 'nmeses']:
            a, b = g[f'{v}_ac'], g[f'{v}_ct']
            sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            bal.append({'cosecha': c, 'categoria': k, 'var': v,
                        'media_ac': round(a.mean(), 2), 'media_ct': round(b.mean(), 2),
                        'SMD': round((a.mean() - b.mean()) / sp if sp > 0 else 0, 3)})
    bdf = pd.DataFrame(bal); bdf['ok'] = bdf.SMD.abs() < 0.10
    print("\nBALANCE (|SMD| < 0.10)"); print(bdf.to_string(index=False))

    print("\nPARES POR NIVEL DE CASCADA")
    print(df_matches.groupby(['cosecha', 'des_categoria', 'nivel_match'])
          .size().rename('pares').to_string())

    print("\nCOBERTURA POR OFICINA (aquí sale el problema real de CM-SUR)")
    if len(df_sin_pareja):
        cob = pd.DataFrame({
            'matcheados': df_matches.groupby(['cosecha', 'des_categoria', 'ofi_ac']).size(),
            'sin_pareja': df_sin_pareja.groupby(['cosecha', 'des_categoria', 'des_oficina']).size()
        }).fillna(0).astype(int)
        cob['cobertura_%'] = (100 * cob.matcheados /
                              (cob.matcheados + cob.sin_pareja)).round(1)
        print(cob.to_string())
    else:
        print("  0 clientes sin pareja -> 100% de la acción quedó matcheada en todas las oficinas.")

    print("\nSESGO DE SELECCIÓN (matcheados vs sin pareja)")
    if len(df_sin_pareja):
        ses = df_matches.groupby(['cosecha', 'des_categoria'])['peso_ac'].mean().rename('peso_match')
        ses = pd.concat([ses, df_sin_pareja.groupby(['cosecha', 'des_categoria'])['peso_12m']
                         .mean().rename('peso_sin_pareja')], axis=1).round(1)
        print(ses.to_string())
    else:
        print("  0 clientes sin pareja -> no aplica comparar sesgo de selección.")

    reuso = df_matches.groupby('cod_control')['cosecha'].nunique()
    print(f"\nControles en >1 cosecha: {(reuso > 1).sum():,}/{len(reuso):,} "
          f"-> CLUSTERIZAR SE por cod_cliente en la regresión DiD")

    reuso_misma_celda = df_matches.groupby(['cosecha', 'des_categoria', 'cod_control']).size()
    reuso_misma_celda = reuso_misma_celda[reuso_misma_celda > 1]
    if len(reuso_misma_celda):
        print(f"\nControles reusados >1 vez DENTRO de la misma celda (MAX_REUSO_CONTROL): "
              f"{len(reuso_misma_celda):,} controles -> ídem, clusterizar SE por cod_control.")
        print(reuso_misma_celda.groupby(['cosecha', 'des_categoria']).agg(['count', 'max'])
              .to_string())

    inval = df_resumen[~df_resumen.valida]
    if len(inval):
        print("\n[CELDAS A EXCLUIR DEL REPORTE]")
        print(inval[['cosecha', 'categoria', 'corr', 'ratio_vol', 'pares']].to_string(index=False))

    print("\nTEST ANCOVA DE PARALELISMO POR CELDA — H0 (homogeneidad de pendientes): no hay interacción grupo:pre")
    print(df_resumen[['cosecha', 'categoria', 'pares', 'ancova_f', 'ancova_pval',
                      'umbral_paralelismo', 'semillas_validas',
                      'semillas_probadas_paralelismo', 'paralelismo_ok']]
          .to_string(index=False))
    no_parall = df_resumen[~df_resumen.paralelismo_ok]
    if len(no_parall):
        print("\n[ALERTA] Celdas que NO lograron paralelismo tras buscar en varias semillas "
              "(quedan fuera de 'reportable' -- ver df_resumen.reportable):")
        print(no_parall[['cosecha', 'categoria', 'ancova_f', 'ancova_pval',
                         'semillas_probadas_paralelismo', 'semillas_validas']].to_string(index=False))

    print("\n" + "=" * 110); print("RESUMEN FINAL — ¿QUÉ CELDAS SON REPORTABLES?"); print("=" * 110)
    print(df_resumen[['grupo', 'categoria', 'pares', 'valida', 'paralelismo_ok', 'reportable']]
          .to_string(index=False))
    print(f"\nReportables: {df_resumen.reportable.sum()} de {len(df_resumen)} celdas.")

# ══════════════ celda 18 del notebook ══════════════
# =============================================================================
# ¿DE DONDE VIENEN LAS DEX DE ACCION Y LAS DEX DE CONTROL?
# Diagnostico -- NO hay una lista fija de distribuidoras (DEX/nom_compania)
# que sea "de accion" o "de control": el rol AC/CT se asigna por CLIENTE (ver
# metodologia, seccion 3.4), no por distribuidora. Un cliente es acción si su
# cosecha real (cli_cosecha) coincide con la cosecha_grupo que se esta
# procesando; es control si NO recibio el piloto en NINGUNA cosecha. Ambos
# roles pueden convivir en la MISMA DEX -- nom_compania es solo uno de los 3
# atributos de estratificacion del matching (junto a oficina y segmento), no
# un criterio de asignacion de rol.
#
# Esta celda muestra, para los pares ya matcheados (df_matches, que ya trae
# 'comp_ac'/'comp_ct' = la DEX de cada cliente del par), de que distribuidoras
# viene realmente la poblacion de accion y la de control, por cosecha y
# categoria.
# =============================================================================

print("=" * 100)
print("DEX (nom_compania) DE LOS CLIENTES DE ACCION, por cosecha x categoria")
print("=" * 100)
dex_accion = (df_matches.groupby(['cosecha', 'des_categoria', 'comp_ac'])
              .size().rename('n_pares').reset_index()
              .sort_values(['cosecha', 'des_categoria', 'n_pares'], ascending=[True, True, False]))
print(dex_accion.to_string(index=False))

print("\n" + "=" * 100)
print("DEX (nom_compania) DE LOS CLIENTES DE CONTROL, por cosecha x categoria")
print("=" * 100)
dex_control = (df_matches.groupby(['cosecha', 'des_categoria', 'comp_ct'])
               .size().rename('n_pares').reset_index()
               .sort_values(['cosecha', 'des_categoria', 'n_pares'], ascending=[True, True, False]))
print(dex_control.to_string(index=False))

print("\n" + "=" * 100)
print("¿CUANTAS DEX SON EXCLUSIVAS DE UN SOLO ROL (solo accion o solo control) "
      "vs. COMPARTIDAS (aportan clientes a ambos roles)?")
print("=" * 100)
dex_ac_set = set(df_matches['comp_ac'].unique())
dex_ct_set = set(df_matches['comp_ct'].unique())
print(f"DEX que aportan clientes de ACCION: {len(dex_ac_set)}")
print(f"DEX que aportan clientes de CONTROL: {len(dex_ct_set)}")
print(f"DEX que aportan a AMBOS roles (comparten clientes de accion y de control "
      f"en algun par): {len(dex_ac_set & dex_ct_set)}")
print(f"DEX SOLO de accion (nunca aportan control en un par matcheado): "
      f"{sorted(dex_ac_set - dex_ct_set)}")
print(f"DEX SOLO de control (nunca aportan accion en un par matcheado): "
      f"{sorted(dex_ct_set - dex_ac_set)}")

# ══════════════ celda 19 del notebook ══════════════
import os, json

# Ruta de salida relativa (portable entre Windows/Linux/Mac) -> antes apuntaba a una
# ruta de OneDrive de otra persona en Windows, que no existe en esta máquina Debian.
ruta_salida = RUTA_SALIDA

# Crear carpeta si no existe
os.makedirs(ruta_salida, exist_ok=True)

# Guardar cada DataFrame
df_matches.to_csv(os.path.join(ruta_salida, 'df_matches.csv'), index=False, encoding='utf-8')
df_sin_pareja.to_csv(os.path.join(ruta_salida, 'df_sin_pareja.csv'), index=False, encoding='utf-8')
df_funnel.to_csv(os.path.join(ruta_salida, 'df_funnel.csv'), index=False, encoding='utf-8')
df_resumen.to_csv(os.path.join(ruta_salida, 'df_resumen.csv'), index=False, encoding='utf-8')
df_semillas.to_csv(os.path.join(ruta_salida, 'df_semillas.csv'), index=False, encoding='utf-8')
df_viabilidad.to_csv(os.path.join(ruta_salida, 'df_viabilidad.csv'), index=False, encoding='utf-8')

print("✓ Archivos guardados en:")
print(f"  {os.path.join(ruta_salida, 'df_matches.csv')}")
print(f"  {os.path.join(ruta_salida, 'df_sin_pareja.csv')}")
print(f"  {os.path.join(ruta_salida, 'df_funnel.csv')}")
print(f"  {os.path.join(ruta_salida, 'df_resumen.csv')}")
print(f"  {os.path.join(ruta_salida, 'df_semillas.csv')}")
print(f"  {os.path.join(ruta_salida, 'df_viabilidad.csv')}")

# Meta de los pares congelados: parámetros del matching + mes de cierre (se valida en la corrida siguiente)
with open(RUTA_META_PARES, 'w', encoding='utf-8') as _f:
    json.dump({**_meta_actual, 'mes_cierre': str(MES_CIERRE.date())}, _f, ensure_ascii=False, indent=2)
print(f"  {RUTA_META_PARES}  (mes_cierre={MES_CIERRE.date()})")

# ══════════════ celda 21 del notebook ══════════════
# =============================================================================
# DISTRIBUCIÓN POR SEGMENTO — acción vs. control, ANTES y DESPUÉS del matching
# =============================================================================
# 1) Pool limpio (antes del matching): expone si un desbalance de escala/volatilidad
#    de la celda (ej. Galletas, ratio_vol fuera de rango) viene de que acción y control
#    están compuestos de segmentos distintos (ej. acción concentrada en Premium/Plus,
#    control en Base/Alto), y no solo de un problema de matching.
# 2) Pares ya matcheados: mismo cálculo, solo sobre los pares 1:1 elegidos -> si el
#    matching corrigió el desbalance, aquí debería verse más parejo que en el pool.
def _tabla_seg(filas):
    d = pd.DataFrame(filas)
    d['pct'] = d.groupby(['cosecha_grupo', 'des_categoria', 'rol'])['n_clientes'].transform(lambda s: round(100 * s / s.sum(), 1))
    return d

filas_pool = []
for (cosecha, cat), pool in pools_out.items():
    for rol, df_r in (('accion', pool['ac']), ('control', pool['ct'])):
        cnt = df_r['des_segmento_transaccional'].fillna('Sin Asignar').value_counts()
        for seg, n in cnt.items():
            filas_pool.append({'cosecha_grupo': cosecha, 'des_categoria': cat, 'rol': rol,
                               'des_segmento_transaccional': seg, 'n_clientes': int(n)})
df_dist_seg_pool = _tabla_seg(filas_pool)

filas_pares = []
for (cosecha, cat), pool in pools_out.items():
    seg_ac, seg_ct = pool['ac']['des_segmento_transaccional'], pool['ct']['des_segmento_transaccional']
    mm = df_matches[(df_matches.cosecha == cosecha) & (df_matches.des_categoria == cat)]
    for rol, ids, seg_map in (('accion', mm.cod_accion, seg_ac), ('control', mm.cod_control, seg_ct)):
        cnt = seg_map.reindex(ids).fillna('Sin Asignar').value_counts()
        for seg, n in cnt.items():
            filas_pares.append({'cosecha_grupo': cosecha, 'des_categoria': cat, 'rol': rol,
                                'des_segmento_transaccional': seg, 'n_clientes': int(n)})
df_dist_seg_pares = _tabla_seg(filas_pares)


def _imprimir(df_dist, titulo):
    print(titulo)
    for cat in CATEGORIAS:
        sub = df_dist[df_dist.des_categoria == cat]
        if not len(sub):
            continue
        n = sub.pivot_table(index='des_segmento_transaccional', columns='rol', values='n_clientes', aggfunc='sum', fill_value=0)
        p = sub.pivot_table(index='des_segmento_transaccional', columns='rol', values='pct', aggfunc='sum', fill_value=0)
        for c in ('accion', 'control'):
            if c not in n: n[c], p[c] = 0, 0.0
        tabla = pd.DataFrame({f'{c}_n': n[c].astype(int) for c in ('accion', 'control')})
        for c in ('accion', 'control'):
            tabla[f'{c}_%'] = p[c]
        print(f"\n{cat}"); print(tabla[['accion_n', 'accion_%', 'control_n', 'control_%']].to_string())

_imprimir(df_dist_seg_pool, "DISTRIBUCIÓN POR SEGMENTO — POOL LIMPIO (antes del matching)")
_imprimir(df_dist_seg_pares, "\n\nDISTRIBUCIÓN POR SEGMENTO — PARES YA MATCHEADOS (1:1, después del matching)")

# Chequeo: el total de clientes en la tabla de pool debe cuadrar con AC_limpio/CT_limpio
for r in df_resumen.itertuples():
    tot_ac = df_dist_seg_pool[(df_dist_seg_pool.cosecha_grupo == r.cosecha) & (df_dist_seg_pool.des_categoria == r.categoria) & (df_dist_seg_pool.rol == 'accion')].n_clientes.sum()
    tot_ct = df_dist_seg_pool[(df_dist_seg_pool.cosecha_grupo == r.cosecha) & (df_dist_seg_pool.des_categoria == r.categoria) & (df_dist_seg_pool.rol == 'control')].n_clientes.sum()
    assert tot_ac == r.AC_limpio and tot_ct == r.CT_limpio, f"distribución de segmento no cuadra con el pool limpio en {r.cosecha.date()}-{r.categoria}"
print("\nOK  distribución de segmento del pool cuadra con AC_limpio / CT_limpio de df_resumen")

# ══════════════ celda 23 del notebook ══════════════
C41_SWEEP_SRC = '# =============================================================================\n# PROYECTO ROMA — PIPELINE DE MATCHING v4 (corregido)\n# =============================================================================\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom dateutil.relativedelta import relativedelta\nfrom scipy.optimize import linear_sum_assignment\nfrom scipy.stats import linregress\nfrom sklearn.metrics import pairwise_distances\nimport statsmodels.formula.api as smf\nfrom statsmodels.stats.anova import anova_lm\n\npd.set_option(\'display.width\', 250)\n\n# ─────────────────────────────────────────────────────────────────────────────\n# TEST FORMAL DE TENDENCIAS PARALELAS (pre-cosecha) — ANCOVA CLASICO, por celda\n# cosecha×categoría. Version 2: a pedido del usuario, se reemplaza el F-test de\n# Wald sobre el modelo clusterizado (trayectoria mes a mes completa) por un\n# ANCOVA clasico de 2 puntos, SIN ajuste por cliente -- decision 2026-09-17.\n#\n# Diseño: el pre-periodo (12 meses) se parte en 2 mitades. \'pre\' = promedio\n# log1p de la PRIMERA mitad (linea base, mas lejos de la cosecha). \'post\' =\n# promedio log1p de la SEGUNDA mitad (mas cerca de la cosecha). Cada cliente\n# (AC o CT) aporta UNA sola fila (pre, post) -> a diferencia de la version\n# anterior (12 filas por cliente, correlacionadas entre si), aqui no hace\n# falta clusterizar: cada observacion es independiente. Eso SI habilita usar\n# la tabla ANCOVA clasica (statsmodels.stats.anova.anova_lm) sin el problema\n# de p-valor artificialmente bajo que tenia con datos repetidos por cliente.\n#\n# Se ajusta post ~ grupo * pre (ANCOVA con interaccion) y se prueba el termino\n# de interaccion grupo:pre con la tabla ANCOVA de Tipo II.\n# H0 (supuesto de homogeneidad de pendientes de ANCOVA): no hay interaccion\n# grupo:pre -> el crecimiento de linea-base a segunda-mitad es igual para AC\n# y CT -> tendencias paralelas.\n#\n# *** Trade-off del nuevo diseño (documentado, no oculto): resumir 12 meses en\n# 2 puntos pierde informacion de la forma de la trayectoria mes a mes (un\n# quiebre puntual a mitad de periodo puede no reflejarse en el promedio de\n# cada mitad). A cambio, el modelo es mas simple y evita clusterizar. ***\n#\n# *** Sigue pendiente confirmar con negocio: UMBRAL_PARALELISMO=0.10 (mismo\n# umbral que tenia el test viejo, pre-TOST) reintroduce a proposito la\n# critica original -- con mas clientes, mas poder para rechazar por\n# diferencias sin relevancia de negocio. ***\n# ─────────────────────────────────────────────────────────────────────────────\nUMBRAL_PARALELISMO = 0.10   # no rechazar H0 con holgura -> exigimos p > 0.10, no solo > 0.05\n\ndef test_ancova_paralelismo(cod_accion, cod_control, ac_df, ct_df, cols_m):\n    """\n    ANCOVA clasico -- un cliente = una fila (sin ajuste por cliente, no hace\n    falta clusterizar). Resume el pre-periodo en 2 puntos: \'pre\' = promedio\n    log1p de la PRIMERA mitad de meses (linea base), \'post\' = promedio log1p\n    de la SEGUNDA mitad (mas cerca de la cosecha). Ajusta post ~ grupo*pre y\n    prueba la interaccion grupo:pre con la tabla ANCOVA clasica (Tipo II) --\n    H0: no hay interaccion -> el crecimiento de linea-base a segunda-mitad es\n    igual entre AC y CT -> tendencias paralelas.\n    Devuelve (f_stat, p_valor, ok).\n    """\n    n = len(cols_m)\n    mitad = n // 2\n    cols_pre, cols_post = cols_m[:mitad], cols_m[mitad:]\n\n    filas = []\n    for cid in cod_accion:\n        pre = np.log1p(ac_df.loc[cid, cols_pre].to_numpy(float)).mean()\n        post = np.log1p(ac_df.loc[cid, cols_post].to_numpy(float)).mean()\n        filas.append((f"AC_{cid}", 1, pre, post))\n    for cid in cod_control:\n        pre = np.log1p(ct_df.loc[cid, cols_pre].to_numpy(float)).mean()\n        post = np.log1p(ct_df.loc[cid, cols_post].to_numpy(float)).mean()\n        filas.append((f"CT_{cid}", 0, pre, post))\n    dfl = pd.DataFrame(filas, columns=[\'cliente\', \'grupo\', \'pre\', \'post\'])\n    try:\n        mod = smf.ols(\'post ~ grupo * pre\', data=dfl).fit()   # OLS clasico, SIN cluster\n        tabla = anova_lm(mod, typ=2)                          # tabla ANCOVA clasica (valida: 1 fila/cliente)\n        f_stat = float(tabla.loc[\'grupo:pre\', \'F\'])\n        p_valor = float(tabla.loc[\'grupo:pre\', \'PR(>F)\'])\n        ok = bool(pd.notna(p_valor) and p_valor > UMBRAL_PARALELISMO)\n        return f_stat, p_valor, ok\n    except Exception:\n        return np.nan, np.nan, False\n\n\n# ─────────────────────────────────────────────────────────────────────────────\n# MATCHING PARA UNA SEMILLA — extraído a función para poder paralelizar (joblib)\n# Cada llamada es independiente entre sí (misma ac/ct, distinta submuestra por seed)\n# -> ideal para ejecutar en paralelo entre los N_JOBS núcleos disponibles.\n# ─────────────────────────────────────────────────────────────────────────────\ndef _correr_semilla(seed, ac, ct, cols_m, estratos_usar, Lac_df, Lct_df, Fac_df, Fct_df,\n                     MED_N, MED_F, NIVELES, MUESTREAR, FRAC_MUESTREO,\n                     W_NIVEL, W_FORMA, CORR_UMBRAL, RATIO_MIN, RATIO_MAX,\n                     max_reuso=1, ac_por_oficina=None,\n                     ka_full_por_claves=None, kc_full_por_claves=None):\n    # ac_por_oficina / ka_full_por_claves / kc_full_por_claves: PRECOMPUTADOS una\n    # sola vez fuera del loop de semillas (no dependen del seed, solo el .sample()\n    # de adentro si) -> evita recalcular lo mismo 50-150 veces por celda. Si vienen\n    # en None (ej. alguien llama a esta funcion suelta, fuera del loop principal),\n    # cae al calculo original -> mismo resultado, solo mas lento.\n    # ---- MUESTREO (FIX 2 + FIX 3) --------------------------------\n    if MUESTREAR:\n        piezas = []\n        fuente_oficina = ac_por_oficina if ac_por_oficina is not None else dict(list(ac.groupby(\'des_oficina\')))\n        for ofi, g_ac in fuente_oficina.items():\n            tgt  = max(1, int(round(len(g_ac) * FRAC_MUESTREO)))\n            frac = tgt / len(g_ac)\n\n            g2 = g_ac.copy()\n            g2[\'_e\'] = g2[estratos_usar].astype(str).agg(\'|\'.join, axis=1)\n            tam_c = g2[\'_e\'].value_counts()\n\n            # asignación por mayor resto: proporcional exacta, sin min-1 ni trim\n            raw   = tam_c * frac\n            base  = np.floor(raw).astype(int)\n            falta = tgt - int(base.sum())\n            resto = (raw - base)\n            if falta > 0:\n                for e in resto.sort_values(ascending=False).index[:falta]:\n                    base[e] += 1\n            elif falta < 0:\n                cand_baja = resto.sort_values().index\n                quitados = 0\n                for e in cand_baja:\n                    if quitados >= -falta:\n                        break\n                    if base[e] > 0:\n                        base[e] -= 1; quitados += 1\n\n            sub = [gg.sample(n=int(base[e]), random_state=seed)\n                   for e, gg in g2.groupby(\'_e\') if base[e] > 0]\n            if not sub:\n                continue\n            sub = pd.concat(sub).drop(columns=\'_e\')\n            sub[\'_peso_pob\'] = len(g_ac) / len(sub)\n            piezas.append(sub)\n        ac_s = pd.concat(piezas) if piezas else ac.iloc[0:0]\n    else:\n        ac_s = ac.copy(); ac_s[\'_peso_pob\'] = 1.0\n    if len(ac_s) == 0:\n        return None\n\n    # ---- MATCHING EN CASCADA, con RONDAS para permitir reuso 1:N -------------\n    # BUG previo: el contador de reuso funcionaba, pero un solo pase de la cascada\n    # (3 niveles, el último con caliper laxo) agota TODO el control disponible en\n    # una sola asignación óptima (linear_sum_assignment nunca repite DENTRO de una\n    # misma llamada) -> nunca había una "segunda vez" en la que reusar un control.\n    # Ahora cada RONDA es un pase completo de la cascada de estratos sobre la\n    # acción aún sin pareja (pend), usando solo controles que no llegaron a su\n    # tope de reuso. Con max_reuso=1 corre una sola ronda -> idéntico al 1:1 original.\n    pend, pares_s = list(ac_s.index), []\n    uso_control = {}\n    ct_ids_all = list(ct.index)\n\n    for ronda in range(max_reuso):\n        if not pend:\n            break\n        disp = {cid for cid in ct_ids_all if uso_control.get(cid, 0) < max_reuso}\n        if not disp:\n            break\n        for nivel, claves, cal in NIVELES:\n            if not pend or not disp:\n                break\n            a_sub, c_sub = ac_s.loc[pend], ct.loc[sorted(disp)]\n            if ka_full_por_claves is not None and kc_full_por_claves is not None:\n                # subset por indice sobre la clave YA calculada -> nada de astype+join\n                # de nuevo (eso es lo que se repetia identico 50-150 veces por celda,\n                # sobre todo del lado control, que ni siquiera cambia con el seed)\n                ka = ka_full_por_claves[tuple(claves)].loc[a_sub.index]\n                kc = kc_full_por_claves[tuple(claves)].loc[c_sub.index]\n            else:\n                ka = a_sub[claves].astype(str).agg(\'|\'.join, axis=1)\n                kc = c_sub[claves].astype(str).agg(\'|\'.join, axis=1)\n            for blk in ka.unique():\n                ia, ic = ka.index[ka == blk], kc.index[kc == blk]\n                if len(ic) == 0:\n                    continue\n                Dn = pairwise_distances(Lac_df.loc[ia].to_numpy(),\n                                        Lct_df.loc[ic].to_numpy()) / MED_N\n                Df = pairwise_distances(Fac_df.loc[ia].to_numpy(),\n                                        Fct_df.loc[ic].to_numpy()) / MED_F\n                D = W_NIVEL * Dn + W_FORMA * Df\n                for i, j in zip(*linear_sum_assignment(D)):\n                    if float(D[i, j]) > cal:\n                        continue\n                    a_id, c_id = ia[i], ic[j]\n                    pares_s.append({\'nivel_match\': nivel, \'ronda_reuso\': ronda + 1,\n                                    \'cod_accion\': a_id, \'cod_control\': c_id,\n                                    \'dist_total\': round(float(D[i, j]), 4),\n                                    \'dist_nivel\': round(float(Dn[i, j]), 4),\n                                    \'dist_forma\': round(float(Df[i, j]), 4),\n                                    \'peso_pob\': ac_s.at[a_id, \'_peso_pob\']})\n                    pend.remove(a_id)\n                    uso_control[c_id] = uso_control.get(c_id, 0) + 1\n                    disp.discard(c_id)\n\n    if not pares_s:\n        return None\n    p_df = pd.DataFrame(pares_s)\n\n    t_ac = ac.loc[p_df.cod_accion, cols_m].mean().to_numpy()\n    t_ct = ct.loc[p_df.cod_control, cols_m].mean().to_numpy()\n    d_ac, d_ct = np.diff(t_ac), np.diff(t_ct)\n    corr  = (np.corrcoef(d_ac, d_ct)[0, 1]\n             if d_ac.std() > 0 and d_ct.std() > 0 else np.nan)\n    ratio = d_ac.std() / d_ct.std() if d_ct.std() > 0 else np.inf\n    brecha = t_ac - t_ct\n    pend_rel = (linregress(np.arange(len(brecha)), brecha).slope / t_ac.mean()\n                if t_ac.mean() > 0 else np.inf)\n    ok = (pd.notna(corr) and corr >= CORR_UMBRAL and RATIO_MIN <= ratio <= RATIO_MAX)\n\n    return {\n        \'seed\': seed, \'n_muestra\': len(ac_s), \'pares\': len(p_df),\n        \'cobertura_pct\': round(100 * len(p_df) / len(ac_s), 1),\n        \'corr\': round(corr, 3) if pd.notna(corr) else np.nan,\n        \'ratio_vol\': round(ratio, 3), \'pend_rel\': round(pend_rel, 5),\n        \'dist_med\': round(p_df.dist_total.median(), 4), \'valida\': ok,\n        \'controles_unicos\': p_df.cod_control.nunique(),\n        \'max_reuso_observado\': int(p_df.cod_control.value_counts().max()),\n        \'_p\': p_df, \'_ac\': ac_s, \'_pend\': pend, \'_tac\': t_ac, \'_tct\': t_ct}\n\n\n# ─────────────────────────────────────────────────────────────────────────────\n# PASO 0 — PARÁMETROS\n# ─────────────────────────────────────────────────────────────────────────────\n# Cosecha ÚNICA agrupada (ancla marzo). Los clientes que entran en abr/may/jun comparten\n# el ancla pero conservan su mes real de entrada (min_periodo_cosecha) -> conteo acumulativo.\nCOSECHAS   = [COSECHA_ANCLA]\nCATEGORIAS = [_SWEEP_CAT]\n\n# Etiquetas de negocio para las cosechas agrupadas (mismo mapeo de la celda 22)\nGRUPO_COSECHA = {COSECHA_ANCLA: \'CHAMPION MM (Mar + altas Abr-Jun, acumulativo)\'}\n\nVENTANA_MESES  = 12\n# MIN_MESES por categoría — PARAMETRIZABLE (dict). Corrida de comparación con\n# Cereales=6 (igual que Galletas), para aislar el efecto de esa relajación del\n# resto de cambios. El valor acordado con el negocio (Cereales=4) queda\n# documentado en las instrucciones del proyecto para cuando se reactive.\nMIN_MESES_POR_CATEGORIA = {_SWEEP_CAT: _mm}\nMIN_MESES_DEFAULT = 6\nEXCLUIR_NUEVOS = True\n\nDROP_FILAS_NEG  = True\nEXIGIR_POSITIVO = True\n\nOUTLIER_METODO = \'iqr_log\'    # \'iqr_log\' | \'iqr_raw\' | \'p99\' | \'ninguno\'\nOUTLIER_K      = 1.5\nOUTLIER_NIVEL  = [\'des_oficina\']\n\n# Muestreo — depende SOLO de la población de acción (FIX 2)\nMUESTREAR     = True\nFRAC_MUESTREO = 0.60\nESTRATOS_FULL = [\'min_periodo_cosecha\', \'des_oficina\',\n                 \'nom_compania\', \'des_segmento_transaccional\']\nMIN_POR_CELDA = 5             # mediana de celda bajo esto -> baja nivel de cascada\n\n# Matching — sin min_periodo_cosecha (los CT no lo tienen)\nMATCH_FULL     = [\'des_oficina\', \'nom_compania\', \'des_segmento_transaccional\']\nNIVEL_MAX      = 3            # 3 = nunca suelta des_oficina\nCALIPER_REL    = 0.60\nCALIPER_ULTIMO = 0.90\n\nUSAR_LOG = True\nW_NIVEL  = 0.5                # cercanía (default)\nW_FORMA  = 0.5                # paralelismo (default)\n\n# ── Cascada de candidatas para el test de paralelismo (ANOVA) ───────────────\n# Antes limitaba cuántas semillas se testeaban formalmente con TOST (más caro\n# que el resto de la selección). El ANCOVA sobre 2 puntos por cliente\n# tiene un costo similar -> se mantiene el mismo tope por razones de tiempo\n# de cómputo, no por necesidad estadística del nuevo test.\nMAX_SEMILLAS_PROBAR_EQUIVALENCIA = 10  # tope de candidatas a testear formalmente por celda\n\nN_SEMILLAS_DEFAULT = 50\nCORR_UMBRAL = 0.55\nRATIO_MIN, RATIO_MAX = 0.70, 1.40\n\n# ── Overrides puntuales — REACTIVADOS: G1-Cereales y G2-Galletas fallan\n# \'valida\' por ratio_vol fuera de rango (0.568 y 1.706 vs [0.70,1.40]) pese a\n# tener corr razonable (0.769 y 0.758) -- mismo síntoma que Junio-Galletas ya\n# documentado en las instrucciones del proyecto (ratio_vol fuera de rango con\n# población balanceada). Se les da más peso a la forma de la trayectoria\n# (W_FORMA=0.7, en vez de matchear casi solo por nivel/estrato) y más semillas\n# (150 en vez de 50) para que la búsqueda tenga más chances de encontrar un\n# control cuya volatilidad pre-cosecha se parezca más a la de acción.\n# Champion: sin overrides de entrada. Se activan por celda solo si el embudo/valida los pide\n# (ej. ratio_vol fuera de rango con población balanceada -> W_FORMA=0.7 y 150 semillas).\nPESOS_OVERRIDE = {}\nN_SEMILLAS_OVERRIDE = {}\n\n# ── Reuso de control 1:N — DESACTIVADO en esta corrida de comparación ──\n# 1:1 estricto en TODAS las cosechas (comportamiento original de v6), para aislar\n# el efecto del resto de cambios sobre el resultado. El reuso 1:N en Grupo 01\n# (justificado por escasez real de control, 96-98% del CT ya en uso en 1:1) queda\n# documentado en las instrucciones del proyecto para cuando se reactive.\nMAX_REUSO_CONTROL_POR_COSECHA = {}\nMAX_REUSO_CONTROL_DEFAULT = 1        # 1:1 estricto (comportamiento original)\n\nCOL_TRATADO    = \'cosecha_agrupada\'\nGRAFICAR       = True\n\n# ─────────────────────────────────────────────────────────────────────────────\n# PASO 1 — ROL A NIVEL CLIENTE\n# ─────────────────────────────────────────────────────────────────────────────\ndf = base_seg_limpio.copy()\ndf[\'periodo\']          = pd.to_datetime(df[\'periodo\'])\ndf[\'cosecha_agrupada\'] = pd.to_datetime(df[\'cosecha_agrupada\'])\nif \'min_periodo_cosecha\' in df.columns:\n    df[\'min_periodo_cosecha\'] = pd.to_datetime(df[\'min_periodo_cosecha\'])\n\ncli_cosecha  = df.dropna(subset=[COL_TRATADO]).groupby(\'cod_cliente\')[COL_TRATADO].min()\nTRATADOS_ALL = set(cli_cosecha.index)\n\n# FIX 1: "nuevo" = primera transacción REAL en toda la base\ncli_primera_compra = df.groupby(\'cod_cliente\')[\'periodo\'].min()\nprint(f"[COBERTURA] La base arranca en {df[\'periodo\'].min().date()}. "\n      f"Un cliente que ya compraba antes de esa fecha se marcará como nuevo por error.")\n\ncli_min_per = (df.dropna(subset=[\'min_periodo_cosecha\'])\n                 .groupby(\'cod_cliente\')[\'min_periodo_cosecha\'].min()\n               if \'min_periodo_cosecha\' in df.columns\n               else pd.Series(dtype=\'datetime64[ns]\'))\n\nprint(f"Clientes en base : {df[\'cod_cliente\'].nunique():,}")\nprint(f"Acciones (todas) : {len(TRATADOS_ALL):,}")\nprint(f"Candidatos espejo: {df[\'cod_cliente\'].nunique() - len(TRATADOS_ALL):,}")\n\nct_ids = set(df[\'cod_cliente\'].unique()) - TRATADOS_ALL\nct_con_min = len(set(cli_min_per.index) & ct_ids)\nprint(f"\\n[CHEQUEO] Controles con min_periodo_cosecha: {ct_con_min:,} de {len(ct_ids):,}")\nprint("  -> es atributo de ACCIÓN: sirve como ESTRATO, no como llave de matching."\n      if ct_con_min < len(ct_ids) * 0.5 else\n      "  -> los CT sí lo tienen: puedes moverlo a MATCH_FULL.")\n\nprint("\\n[ESTRUCTURA] min_periodo_cosecha por cosecha_agrupada:")\nprint(df.dropna(subset=[COL_TRATADO]).groupby(COL_TRATADO)[\'min_periodo_cosecha\']\n        .agg(n_valores=\'nunique\',\n             valores=lambda s: [str(x.date()) for x in sorted(s.dropna().unique())])\n        .to_string())\n\n# Cascada de matching (FIX 4)\nNIVELES = [[len(MATCH_FULL) - k + 1, MATCH_FULL[:k], CALIPER_REL]\n           for k in range(len(MATCH_FULL), 0, -1)][:NIVEL_MAX]\nNIVELES[-1][2] = CALIPER_ULTIMO\n\n# ─────────────────────────────────────────────────────────────────────────────\n# LOOP PRINCIPAL\n# ─────────────────────────────────────────────────────────────────────────────\nmatches_out, sinpar_out, funnel_out, resumen_out, semillas_out, pools_out, viabilidad_out = [], [], [], [], [], {}, []\n\nfor cosecha in COSECHAS:\n    for cat in CATEGORIAS:\n\n        ini   = cosecha - relativedelta(months=VENTANA_MESES)\n        fin   = cosecha - relativedelta(months=1)\n        meses = pd.date_range(ini, fin, freq=\'MS\')\n\n        print("\\n" + "=" * 110)\n        print(f"COSECHA {cosecha.date()} [{GRUPO_COSECHA.get(cosecha, \'?\')}]  |  {cat}  |  "\n              f"ventana {ini.date()} .. {fin.date()}")\n        print("=" * 110)\n\n        min_meses_celda = MIN_MESES_POR_CATEGORIA.get(cat, MIN_MESES_DEFAULT)\n        w_nivel_celda = PESOS_OVERRIDE.get((cosecha, cat), {}).get(\'W_NIVEL\', W_NIVEL)\n        w_forma_celda = PESOS_OVERRIDE.get((cosecha, cat), {}).get(\'W_FORMA\', W_FORMA)\n        n_semillas_celda = N_SEMILLAS_OVERRIDE.get((cosecha, cat), N_SEMILLAS_DEFAULT)\n        semillas_celda = list(range(1, n_semillas_celda + 1))\n        max_reuso_celda = MAX_REUSO_CONTROL_POR_COSECHA.get(cosecha, MAX_REUSO_CONTROL_DEFAULT)\n        if (cosecha, cat) in PESOS_OVERRIDE or (cosecha, cat) in N_SEMILLAS_OVERRIDE:\n            print(f"  [OVERRIDE] W_NIVEL={w_nivel_celda} W_FORMA={w_forma_celda} "\n                  f"N_SEMILLAS={n_semillas_celda} (celda con override específico)")\n        if max_reuso_celda > 1:\n            print(f"  [REUSO CONTROL] max_reuso_celda={max_reuso_celda} "\n                  f"(un control puede matchear hasta {max_reuso_celda} veces en esta cosecha)")\n\n        w = df[(df[\'periodo\'] >= ini) & (df[\'periodo\'] <= fin) &\n               (df[\'des_categoria\'] == cat)].copy()\n        if w.empty:\n            print("  [SKIP] sin datos"); continue\n\n        n_neg = (w[\'peso\'] < 0).sum()\n        if DROP_FILAS_NEG and n_neg:\n            print(f"  [NOTA] {n_neg:,} filas peso<0 eliminadas "\n                  f"({w.loc[w.peso < 0, \'peso\'].sum():,.1f} kg). Compra BRUTA "\n                  f"-> misma regla en el post-período.")\n            w = w[w[\'peso\'] >= 0]\n\n        wp  = w[w[\'peso\'] > 0]\n        agg = (w.groupby(\'cod_cliente\').agg(peso_12m=(\'peso\', \'sum\'))\n                 .join(wp.groupby(\'cod_cliente\').agg(n_meses=(\'periodo\', \'nunique\')))\n                 .fillna({\'n_meses\': 0}))\n\n        tray = (w.pivot_table(index=\'cod_cliente\', columns=\'periodo\',\n                              values=\'peso\', aggfunc=\'sum\')\n                  .reindex(columns=meses).fillna(0.0).clip(lower=0))\n        cols_m = [f"m{i:02d}" for i in range(len(meses))]\n        tray.columns = cols_m\n\n        attrs = (w.sort_values(\'periodo\').groupby(\'cod_cliente\').tail(1)\n                   .set_index(\'cod_cliente\')[[\'des_oficina\', \'nom_compania\',\n                                              \'des_segmento_transaccional\']])\n\n        pool = agg.join(tray).join(attrs)\n        pool[\'min_periodo_cosecha\'] = cli_min_per.reindex(pool.index)\n\n        es_ac = pool.index.map(lambda i: cli_cosecha.get(i, pd.NaT) == cosecha)\n        pool[\'rol\'] = np.where(es_ac, \'AC\',\n                        np.where(pool.index.isin(TRATADOS_ALL), \'EXCL\', \'CT\'))\n\n        # ── EMBUDO ───────────────────────────────────────────────────────────\n        fn = []\n        _snap = lambda e, p: fn.append({\'etapa\': e,\n                                        \'AC\': int((p.rol == \'AC\').sum()),\n                                        \'CT\': int((p.rol == \'CT\').sum())})\n        _snap(\'0. Con datos en ventana\', pool)\n        pool = pool[pool[\'rol\'] != \'EXCL\'];  _snap(\'1. Sin acción de otra cosecha\', pool)\n        if EXCLUIR_NUEVOS:\n            nuevos = pool[\'n_meses\'] == 0    # no compró en los 12 meses pre-cosecha\n            pool = pool[~nuevos]\n        _snap(\'2. Sin clientes nuevos\', pool)\n        if EXIGIR_POSITIVO:\n            pool = pool[pool[\'peso_12m\'] > 0]\n        _snap(\'3. Con volumen > 0\', pool)\n        pool = pool[pool[\'n_meses\'] >= min_meses_celda]\n        _snap(f\'4. Recurrencia >= {min_meses_celda}\', pool)\n\n        if OUTLIER_METODO != \'ninguno\' and len(pool):\n            v = np.log1p(pool[\'peso_12m\']) if OUTLIER_METODO == \'iqr_log\' else pool[\'peso_12m\']\n            tmp = pool[OUTLIER_NIVEL].copy(); tmp[\'_v\'] = v.values\n            if OUTLIER_METODO in (\'iqr_log\', \'iqr_raw\'):\n                q1 = tmp.groupby(OUTLIER_NIVEL)[\'_v\'].transform(lambda s: s.quantile(.25))\n                q3 = tmp.groupby(OUTLIER_NIVEL)[\'_v\'].transform(lambda s: s.quantile(.75))\n                lo, hi = q1 - OUTLIER_K * (q3 - q1), q3 + OUTLIER_K * (q3 - q1)\n            else:\n                hi = tmp.groupby(OUTLIER_NIVEL)[\'_v\'].transform(lambda s: s.quantile(.99))\n                lo = pd.Series(-np.inf, index=tmp.index)\n            out = (tmp[\'_v\'] < lo) | (tmp[\'_v\'] > hi)\n            print("\\n  Tasa de outlier por oficina (pool combinado AC+CT, en log):")\n            print(pd.DataFrame({\n                \'n_pool\'   : pool.groupby(\'des_oficina\').size(),\n                \'n_outlier\': out.groupby(pool[\'des_oficina\']).sum(),\n                \'pct\'      : (out.groupby(pool[\'des_oficina\']).mean() * 100).round(1),\n                \'lim_sup_kg\': (np.expm1(hi.groupby(pool[\'des_oficina\']).first())\n                               if OUTLIER_METODO == \'iqr_log\'\n                               else hi.groupby(pool[\'des_oficina\']).first()).round(1),\n            }).to_string().replace(\'\\n\', \'\\n  \'))\n            pool = pool[~out]\n        _snap(\'5. Sin outliers\', pool)\n\n        fdf = pd.DataFrame(fn)\n        fdf[\'AC_pct\'] = (100 * fdf.AC / max(1, fdf.AC.iloc[0])).round(1)\n        fdf[\'CT_pct\'] = (100 * fdf.CT / max(1, fdf.CT.iloc[0])).round(1)\n        print("\\n  EMBUDO DE LIMPIEZA")\n        print(fdf.to_string(index=False).replace(\'\\n\', \'\\n  \'))\n        for r in fn:\n            funnel_out.append({\'cosecha\': cosecha, \'categoria\': cat, **r})\n\n        ac = pool[pool.rol == \'AC\'].copy()\n        ct = pool[pool.rol == \'CT\'].copy()\n        if len(ac) == 0 or len(ct) == 0:\n            print("  [SKIP] celda sin acción o sin control"); continue\n\n        viab = pd.DataFrame({\'AC\': ac.groupby(\'des_oficina\').size(),\n                             \'CT\': ct.groupby(\'des_oficina\').size()}).fillna(0).astype(int)\n        viab[\'ratio_CT_AC\'] = (viab.CT / viab.AC.replace(0, np.nan)).round(3)\n        # Viabilidad = MIN_CT_OFICINA=30 fijo, IGUAL que v6 (piso fijo, no dinámico).\n        # n_min_requerido (fórmula con corrección por población finita) se deja como\n        # columna INFORMATIVA adicional -- no decide \'viable\', solo da contexto extra\n        # de qué tan lejos/cerca está esa oficina del estándar 5%/95% del proyecto.\n        MIN_CT_OFICINA = 30\n        viab[\'n_min_requerido\'] = [n_minimo_requerido(a, c_) for a, c_ in zip(viab.AC, viab.CT)]\n        viab[\'viable\'] = viab.CT >= MIN_CT_OFICINA\n        print("\\n  VIABILIDAD POR OFICINA")\n        print(viab.to_string().replace(\'\\n\', \'\\n  \'))\n        for ofi, r in viab[~viab.viable].iterrows():\n            print(f"  [NO VIABLE] {ofi}: {r.CT} controles para {r.AC:,} acciones. "\n                  f"La muestra sí se toma; el problema saldrá en cobertura de matching.")\n\n        # Guardamos la viabilidad de TODAS las oficinas (viables y no viables) —\n        # esto es lo que sustenta, más adelante, si se puede reportar la brecha\n        # partida por oficina o si hay que decir explícitamente "sin control suficiente".\n        for ofi, r in viab.iterrows():\n            viabilidad_out.append({\n                \'cosecha\': cosecha, \'grupo\': GRUPO_COSECHA.get(cosecha, \'?\'), \'categoria\': cat,\n                \'des_oficina\': ofi,\n                \'AC\': int(r.AC), \'CT\': int(r.CT), \'n_min_requerido\': int(r.n_min_requerido),\n                \'ratio_CT_AC\': float(r.ratio_CT_AC) if pd.notna(r.ratio_CT_AC) else np.nan,\n                \'viable\': bool(r.viable)})\n\n        # ── CASCADA DE ESTRATOS (FIX 5) ──────────────────────────────────────\n        estratos_usar, nivel_estrato = ESTRATOS_FULL[:], 1\n        for k in range(len(ESTRATOS_FULL), 0, -1):\n            cand = ESTRATOS_FULL[:k]\n            tam = ac.groupby([ac[c].astype(str) for c in cand]).size()\n            estratos_usar = cand\n            nivel_estrato = len(ESTRATOS_FULL) - k + 1\n            if len(tam) and tam.median() >= MIN_POR_CELDA:\n                break\n        tam = ac.groupby([ac[c].astype(str) for c in estratos_usar]).size()\n        print(f"\\n  ESTRATOS (nivel {nivel_estrato}): {\' × \'.join(estratos_usar)}")\n        print(f"    celdas={len(tam)} | mediana={tam.median():.0f} | "\n              f"celdas<{MIN_POR_CELDA}: {(tam < MIN_POR_CELDA).sum()}")\n\n        # ── ESPACIO DE MATCHING ──────────────────────────────────────────────\n        f_log = (lambda M: np.log1p(M)) if USAR_LOG else (lambda M: M)\n        Lac, Lct = f_log(ac[cols_m].to_numpy(float)), f_log(ct[cols_m].to_numpy(float))\n        Fac, Fct = np.diff(Lac, axis=1), np.diff(Lct, axis=1)\n        Lac_df, Lct_df = pd.DataFrame(Lac, index=ac.index), pd.DataFrame(Lct, index=ct.index)\n        Fac_df, Fct_df = pd.DataFrame(Fac, index=ac.index), pd.DataFrame(Fct, index=ct.index)\n\n        _r = np.random.default_rng(0)\n        sa = _r.choice(len(ac), min(600, len(ac)), replace=False)\n        sc = _r.choice(len(ct), min(600, len(ct)), replace=False)\n        MED_N = float(np.median(pairwise_distances(Lac[sa], Lct[sc]))) or 1.0\n        MED_F = float(np.median(pairwise_distances(Fac[sa], Fct[sc]))) or 1.0\n\n        # ── LOOP DE SEMILLAS — PARALELIZADO CON JOBLIB (N_JOBS núcleos) ────────\n        # Cada semilla es independiente -> se reparte entre los núcleos disponibles.\n        # backend=\'threads\' (no procesos): pairwise_distances/linear_sum_assignment son\n        # C/Fortran y liberan el GIL, y así evitamos el costo de volver a serializar\n        # ac/ct/Lac_df/... (varios miles de filas) en cada una de las 50 semillas.\n        # Medido: ~15x más rápido que \'processes\' para este tamaño de pool.\n        # ── PRECOMPUTO por celda (cosecha x categoria) — una vez, no una vez por semilla ──\n        # ac_por_oficina: el agrupamiento por oficina no depende del seed -> se arma\n        # una vez y se comparte entre threads (memoria compartida, solo lectura, seguro).\n        # ka_full/kc_full: la clave de estrato de cada cliente (ej. "oficina|compania|\n        # segmento") tampoco depende del seed -> se calcula una vez por combinacion de\n        # columnas usada en NIVELES, en vez de recalcularla con astype+join en cada una\n        # de las N semillas (para celdas 1:1, el lado control se repetia identico).\n        ac_por_oficina_pre = dict(list(ac.groupby(\'des_oficina\')))\n        claves_unicas = []\n        for _, claves_n, _ in NIVELES:\n            if claves_n not in claves_unicas:\n                claves_unicas.append(claves_n)\n        ka_full_pre = {tuple(cl): ac[cl].astype(str).agg(\'|\'.join, axis=1) for cl in claves_unicas}\n        kc_full_pre = {tuple(cl): ct[cl].astype(str).agg(\'|\'.join, axis=1) for cl in claves_unicas}\n\n        print(f"\\n  ITERANDO {n_semillas_celda} SEMILLAS  (paralelo, N_JOBS={N_JOBS}, backend=threads)")\n\n        resultados_semilla = Parallel(n_jobs=N_JOBS, prefer=\'threads\')(\n            delayed(_correr_semilla)(\n                seed, ac, ct, cols_m, estratos_usar, Lac_df, Lct_df, Fac_df, Fct_df,\n                MED_N, MED_F, NIVELES, MUESTREAR, FRAC_MUESTREO,\n                w_nivel_celda, w_forma_celda, CORR_UMBRAL, RATIO_MIN, RATIO_MAX,\n                max_reuso=max_reuso_celda,\n                ac_por_oficina=ac_por_oficina_pre,\n                ka_full_por_claves=ka_full_pre, kc_full_por_claves=kc_full_pre)\n            for seed in semillas_celda)\n        resultados_semilla = [r for r in resultados_semilla if r is not None]\n\n        if not resultados_semilla:\n            print("  [SKIP] ninguna semilla produjo pares"); continue\n\n        rs_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith(\'_\')}\n                              for r in resultados_semilla])\n        print(rs_df.to_string(index=False).replace(\'\\n\', \'\\n  \'))\n        for r in rs_df.to_dict(\'records\'):\n            semillas_out.append({\'cosecha\': cosecha, \'categoria\': cat, **r})\n\n        val = [r for r in resultados_semilla if r[\'valida\']]\n\n        # ── BÚSQUEDA DE LA MEJOR SEMILLA QUE TAMBIÉN SEA PARALELA (ANCOVA) ──────\n        # AJUSTE (Champion): el ANCOVA se corre SIEMPRE, incluso si NINGUNA semilla\n        # cumple corr/ratio_vol (val vacío) -- antes, en ese caso, el ANCOVA nunca se\n        # calculaba y paralelismo_ok quedaba en False "por defecto", no por resultado\n        # de un test. Ahora se buscan candidatas en \'val\' si hay, y si no, en TODAS las\n        # semillas (resultados_semilla), siempre ordenadas por pend_rel (mejor proxy\n        # primero) y probadas en cascada hasta encontrar la primera paralela.\n        # \'valida\' (corr/ratio_vol) queda igual de informativa que antes -- solo deja\n        # de decidir si el ANCOVA se corre o no.\n        ancova_f = ancova_p = np.nan\n        paralelismo_ok = False\n        semillas_probadas = 0\n        best = None\n\n        candidatas = sorted(val if val else resultados_semilla, key=lambda r: abs(r[\'pend_rel\']))\n        tag_val = "" if val else " [ninguna semilla es \'valida\' por corr/ratio_vol -- ANCOVA corrido igual]"\n        for cand in candidatas[:MAX_SEMILLAS_PROBAR_EQUIVALENCIA]:\n            semillas_probadas += 1\n            f_stat, p_valor, ok = test_ancova_paralelismo(\n                cand[\'_p\'].cod_accion, cand[\'_p\'].cod_control, ac, ct, cols_m)\n            if ok:\n                best = cand\n                ancova_f, ancova_p, paralelismo_ok = f_stat, p_valor, ok\n                print(f"\\n  >> SEMILLA {best[\'seed\']} | corr={best[\'corr\']} "\n                      f"ratio={best[\'ratio_vol\']} pend_rel={best[\'pend_rel\']} "\n                      f"pares={best[\'pares\']:,} cobertura={best[\'cobertura_pct\']}% "\n                      f"| PARALELA tras probar {semillas_probadas} semilla(s) "\n                      f"(F={f_stat:.3f}, p={p_valor:.4f}, umbral>{UMBRAL_PARALELISMO})" + tag_val)\n                break\n        if best is None:\n            # Ninguna de las candidatas probadas fue paralela -> nos quedamos con la\n            # mejor por pend_rel (igual que antes), pero documentado como NO paralela,\n            # no se reporta como si hubiera pasado.\n            best = candidatas[0]\n            ancova_f, ancova_p, paralelismo_ok = test_ancova_paralelismo(\n                best[\'_p\'].cod_accion, best[\'_p\'].cod_control, ac, ct, cols_m)\n            print(f"\\n  >> SEMILLA {best[\'seed\']} | corr={best[\'corr\']} "\n                  f"ratio={best[\'ratio_vol\']} pend_rel={best[\'pend_rel\']} "\n                  f"pares={best[\'pares\']:,} cobertura={best[\'cobertura_pct\']}% "\n                  f"| [NO PARALELA] ninguna de las {semillas_probadas} semilla(s) "\n                  f"probadas (de {len(candidatas)} candidatas) superó el umbral "\n                  f"p>{UMBRAL_PARALELISMO} -> se documenta, no se descarta la celda." + tag_val)\n\n        if best.get(\'controles_unicos\', best[\'pares\']) < best[\'pares\']:\n            print(f"  [REUSO CONTROL EN USO] {best[\'pares\']:,} pares usan solo "\n                  f"{best[\'controles_unicos\']:,} controles únicos (máx. reuso observado: "\n                  f"{best[\'max_reuso_observado\']}x) -> clusterizar SE por cod_control en el DiD.")\n\n        for r in best[\'_p\'].to_dict(\'records\'):\n            a_id, c_id = r[\'cod_accion\'], r[\'cod_control\']\n            matches_out.append({\n                \'cosecha\': cosecha, \'des_categoria\': cat, \'seed\': best[\'seed\'],\n                \'celda_valida\': best[\'valida\'], **r,\n                \'ofi_ac\': ac.at[a_id, \'des_oficina\'],  \'ofi_ct\': ct.at[c_id, \'des_oficina\'],\n                \'comp_ac\': ac.at[a_id, \'nom_compania\'], \'comp_ct\': ct.at[c_id, \'nom_compania\'],\n                \'seg_ac\': ac.at[a_id, \'des_segmento_transaccional\'],\n                \'seg_ct\': ct.at[c_id, \'des_segmento_transaccional\'],\n                \'min_per_ac\': ac.at[a_id, \'min_periodo_cosecha\'],\n                \'peso_ac\': ac.at[a_id, \'peso_12m\'],   \'peso_ct\': ct.at[c_id, \'peso_12m\'],\n                \'nmeses_ac\': ac.at[a_id, \'n_meses\'],  \'nmeses_ct\': ct.at[c_id, \'n_meses\']})\n\n        for a_id in best[\'_pend\']:\n            sinpar_out.append({\n                \'cosecha\': cosecha, \'des_categoria\': cat, \'cod_accion\': a_id,\n                \'des_oficina\': ac.at[a_id, \'des_oficina\'],\n                \'nom_compania\': ac.at[a_id, \'nom_compania\'],\n                \'des_segmento_transaccional\': ac.at[a_id, \'des_segmento_transaccional\'],\n                \'min_periodo_cosecha\': ac.at[a_id, \'min_periodo_cosecha\'],\n                \'peso_12m\': ac.at[a_id, \'peso_12m\'], \'n_meses\': ac.at[a_id, \'n_meses\']})\n\n        # ── RESULTADO DE PARALELISMO (ANOVA) sobre la semilla finalmente elegida ──\n        # Ya se calculó dentro de la búsqueda de arriba (ancova_f/ancova_p/ok) -- no\n        # se vuelve a correr la regresión, solo se deja registrado para el resumen.\n        print(f"\\n  [TEST ANCOVA PARALELISMO] H0 (homogeneidad de pendientes): no hay interacción grupo:pre | "\n              f"F={ancova_f:.3f}, p={ancova_p:.4f}" if pd.notna(ancova_p) else "  [TEST ANCOVA] no calculable")\n        print(f"    -> {\'OK: no se rechaza H0, tendencias paralelas\' if paralelismo_ok else \'[ALERTA] se rechaza H0: tendencias NO paralelas, revisar celda\'} "\n              f"(umbral p>{UMBRAL_PARALELISMO}, semillas válidas probadas: {semillas_probadas} de {len(val)})")\n\n        resumen_out.append({\n            \'cosecha\': cosecha, \'grupo\': GRUPO_COSECHA.get(cosecha, \'?\'), \'categoria\': cat,\n            \'AC_limpio\': len(ac), \'CT_limpio\': len(ct),\n            \'AC_muestra\': best[\'n_muestra\'], \'seed\': best[\'seed\'], \'pares\': best[\'pares\'],\n            \'controles_unicos\': best.get(\'controles_unicos\', best[\'pares\']),\n            \'max_reuso_observado\': best.get(\'max_reuso_observado\', 1),\n            \'min_meses_usado\': min_meses_celda,\n            \'cobertura_pct\': best[\'cobertura_pct\'], \'corr\': best[\'corr\'],\n            \'ratio_vol\': best[\'ratio_vol\'], \'pend_rel\': best[\'pend_rel\'],\n            \'valida\': best[\'valida\'], \'nivel_estrato\': nivel_estrato,\n            \'ancova_f\': round(ancova_f, 4) if pd.notna(ancova_f) else np.nan,\n            \'ancova_pval\': round(ancova_p, 4) if pd.notna(ancova_p) else np.nan,\n            \'umbral_paralelismo\': UMBRAL_PARALELISMO,\n            \'semillas_validas\': len(val), \'semillas_probadas_paralelismo\': semillas_probadas,\n            \'paralelismo_ok\': paralelismo_ok})\n\n        pools_out[(cosecha, cat)] = {\'ac\': ac, \'ct\': ct, \'meses\': meses,\n                                     \'ac_s\': best[\'_ac\'], \'t_ac\': best[\'_tac\'],\n                                     \'t_ct\': best[\'_tct\'], \'best\': best}\n\ndf_matches     = pd.DataFrame(matches_out)\ndf_sin_pareja  = pd.DataFrame(sinpar_out)\ndf_funnel      = pd.DataFrame(funnel_out)\ndf_resumen     = pd.DataFrame(resumen_out)\n# AJUSTE (Champion): reportable = SOLO paralelismo_ok (test ANCOVA formal, p > umbral).\n# \'valida\' (corr >= 0.55 y ratio_vol en 0.70-1.40) queda como columna INFORMATIVA en\n# df_resumen, ya no es un gate para reportable. Motivo: en Galletas \'valida\' es False\n# por diferencia de ESCALA entre acción y control (ver distribución por segmento más\n# abajo), no por falta de paralelismo -- y el criterio pedido es el ANCOVA únicamente.\n# Advertencia que se mantiene igual: una celda con corr/ratio_vol fuera de rango puede\n# pasar paralelismo_ok con un coeficiente inestable si acción y control difieren mucho\n# en escala (ver \'Brecha ANCOVA\' más abajo) -- por eso \'valida\' se deja visible, no se\n# borra, aunque ya no bloquee.\ndf_resumen[\'reportable\'] = df_resumen[\'paralelismo_ok\']\ndf_semillas    = pd.DataFrame(semillas_out)\ndf_viabilidad  = pd.DataFrame(viabilidad_out)\n\nprint("\\n" + "=" * 110); print("VIABILIDAD POR OFICINA — TODAS LAS CELDAS"); print("=" * 110)\nprint(df_viabilidad.to_string(index=False))\nprint(f"\\nCeldas cosecha×categoría×oficina SIN control suficiente para {MARGEN_ERROR_OBJETIVO*100:.0f}% "\n      f"de margen de error: {(~df_viabilidad.viable).sum():,} de {len(df_viabilidad):,}")\n\nprint("\\n" + "=" * 110); print("RESUMEN GENERAL"); print("=" * 110)\nprint(df_resumen.to_string(index=False))\n\n# ─────────────────────────────────────────────────────────────────────────────\n# TABLAS DE COMPARACIÓN\n# ─────────────────────────────────────────────────────────────────────────────\ncomp_pob_mue, comp_ac_ct = [], []\n\nfor (cosecha, cat), inf in pools_out.items():\n    ac_pob, ac_mue, ct_pob = inf[\'ac\'], inf[\'ac_s\'], inf[\'ct\']\n    print("\\n" + "=" * 110)\n    print(f"COMPARACIONES — {cosecha.date()} | {cat}")\n    print("=" * 110)\n\n    # (A) Población vs Muestra\n    for col in ESTRATOS_FULL:\n        p = ac_pob[col].astype(str).value_counts(normalize=True).mul(100)\n        m = ac_mue[col].astype(str).value_counts(normalize=True).mul(100)\n        t = pd.DataFrame({\'pob_%\': p.round(1), \'muestra_%\': m.round(1)}).fillna(0.0)\n        t[\'dif_pp\']    = (t[\'muestra_%\'] - t[\'pob_%\']).round(1)\n        t[\'n_pob\']     = ac_pob[col].astype(str).value_counts()\n        t[\'n_muestra\'] = ac_mue[col].astype(str).value_counts()\n        t = t.fillna(0).sort_values(\'n_pob\', ascending=False)\n        t.index.name = col\n        print(f"\\n  [A] POBLACIÓN vs MUESTRA — {col}")\n        print(t.to_string().replace(\'\\n\', \'\\n  \'))\n        for idx, r in t.iterrows():\n            if abs(r[\'dif_pp\']) > 5:\n                print(f"     [ALERTA] {idx}: desvío {r[\'dif_pp\']}pp")\n        comp_pob_mue.append(t.assign(cosecha=cosecha, categoria=cat, variable=col))\n\n    # (B) Acción vs Control (post-limpieza, pre-matching)\n    for col in MATCH_FULL:\n        a = ac_pob[col].astype(str).value_counts(normalize=True).mul(100)\n        c = ct_pob[col].astype(str).value_counts(normalize=True).mul(100)\n        t = pd.DataFrame({\'accion_%\': a.round(1), \'control_%\': c.round(1)}).fillna(0.0)\n        t[\'dif_pp\']    = (t[\'control_%\'] - t[\'accion_%\']).round(1)\n        t[\'n_accion\']  = ac_pob[col].astype(str).value_counts()\n        t[\'n_control\'] = ct_pob[col].astype(str).value_counts()\n        t = t.fillna(0).sort_values(\'n_accion\', ascending=False)\n        t.index.name = col\n        print(f"\\n  [B] ACCIÓN vs CONTROL — {col}")\n        print(t.to_string().replace(\'\\n\', \'\\n  \'))\n        comp_ac_ct.append(t.assign(cosecha=cosecha, categoria=cat, variable=col))\n\ndf_comp_pob_mue = pd.concat(comp_pob_mue) if comp_pob_mue else pd.DataFrame()\ndf_comp_ac_ct   = pd.concat(comp_ac_ct)   if comp_ac_ct   else pd.DataFrame()\n\n# ─────────────────────────────────────────────────────────────────────────────\n# GRÁFICO: trayectoria 12M por cosecha_agrupada\n# ─────────────────────────────────────────────────────────────────────────────\nif GRAFICAR and pools_out:\n    for cosecha in COSECHAS:\n        cells = [k for k in pools_out if k[0] == cosecha]\n        if not cells:\n            continue\n        fig, axes = plt.subplots(1, len(cells), figsize=(7 * len(cells), 4.6), squeeze=False)\n        for axi, key in enumerate(cells):\n            inf, ax = pools_out[key], axes[0][axi]\n            x = [m.strftime(\'%b-%y\') for m in inf[\'meses\']]\n            ax.plot(x, inf[\'t_ac\'], marker=\'o\', lw=2.2, color=\'#1f3864\', label=\'Acción\')\n            ax.plot(x, inf[\'t_ct\'], marker=\'s\', lw=2.2, ls=\'--\', color=\'#c00000\', label=\'Control\')\n            ax.fill_between(x, inf[\'t_ac\'], inf[\'t_ct\'], alpha=0.12, color=\'grey\')\n            r = inf[\'best\']\n            ax.set_title(f"{key[0].date()} | {key[1]}  "\n                         f"[{\'VÁLIDA\' if r[\'valida\'] else \'INCOMPARABLE\'}]\\n"\n                         f"seed={r[\'seed\']} corr={r[\'corr\']} ratio={r[\'ratio_vol\']} "\n                         f"n={r[\'pares\']:,} cob={r[\'cobertura_pct\']}%",\n                         fontsize=9.5, color=\'#1f3864\' if r[\'valida\'] else \'#c00000\')\n            ax.set_ylabel(\'Peso promedio (kg)\'); ax.legend(fontsize=8)\n            ax.grid(alpha=0.25); ax.tick_params(axis=\'x\', rotation=45, labelsize=8)\n        fig.suptitle(f"Trayectoria pre-cosecha 12M — cosecha_agrupada {cosecha.date()}",\n                     fontsize=12, weight=\'bold\')\n        plt.tight_layout(); plt.show()\n\n# ─────────────────────────────────────────────────────────────────────────────\n# DIAGNÓSTICO\n# ─────────────────────────────────────────────────────────────────────────────\nif len(df_matches):\n    print("\\nESTABILIDAD ENTRE SEMILLAS")\n    print(df_semillas.groupby([\'cosecha\', \'categoria\'])\n          .agg(n_validas=(\'valida\', \'sum\'), pares_min=(\'pares\', \'min\'),\n               pares_max=(\'pares\', \'max\'), corr_min=(\'corr\', \'min\'),\n               corr_max=(\'corr\', \'max\'), cob_min=(\'cobertura_pct\', \'min\'),\n               cob_max=(\'cobertura_pct\', \'max\')).to_string())\n\n    bal = []\n    for (c, k), g in df_matches.groupby([\'cosecha\', \'des_categoria\']):\n        for v in [\'peso\', \'nmeses\']:\n            a, b = g[f\'{v}_ac\'], g[f\'{v}_ct\']\n            sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)\n            bal.append({\'cosecha\': c, \'categoria\': k, \'var\': v,\n                        \'media_ac\': round(a.mean(), 2), \'media_ct\': round(b.mean(), 2),\n                        \'SMD\': round((a.mean() - b.mean()) / sp if sp > 0 else 0, 3)})\n    bdf = pd.DataFrame(bal); bdf[\'ok\'] = bdf.SMD.abs() < 0.10\n    print("\\nBALANCE (|SMD| < 0.10)"); print(bdf.to_string(index=False))\n\n    print("\\nPARES POR NIVEL DE CASCADA")\n    print(df_matches.groupby([\'cosecha\', \'des_categoria\', \'nivel_match\'])\n          .size().rename(\'pares\').to_string())\n\n    print("\\nCOBERTURA POR OFICINA (aquí sale el problema real de CM-SUR)")\n    if len(df_sin_pareja):\n        cob = pd.DataFrame({\n            \'matcheados\': df_matches.groupby([\'cosecha\', \'des_categoria\', \'ofi_ac\']).size(),\n            \'sin_pareja\': df_sin_pareja.groupby([\'cosecha\', \'des_categoria\', \'des_oficina\']).size()\n        }).fillna(0).astype(int)\n        cob[\'cobertura_%\'] = (100 * cob.matcheados /\n                              (cob.matcheados + cob.sin_pareja)).round(1)\n        print(cob.to_string())\n    else:\n        print("  0 clientes sin pareja -> 100% de la acción quedó matcheada en todas las oficinas.")\n\n    print("\\nSESGO DE SELECCIÓN (matcheados vs sin pareja)")\n    if len(df_sin_pareja):\n        ses = df_matches.groupby([\'cosecha\', \'des_categoria\'])[\'peso_ac\'].mean().rename(\'peso_match\')\n        ses = pd.concat([ses, df_sin_pareja.groupby([\'cosecha\', \'des_categoria\'])[\'peso_12m\']\n                         .mean().rename(\'peso_sin_pareja\')], axis=1).round(1)\n        print(ses.to_string())\n    else:\n        print("  0 clientes sin pareja -> no aplica comparar sesgo de selección.")\n\n    reuso = df_matches.groupby(\'cod_control\')[\'cosecha\'].nunique()\n    print(f"\\nControles en >1 cosecha: {(reuso > 1).sum():,}/{len(reuso):,} "\n          f"-> CLUSTERIZAR SE por cod_cliente en la regresión DiD")\n\n    reuso_misma_celda = df_matches.groupby([\'cosecha\', \'des_categoria\', \'cod_control\']).size()\n    reuso_misma_celda = reuso_misma_celda[reuso_misma_celda > 1]\n    if len(reuso_misma_celda):\n        print(f"\\nControles reusados >1 vez DENTRO de la misma celda (MAX_REUSO_CONTROL): "\n              f"{len(reuso_misma_celda):,} controles -> ídem, clusterizar SE por cod_control.")\n        print(reuso_misma_celda.groupby([\'cosecha\', \'des_categoria\']).agg([\'count\', \'max\'])\n              .to_string())\n\n    inval = df_resumen[~df_resumen.valida]\n    if len(inval):\n        print("\\n[CELDAS A EXCLUIR DEL REPORTE]")\n        print(inval[[\'cosecha\', \'categoria\', \'corr\', \'ratio_vol\', \'pares\']].to_string(index=False))\n\n    print("\\nTEST ANCOVA DE PARALELISMO POR CELDA — H0 (homogeneidad de pendientes): no hay interacción grupo:pre")\n    print(df_resumen[[\'cosecha\', \'categoria\', \'pares\', \'ancova_f\', \'ancova_pval\',\n                      \'umbral_paralelismo\', \'semillas_validas\',\n                      \'semillas_probadas_paralelismo\', \'paralelismo_ok\']]\n          .to_string(index=False))\n    no_parall = df_resumen[~df_resumen.paralelismo_ok]\n    if len(no_parall):\n        print("\\n[ALERTA] Celdas que NO lograron paralelismo tras buscar en varias semillas "\n              "(quedan fuera de \'reportable\' -- ver df_resumen.reportable):")\n        print(no_parall[[\'cosecha\', \'categoria\', \'ancova_f\', \'ancova_pval\',\n                         \'semillas_probadas_paralelismo\', \'semillas_validas\']].to_string(index=False))\n\n    print("\\n" + "=" * 110); print("RESUMEN FINAL — ¿QUÉ CELDAS SON REPORTABLES?"); print("=" * 110)\n    print(df_resumen[[\'grupo\', \'categoria\', \'pares\', \'valida\', \'paralelismo_ok\', \'reportable\']]\n          .to_string(index=False))\n    print(f"\\nReportables: {df_resumen.reportable.sum()} de {len(df_resumen)} celdas.")'

import io, contextlib

SWEEP_MIN_MESES  = [4, 6, 8, 10, 12]     # editar para probar otros puntos
SWEEP_CATEGORIAS = ['Galletas', 'Cereales']

# Snapshot de los resultados DEFINITIVOS (MIN_MESES_POR_CATEGORIA actual, ya corridos
# arriba) -- se restauran al final para que el resto del notebook no use el barrido.
_snap_definitivo = {
    'df_matches': df_matches.copy(), 'df_sin_pareja': df_sin_pareja.copy(),
    'df_funnel': df_funnel.copy(), 'df_resumen': df_resumen.copy(),
    'df_semillas': df_semillas.copy(), 'df_viabilidad': df_viabilidad.copy(),
    'pools_out': dict(pools_out), 'CATEGORIAS': list(CATEGORIAS),
    'MIN_MESES_POR_CATEGORIA': dict(MIN_MESES_POR_CATEGORIA)}

print(f"Corriendo {len(SWEEP_CATEGORIAS) * len(SWEEP_MIN_MESES)} combinaciones "
      f"(categoría × MIN_MESES) -- esto re-ejecuta el matching completo cada vez.")

filas_barrido = []
for _SWEEP_CAT in SWEEP_CATEGORIAS:
    for _mm in SWEEP_MIN_MESES:
        MIN_MESES_POR_CATEGORIA = {_SWEEP_CAT: _mm}
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            exec(compile(C41_SWEEP_SRC, f'<barrido {_SWEEP_CAT} min_meses={_mm}>', 'exec'), globals())
        if not len(df_resumen):
            print(f"  {_SWEEP_CAT:<10} min_meses={_mm:>2} | [SKIP] celda sin acción o sin control tras el filtro")
            continue
        r = df_resumen.iloc[0]   # 1 sola cosecha x 1 sola categoría por corrida
        filas_barrido.append({
            'categoria': _SWEEP_CAT, 'min_meses': _mm,
            'AC_limpio': int(r.AC_limpio), 'CT_limpio': int(r.CT_limpio), 'pares': int(r.pares),
            'corr': float(r['corr']), 'ratio_vol': float(r.ratio_vol), 'valida': bool(r.valida),
            'ancova_pval': (None if pd.isna(r.ancova_pval) else float(r.ancova_pval)),
            'paralelismo_ok': bool(r.paralelismo_ok), 'reportable': bool(r.reportable),
            'n_min_requerido': n_minimo_requerido(int(r.AC_limpio), int(r.CT_limpio)),
            'margen_error_real_pct': margen_error_real(int(r.pares), int(r.AC_limpio))})
        print(f"  {_SWEEP_CAT:<10} min_meses={_mm:>2} | AC={int(r.AC_limpio):>4} CT={int(r.CT_limpio):>5} "
              f"pares={int(r.pares):>4} | corr={r['corr']:.3f} ratio_vol={r.ratio_vol:.3f} | "
              f"valida={bool(r.valida)} reportable={bool(r.reportable)}")

df_barrido_min_meses = pd.DataFrame(filas_barrido)

# Restaurar los resultados DEFINITIVOS -- el resto del notebook usa estos, no el barrido
df_matches, df_sin_pareja = _snap_definitivo['df_matches'], _snap_definitivo['df_sin_pareja']
df_funnel, df_resumen     = _snap_definitivo['df_funnel'], _snap_definitivo['df_resumen']
df_semillas, df_viabilidad = _snap_definitivo['df_semillas'], _snap_definitivo['df_viabilidad']
pools_out            = _snap_definitivo['pools_out']
CATEGORIAS           = _snap_definitivo['CATEGORIAS']
MIN_MESES_POR_CATEGORIA = _snap_definitivo['MIN_MESES_POR_CATEGORIA']
print(f"\n[RESTAURADO] df_resumen, df_matches, pools_out, etc. vuelven a los resultados "
      f"definitivos (MIN_MESES={MIN_MESES_POR_CATEGORIA}) -- el resto del notebook usa "
      f"estos, no el barrido.")

print("\n" + "=" * 100); print("BARRIDO MIN_MESES -- RESUMEN"); print("=" * 100)
if len(df_barrido_min_meses):
    print(df_barrido_min_meses.to_string(index=False))
else:
    print("[SIN RESULTADOS] ninguna combinación del barrido produjo pares.")

# ══════════════ celda 25 del notebook ══════════════
# =============================================================================
# PASO A — LLAVE DE PARES (acción + control) CON MES REAL DE ENTRADA
# =============================================================================
from dateutil.relativedelta import relativedelta

ALPHA       = 0.05
MODO_PRE    = 'mismo_mes'     # 'mismo_mes' | 'promedio'  (oficial: mismo_mes)
# MESES_POST ya viene de los parámetros globales (ancla mar -> corte ago = 6 meses)

assert len(df_matches), "df_matches vacío: el matching no produjo pares -> revisar embudo y semillas"

# Llave larga: una fila por par y rol. 'entrada' = mes REAL en que el cliente de ACCIÓN
# entró al piloto (propiedad del PAR: el control "arranca" cuando arranca su espejo).
llave = []
for r in df_matches.itertuples():
    entrada = r.min_per_ac if pd.notna(r.min_per_ac) else r.cosecha
    par_id = f"{r.cosecha.date()}|{r.des_categoria}|{r.cod_accion}"
    for cod, rol in ((r.cod_accion, 'accion'), (r.cod_control, 'control')):
        llave.append({'cod_cliente': cod, 'rol': rol,
                      'cosecha_grupo': r.cosecha, 'des_categoria': r.des_categoria,
                      'des_oficina': r.ofi_ac, 'entrada': entrada, 'par_id': par_id})
df_llave = pd.DataFrame(llave)
df_llave['entrada'] = pd.to_datetime(df_llave['entrada'])

print(f"Filas de llave: {len(df_llave):,}  (pares × 2 = {len(df_matches)*2:,})")
print(df_llave.groupby(['cosecha_grupo', 'des_categoria', 'rol']).size().unstack().to_string())
print("\nPares por mes REAL de entrada (lado acción):")
print(df_llave[df_llave.rol == 'accion'].groupby(['des_categoria', 'entrada']).size().unstack(0).to_string())

# Un control puede servir en varias categorías -> el merge lo duplica a propósito
base = base_seg_limpio.copy()
base['periodo'] = pd.to_datetime(base['periodo'])
df_trx = base.merge(df_llave, on=['cod_cliente', 'des_categoria'], how='inner')
df_trx['flg_accion'] = (df_trx['rol'] == 'accion').astype(int)
print(f"\nTransaccional marcada: {len(df_trx):,} filas | {df_trx.cod_cliente.nunique():,} clientes únicos")

# ══════════════ celda 26 del notebook ══════════════
# =============================================================================
# PASO B — PANEL MENSUAL POR PAR, ACUMULATIVO (función reutilizable)
# =============================================================================
# En cada mes calendario k (0 = ancla marzo) entran SOLO los pares cuyo cliente de acción
# ya entró (entrada <= mes). El panel no es balanceado a propósito: el n crece mes a mes.
#   mes_rel        = meses desde el ancla (calendario; es la llave de las bandas IC)
#   mes_exposicion = meses desde la entrada REAL del cliente (0 = su mes de alta)
CAP_PCTL = 99.5   # winsorización mensual por celda (mismo criterio que el surtido)

def construir_panel_pares(modo_pre='mismo_mes', verbose=True):
    paneles = []
    for (cos, cat), g in df_trx.groupby(['cosecha_grupo', 'des_categoria']):
        ini_pre = cos - relativedelta(months=12)
        fin_pre = cos - relativedelta(months=1)

        pares_celda = (df_llave[(df_llave.cosecha_grupo == cos) & (df_llave.des_categoria == cat)]
                       [['par_id', 'rol', 'des_oficina', 'entrada']].drop_duplicates())
        par_info = pares_celda.drop_duplicates('par_id').set_index('par_id')
        oficina_map, entrada_map = par_info['des_oficina'], par_info['entrada']

        px = (g.groupby(['par_id', 'rol', 'periodo'])['peso'].sum().clip(lower=0).reset_index())

        # Winsorización de outliers mensuales (p99.5 de la celda, igual a TODOS los meses,
        # pre y post, acción y control -> sin sesgo direccional)
        tope_peso = px.loc[px.peso > 0, 'peso'].quantile(CAP_PCTL / 100)
        filas_capadas = px[px.peso > tope_peso]
        if len(filas_capadas):
            if verbose:
                clientes_capados = (
                    g[g.peso > 0].groupby(['cod_cliente', 'par_id', 'rol', 'periodo'])['peso'].sum()
                     .reset_index()
                     .merge(filas_capadas[['par_id', 'rol', 'periodo']], on=['par_id', 'rol', 'periodo'])
                     [['cod_cliente', 'rol', 'periodo', 'peso']].sort_values('peso', ascending=False))
                print(f"  [WINSORIZACION] {cos.date()} | {cat}: {len(filas_capadas)} fila(s) mes-par con "
                      f"peso > p{CAP_PCTL} ({tope_peso:.1f} kg) -> capadas. Clientes:")
                print(clientes_capados.head(10).to_string(index=False))
            px['peso'] = px['peso'].clip(upper=tope_peso)

        for k in range(MESES_POST):
            mes_post = cos + relativedelta(months=k)
            if mes_post > g['periodo'].max():
                break
            activos = entrada_map.index[entrada_map <= mes_post]     # ACUMULATIVO
            if len(activos) == 0:
                continue

            if modo_pre == 'mismo_mes':
                mes_base = mes_post - relativedelta(months=12)
                if mes_base < ini_pre:
                    continue
                base_v = px[px.periodo == mes_base].set_index(['par_id', 'rol'])['peso']
            else:
                base_v = (px[(px.periodo >= ini_pre) & (px.periodo <= fin_pre)]
                            .groupby(['par_id', 'rol'])['peso'].sum() / 12)
            post_v = px[px.periodo == mes_post].set_index(['par_id', 'rol'])['peso']

            # REINDEX explícito sobre los pares ACTIVOS × rol -> ausencia = 0, no ausencia de fila
            idx_completo = pd.MultiIndex.from_frame(
                pares_celda[pares_celda.par_id.isin(activos)][['par_id', 'rol']])
            base_v = base_v.reindex(idx_completo, fill_value=0.0)
            post_v = post_v.reindex(idx_completo, fill_value=0.0)

            d = pd.DataFrame({'pre': base_v, 'post': post_v}).unstack('rol').fillna(0.0)
            d.columns = [f"{a}_{b}" for a, b in d.columns]
            for c in ['pre_accion', 'pre_control', 'post_accion', 'post_control']:
                if c not in d:
                    d[c] = 0.0

            d['delta_accion']  = d.post_accion  - d.pre_accion
            d['delta_control'] = d.post_control - d.pre_control
            d['did_kg']        = d.delta_accion - d.delta_control
            d = d.assign(cosecha_grupo=cos, des_categoria=cat,
                         periodo=mes_post, mes_rel=k).reset_index()
            d['des_oficina'] = d['par_id'].map(oficina_map)
            d['entrada'] = d['par_id'].map(entrada_map)
            d['mes_exposicion'] = ((mes_post.year - d['entrada'].dt.year) * 12
                                   + (mes_post.month - d['entrada'].dt.month))

            assert d.par_id.nunique() == len(activos), \
                f"n_pares inconsistente: {d.par_id.nunique()} vs {len(activos)} activos"
            paneles.append(d)

    return pd.concat(paneles, ignore_index=True)

df_panel = construir_panel_pares(MODO_PRE)
print(f"Panel: {len(df_panel):,} filas")
print("Pares ACTIVOS por mes calendario (crece: cada mes suma los que ya entraron):")
print(df_panel.groupby(['cosecha_grupo', 'des_categoria', 'mes_rel']).par_id.nunique().unstack().to_string())

# ══════════════ celda 27 del notebook ══════════════
# =============================================================================
# PASO C — BRECHA + IC BOOTSTRAP (función reutilizable)
# =============================================================================
# Antes esta lógica estaba duplicada (una vez por cada corte que se quisiera
# reportar: agregado, por oficina, MODO_PRE alterno...). Se extrae a función
# para poder reutilizarla con distintos group_cols sin repetir el bootstrap.

from mediciones.metodologia.inferencia import bootstrap_por_par as calcular_brechas_bootstrap, distribucion_brechas_cosechas as calcular_distribucion_brechas   # idénticas a las de surtido




# ── n mínimo dinámico por celda (población real de esa cosecha×categoría) ──
n_min_dict_agregado = {
    (row.cosecha, row.categoria): n_minimo_requerido(row.AC_limpio, row.CT_limpio)
    for row in df_resumen.itertuples()
}
print("n mínimo requerido por celda (5% margen / 95% confianza):")
for k, v in n_min_dict_agregado.items():
    print(f"  {k[0].date()} | {k[1]}: n_min={v}")

# ── Cálculo agregado: cosecha × categoría × mes_rel (el nivel "oficial" del reporte) ──
df_brechas = calcular_brechas_bootstrap(
    df_panel, ['cosecha_grupo','des_categoria','mes_rel'], n_min_dict=n_min_dict_agregado)
df_distribucion = calcular_distribucion_brechas(
    df_panel, ['cosecha_grupo','des_categoria','mes_rel'], n_min_dict=n_min_dict_agregado)

# ── Propagar 'reportable' (validez del matching + paralelismo) a las brechas ──
reportable_map = {(row.cosecha, row.categoria): row.reportable for row in df_resumen.itertuples()}
df_brechas['reportable'] = df_brechas.apply(
    lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
df_distribucion['reportable'] = df_distribucion.apply(
    lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
print(f"\nFilas de df_brechas que SÍ son reportables (grupo comparable): "
      f"{df_brechas.reportable.sum()} de {len(df_brechas)}")

print("\n" + "="*110)
print("BRECHAS DiD — RESUMEN (IC 95% para lectura rápida; ver bandas completas abajo)")
print("="*110)
print(df_brechas[['cosecha_grupo','des_categoria','mes_rel','n_pares',
                  'brecha_kg','brecha_pct','ic_pct_inf','ic_pct_sup',
                  'p_valor','significativo']].to_string(index=False))

print("\n" + "="*110)
print("BRECHAS DiD — INTERVALOS DE CONFIANZA 80% / 90% / 95% (mismo bootstrap, 3 anchos)")
print("="*110)
cols_ic = ['cosecha_grupo','des_categoria','mes_rel','n_pares','brecha_pct']
for nivel in NIVELES_IC:
    cols_ic += [f'ic_pct_inf_{nivel}', f'ic_pct_sup_{nivel}']
print(df_brechas[cols_ic].to_string(index=False))

# Consolidado por cosecha × categoría (promedio de meses post)
agg_dict = dict(meses=('mes_rel','count'), n_pares=('n_pares','mean'),
                brecha_pct_prom=('brecha_pct','mean'))
for nivel in NIVELES_IC:
    agg_dict[f'meses_signif_{nivel}'] = (f'significativo_{nivel}', 'sum')

df_brecha_resumen = (df_brechas.groupby(['cosecha_grupo','des_categoria'])
    .agg(**agg_dict).round(2).reset_index())
for nivel in NIVELES_IC:
    df_brecha_resumen[f'pct_meses_signif_{nivel}'] = (
        100*df_brecha_resumen[f'meses_signif_{nivel}']/df_brecha_resumen.meses).round(1)

print("\nCONSOLIDADO (significancia a 80% / 90% / 95%)")
print(df_brecha_resumen.to_string(index=False))

# ══════════════ celda 28 del notebook ══════════════
# =============================================================================
# REMUESTREO BOOTSTRAP — distribución de las 2,000 simulaciones (lámina "Distribución de Remuestreo")
# =============================================================================
# Reproduce EXACTAMENTE el flujo de números aleatorios de calcular_brechas_bootstrap (misma
# semilla, mismos grupos en el mismo orden, mismo salto n<30) -> los IC del gráfico coinciden
# con los de df_brechas al decimal. Se guardan las simulaciones para armar el gráfico nativo en PPT.
import os

def simulaciones_bootstrap(panel_df, group_cols, n_boot=N_BOOT, seed=SEED_BOOT):
    rng = np.random.default_rng(seed)
    out = {}
    for keys, g in panel_df.groupby(group_cols):
        n = len(g)
        if n < 30:                       # mismo corte duro que calcular_brechas_bootstrap
            continue
        did, base_ac = g['did_kg'].to_numpy(), g['pre_accion'].to_numpy()
        idx = rng.integers(0, n, size=(n_boot, n))
        did_b, base_b = did[idx].mean(axis=1), base_ac[idx].mean(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_b = np.where(base_b > 0, 100 * did_b / base_b, np.nan)
        out[keys if isinstance(keys, tuple) else (keys,)] = (
            pct_b, (100 * did.mean() / base_ac.mean() if base_ac.mean() > 0 else np.nan))
    return out

SIMS = simulaciones_bootstrap(df_panel, ['cosecha_grupo', 'des_categoria', 'mes_rel'])

# Tabla larga (cosecha, categoría, mes_rel, sim_id, brecha_pct simulada)
df_sim_bootstrap = pd.concat([
    pd.DataFrame({'cosecha_grupo': k[0], 'des_categoria': k[1], 'mes_rel': k[2],
                  'sim_id': np.arange(1, len(v[0]) + 1), 'brecha_pct_simulada': np.round(v[0], 4)})
    for k, v in SIMS.items()], ignore_index=True) if SIMS else pd.DataFrame()

# Chequeo: los percentiles de las simulaciones deben coincidir con df_brechas
for k, (pct_b, obs) in SIMS.items():
    fila = df_brechas[(df_brechas.cosecha_grupo == k[0]) & (df_brechas.des_categoria == k[1]) &
                      (df_brechas.mes_rel == k[2])].iloc[0]
    lo80, hi80 = np.nanpercentile(pct_b, [10, 90])
    assert abs(lo80 - fila['ic_pct_inf_80']) < 0.01 and abs(hi80 - fila['ic_pct_sup_80']) < 0.01, \
        f"IC80 del remuestreo no coincide con df_brechas en {k}"
print(f"Simulaciones guardadas: {len(df_sim_bootstrap):,} filas ({len(SIMS)} celdas × {N_BOOT:,}) | "
      f"IC del gráfico = IC de df_brechas (verificado)")

# Gráfico: una figura por cosecha, filas = categoría, columnas = mes calendario
OUTPUT_DIR = 'graficos_bootstrap'
os.makedirs(OUTPUT_DIR, exist_ok=True)
for cos in df_panel['cosecha_grupo'].unique():
    cats = sorted(df_panel[df_panel.cosecha_grupo == cos]['des_categoria'].unique())
    meses = sorted(df_panel[df_panel.cosecha_grupo == cos]['mes_rel'].unique())
    fig, axes = plt.subplots(len(cats), len(meses), figsize=(3.6 * len(meses), 3.4 * len(cats)), squeeze=False)
    for fi, cat in enumerate(cats):
        for ci, m in enumerate(meses):
            ax = axes[fi][ci]
            if (cos, cat, m) not in SIMS:
                ax.axis('off'); continue
            pct_b, obs = SIMS[(cos, cat, m)]
            pct_b = pct_b[~np.isnan(pct_b)]
            lo80, hi80 = np.percentile(pct_b, [10, 90]); lo95, hi95 = np.percentile(pct_b, [2.5, 97.5])
            ax.hist(pct_b, bins=40, color='#1f3864', alpha=0.75, edgecolor='white')
            ax.axvline(0, color='red', ls='--', lw=1.3, label='0')
            ax.axvline(obs, color='#c00000', ls='-', lw=1.6, label='observado')
            ax.axvline(lo80, color='#1b5e20', ls='-', lw=1.2); ax.axvline(hi80, color='#1b5e20', ls='-', lw=1.2, label='IC 80%')
            ax.axvline(lo95, color='#2e7d32', ls=':', lw=1.3); ax.axvline(hi95, color='#2e7d32', ls=':', lw=1.3, label='IC 95%')
            sig = 'SIG' if (lo95 > 0 or hi95 < 0) else 'ns'
            mes_txt = (cos + pd.DateOffset(months=int(m))).strftime('%b-%y')
            ax.set_title(f"{cat} | {mes_txt}\nobs={obs:+.1f}% | IC80[{lo80:+.1f}%,{hi80:+.1f}%] "
                         f"IC95[{lo95:+.1f}%,{hi95:+.1f}%] | {sig}", fontsize=7.2)
            if fi == len(cats) - 1: ax.set_xlabel("brecha % (promedio simulado)")
            if ci == 0: ax.set_ylabel("n° simulaciones")
            ax.tick_params(labelsize=7)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Distribución de las {N_BOOT:,} simulaciones bootstrap — {GRUPO_COSECHA.get(cos, cos.date())}\n"
                 f"(rojo = 0 | rojo oscuro = brecha observada | verde = IC 80% | verde punteado = IC 95%)",
                 fontsize=10, weight='bold')
    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, f"bootstrap_{cos.strftime('%Y-%m-%d')}.png")
    fig.savefig(ruta, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PNG: {ruta}")
    plt.show()

# ══════════════ celda 30 del notebook ══════════════
# =============================================================================
# BRECHA VIA ANCOVA — validación ADICIONAL a la brecha por bootstrap
# =============================================================================
# Corre sobre TODA la población limpia de acción y control de la celda (pools_out, antes
# de aparear 1:1), así que no depende de que el matching pase el filtro de ratio_vol.
# ACUMULATIVO: en el mes k solo cuentan los clientes de acción que ya entraron.
#   post = volumen del cliente en el mes; pre = MISMO cliente, mismo mes, año anterior
#   log1p(post) ~ grupo + log1p(pre); coef de 'grupo' -> brecha % (expm1), IC 80%
import statsmodels.formula.api as smf

filas_ancova = []
for (cosecha, cat), pool in pools_out.items():
    ac_ids = list(pool['ac'].index)
    ct_ids = list(pool['ct'].index)

    w = base_seg_limpio[(base_seg_limpio.des_categoria == cat) &
                         (base_seg_limpio.cod_cliente.isin(ac_ids + ct_ids))].copy()
    w['periodo'] = pd.to_datetime(w['periodo'])
    piv_a = w.pivot_table(index='cod_cliente', columns='periodo', values='peso', aggfunc='sum')

    for k in range(MESES_POST):
        mes_post = cosecha + relativedelta(months=k)
        mes_pre = mes_post - relativedelta(months=12)
        if mes_post not in piv_a.columns or mes_pre not in piv_a.columns:
            continue
        ac_k = [c for c in ac_ids if cli_min_per.get(c, cosecha) <= mes_post]   # ya entraron
        if len(ac_k) < 3:
            continue
        todos_ids = ac_k + ct_ids
        grupo = np.r_[np.ones(len(ac_k)), np.zeros(len(ct_ids))]

        dfl = pd.DataFrame({
            'grupo': grupo,
            'post': piv_a.reindex(todos_ids)[mes_post].fillna(0.0).clip(lower=0).to_numpy(),
            'pre':  piv_a.reindex(todos_ids)[mes_pre].fillna(0.0).clip(lower=0).to_numpy()})
        for col in ('post', 'pre'):   # winsorización p99.5, mismo criterio que el panel DiD
            dfl[col] = dfl[col].clip(upper=dfl[col].quantile(0.995))
        dfl['log_post'] = np.log1p(dfl['post'])
        dfl['log_pre'] = np.log1p(dfl['pre'])

        try:
            mod = smf.ols('log_post ~ grupo + log_pre', data=dfl).fit()
            coef = mod.params['grupo']
            ic = mod.conf_int(alpha=0.20)   # IC 80%
            p_valor = mod.pvalues['grupo']
            brecha_pct = np.expm1(coef) * 100
            ic_inf_pct = np.expm1(ic.loc['grupo', 0]) * 100
            ic_sup_pct = np.expm1(ic.loc['grupo', 1]) * 100
            signif = 'Sí' if p_valor < 0.05 else 'No'
        except Exception:
            brecha_pct = ic_inf_pct = ic_sup_pct = np.nan
            signif = 'No'

        filas_ancova.append({
            'Grupo': GRUPO_COSECHA.get(cosecha, '?'), 'Categoría': cat,
            'Mes': f'mes +{k}', 'Periodo': mes_post, 'N acción': len(ac_k), 'N control': len(ct_ids),
            'Brecha % (ANCOVA)': round(brecha_pct, 2) if pd.notna(brecha_pct) else np.nan,
            'IC 80% inf': round(ic_inf_pct, 2) if pd.notna(ic_inf_pct) else np.nan,
            'IC 80% sup': round(ic_sup_pct, 2) if pd.notna(ic_sup_pct) else np.nan,
            'Signif.': signif})

df_brecha_ancova = pd.DataFrame(filas_ancova)
print(df_brecha_ancova.to_string(index=False))

# ══════════════ celda 31 del notebook ══════════════
# =============================================================================
# BRECHA ACUMULADA — Método 2 (maduración): pool de pares-mes con >= 2 meses de EXPOSICIÓN
# =============================================================================
# En el surtido la maduración se medía como mes_rel >= 2 desde la cosecha. Aquí, con
# entradas escalonadas, la maduración se mide desde la entrada REAL de cada cliente
# (mes_exposicion >= 2), que es la generalización natural: cada par aporta solo los
# meses en que su cliente ya lleva >= 2 meses en el piloto.
# Se apoya en calcular_brechas_bootstrap (misma función, agrupa filas par×mes directo).
panel_metodo2 = df_panel[df_panel.mes_exposicion >= 2]

reportable_map = {(row.cosecha, row.categoria): row.reportable for row in df_resumen.itertuples()}

if len(panel_metodo2):
    df_brecha_acumulada = calcular_brechas_bootstrap(panel_metodo2, ['cosecha_grupo', 'des_categoria'])
    df_brecha_acumulada['metodo'] = 'maduracion_exposicion_>=2'
    df_brecha_acumulada['reportable'] = df_brecha_acumulada.apply(
        lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
    df_brecha_acumulada = df_brecha_acumulada.drop(columns='periodo', errors='ignore')
else:
    df_brecha_acumulada = pd.DataFrame()
    print("[SIN RESULTADOS] Ningún par alcanza todavía 2 meses de exposición.")

print(f"df_brecha_acumulada (Método 2): {len(df_brecha_acumulada):,} filas")
if len(df_brecha_acumulada):
    print(df_brecha_acumulada[['cosecha_grupo', 'des_categoria', 'n_pares', 'brecha_kg', 'brecha_pct',
                               'ic_pct_inf_80', 'ic_pct_sup_80', 'ic_pct_inf_95', 'ic_pct_sup_95',
                               'p_valor', 'reportable']].to_string(index=False))

# Vista adicional: brecha por MESES DE EXPOSICIÓN (0 = mes de alta, 1, 2...). Muestra cómo
# madura el efecto independientemente de cuándo entró cada cliente.
df_brechas_exposicion = calcular_brechas_bootstrap(
    df_panel, ['cosecha_grupo', 'des_categoria', 'mes_exposicion'])
if len(df_brechas_exposicion):
    df_brechas_exposicion = df_brechas_exposicion.drop(columns='periodo', errors='ignore')
    df_brechas_exposicion['reportable'] = df_brechas_exposicion.apply(
        lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
    print("\nBrecha por meses de exposición:")
    print(df_brechas_exposicion[['des_categoria', 'mes_exposicion', 'n_pares', 'brecha_pct',
                                 'ic_pct_inf_80', 'ic_pct_sup_80', 'significativo_80']].to_string(index=False))

# ══════════════ celda 32 del notebook ══════════════
# =============================================================================
# CHEQUEO DE SENSIBILIDAD — pre-período = PROMEDIO 12M (en vez de mismo mes)
# =============================================================================
# Reusa construir_panel_pares (mismo panel acumulativo) con modo_pre='promedio'. Guarda en
# df_panel_promedio / df_brechas_promedio: NO pisa df_panel / df_brechas (oficiales).
df_panel_promedio = construir_panel_pares('promedio', verbose=False)
df_brechas_promedio = calcular_brechas_bootstrap(
    df_panel_promedio, ['cosecha_grupo', 'des_categoria', 'mes_rel'])

comp = (df_brechas[['cosecha_grupo', 'des_categoria', 'mes_rel', 'brecha_pct', 'significativo']]
        .merge(df_brechas_promedio[['cosecha_grupo', 'des_categoria', 'mes_rel', 'brecha_pct', 'significativo']],
               on=['cosecha_grupo', 'des_categoria', 'mes_rel'], suffixes=('_mismo_mes', '_promedio')))
comp['cambia_significancia'] = comp.significativo_mismo_mes != comp.significativo_promedio

print("=" * 110)
print("SENSIBILIDAD: mismo_mes (oficial) vs promedio 12M — ¿cambia la conclusión?")
print("=" * 110)
print(comp.to_string(index=False))
if len(comp) and comp.cambia_significancia.any():
    print(f"\n[AVISO] {comp.cambia_significancia.sum()} celda(s) cambian de significativo/no-significativo "
          f"según el método de pre-período. Revisar antes de reportar.")
else:
    print("\nAmbos métodos coinciden en significancia para todas las celdas -> resultado robusto a esta decisión.")

# ══════════════ celda 33 del notebook ══════════════
# =============================================================================
# DISTRIBUCIÓN DE LAS BRECHAS POR PAR, EN CADA COSECHA × PERIODO
# =============================================================================
# df_distribucion ya se calculó en la celda anterior con calcular_distribucion_brechas()
# (antes esta celda repetía la misma lógica por separado).
print("\n" + "="*110)
print("DISTRIBUCIÓN DE BRECHAS POR PAR (no el promedio — la dispersión individual)")
print("="*110)
print(df_distribucion.to_string(index=False))

# ══════════════ celda 35 del notebook ══════════════
# =============================================================================
# BRECHAS POR OFICINA — mismo bootstrap, partido por des_oficina, filtrado a:
#  (a) celdas cosecha×categoría que son 'reportable' a nivel agregado (si el grupo
#      completo no es comparable, partirlo por oficina no lo arregla), Y
#  (b) oficinas con MIN_CT_OFICINA=30 alcanzado (piso fijo, IGUAL que v6 -- no la
#      fórmula dinámica, que aquí queda solo como columna informativa en df_viabilidad).
# Este filtro por oficina SÍ excluye: replica lo que hacía el notebook original (v6)
# con el chequeo de viabilidad por oficina -> evidenciar que no hay control suficiente
# y, por eso mismo, NO calcular una brecha con él.
# =============================================================================

viab_ok = df_viabilidad.rename(columns={'cosecha': 'cosecha_grupo', 'categoria': 'des_categoria'})
viab_ok['reportable_padre'] = viab_ok.apply(
    lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
viab_ok = viab_ok[viab_ok.viable & viab_ok.reportable_padre][
    ['cosecha_grupo', 'des_categoria', 'des_oficina', 'grupo']]

print(f"Celdas cosecha×categoría×oficina que pasan AMBOS filtros (comparable + n mínimo): "
      f"{len(viab_ok):,} de {len(df_viabilidad):,}")

panel_oficina_ok = df_panel.merge(
    viab_ok, on=['cosecha_grupo', 'des_categoria', 'des_oficina'], how='inner')

print(f"Panel filtrado: {len(panel_oficina_ok):,} de {len(df_panel):,} filas "
      f"({100*len(panel_oficina_ok)/max(1,len(df_panel)):.1f}%)")

# n mínimo dinámico por oficina (ya viene calculado en df_viabilidad, celda del matching)
n_min_dict_oficina = {
    (row.cosecha, row.categoria, row.des_oficina): row.n_min_requerido
    for row in df_viabilidad.itertuples()
}

df_brechas_oficina = calcular_brechas_bootstrap(
    panel_oficina_ok, ['cosecha_grupo', 'des_categoria', 'des_oficina', 'mes_rel'],
    n_min_dict=n_min_dict_oficina)
df_distribucion_oficina = calcular_distribucion_brechas(
    panel_oficina_ok, ['cosecha_grupo', 'des_categoria', 'des_oficina', 'mes_rel'],
    n_min_dict=n_min_dict_oficina)

print("\n" + "=" * 110)
print("BRECHAS DiD POR OFICINA (solo celdas comparables y con n mínimo real alcanzado)")
print("=" * 110)
if len(df_brechas_oficina):
    print(df_brechas_oficina[['cosecha_grupo', 'des_categoria', 'des_oficina', 'mes_rel', 'n_pares',
                              'brecha_kg', 'brecha_pct', 'ic_pct_inf', 'ic_pct_sup',
                              'p_valor', 'significativo']].to_string(index=False))
else:
    print("[SIN RESULTADOS] Ninguna combinación cosecha×categoría×oficina pasó ambos filtros. "
          "Reportar la brecha SOLO a nivel cosecha×categoría (ver df_brechas) y documentar "
          "la limitación por oficina abajo.")

# ── Documentar explícitamente qué combinaciones NO se pueden reportar por oficina,
#    separando la razón: celda padre no comparable vs. oficina sin n mínimo ──
todas_combos = df_panel[['cosecha_grupo', 'des_categoria', 'des_oficina']].drop_duplicates()
df_sin_control_oficina = todas_combos.merge(
    viab_ok[['cosecha_grupo', 'des_categoria', 'des_oficina']],
    on=['cosecha_grupo', 'des_categoria', 'des_oficina'], how='left', indicator=True)
df_sin_control_oficina = (df_sin_control_oficina[df_sin_control_oficina._merge == 'left_only']
                          .drop(columns='_merge'))
df_sin_control_oficina['motivo'] = df_sin_control_oficina.apply(
    lambda r: 'celda padre no comparable (matching estadístico: corr/ratio_vol)'
    if not reportable_map.get((r.cosecha_grupo, r.des_categoria), False)
    else 'oficina sin n mínimo suficiente (evidencia falta de control, como en v6)', axis=1)

print(f"\ncosecha×categoría×oficina SIN brecha propia reportable: {len(df_sin_control_oficina):,}")
print(df_sin_control_oficina.to_string(index=False))

# ══════════════ celda 37 del notebook ══════════════
# =============================================================================
# ATRIBUIBLE POR CLIENTE-CATEGORÍA (YoY capado a la banda IC80 de la brecha DiD)
# =============================================================================
# Lado ACCIÓN = TODO cliente tratado (matcheado + sin pareja), medido desde su ENTRADA REAL.
#   - antiguos: yoy_pct capado a [ic_pct_inf_80, ic_pct_sup_80] de su mes calendario (mes_rel)
#   - reactivados (existían, compraron 0 el año pasado): tope = ic_pct_sup_80 (>=0)
#   - NUEVOS ('no_existia_periodo_pasado'): SIN contrafactual -> volumen puro (toneladas_actual),
#     NUNCA una brecha ni una comparación contra un benchmark.
#   - clientes que existían pero compraron 0 este mes: fuera del atribuible (no es lo mismo
#     caer a cero que crecer menos de lo esperado).
# Se calcula también con IC95 como referencia; el oficial es IC80.

piv = (base_seg_limpio.groupby(['cod_cliente', 'des_categoria', 'periodo'])['peso']
       .sum().clip(lower=0).reset_index())
periodos_all = pd.date_range(piv['periodo'].min(), MAX_PERIODO, freq='MS')

cli_cos = base_seg_limpio.dropna(subset=['cosecha_agrupada']).groupby('cod_cliente')['cosecha_agrupada'].min()
cli_ent = base_seg_limpio.dropna(subset=['min_periodo_cosecha']).groupby('cod_cliente')['min_periodo_cosecha'].min()
print(f"Clientes de acción: {len(cli_cos):,}")

# fecha de creación (todos los MA) -> distingue 'no existía' de 'reactivado'
fec_creacion_map = fec_df.drop_duplicates('cod_cliente').set_index('cod_cliente')['fec_creacion']


def construir_panel_yoy(combos):
    """combos: cod_cliente, des_categoria, entrada, cosecha_grupo (1 fila por cliente×categoría).
    Devuelve el panel cliente×categoría×periodo desde la entrada, con año anterior (lag 12)."""
    combos = combos.drop_duplicates(['cod_cliente', 'des_categoria'])
    idx = pd.MultiIndex.from_frame(
        combos[['cod_cliente', 'des_categoria']].merge(pd.DataFrame({'periodo': periodos_all}), how='cross'))
    p = (piv.set_index(['cod_cliente', 'des_categoria', 'periodo'])['peso']
            .reindex(idx, fill_value=0.0).reset_index()
            .sort_values(['cod_cliente', 'des_categoria', 'periodo']))
    # lag 12 sobre malla mensual completa sin huecos
    p['peso_ant'] = p.groupby(['cod_cliente', 'des_categoria'])['peso'].shift(12)
    p = p.merge(combos[['cod_cliente', 'des_categoria', 'entrada', 'cosecha_grupo']],
                on=['cod_cliente', 'des_categoria'], how='left')
    p = p[p['periodo'] >= p['entrada']].dropna(subset=['peso_ant']).copy()
    p['mes_rel'] = ((p.periodo.dt.year - p.cosecha_grupo.dt.year) * 12
                    + (p.periodo.dt.month - p.cosecha_grupo.dt.month))
    p['mes_exposicion'] = ((p.periodo.dt.year - p.entrada.dt.year) * 12
                           + (p.periodo.dt.month - p.entrada.dt.month))
    p['toneladas_actual'] = p['peso'] / 1000
    p['toneladas_ant'] = p['peso_ant'] / 1000
    p['toneladas_variacion'] = p['toneladas_actual'] - p['toneladas_ant']
    p['flg_tendencia'] = np.select([p['toneladas_variacion'] > 0, p['toneladas_variacion'] < 0],
                                   ['crece', 'cae'], default='estable')
    return p


# ── LADO ACCIÓN ──────────────────────────────────────────────────────────────
combos_ac = (pd.DataFrame({'cod_cliente': cli_cos.index, 'cosecha_grupo': cli_cos.values})
             .assign(entrada=lambda d: d.cod_cliente.map(cli_ent))
             .merge(pd.DataFrame({'des_categoria': CATEGORIAS}), how='cross'))
combos_ac['entrada'] = combos_ac['entrada'].fillna(combos_ac['cosecha_grupo'])
panel = construir_panel_yoy(combos_ac)
print(f"Panel acción (desde entrada, con año anterior): {len(panel):,} filas | "
      f"mes_rel {panel.mes_rel.min()}..{panel.mes_rel.max()}")

# ── Clasificación del estado YoY ─────────────────────────────────────────────
panel['fec_creacion'] = panel['cod_cliente'].map(fec_creacion_map)
print(f"Clientes en panel sin fecha de creación: {panel.loc[panel.fec_creacion.isna(), 'cod_cliente'].nunique():,} "
      f"-> 'creación desconocida', no se asume nada")
panel['periodo_ant'] = panel['periodo'] - pd.DateOffset(months=12)

cond_no_existia = ((panel['peso_ant'] == 0) & (panel['peso'] > 0) &
                   panel['fec_creacion'].notna() & (panel['fec_creacion'] > panel['periodo_ant']))
cond_reactivado = (panel['peso_ant'] == 0) & (panel['peso'] > 0) & (~cond_no_existia)
cond_sigue_cero = (panel['peso_ant'] == 0) & (panel['peso'] == 0)
panel['flg_estado_yoy'] = np.select(
    [cond_no_existia, cond_reactivado, cond_sigue_cero],
    ['no_existia_periodo_pasado', 'reactivado', 'sin_compra_ambos'], default='normal')

panel['yoy_pct_raw'] = np.where(
    panel['peso_ant'] > 0, 100 * panel['toneladas_variacion'] / panel['toneladas_ant'],
    np.where(cond_reactivado, np.inf, np.where(cond_no_existia, np.nan, 0.0)))

# ── Bandas IC de la brecha DiD (por mes calendario) ──────────────────────────
cols_bandas = ['cosecha_grupo', 'des_categoria', 'mes_rel', 'reportable'] + \
              [f'ic_pct_{lado}_{n}' for n in (80, 90, 95) for lado in ('inf', 'sup')]
panel = panel.merge(df_brechas[cols_bandas], on=['cosecha_grupo', 'des_categoria', 'mes_rel'], how='left')
panel['flg_sin_banda'] = panel['ic_pct_inf_80'].isna() | panel['ic_pct_sup_80'].isna()
print(f"\nFilas SIN banda IC (celda cosecha×categoría×mes con <30 pares): "
      f"{panel.flg_sin_banda.sum():,} de {len(panel):,} ({100*panel.flg_sin_banda.mean():.1f}%) "
      f"-> no se les atribuye nada (NaN), no se les deja el YoY sin capar")


def aplicar_banda(p, nivel):
    """crec_atrib_{nivel}_pct y ton_atrib_{nivel}: YoY capado a la banda ICnivel."""
    inf, sup = p[f'ic_pct_inf_{nivel}'], p[f'ic_pct_sup_{nivel}']
    yoy = p['yoy_pct_raw'].replace(np.inf, np.nan)
    crec = yoy.clip(lower=inf, upper=sup)
    es_react = p['flg_estado_yoy'] == 'reactivado'
    es_nuevo = p['flg_estado_yoy'] == 'no_existia_periodo_pasado'
    crec = crec.where(~es_react, sup.clip(lower=0))     # reactivado: tope superior (>= 0)
    crec = crec.where(~es_nuevo, np.nan)                # nuevo: NUNCA atribuible por brecha
    crec = crec.where(inf.notna() & sup.notna(), np.nan)  # sin banda -> sin atribuible
    ton = pd.Series(np.where(p['toneladas_ant'] > 0, p['toneladas_ant'] * crec / 100, 0.0), index=p.index)
    ton = ton.where(crec.notna(), np.nan)
    p[f'crec_atrib_{nivel}_pct'] = crec
    p[f'ton_atrib_{nivel}'] = ton
    return p

panel = aplicar_banda(panel, 80)   # OFICIAL
panel = aplicar_banda(panel, 90)   # escenarios de IC (lámina de escenarios del PPT)
panel = aplicar_banda(panel, 95)   # referencia

yoy_c = panel['yoy_pct_raw'].replace(np.inf, np.nan)
# dtype object: True / False / NaN (sin banda o cliente nuevo -> no aplica). Un bool puro no admite NaN.
panel['flg_bajo_esperado']  = (yoy_c < panel['ic_pct_inf_80']).astype(object)
panel['flg_sobre_esperado'] = (yoy_c > panel['ic_pct_sup_80']).astype(object)
panel.loc[panel.flg_sin_banda | (panel.flg_estado_yoy == 'no_existia_periodo_pasado'),
          ['flg_bajo_esperado', 'flg_sobre_esperado']] = np.nan

# ── valor_atribuible: el número final, Suma directa en tabla dinámica ────────
es_nuevo = panel['flg_estado_yoy'] == 'no_existia_periodo_pasado'
es_activo = panel['toneladas_actual'] > 0
panel['valor_atribuible'] = np.select(
    [es_nuevo, es_activo], [panel['toneladas_actual'], panel['ton_atrib_80'].fillna(0.0)], default=0.0)
# Escenarios: mismo valor_atribuible pero topando a la banda IC90 / IC95 (80 = oficial)
for _n in (90, 95):
    panel[f'valor_atribuible_{_n}'] = np.select(
        [es_nuevo, es_activo], [panel['toneladas_actual'], panel[f'ton_atrib_{_n}'].fillna(0.0)], default=0.0)
panel['ton_con_crecimiento'] = np.where(~es_nuevo & es_activo & (panel['ton_atrib_80'] > 0), panel['valor_atribuible'], 0.0)
panel['ton_sin_crecimiento'] = np.where(~es_nuevo & es_activo & (panel['ton_atrib_80'] < 0), panel['valor_atribuible'], 0.0)
panel['ton_nuevos'] = np.where(es_nuevo, panel['valor_atribuible'], 0.0)
panel['ton_neto_sin_nuevos'] = panel['ton_con_crecimiento'] + panel['ton_sin_crecimiento']

desf = (panel['ton_neto_sin_nuevos'] + panel['ton_nuevos'] - panel['valor_atribuible']).abs()
assert (desf > 1e-6).sum() == 0, "con_crecimiento + sin_crecimiento + nuevos != valor_atribuible"
# Verificación del punto 3: nuevos con atribuible por brecha (debe ser 0)
assert panel.loc[es_nuevo, 'ton_atrib_80'].notna().sum() == 0, "un cliente nuevo tiene brecha atribuible"

df_incremental = panel.copy()
print(f"\ndf_incremental: {len(df_incremental):,} filas | {df_incremental.cod_cliente.nunique():,} clientes")
print("\nDistribución flg_estado_yoy:")
print(df_incremental['flg_estado_yoy'].value_counts(normalize=True).mul(100).round(1).to_string())
print(f"\nToneladas atribuibles (IC80) acumuladas: {df_incremental.valor_atribuible.sum():,.1f} t "
      f"| con crecimiento {df_incremental.ton_con_crecimiento.sum():,.1f} | "
      f"sin crecimiento {df_incremental.ton_sin_crecimiento.sum():,.1f} | "
      f"nuevos (volumen puro) {df_incremental.ton_nuevos.sum():,.1f}")

# ══════════════ celda 38 del notebook ══════════════
# =============================================================================
# TABLAS PARA EL PPT: escenarios de IC, nuevos vs antiguos, neteo, kg/cliente, cohorte de entrada
# =============================================================================
di = df_incremental.copy()
di['cohorte_entrada'] = di['entrada'].dt.strftime('%Y-%m')
es_nue = di['flg_estado_yoy'] == 'no_existia_periodo_pasado'
es_ant = (~es_nue) & (di['toneladas_actual'] > 0)         # antiguos activos (mismo criterio que las vistas)
di['_ant_bruto']  = np.where(es_ant, di['toneladas_variacion'], 0.0)
di['_nuevos']     = np.where(es_nue, di['toneladas_actual'], 0.0)
for n in (80, 90, 95):
    di[f'_ant_atrib_{n}'] = np.where(es_ant, di[f'ton_atrib_{n}'].fillna(0.0), 0.0)


def resumen_escenarios(grupo):
    g = di.groupby(grupo).agg(
        n_clientes=('cod_cliente', 'nunique'),
        ton_nuevos_bruto_eq_atrib=('_nuevos', 'sum'), ton_antiguos_bruto=('_ant_bruto', 'sum'),
        ton_antiguos_atrib_80=('_ant_atrib_80', 'sum'), ton_antiguos_atrib_90=('_ant_atrib_90', 'sum'),
        ton_antiguos_atrib_95=('_ant_atrib_95', 'sum')).reset_index()
    for n in (80, 90, 95):
        g[f'ton_total_atrib_{n}'] = g['ton_nuevos_bruto_eq_atrib'] + g[f'ton_antiguos_atrib_{n}']
    g['dif_pct_90_vs_80'] = 100 * (g['ton_total_atrib_90'] / g['ton_total_atrib_80'] - 1)
    g['dif_pct_95_vs_80'] = 100 * (g['ton_total_atrib_95'] / g['ton_total_atrib_80'] - 1)
    return g.round(3)

df_escenarios_ic = resumen_escenarios(['des_categoria'])
_tot = resumen_escenarios(['cosecha_grupo']).assign(des_categoria='TOTAL').drop(columns='cosecha_grupo')
_tot['n_clientes'] = di['cod_cliente'].nunique()   # únicos globales (no suma de categorías)
df_escenarios_ic = pd.concat([df_escenarios_ic, _tot], ignore_index=True)
df_escenarios_ic_mes = resumen_escenarios(['des_categoria', 'periodo'])

# Cuadre: el escenario 80 es el atribuible oficial (tolerancia 1e-3: la tabla va redondeada a 3 decimales)
_tot80 = df_escenarios_ic.loc[df_escenarios_ic.des_categoria == 'TOTAL', 'ton_total_atrib_80'].iloc[0]
assert abs(_tot80 - df_incremental['valor_atribuible'].sum()) < 1e-3, "escenario IC80 no cuadra con valor_atribuible"

# Neteo de antiguos: quién crece (+) y quién cae (-), topados a IC80
_ant = di[es_ant]
df_neteo = (_ant.groupby(['des_categoria', 'flg_tendencia'])
            .agg(n_clientes=('cod_cliente', 'nunique'), n_cliente_mes=('cod_cliente', 'size'),
                 ton_atrib_80=('ton_atrib_80', 'sum'), ton_variacion_bruta=('toneladas_variacion', 'sum'))
            .reset_index())
_neto = (df_neteo.groupby('des_categoria')[['n_cliente_mes', 'ton_atrib_80', 'ton_variacion_bruta']].sum().reset_index()
         .assign(flg_tendencia='NETO'))
_neto['n_clientes'] = _neto['des_categoria'].map(_ant.groupby('des_categoria')['cod_cliente'].nunique())
df_neteo = pd.concat([df_neteo, _neto], ignore_index=True).round(3)

# Nuevos (volumen puro): N clientes, kg/cliente y toneladas por cohorte de entrada × mes
_nue = di[es_nue]
df_nuevos_kg = (_nue.groupby(['des_categoria', 'cohorte_entrada', 'periodo'])
                .agg(n_clientes=('cod_cliente', 'nunique'), toneladas=('toneladas_actual', 'sum')).reset_index())
df_nuevos_kg['kg_por_cliente'] = 1000 * df_nuevos_kg['toneladas'] / df_nuevos_kg['n_clientes']
# TOTAL: n_clientes = suma de clientes-mes (igual que el PPT: 8,023 = suma de los N mensuales), no únicos
_tn = (_nue.groupby('des_categoria').agg(n_clientes=('cod_cliente', 'size'), toneladas=('toneladas_actual', 'sum'))
       .reset_index().assign(cohorte_entrada='TOTAL', periodo=pd.NaT))
_tn['kg_por_cliente'] = 1000 * _tn['toneladas'] / _tn['n_clientes']
df_nuevos_kg = pd.concat([df_nuevos_kg, _tn], ignore_index=True).round(3)

# Toneladas atribuibles (IC80) por COHORTE DE ENTRADA × mes calendario (equivale a las filas G01/G02/G03
# del PPT). La brecha % por cohorte de entrada NO se calcula: cada cohorte de abr-jun tiene <30 pares.
di['_ton_total_80'] = di['_nuevos'] + di['_ant_atrib_80']
df_ton_cohorte = (di.groupby(['des_categoria', 'cohorte_entrada', 'periodo'])
                  .agg(n_clientes=('cod_cliente', 'nunique'), ton_atrib_antiguos_80=('_ant_atrib_80', 'sum'),
                       ton_nuevos_puro=('_nuevos', 'sum'), ton_atrib_total_80=('_ton_total_80', 'sum'))
                  .reset_index().round(3))

print("ESCENARIOS DE IC — toneladas atribuibles totales (nuevos + antiguos topados a cada banda)")
print(df_escenarios_ic[['des_categoria', 'ton_nuevos_bruto_eq_atrib', 'ton_antiguos_bruto', 'ton_total_atrib_80',
                        'ton_total_atrib_90', 'ton_total_atrib_95', 'dif_pct_90_vs_80', 'dif_pct_95_vs_80']].to_string(index=False))
print("\nNETEO DE ANTIGUOS (IC80)"); print(df_neteo.to_string(index=False))
print("\nNUEVOS — kg por cliente (total por categoría)"); print(df_nuevos_kg[df_nuevos_kg.cohorte_entrada == 'TOTAL'].to_string(index=False))

# ══════════════ celda 40 del notebook ══════════════
# =============================================================================
# BASE COMPLETA — accion_matcheado + accion_sin_pareja + control_espejo (punto 5)
# =============================================================================
# Atributos por cliente (última fila observada): oficina, distribuidora, segmento
cli_attrs = (base_seg_limpio.sort_values('periodo').groupby('cod_cliente').tail(1)
             .set_index('cod_cliente')[['des_oficina', 'nom_compania', 'des_segmento_transaccional']])

# ── 1) ACCIÓN: rol según si tiene par en el matching de ESA categoría ────────
match_keys = (df_matches[['cod_accion', 'des_categoria', 'cosecha', 'cod_control']].drop_duplicates()
              .rename(columns={'cod_accion': 'cod_cliente', 'cosecha': 'cosecha_grupo',
                               'cod_control': 'cod_cliente_par'}))
match_keys['rol'] = 'accion_matcheado'
df_inc_rol = df_incremental.merge(match_keys, on=['cod_cliente', 'des_categoria', 'cosecha_grupo'], how='left')
df_inc_rol['rol'] = df_inc_rol['rol'].fillna('accion_sin_pareja')
print("Rol dentro del lado acción (cliente-categoría-mes):")
print(df_inc_rol['rol'].value_counts().to_string())

# ── 2) CONTROL: mismo panel YoY, desde la entrada de SU espejo ───────────────
ctl = df_llave[df_llave.rol == 'control']
combos_ct = (ctl.groupby(['cod_cliente', 'des_categoria'], as_index=False)
                .agg(entrada=('entrada', 'min'), cosecha_grupo=('cosecha_grupo', 'min')))
panel_ct = construir_panel_yoy(combos_ct)
panel_ct['rol'] = 'control_espejo'
par_ct = (df_matches[['cod_control', 'des_categoria', 'cod_accion']].drop_duplicates(['cod_control', 'des_categoria'])
          .rename(columns={'cod_control': 'cod_cliente', 'cod_accion': 'cod_cliente_par'}))
panel_ct = panel_ct.merge(par_ct, on=['cod_cliente', 'des_categoria'], how='left')
print(f"\nPanel de control: {len(panel_ct):,} filas | {panel_ct.cod_cliente.nunique():,} controles únicos")

# ── 3) Unificar esquema ──────────────────────────────────────────────────────
df_inc_rol = df_inc_rol.rename(columns={'entrada': 'min_periodo_cosecha_real'})
panel_ct = panel_ct.rename(columns={'entrada': 'min_periodo_cosecha_real'})
COLS_BASE_COMPLETA = [
    'cod_cliente', 'rol', 'cod_cliente_par', 'des_categoria', 'des_oficina', 'nom_compania',
    'des_segmento_transaccional', 'cosecha_grupo', 'min_periodo_cosecha_real', 'periodo', 'mes_rel',
    'mes_exposicion', 'toneladas_ant', 'toneladas_actual', 'toneladas_variacion', 'yoy_pct_raw',
    'ic_pct_inf_80', 'ic_pct_sup_80', 'ic_pct_inf_90', 'ic_pct_sup_90', 'ic_pct_inf_95', 'ic_pct_sup_95',
    'crec_atrib_80_pct', 'ton_atrib_80', 'crec_atrib_90_pct', 'ton_atrib_90', 'crec_atrib_95_pct', 'ton_atrib_95',
    'valor_atribuible', 'valor_atribuible_90', 'valor_atribuible_95',
    'ton_con_crecimiento', 'ton_sin_crecimiento', 'ton_nuevos', 'ton_neto_sin_nuevos',
    'flg_estado_yoy', 'flg_bajo_esperado', 'flg_sobre_esperado', 'flg_sin_banda', 'flg_tendencia', 'reportable']
for d_ in (df_inc_rol, panel_ct):
    for c in ('des_oficina', 'nom_compania', 'des_segmento_transaccional'):
        d_[c] = d_['cod_cliente'].map(cli_attrs[c])
    for c in COLS_BASE_COMPLETA:
        if c not in d_.columns:
            d_[c] = np.nan
df_base_completa = pd.concat([df_inc_rol[COLS_BASE_COMPLETA], panel_ct[COLS_BASE_COMPLETA]], ignore_index=True)

# 'reportable' del lado control: el de su celda cosecha×categoría (informativo, no filtra)
rep_df = df_resumen[['cosecha', 'categoria', 'reportable']].rename(
    columns={'cosecha': 'cosecha_grupo', 'categoria': 'des_categoria', 'reportable': 'rep_celda'})
df_base_completa = df_base_completa.merge(rep_df, on=['cosecha_grupo', 'des_categoria'], how='left')
df_base_completa['reportable'] = df_base_completa['reportable'].fillna(df_base_completa['rep_celda'])
df_base_completa = df_base_completa.drop(columns='rep_celda')

print(f"\ndf_base_completa: {len(df_base_completa):,} filas | {df_base_completa.cod_cliente.nunique():,} clientes únicos")
print(df_base_completa.groupby('rol').agg(clientes=('cod_cliente', 'nunique'),
                                          toneladas_actual=('toneladas_actual', 'sum')).round(1).to_string())

# ══════════════ celda 42 del notebook ══════════════
# =============================================================================
# VISTAS EXCEL — BRECHA y TONELADAS: global / categoría / oficina×categoría /
# oficina×segmento×categoría / segmento×categoría. Formato largo (periodo + mes_rel juntos).
# Sin retroactivo (no aplica: marzo es la cosecha más temprana).
# =============================================================================
segmento_map = cli_attrs['des_segmento_transaccional']
nom_map = cli_attrs['nom_compania']

# par -> cliente de acción (segmento de quien recibió el piloto)
par_accion_map = df_llave[df_llave.rol == 'accion'].drop_duplicates('par_id').set_index('par_id')['cod_cliente']
df_panel_seg = df_panel.copy()
df_panel_seg['cod_cliente_accion'] = df_panel_seg['par_id'].map(par_accion_map)
df_panel_seg['des_segmento_transaccional'] = df_panel_seg['cod_cliente_accion'].map(segmento_map)
df_panel_seg['nom_compania_accion'] = df_panel_seg['cod_cliente_accion'].map(nom_map)
print(f"Pares sin segmento: {df_panel_seg['des_segmento_transaccional'].isna().sum():,} de {len(df_panel_seg):,}")

base_vistas = df_incremental.copy()
base_vistas['des_oficina'] = base_vistas['cod_cliente'].map(cli_attrs['des_oficina'])
base_vistas['des_segmento_transaccional'] = base_vistas['cod_cliente'].map(segmento_map)

COLS_IC = (['n_pares', 'brecha_kg', 'brecha_pct', 'p_valor'] +
           [f'ic_pct_inf_{n}' for n in NIVELES_IC] + [f'ic_pct_sup_{n}' for n in NIVELES_IC] +
           [f'significativo_{n}' for n in NIVELES_IC])


def construir_vista_brecha(extra_cols, nombre_vista):
    group_cols = ['cosecha_grupo'] + extra_cols + ['mes_rel']
    panel_v = df_panel_seg
    if 'des_segmento_transaccional' in extra_cols:
        panel_v = panel_v.dropna(subset=['des_segmento_transaccional'])
    out = calcular_brechas_bootstrap(panel_v, group_cols)
    if out.empty:
        print(f"Vista brecha '{nombre_vista}': 0 filas (ninguna combinación alcanza n>=30)")
        return out
    out['reportable'] = (out.apply(lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
                         if 'des_categoria' in extra_cols else np.nan)
    out = out.drop(columns=['periodo'], errors='ignore')
    out['periodo'] = out.apply(lambda r: r.cosecha_grupo + pd.DateOffset(months=int(r.mes_rel)), axis=1)
    out['cosecha_label'] = out['cosecha_grupo'].map(GRUPO_COSECHA)
    out = out.sort_values(extra_cols + ['cosecha_grupo', 'mes_rel']).reset_index(drop=True)
    print(f"Vista brecha '{nombre_vista}': {len(out):,} filas")
    return out


def construir_vista_toneladas(extra_cols, nombre_vista):
    group_cols = ['cosecha_grupo'] + extra_cols + ['periodo']
    bv = base_vistas
    if 'des_segmento_transaccional' in extra_cols:
        bv = bv.dropna(subset=['des_segmento_transaccional'])
    es_activo = bv['toneladas_actual'] > 0
    ant = bv[(bv.flg_estado_yoy != 'no_existia_periodo_pasado') & es_activo]   # antiguos activos (capados IC80)
    nue = bv[bv.flg_estado_yoy == 'no_existia_periodo_pasado']                 # nuevos: volumen puro
    g_ant = ant.groupby(group_cols)['ton_atrib_80'].sum().rename('ton_atrib_antiguos_80')
    g_nue = nue.groupby(group_cols)['toneladas_actual'].sum().rename('ton_nuevos_puro')
    n_ant = ant.groupby(group_cols)['cod_cliente'].nunique().rename('n_clientes_antiguos')
    n_nue = nue.groupby(group_cols)['cod_cliente'].nunique().rename('n_clientes_nuevos')
    out = pd.concat([g_ant, g_nue, n_ant, n_nue], axis=1).fillna(0.0).reset_index()
    out['ton_atrib_total_80'] = out['ton_atrib_antiguos_80'] + out['ton_nuevos_puro']
    out['mes_rel'] = ((out.periodo.dt.year - out.cosecha_grupo.dt.year) * 12
                      + (out.periodo.dt.month - out.cosecha_grupo.dt.month))
    if 'des_categoria' in extra_cols:
        out['reportable'] = out.apply(lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
    out['cosecha_label'] = out['cosecha_grupo'].map(GRUPO_COSECHA)
    out = out.sort_values(extra_cols + ['cosecha_grupo', 'mes_rel']).reset_index(drop=True)
    print(f"Vista toneladas '{nombre_vista}': {len(out):,} filas")
    return out


NIVELES_VISTA = {
    'Global': [], 'Categoria': ['des_categoria'],
    'Oficina x Cat': ['des_oficina', 'des_categoria'],
    'Ofi x Seg x Cat': ['des_oficina', 'des_segmento_transaccional', 'des_categoria'],
    'Segmento x Cat': ['des_segmento_transaccional', 'des_categoria']}
vistas_brecha = {n: construir_vista_brecha(c, n) for n, c in NIVELES_VISTA.items()}
vistas_ton    = {n: construir_vista_toneladas(c, n) for n, c in NIVELES_VISTA.items()}

# ══════════════ celda 44 del notebook ══════════════
# =============================================================================
# INKA CHIPS -- histórico de ventas (canal MA), para cruzar con la población de
# acción YA DEFINIDA en el notebook (df_cosecha) -- sin volver a consultar
# población/cosecha en BigQuery. Histórico completo (no solo periodo atribuible)
# porque hace falta el AÑO ANTERIOR de cada cliente-mes para el diferencial YoY.
# =============================================================================

sql_inkachips_ventas = f"""
SELECT
  periodo,
  cod_cliente_dex AS cod_cliente,
  des_categoria,
  SUM(num_peso_real) AS peso
FROM `acpe-prod-dlk-shrd-gld.delivery_reporte_comercial.cr_venta_sell_out`
WHERE cod_grupo_precio_dex = 'MA'
  AND periodo BETWEEN '2024-01-01' AND '{MES_CIERRE:%Y-%m-%d}'
  AND des_marca = 'Inka Chips'   -- único filtro: toda categoría donde vende esta marca
GROUP BY periodo, cod_cliente_dex, des_categoria
"""

inkachips_ventas = ejecutar_grilla(sql_inkachips_ventas, nombre_cache='champion_inkachips_ventas' + _SUF)
inkachips_ventas['periodo'] = pd.to_datetime(inkachips_ventas['periodo'])
# BigQuery devuelve SUM(num_peso_real) como decimal.Decimal (columna object), no float ->
# revienta la resta toneladas_actual - toneladas_ant más abajo si no se convierte acá.
inkachips_ventas['peso'] = inkachips_ventas['peso'].astype(float)

print(f"Inka Chips -- histórico de ventas (canal MA, 2024-01 a {MAX_PERIODO.date()}): "
      f"{len(inkachips_ventas):,} filas cliente-categoría-mes | "
      f"{inkachips_ventas.cod_cliente.nunique():,} clientes distintos con alguna compra")

# ── Población: la MISMA de Champion (df_cosecha), NO cr_surtido_ideal ──────────
# df_cosecha ya tiene cod_cliente_dex + min_periodo_cosecha para los 334 clientes que
# recibieron el piloto (viene de la celda TAG, más arriba) -- en formato DEX, compatible
# con cod_cliente_dex de la tabla de ventas (a diferencia de cr_surtido_ideal.cod_cliente).
poblacion_inka = df_cosecha.rename(columns={'cod_cliente_dex': 'cod_cliente'})[['cod_cliente', 'min_periodo_cosecha']]
print(f"Población de acción (df_cosecha, reutilizada): {len(poblacion_inka):,} clientes | "
      f"entrada mín={poblacion_inka.min_periodo_cosecha.min().date()} máx={poblacion_inka.min_periodo_cosecha.max().date()}")

n_interseccion = inkachips_ventas.cod_cliente.isin(poblacion_inka.cod_cliente).sum()
print(f"Filas de venta Inka Chips que cruzan con la población de acción: {n_interseccion:,} de {len(inkachips_ventas):,}")

# ══════════════ celda 45 del notebook ══════════════
# =============================================================================
# INKA CHIPS -- SIN grupo control, atribuible = DIFERENCIAL YoY (este mes vs el
# mismo mes del año pasado, sin piso en 0: si el cliente compró MENOS que el año
# pasado, el diferencial sale negativo y se reporta tal cual). Población = acción
# de Champion (df_cosecha), acumulativa desde el mes real de entrada de cada cliente.
# =============================================================================

# ── Restringir a la población de acción, agregar su mes de entrada ─────────────
inka = inkachips_ventas.merge(poblacion_inka, on='cod_cliente', how='inner')

# ── Año anterior: mismo cliente-categoría, periodo - 12 meses -> self-merge sobre
# el HISTÓRICO completo (por eso la query trae desde 2024-01, no solo el periodo
# atribuible -- sin esas filas de "año pasado" no habría con qué calcular el diferencial).
inka['periodo_ant'] = inka['periodo'] - pd.DateOffset(months=12)
peso_ant_lookup = inkachips_ventas[['cod_cliente', 'des_categoria', 'periodo', 'peso']].rename(
    columns={'periodo': 'periodo_ant', 'peso': 'peso_ant'})
inka = inka.merge(peso_ant_lookup, on=['cod_cliente', 'des_categoria', 'periodo_ant'], how='left')
inka['peso_ant'] = inka['peso_ant'].fillna(0.0)

# ── Filtrar a las filas ATRIBUIBLES: periodo >= mes real de entrada del cliente
# (acumulativo, mismo criterio que el resto del notebook) Y periodo <= corte.
papas = inka[(inka['periodo'] >= inka['min_periodo_cosecha']) & (inka['periodo'] <= MAX_PERIODO)].copy()

papas['toneladas_actual'] = papas['peso'] / 1000
papas['toneladas_ant'] = papas['peso_ant'] / 1000
papas['valor_atribuible'] = papas['toneladas_actual'] - papas['toneladas_ant']  # diferencial YoY, con signo

# Agrupar estas categorías reales de venta como una sola 'Inka Chips' -- el resto de
# des_categoria que pueda traer la marca se deja tal cual viene (sin recodificar).
# Ajustar esta lista si BigQuery trae otras categorías para esta marca en Minimayorista.
CATEGORIAS_INKACHIPS = ['Golosinas y Snacks SSCC', 'Papas Fritas', 'Platano Frito', 'Vegetales']
papas['des_categoria'] = papas['des_categoria'].where(
    ~papas['des_categoria'].isin(CATEGORIAS_INKACHIPS), 'Inka Chips')

# Diagnóstico informativo (NO filtra población): overlap con los clientes acción de
# Galletas/Cereales (que ahora es, por construcción, un subconjunto de la MISMA
# población df_cosecha -- todo cliente Inka Chips atribuible YA recibió el piloto).
rol_gc_map = (df_base_completa[df_base_completa.rol.isin(['accion_matcheado', 'accion_sin_pareja'])]
              .drop_duplicates('cod_cliente').set_index('cod_cliente')['rol'])
papas['rol_gc'] = papas['cod_cliente'].map(rol_gc_map)  # NaN = no tiene compra de Galletas/Cereales (pero sí recibió el piloto)

n_total_inka = papas['cod_cliente'].nunique()
n_solo_inka = papas.loc[papas['rol_gc'].isna(), 'cod_cliente'].nunique()
print(f"Inka Chips -- clientes con compra atribuible: {n_total_inka:,} de {len(poblacion_inka):,} en la población | "
      f"{n_solo_inka:,} sin compra de Galletas/Cereales (pero sí recibieron el piloto)")

# 'mes_rel' referencial (vs. el ancla única de Champion, para comparar con las demás
# vistas) -- no determina el filtro de atribución, que ya se hizo por entrada real arriba.
papas['mes_rel'] = ((papas.periodo.dt.year - COSECHA_ANCLA.year) * 12 + (papas.periodo.dt.month - COSECHA_ANCLA.month))

vista_inkachips = (papas.groupby('periodo')['valor_atribuible'].sum()).rename('toneladas_inkachips').reset_index()
vista_inkachips = vista_inkachips.sort_values('periodo').reset_index(drop=True)

print(f"\nToneladas atribuibles Inka Chips (diferencial YoY, sin capado ni IC): "
      f"{vista_inkachips.toneladas_inkachips.sum():,.2f} t")
print(vista_inkachips.to_string(index=False))

# ── Diagnóstico: desglose por des_categoria real (ya con 'Inka Chips' agrupado) ──
diag_cat_inka = (papas.groupby(['periodo', 'des_categoria'])['valor_atribuible'].sum()).rename('t').reset_index()
print("\nDesglose por des_categoria (diferencial YoY):")
print(diag_cat_inka.pivot_table(index='periodo', columns='des_categoria', values='t').round(2).to_string())

# ── Se coloca como TOTAL: se suma al TOTAL de toneladas de la vista Global ──
# (no reemplaza a Cereales/Galletas, se agrega como componente adicional, igual
# que "nuevos" se agrega a "antiguos" en las demás vistas).
vistas_ton['Global'] = vistas_ton['Global'].merge(vista_inkachips, on='periodo', how='left')
vistas_ton['Global']['toneladas_inkachips'] = vistas_ton['Global']['toneladas_inkachips'].fillna(0.0)
vistas_ton['Global']['ton_atrib_total_80'] = (
    vistas_ton['Global']['ton_atrib_total_80'] + vistas_ton['Global']['toneladas_inkachips'])

print(f"\nTon Global CON Inka Chips sumado al TOTAL: "
      f"{vistas_ton['Global']['toneladas_inkachips'].sum():,.2f} t agregadas")

# ══════════════ celda 46 del notebook ══════════════
# =============================================================================
# BASES DE CÁLCULO (para tablas dinámicas: la Suma / Promedio simple = número final)
# =============================================================================
_bt = base_vistas.copy()
_bt['nom_compania'] = _bt['cod_cliente'].map(nom_map)
_bt = _bt.rename(columns={'entrada': 'min_periodo_cosecha_real'})
base_calculo_toneladas = _bt[[
    'cod_cliente', 'des_categoria', 'des_oficina', 'des_segmento_transaccional', 'nom_compania',
    'cosecha_grupo', 'min_periodo_cosecha_real', 'periodo', 'mes_rel', 'mes_exposicion',
    'toneladas_ant', 'toneladas_actual', 'yoy_pct_raw', 'ic_pct_inf_80', 'ic_pct_sup_80',
    'flg_estado_yoy', 'crec_atrib_80_pct', 'ton_atrib_80', 'valor_atribuible',
    'ton_con_crecimiento', 'ton_sin_crecimiento', 'ton_nuevos', 'ton_neto_sin_nuevos',
    'flg_sin_banda', 'reportable']].sort_values(['cod_cliente', 'des_categoria', 'periodo']).reset_index(drop=True)
del _bt

# Base BRECHA (par × mes): brecha_pct de la celda repetida en cada fila -> el PROMEDIO en la
# dinámica equivale al total ponderado por n_pares
brecha_celda = df_brechas[['cosecha_grupo', 'des_categoria', 'mes_rel', 'brecha_pct', 'reportable']].rename(
    columns={'brecha_pct': 'brecha_pct_celda'})
base_calculo_brecha = (df_panel_seg[['par_id', 'cod_cliente_accion', 'nom_compania_accion', 'cosecha_grupo',
                                     'des_categoria', 'des_oficina', 'des_segmento_transaccional', 'entrada',
                                     'periodo', 'mes_rel', 'mes_exposicion', 'pre_accion', 'post_accion',
                                     'pre_control', 'post_control', 'did_kg']]
                       .merge(brecha_celda, on=['cosecha_grupo', 'des_categoria', 'mes_rel'], how='left')
                       .sort_values(['des_categoria', 'periodo', 'par_id']).reset_index(drop=True))
base_calculo_brecha['cosecha_label'] = base_calculo_brecha['cosecha_grupo'].map(GRUPO_COSECHA)

print(f"base_calculo_toneladas: {len(base_calculo_toneladas):,} filas | base_calculo_brecha: {len(base_calculo_brecha):,} filas")

# ══════════════ celda 48 del notebook ══════════════
# =============================================================================
# HELPERS DE EXPORTACION A EXCEL — estilos + escribir_hoja(), usados por la
# 'Base de calculo' y las 'Vistas Excel' mas abajo. Ya no arma un Excel propio
# aqui -- los unicos 2 exportables del notebook son esos dos.
# =============================================================================
from mediciones.excel_estilos import (Workbook, Font, PatternFill, Alignment, Border, Side, get_column_letter, CellIsRule, FONT_NAME, HEADER_FILL, HEADER_FONT, BODY_FONT, ALERT_FILL, OK_FILL, THIN, BORDER, UMBRAL_ESTILO_CELDA, escribir_hoja)   # compartido con la familia de cosechas (antes copiado aquí, idéntico)

# ══════════════ celda 49 del notebook ══════════════
# =============================================================================
# EXPORT — 3 archivos Excel
#   1) metricas_champion_mm.xlsx          : nivel agregado (embudo, viabilidad, paralelismo, brechas + IC, ...)
#   2) base_completa_clientes_champion.xlsx : nivel cliente (base completa + bases de cálculo + sin pareja)
#   3) vistas_champion_mm.xlsx            : global / categoría / oficina×cat / oficina×seg×cat / segmento×cat
# =============================================================================
_MES_TXT = MES_CIERRE.strftime('%Y-%m')   # los 3 Excel llevan el mes de cierre (no se pisa la corrida anterior)
RUTA_EXCEL_METRICAS = f'metricas_champion_mm_{_MES_TXT}.xlsx'
RUTA_EXCEL_BASE     = f'base_completa_clientes_champion_inka_{_MES_TXT}.xlsx'
RUTA_EXCEL_VISTAS   = f'vistas_champion_mm_{_MES_TXT}.xlsx'

# n mínimo requerido por celda (5% / 95%) y margen real logrado, para reportar junto al embudo
df_tamano_muestra = df_resumen[['grupo', 'categoria', 'AC_limpio', 'CT_limpio', 'AC_muestra', 'pares']].copy()
df_tamano_muestra['n_min_requerido_5pct_95conf'] = [n_minimo_requerido(a, c) for a, c in
                                                     zip(df_tamano_muestra.AC_limpio, df_tamano_muestra.CT_limpio)]
df_tamano_muestra['margen_error_real_pct_pares'] = [margen_error_real(p, a) for p, a in
                                                     zip(df_tamano_muestra.pares, df_tamano_muestra.AC_limpio)]

hojas_metricas = {
    '0 Embudo acumulado GA-GC': df_embudo_acum,
    '1 Embudo de limpieza': df_funnel,
    '2 Viabilidad x oficina': df_viabilidad,
    '3 Tamano de muestra': df_tamano_muestra,
    '4 Paralelismo y validez': df_resumen,
    '5 Brechas DiD + IC': df_brechas,
    '6 Consolidado brechas': df_brecha_resumen,
    '7 Brecha acumulada (M2)': df_brecha_acumulada,
    '8 Brecha x exposicion': df_brechas_exposicion,
    '9 Brecha ANCOVA': df_brecha_ancova,
    '10 Brechas x oficina': df_brechas_oficina,
    '11 Sin brecha x oficina': df_sin_control_oficina,
    '12 Distribucion brechas': df_distribucion,
    '13 Distribucion x oficina': df_distribucion_oficina,
    '14 Sensibilidad pre-periodo': comp,
    '15 Semillas': df_semillas.drop(columns=[c for c in df_semillas.columns if c.startswith('_')], errors='ignore'),
    '16 Remuestreo (simulaciones)': df_sim_bootstrap,
    '17 Escenarios IC (ton)': df_escenarios_ic,
    '18 Escenarios IC x mes': df_escenarios_ic_mes,
    '19 Neteo antiguos': df_neteo,
    '20 Nuevos kg x cliente': df_nuevos_kg,
    '21 Ton x cohorte entrada': df_ton_cohorte,
    '22 Dist segmento pool': df_dist_seg_pool,
    '23 Dist segmento pares': df_dist_seg_pares,
    '24 Dist segmento implementados': df_dist_seg_implementados,
    '25 Barrido min_meses': df_barrido_min_meses,
    '26 Inka Chips (YoY)': vista_inkachips,
}
wb_m = Workbook(); wb_m.remove(wb_m.active)
for nombre, d_ in hojas_metricas.items():
    escribir_hoja(wb_m, nombre[:31], d_)
wb_m.save(RUTA_EXCEL_METRICAS)

# Inka Chips, adaptada a las columnas de 'base_calculo_toneladas' para poder ir en la
# MISMA hoja (pedido explícito) -- las columnas que no aplican (matching: oficina,
# segmento, IC, flg_estado_yoy, reportable, etc.) quedan en blanco, no inventadas.
# 'base_calculo_toneladas' en sí NO se toca (sigue siendo solo Cereales/Galletas, para
# no romper los chequeos de cuadre más arriba, que ya corrieron antes de este punto);
# el merge es solo para la hoja exportada.
cli_attrs_map = cli_attrs  # ya existe (des_oficina, nom_compania, des_segmento_transaccional por cod_cliente)
base_inkachips_export = pd.DataFrame({
    'cod_cliente': papas['cod_cliente'],
    'des_categoria': papas['des_categoria'],
    'des_oficina': papas['cod_cliente'].map(cli_attrs_map['des_oficina']),               # NaN si el cliente no está en Galletas/Cereales
    'des_segmento_transaccional': papas['cod_cliente'].map(cli_attrs_map['des_segmento_transaccional']),
    'nom_compania': papas['cod_cliente'].map(cli_attrs_map['nom_compania']),
    'cosecha_grupo': COSECHA_ANCLA,             # mismo ancla única que Cereales/Galletas (población = df_cosecha)
    'min_periodo_cosecha_real': papas['min_periodo_cosecha'],   # mes real de entrada DEL CLIENTE (ya no a nivel de grupo)
    'periodo': papas['periodo'],
    'mes_rel': papas['mes_rel'],                # vs COSECHA_ANCLA, referencial (no determina el filtro de atribución)
    'mes_exposicion': ((papas.periodo.dt.year - papas.min_periodo_cosecha.dt.year) * 12
                       + (papas.periodo.dt.month - papas.min_periodo_cosecha.dt.month)),  # meses desde la entrada real del cliente
    'toneladas_ant': papas['toneladas_ant'],
    'toneladas_actual': papas['toneladas_actual'],
    'yoy_pct_raw': np.where(papas['toneladas_ant'] > 0, 100 * papas['valor_atribuible'] / papas['toneladas_ant'], np.nan),
    'ic_pct_inf_80': np.nan, 'ic_pct_sup_80': np.nan,          # no aplica: sin bootstrap/IC (sin grupo control)
    'flg_estado_yoy': np.nan, 'crec_atrib_80_pct': np.nan, 'ton_atrib_80': np.nan,  # no aplica: no se capa a ningún IC
    'valor_atribuible': papas['valor_atribuible'],              # diferencial YoY, con signo -- el atribuible real de Inka Chips
    'ton_con_crecimiento': np.nan, 'ton_sin_crecimiento': np.nan, 'ton_nuevos': np.nan, 'ton_neto_sin_nuevos': np.nan,
    'flg_sin_banda': np.nan, 'reportable': np.nan})

base_toneladas_export = pd.concat([base_calculo_toneladas, base_inkachips_export], ignore_index=True)
print(f"Base toneladas (cli-mes) exportada: {len(base_calculo_toneladas):,} filas Cereales/Galletas + "
      f"{len(base_inkachips_export):,} filas Inka Chips = {len(base_toneladas_export):,} filas totales")

hojas_base = {
    'Base completa': df_base_completa,
    'Base toneladas (cli-mes)': base_toneladas_export,   # incluye Inka Chips (filas con columnas de matching en blanco)
    'Base brecha (par-mes)': base_calculo_brecha,
    'Sin pareja (en muestra)': df_sin_pareja,   # solo los de la muestra sin par; el universo completo esta en 'Base completa' (rol)
    'Matches': df_matches,
}
wb_b = Workbook(); wb_b.remove(wb_b.active)
for nombre, d_ in hojas_base.items():
    escribir_hoja(wb_b, nombre[:31], d_)
wb_b.save(RUTA_EXCEL_BASE)

wb_v = Workbook(); wb_v.remove(wb_v.active)
for nombre, d_ in vistas_brecha.items():
    escribir_hoja(wb_v, f'Brecha {nombre}'[:31], d_)
for nombre, d_ in vistas_ton.items():
    escribir_hoja(wb_v, f'Ton {nombre}'[:31], d_)
wb_v.save(RUTA_EXCEL_VISTAS)

for ruta in (RUTA_EXCEL_METRICAS, RUTA_EXCEL_BASE, RUTA_EXCEL_VISTAS):
    print(f"Guardado: {ruta}")

# ══════════════ celda 50 del notebook ══════════════
# =============================================================================
# CONTROL DE CALIDAD FINAL — cuadres que deben cumplirse antes de entregar
# =============================================================================
print("=" * 100); print("CHEQUEOS FINALES"); print("=" * 100)

# 1) El pipeline de matching (df_base_completa, base_calculo_toneladas) es SOLO
# Galletas y Cereales -- Inka Chips tiene su propia población/atribuible (papas,
# vista_inkachips) y nunca se mezcla con esta base, solo se suma al TOTAL de la
# vista Global (chequeo 5, más abajo).
cats = set(df_base_completa.des_categoria.unique()) | set(base_calculo_toneladas.des_categoria.unique())
assert cats <= set(CATEGORIAS), f"categorías fuera de alcance: {cats - set(CATEGORIAS)}"
print("OK  pipeline de matching (base_completa) es solo Galletas y Cereales -- Inka Chips no se mezcla acá")

# 2) Corte de datos
assert df_panel.periodo.max() <= MAX_PERIODO and df_incremental.periodo.max() <= MAX_PERIODO
print(f"OK  ningún periodo posterior a {MAX_PERIODO.date()}")

# 3) Acumulativo: ningún par-mes anterior a la entrada real del cliente de acción
assert (df_panel.mes_exposicion >= 0).all(), "hay pares contados ANTES de su entrada real"
assert (df_incremental.mes_exposicion >= 0).all(), "hay clientes-mes anteriores a su entrada real"
print("OK  acumulativo: ningún cliente cuenta antes de su mes de entrada")

# 4) Nuevos = volumen puro, sin brecha
nuevos = df_incremental[df_incremental.flg_estado_yoy == 'no_existia_periodo_pasado']
assert nuevos.ton_atrib_80.notna().sum() == 0 and (nuevos.valor_atribuible == nuevos.toneladas_actual).all()
print(f"OK  nuevos ({nuevos.cod_cliente.nunique():,} clientes): atribuible = toneladas_actual, sin brecha")

# 5) Cuadre de toneladas entre vistas y base de cálculo
# vistas_ton['Global'] YA incluye Inka Chips (se le sumó más abajo, antes de este
# QA) -> el cuadre correcto es contra tot_base + Inka Chips, no solo tot_base.
tot_base = base_calculo_toneladas.valor_atribuible.sum()
tot_vista = vistas_ton['Global'].ton_atrib_total_80.sum()
tot_inka_qa = vistas_ton['Global']['toneladas_inkachips'].sum()
print(f"    valor_atribuible (base de cálculo, Cereales+Galletas): {tot_base:,.2f} t | "
      f"vista Global (con Inka Chips): {tot_vista:,.2f} t | Inka Chips: {tot_inka_qa:,.2f} t")
assert abs((tot_base + tot_inka_qa) - tot_vista) < 1e-3 * max(1, abs(tot_base)),     "la vista Global no cuadra con la base de cálculo + Inka Chips"
print("OK  vista Global cuadra con la base de cálculo + Inka Chips")

# 5b) Embudo acumulado: GA + GC = universo (constante) en cada mes
_chk = df_embudo_acum.assign(suma=lambda d: d.GA_acumulado + d.GC_disponible)
assert (_chk.suma == _chk.universo_categoria).all(), "GA + GC no suma el universo"
assert df_embudo_acum.groupby('categoria').GA_acumulado.apply(lambda s: s.is_monotonic_increasing).all(), "GA no es acumulativo"
print("OK  embudo acumulado: GA + GC = universo y GA solo crece")

# 5c) Escenarios de IC: el 80% es exactamente el atribuible oficial
_t = df_escenarios_ic[df_escenarios_ic.des_categoria == 'TOTAL'].iloc[0]
print(f"    escenarios IC (ton): 80% {_t.ton_total_atrib_80:,.2f} | 90% {_t.ton_total_atrib_90:,.2f} | 95% {_t.ton_total_atrib_95:,.2f}")
assert abs(_t.ton_total_atrib_80 - tot_base) < 1e-3
print("OK  escenario IC80 = atribuible oficial")

# 6) Cobertura de banda
print(f"    filas sin banda IC (sin atribuible): {df_incremental.flg_sin_banda.sum():,} de {len(df_incremental):,}")

# 7) Resumen ejecutivo por celda
print("\nRESUMEN POR CATEGORÍA")
print(df_resumen[['categoria', 'AC_limpio', 'CT_limpio', 'pares', 'valida', 'paralelismo_ok', 'reportable']].to_string(index=False))
print("\nTONELADAS ATRIBUIBLES (IC80) POR CATEGORÍA Y MES")
print(vistas_ton['Categoria'][['des_categoria', 'periodo', 'ton_atrib_antiguos_80', 'ton_nuevos_puro',
                               'ton_atrib_total_80', 'reportable']].round(2).to_string(index=False))
