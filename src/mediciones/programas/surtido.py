# flake8: noqa
# ruff: noqa
# Script de PROGRAMA: configuración, SQL, universo y series. Se ejecuta con exec() antes que la metodología de su familia.
# MANTENIDO A MANO: este archivo es la fuente de verdad (los notebooks originales, fuera de este paquete, quedan solo como referencia histórica).
# =============================================================================
# PROGRAMA SURTIDO (piloto CMP) — configuración, datos, cosechas e Inka Chips
# =============================================================================
from mediciones.parametros import (  # parámetros metodológicos comunes a las dos familias
    VENTANA_MESES,
    OUTLIER_METODO,
    OUTLIER_K,
    OUTLIER_NIVEL,
    USAR_LOG,
    CALIPER_REL,
    CALIPER_ULTIMO,
    NIVEL_MAX,
    CORR_UMBRAL,
    N_SEMILLAS_DEFAULT,
    MAX_SEMILLAS_PROBAR_EQUIVALENCIA,
    MARGEN_ERROR_OBJETIVO,
    HORIZONTE_MESES_REPORTE,
    MARGEN_EQUIVALENCIA_PEND,
    ALPHA_TOST,
    NIVELES_IC,
    N_BOOT,
    SEED_BOOT,
)
from mediciones.metodologia.bodega import periodo_max_bodega, sql_surtido_bodeg   # exclusión de surtido de bodega, congelada por cierre
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

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('mode.copy_on_write', True)   # pandas 2.x: menos copias implícitas -> menos RAM/tiempo

# ── N_JOBS para las etapas paralelizables (matching por semilla, ver celda del pipeline) ──
N_JOBS = max(1, os.cpu_count() - 2)   # deja 2 núcleos libres para el SO / Jupyter
print(f"[PERFORMANCE] CPUs detectados: {os.cpu_count()} | N_JOBS configurado para paralelismo: {N_JOBS}")

# =============================================================================
# CONFIGURACIÓN DE EJECUCIÓN MENSUAL — ÚNICO LUGAR QUE SE EDITA CADA MES
# =============================================================================
import os, json, shutil
import pandas as pd

MES_CIERRE = pd.Timestamp(globals().get('MES_CIERRE_OVERRIDE', '2026-09-01'))        # <-- CAMBIAR CADA MES (siempre día 1)
assert MES_CIERRE.day == 1, "MES_CIERRE debe ser el día 1 del mes (ej. '2026-09-01')"
MES_CIERRE_STR = MES_CIERRE.strftime('%Y-%m-%d')
SUFIJO_CACHE   = MES_CIERRE.strftime('%Y%m')   # data_cache/<nombre>_<AAAAMM>.csv -> un snapshot por cierre
# Lista de exclusión de surtido_ideal_bodeg: None = congelada al cierre (solo cuentan los periodos <= MES_CIERRE, como el resto de las consultas). Un 'AAAA-MM-01' <= cierre
# fija un corte anterior (p. ej. para reproducir un reportado generado antes de que se cargara el último periodo). Ver docs/decisiones.md.
PERIODO_MAX_BODEG = None
# Controles ya emparejados (df_matches.csv congelado): True = conservan su serie aunque hoy los saque un filtro de la base (territorio de impulso, compañía
# excluida o lista de bodega); los filtros siguen aplicando a todos los demás clientes y a los candidatos de pares nuevos. Evita ceros artificiales en pares
# congelados (19 controles salieron de terr_impulso después del reporte de setiembre). False = comportamiento anterior. Ver docs/decisiones.md.
CONSERVAR_CONTROLES_CONGELADOS = True

# Cosechas: salen del calendario (config/programas.yaml -> surtido_motor.calendario_cosechas; ver mediciones/calendario.py).
#   G01 (altas feb-abr, cosecha abr), G02 (may-jun, jun), G03 (jul-set, cosecha ago; cerró altas en setiembre),
#   G04 (oct-dic, cosecha dic): PRELIMINAR en oct-nov (anclado en oct, incremental), DEFINITIVO desde dic (emparejamiento completo, se congela).
from mediciones.calendario import Calendario
CALENDARIO = Calendario(globals().get('CALENDARIO_COSECHAS'))      # el runner lo inyecta desde programas.yaml; si no, el calendario por defecto
COSECHA_ABIERTA  = CALENDARIO.abierta(MES_CIERRE)                  # grupo que recibe altas este cierre (None si ninguno)
COSECHAS_REHACER   = CALENDARIO.a_rehacer(MES_CIERRE)              # G04 al pasar a definitivo: emparejamiento completo (si aún no tiene pares definitivos)
COSECHAS_DESCARTAR = CALENDARIO.a_descartar(MES_CIERRE)            # pares preliminares del G04 que se descartan al pasar a definitivo
ALIAS_COSECHA = CALENDARIO.alias(MES_CIERRE)                       # en las salidas, el G04 preliminar se rotula con su cosecha definitiva
print(f"[CALENDARIO] cierre {MES_CIERRE.date()} | grupo abierto: {COSECHA_ABIERTA.date() if COSECHA_ABIERTA is not None else 'ninguno'} | "
      f"cosechas: {[c.strftime('%Y-%m') for c in CALENDARIO.cosechas(MES_CIERRE)]} | "
      f"rehacer: {[c.strftime('%Y-%m') for c in COSECHAS_REHACER]} | alias salidas: { {k.strftime('%Y-%m'): v.strftime('%Y-%m') for k, v in ALIAS_COSECHA.items()} }")
# Último min_periodo_cosecha ya cubierto por la corrida base (Feb-Ago). Solo se usa si aún no existe estado guardado.
FECHA_CORTE_BASE = pd.Timestamp('2026-08-01')

# Matching
#   False (normal, cada mes): congela G01/G02/G03-base y solo matchea a los entrantes del G03.
#   True : rehace TODO el matching desde cero (comportamiento v7). Úsalo solo si cambias criterios.
REHACER_MATCHING  = False
RUTA_SALIDA_MATCH = os.path.join('outputs', 'Surtido Ideal - Medición')
RUTA_ESTADO       = os.path.join(RUTA_SALIDA_MATCH, 'estado_matching.json')









# ── Configuración propia del programa (antes dentro del bloque de matching) ──
COSECHAS   = CALENDARIO.cosechas(MES_CIERRE)                     # fechas internas (mes 0 del panel) de los grupos que ya existen en este cierre
CATEGORIAS = ['Galletas', 'Cereales']
GRUPO_COSECHA = CALENDARIO.nombres(MES_CIERRE)
MIN_MESES_POR_CATEGORIA = {'Galletas': 6, 'Cereales': 6}
MIN_MESES_DEFAULT = 6
PESOS_OVERRIDE = {
    (pd.Timestamp('2026-04-01'), 'Cereales'): {'W_NIVEL': 0.3, 'W_FORMA': 0.7},
    (pd.Timestamp('2026-06-01'), 'Galletas'): {'W_NIVEL': 0.3, 'W_FORMA': 0.7},
}
N_SEMILLAS_OVERRIDE = {
    (pd.Timestamp('2026-04-01'), 'Cereales'): 150,
    (pd.Timestamp('2026-06-01'), 'Galletas'): 150,
}
MAX_REUSO_CONTROL_POR_COSECHA = {}
MAX_REUSO_CONTROL_DEFAULT = 1        # 1:1 estricto (comportamiento original)

import decimal
# =============================================================================
# CACHÉ LOCAL DE DESCARGAS — evita volver a golpear BigQuery si el CSV ya existe
# =============================================================================
CACHE_DIR = 'data_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Columnas de código de cliente -> SIEMPRE se normalizan a string (merge('cod_cliente') truena si una tabla llega int64 y otra object).
# La lógica vive en mediciones.metodologia.ids; la lectura con caché CSV, en mediciones.grilla (compartida por surtido y Champion).
from mediciones.metodologia.ids import es_columna_id_cliente as _es_columna_id_cliente, normalizar_ids as _normalizar_ids
from mediciones.grilla import ejecutar_grilla as _grilla


def ejecutar_grilla(query: str, nombre_cache: str, project_id: str = "acpe-dev-uc-sandbox-aa",
                     verbose: bool = True, forzar_descarga: bool = True):   # el motor usa mediciones.insumos (snapshots con hash de SQL); caché CSV del notebook desactivado
    """Consulta BigQuery con caché CSV local (ver mediciones.grilla). Lo propio de este programa: caché con sufijo AAAAMM del cierre; convierte fechas y Decimal."""
    return _grilla(query, nombre_cache, cache_dir=CACHE_DIR, project_id=project_id, verbose=verbose, forzar_descarga=forzar_descarga,
                   sufijo_cache=SUFIJO_CACHE,
                   ids_como_texto_en_cache=False, fechas_en_consulta=True, decimales_en_consulta=True)

sql = f"""
SELECT 
    MIN(s.periodo) min_periodo_cosecha,
    s.cod_cliente,
    CASE WHEN s.nom_compania='DISTRIBUIDORA ALIMENTOS DEL VALLE S.A.C.' THEN 'DISTRIBUIDORA SOL DEL VALLE S.A.C.' 
    ELSE s.nom_compania END nom_compania
FROM `acpe-prod-dlk-shrd-gld.delivery_comercial_dex.cr_surtido_ideal` s
WHERE  des_estrategia= 'CMP Impulso'
    AND periodo BETWEEN '2026-02-01' AND '{MES_CIERRE_STR}'
GROUP BY ALL;
"""

cosecha_surtido = ejecutar_grilla(sql, nombre_cache='cosecha_surtido')

cosecha_surtido = cosecha_surtido.drop_duplicates('cod_cliente')

# pantalla surtido bodeg: lista congelada al cierre (ver metodologia/bodega.py)
PERIODO_MAX_BODEG_USADO = periodo_max_bodega(MES_CIERRE, PERIODO_MAX_BODEG)
sql_ps = sql_surtido_bodeg(PERIODO_MAX_BODEG_USADO)

surtido_bodeg = ejecutar_grilla(sql_ps, nombre_cache='surtido_bodeg')
print(f"  [BODEGA] lista de exclusión congelada al periodo {PERIODO_MAX_BODEG_USADO}: "
      f"{surtido_bodeg['cod_cliente_dex'].nunique() if 'cod_cliente_dex' in surtido_bodeg.columns else 0:,} clientes de surtido de bodega")

sql_sell_out = f"""
SELECT
   periodo,
   cod_cliente_dex cod_cliente,
   est_cliente,
   c.nom_compania,
   CASE WHEN des_categoria='Golosinas y Snacks SSCC' THEN 'Papas Fritas' ELSE des_categoria END AS des_categoria,
   COUNT(DISTINCT cod_material_actual) mix_sku,
   COUNT(DISTINCT cod_familia) mix_familia,
   SUM(num_peso_real)  AS peso,
   SUM(mnt_venta_real) AS monto
FROM `acpe-prod-dlk-shrd-gld.delivery_reporte_comercial.cr_venta_sell_out` s
LEFT JOIN `acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente` c
  ON s.cod_cliente_dex=c.id_interlocutor_origen
WHERE cod_grupo_precio_dex = 'BO'
    AND periodo >= '2025-01-01'  AND periodo<='{MES_CIERRE_STR}'
     AND (
          des_categoria IN ('Galletas','Cereales','Papas Fritas')
          OR (des_categoria = 'Golosinas y Snacks SSCC' AND des_marca = 'Inka Chips')
        )
GROUP BY ALL
"""

sell_out_jun = ejecutar_grilla(sql_sell_out, nombre_cache='sell_out_jun')
# [sell_out_jun:inicio] (celda 14 del notebook) tipos numéricos y de fecha antes de cualquier cruce
sell_out_jun['peso'] = sell_out_jun['peso'].astype(float)
sell_out_jun['monto'] = sell_out_jun['monto'].astype(float)
sell_out_jun['periodo'] = pd.to_datetime(sell_out_jun['periodo'])
# [sell_out_jun:fin]

# la base cambia
sql_terr = """
SELECT DISTINCT
  v.id_interlocutor_origen cod_cliente, 
  id_fuerza_venta_origen
FROM `acpe-prod-dlk-shrd-gld.gld_cliente.dex_interlocutor_territorio` v
LEFT JOIN `acpe-prod-dlk-shrd-gld.gld_cliente.dex_personal_territorio`t 
  ON v.id_territorio_origen=t.id_territorio_origen
WHERE  v.id_territorio IS NOT NULL AND id_fuerza_venta_origen='IMPU';
"""
terr = ejecutar_grilla(sql_terr, nombre_cache='terr_impulso')

# *** AJUSTE: cr_segmentacion usa cod_interlocutor, pero sell_out usa
# cod_cliente_dex -- son DOS sistemas de código distintos (confirmado: 0 de 29
# códigos de muestra coincidian directo). Se tiende el puente vía dex_cliente
# (misma tabla que ya usa sql_sell_out para mapear oficina): sql_sell_out
# mergea cod_cliente_dex contra dex_cliente.id_interlocutor_origen, o sea que
# el 'cod_cliente' que se usa en TODO el pipeline ya vive en el espacio de
# id_interlocutor_origen, no en un campo propio de dex_cliente. Por eso el
# SELECT correcto es c.id_interlocutor_origen (no c.cod_cliente_dex, que era
# el supuesto equivocado de la versión anterior). ***
sql_seg = f"""
SELECT
       c.id_interlocutor_origen AS cod_cliente,
       s.des_segmento_transaccional,
       s.des_segmento_estrategica,
       COALESCE(s.flg_potential,False) flg_potential
FROM `acpe-prod-dlk-shrd-gld.delivery_cliente_gtm.cr_segmentacion` s
JOIN `acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente` c
  ON s.cod_interlocutor = c.id_interlocutor_origen
WHERE s.periodo = (
       SELECT MAX(periodo)
       FROM `acpe-prod-dlk-shrd-gld.delivery_cliente_gtm.cr_segmentacion`
       WHERE periodo >= '2025-06-06'
         AND periodo < DATE_ADD(DATE '{MES_CIERRE_STR}', INTERVAL 1 MONTH)   -- la segmentación vigente AL CIERRE, no la última que exista hoy
     )
     AND s.periodo >= '2025-06-06'
   GROUP BY 1,2,3,4
"""
df_seg = ejecutar_grilla(sql_seg, nombre_cache='df_seg_v3')


sql_roma = f"""
SELECT flg_cliente_roma,
       MIN(fec_periodo_alta) cosecha_roma,
       cod_cliente
FROM `acpe-prod-dlk-shrd-gld.gld_cliente.gtm_cliente_roma`
WHERE periodo>='2025-10-01' AND periodo <= '{MES_CIERRE_STR}' AND flg_cliente_roma=True   -- tope al cierre
GROUP BY ALL
"""
df_roma = ejecutar_grilla(sql_roma, nombre_cache='df_roma')

sell_out = sell_out_jun.copy()

dex_oficina = {
    "MBA DEX SOCIEDAD ANONIMA CERRADA": "CM-SUR / CENTRO",
    "RTDEX E.I.R.L.": "CM-SUR / CENTRO",
    "DEX APURIMAC S.A.C.": "CM-SUR / CENTRO",
    "DISTRIBUIDORA RODRIGUEZ S.A.": "CM-NORTE / ORIENTE",
    "DISTRIBUIDORA COMERCIAL LUIS ARTURO & XIMENA E.I.R.L.": "CM-NORTE / ORIENTE",
    "JM GENG DISTRIBUCIONES S.A.C.": "CM-SUR / CENTRO",
    "DISTRIBUIDORA MADEX SOCIEDAD ANONIMA CERRADA": "CM-SUR / CENTRO",  # ojo: aparece dos veces con oficinas distintas
    "FUERZADEX S.A.C.": "CM-LIMA / NS CHICO",
    "DISTRIBUIDORA CUNZA S.A.": "CM-LIMA / NS CHICO",
    "SELVA DEX S.A.C": "CM-SUR / CENTRO",
    "D.L.F. MEDINA RIVERA S.A.": "CM-LIMA / NS CHICO",
    "DISTRIBUIDORA DE PRODUCTOS DE CONSUMO MASIVO SOCIEDAD ANONIMA CERRADA": "CM-NORTE / ORIENTE",
    "DALIMNORT S.A.C.": "CM-LIMA / NS CHICO",
    "PACIFICO DEX SOCIEDAD ANONIMA CERRADA": "CM-LIMA / NS CHICO",
    "ATUQ DEX S.A.C.": "CM-NORTE / ORIENTE",
    "DISTRIBUIDORA AURELIO SOCIEDAD ANONIMA CERRADA": "CM-LIMA / NS CHICO",
    "MT DISTRIBUCIONES S.A.C.": "CM-SUR / CENTRO",
    "DISTRIBUCIONES EXCLUSIVAS DEL SUR SOCIEDAD ANONIMA CERRADA": "CM-SUR / CENTRO",
    "COBERDEX SOCIEDAD ANONIMA CERRADA": "CM-LIMA / NS CHICO",
    "DISTRIBUIDORA ALIMENTOS DEL VALLE S.A.C.": "CM-NORTE / ORIENTE",
    "ALMA DISTRIBUCIONES E.I.R.L.": "CM-SUR / CENTRO",
    "DALIDELO E.I.R.L.": "CM-NORTE / ORIENTE",
    "F & B SUR S.A.C": "CM-SUR / CENTRO",
    "DISTRIBUIDORA SOL DEL VALLE S.A.C.": "CM-NORTE / ORIENTE",
    "DEX MINALI SOCIEDAD ANONIMA CERRADA": "CM-SUR / CENTRO",
    "DISTRIBUIDORA ATIX S.A.C.": "CM-LIMA / NS CHICO",
    "DISTRIBUCIONES DEL NORTE ORIENTE SOCIEDAD ANONIMA CERRADA": "CM-NORTE / ORIENTE",
    "DISTRIBUIDORA MKM S.A.C": "CM-NORTE / ORIENTE",
    "VITALE DEX S.A.C.": "CM-NORTE / ORIENTE",
    "MED DISTRIBUCIONES EMPRESA INDIVIDUAL DE RESPONSABILIDAD LIMITADA": "CM-NORTE / ORIENTE",
    "ATIPANA DEX S.A.C.": "CM-LIMA / NS CHICO",
    "ROSET DISTRIBUCIONES EMPRESA INDIVIDUAL DE RESPONSABILIDAD LIMITADA": "CM-SUR / CENTRO",
    "DIXIAL S.A.C.": "CM-LIMA / NS CHICO",
    "DEX PARDADEX S.A.C.": "CM-NORTE / ORIENTE",
}

# filtro clientes impulso
clientes_impulso = terr[terr.id_fuerza_venta_origen=='IMPU'].cod_cliente.unique()

sell_out['des_oficina'] = sell_out['nom_compania'].map(dex_oficina)

# tageamos si tiene surtido
sell_out['tiene_surtido'] = (sell_out.cod_cliente.isin(cosecha_surtido.cod_cliente.unique())).astype(bool)

# primera cosecha
sell_out_cosecha = sell_out.merge(cosecha_surtido[['cod_cliente','min_periodo_cosecha']], on='cod_cliente', how='left')

df_roma['es_roma'] = True

sell_out_cosecha_roma = sell_out_cosecha.merge(df_roma, on='cod_cliente', how='left')
sell_out_cosecha_roma['es_roma'] = sell_out_cosecha_roma['es_roma'].fillna(False)

#base = sell_out_cosecha_roma.copy()
sell_out_cosecha_roma['periodo'] = pd.to_datetime(sell_out_cosecha_roma['periodo'])
sell_out_cosecha_roma['min_periodo_cosecha']=pd.to_datetime(sell_out_cosecha_roma['min_periodo_cosecha'])
sell_out_cosecha_roma['cosecha_roma']=pd.to_datetime(sell_out_cosecha_roma['cosecha_roma'])

sell_out_cosecha_roma['surtido_bodeg'] = (sell_out_cosecha_roma['cod_cliente'].isin(surtido_bodeg.cod_cliente_dex.unique())).astype(bool)
sell_out_cosecha_roma['surtido_bodeg']=sell_out_cosecha_roma['surtido_bodeg'].fillna(False)

# ── DIAGNÓSTICO: casi todo cae en 'Por definir' -> el sospechoso #1 es un
# mismatch de dtype en cod_cliente entre sell_out_cosecha_roma y df_seg (mismo
# patrón ya documentado con _normalizar_ids, pero df_seg llega por ejecutar_grilla
# mientras que sell_out_cosecha_roma pasó por varios merges intermedios -- cosecha_surtido,
# df_roma, surtido_bodeg -- que pueden haber revertido el tipo a int64/float64).
# Se normaliza defensivamente ANTES del merge (idempotente si ya estaba bien) y se
# imprime la tasa de match real para confirmar si esto resuelve el problema o si el
# mismatch viene de más atrás en la cadena.
if 'cod_cliente' not in df_seg.columns:
    raise ValueError(
        "df_seg no tiene columna 'cod_cliente' -> la query sql_seg (celda 11) "
        "probablemente falló en BigQuery (revisar el print '[BIGQUERY]...Error al "
        "ejecutar la grilla' de esa celda) o devolvió un DataFrame vacío. "
        f"Columnas actuales de df_seg: {list(df_seg.columns)}")
_dtype_izq = sell_out_cosecha_roma['cod_cliente'].dtype
_dtype_der = df_seg['cod_cliente'].dtype
if _dtype_izq != _dtype_der:
    print(f"  [ALERTA] cod_cliente con dtypes distintos antes del merge: "
          f"sell_out_cosecha_roma={_dtype_izq} vs df_seg={_dtype_der} -> "
          f"normalizando ambos a string antes de mergear.")
sell_out_cosecha_roma = _normalizar_ids(sell_out_cosecha_roma.copy())
df_seg = _normalizar_ids(df_seg.copy())

# ── FIX REAL: no era dtype, era ANCHO -- dex_cliente.id_interlocutor_origen es
# numérico en BigQuery (pierde ceros a la izquierda al castear a string: '4164673'),
# mientras que cr_venta_sell_out.cod_cliente_dex ya viene como string de 10 dígitos
# con ceros ('0004164673'). Mismo cliente, distinto ancho de texto. Confirmado con
# datos reales: los prefijos numéricos calzan uno a uno entre ambos lados, y la
# intersección pasó de 2 a la cifra real una vez rellenado a 10 dígitos.
# Se rellena SOLO el lado que venía de un campo numérico (df_seg); sell_out_cosecha_roma
# ya tiene el ancho correcto nativamente, pero se le aplica zfill también por
# idempotencia (no cambia nada si ya mide 10).
sell_out_cosecha_roma['cod_cliente'] = sell_out_cosecha_roma['cod_cliente'].str.zfill(10)
df_seg['cod_cliente'] = df_seg['cod_cliente'].str.zfill(10)

sell_out_cosecha_roma_seg = (
    sell_out_cosecha_roma.merge(
        df_seg,
        on='cod_cliente',
        how='left'
    )
)

_tasa_match = sell_out_cosecha_roma_seg['des_segmento_transaccional'].notna().mean()
if _tasa_match < 0.5:
    _msg_diag_seg = "revisar más atrás en la cadena de merges si esto sigue bajo"
else:
    _msg_diag_seg = "OK, la normalización resolvió el mismatch"
print(f"  [SEGMENTACIÓN] tasa de match con df_seg (antes de fillna): {_tasa_match:.1%} "
      f"de las filas encontraron segmento -> {_msg_diag_seg}")

sell_out_cosecha_roma_seg['des_segmento_transaccional']=sell_out_cosecha_roma_seg['des_segmento_transaccional'].fillna('Por definir')
sell_out_cosecha_roma_seg['des_segmento_estrategica']=sell_out_cosecha_roma_seg['des_segmento_estrategica'].fillna('Por definir')

# quitar dex con surtido que no tienen surtido
dex = ['D.L.F. MEDINA RIVERA S.A.', 'DISTRIBUIDORA CUNZA S.A.',
       'DISTRIBUCIONES EXCLUSIVAS DEL SUR SOCIEDAD ANONIMA CERRADA',
       'ATIPANA DEX S.A.C.', 'DEX PARDADEX S.A.C.', 'FUERZADEX S.A.C.',
       'MBA DEX SOCIEDAD ANONIMA CERRADA',
       'ROSET DISTRIBUCIONES EMPRESA INDIVIDUAL DE RESPONSABILIDAD LIMITADA',
       'ATUQ DEX S.A.C.', 'DISTRIBUIDORA SOL DEL VALLE S.A.C.']

# [controles_congelados:inicio] Controles ya emparejados (df_matches.csv congelado): el grupo de control ya está decidido, así que su serie NO depende de los
# filtros del día (territorio de impulso, compañía excluida, lista de bodega), que salen de tablas maestras que cambian. Ver CONSERVAR_CONTROLES_CONGELADOS.
_ctrl_congelados = set()
_ruta_dm_cong = os.path.join(RUTA_SALIDA_MATCH, 'df_matches.csv')
if CONSERVAR_CONTROLES_CONGELADOS and not REHACER_MATCHING and os.path.exists(_ruta_dm_cong):
    _ctrl_congelados = set(pd.read_csv(_ruta_dm_cong, usecols=['cod_control'], dtype=str)['cod_control']
                           .dropna().str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(10))
# [controles_congelados:fin]

# base de clientes que tienen surtido o impulsos (+ los controles congelados, aunque hoy hayan salido del territorio de impulso o de la compañía)
_filtro_dia = (
    ~((sell_out_cosecha_roma_seg.nom_compania.isin(dex)) &
      (sell_out_cosecha_roma_seg.tiene_surtido == False)) &
    (
        (sell_out_cosecha_roma_seg.tiene_surtido == True) |
        (sell_out_cosecha_roma_seg.cod_cliente.isin(clientes_impulso))
    )
)
_es_ctrl_congelado = sell_out_cosecha_roma_seg.cod_cliente.isin(_ctrl_congelados)
print(f"  [CONTROLES CONGELADOS] {len(_ctrl_congelados):,} controles en df_matches.csv | conservan su serie aunque hoy los saque un filtro "
      f"(territorio/compañía): {sell_out_cosecha_roma_seg.loc[_es_ctrl_congelado & ~_filtro_dia, 'cod_cliente'].nunique():,}")
base_seg_limpio = sell_out_cosecha_roma_seg[_filtro_dia | _es_ctrl_congelado]

# Opción 1: con boolean indexing (más claro)
mask = (base_seg_limpio['min_periodo_cosecha'] == pd.Timestamp('2026-02-01')) & (base_seg_limpio['des_oficina'] == 'CM-SUR / CENTRO')
base_seg_limpio = base_seg_limpio[~mask]

print(f"Filas eliminadas: {mask.sum()}")
print(f"Filas restantes: {len(base_seg_limpio)}")

# [cosecha_agrupada:inicio] (celda 23 del notebook) homologar cosecha: mes de entrada al piloto -> grupo, según el calendario (la SQL de Inka Chips usa el mismo).
base_seg_limpio['min_periodo_cosecha'] = pd.to_datetime(base_seg_limpio['min_periodo_cosecha'])
mes = base_seg_limpio['min_periodo_cosecha'].dt.month

base_seg_limpio['cosecha_agrupada'] = CALENDARIO.asignar(base_seg_limpio['min_periodo_cosecha'], MES_CIERRE)   # según calendario (G01..G04)
# [cosecha_agrupada:fin]


import pandas as pd
import numpy as np

# Definir cosechas y sus rangos de 12 meses atrás
cosechas_config = CALENDARIO.rangos_12m(MES_CIERRE)                # 12 meses previos a cada cosecha (fecha interna)

print("\n" + "="*80)
print("DISTRIBUCIÓN DE PERÍODOS POR COSECHA (12 meses antes)")
print("="*80)

for cosecha, (fecha_inicio, fecha_fin) in cosechas_config.items():
    print(f"\n{'='*80}")
    print(f"COSECHA: {cosecha} | Rango: {fecha_inicio.date()} a {fecha_fin.date()}")
    print(f"{'='*80}")
    
    # Filtrar clientes con esta cosecha_agrupada
    clientes_cosecha = base_seg_limpio[
        base_seg_limpio['cosecha_agrupada'] == cosecha
    ]['cod_cliente'].unique()
    
    # Filtrar datos en el rango de 12 meses
    df_rango = base_seg_limpio[
        (base_seg_limpio['cod_cliente'].isin(clientes_cosecha)) &
        (base_seg_limpio['periodo'] >= fecha_inicio) &
        (base_seg_limpio['periodo'] <= fecha_fin)
    ]
    
    # Contar períodos distintos por cliente
    periodos_por_cliente = df_rango.groupby('cod_cliente')['periodo'].nunique()
    
    # Distribución ordenada de mayor a menor
    distribucion = periodos_por_cliente.value_counts().sort_index(ascending=False)
    
    print(f"\nTotal clientes con esta cosecha: {len(clientes_cosecha):,}")
    print(f"Clientes con datos en rango: {len(periodos_por_cliente):,}")
    print(f"Clientes SIN datos en rango: {len(clientes_cosecha) - len(periodos_por_cliente):,}")
    
    print(f"\n{'Nº Períodos':>12} | {'Nº Clientes':>12} | {'% Clientes':>12} | {'% Acumulado':>12}")
    print("-" * 54)
    
    acumulado = 0
    for num_periodos, num_clientes in distribucion.items():
        pct = (num_clientes / len(periodos_por_cliente) * 100) if len(periodos_por_cliente) > 0 else 0
        acumulado += pct
        print(f"{num_periodos:>12} | {num_clientes:>12,} | {pct:>11.1f}% | {acumulado:>11.1f}%")


# eliminamos surtido_bodeg (salvo los controles ya emparejados en el estado congelado: _ctrl_congelados, ver [controles_congelados])
_en_bodega = base_seg_limpio['surtido_bodeg'].astype(bool)
_conserva_bodega = _en_bodega & base_seg_limpio['cod_cliente'].isin(_ctrl_congelados)
print(f"  [BODEGA] clientes en la lista: {base_seg_limpio.loc[_en_bodega, 'cod_cliente'].nunique():,} | se conservan por ser controles congelados: "
      f"{base_seg_limpio.loc[_conserva_bodega, 'cod_cliente'].nunique():,} | se excluyen: {base_seg_limpio.loc[_en_bodega & ~_conserva_bodega, 'cod_cliente'].nunique():,}")
base_seg_limpio = base_seg_limpio[(~_en_bodega) | _conserva_bodega]

# Definir cosechas y sus rangos de 12 meses atrás
cosechas_config = CALENDARIO.rangos_12m(MES_CIERRE)                # 12 meses previos a cada cosecha (fecha interna)


resultados_outliers = {}

for cosecha, (fecha_inicio, fecha_fin) in cosechas_config.items():
    print(f"\n{'='*80}")
    print(f"COSECHA: {cosecha} | Rango: {fecha_inicio.date()} a {fecha_fin.date()}")
    print(f"{'='*80}")

    # Clientes con esta cosecha_agrupada
    clientes_cosecha = base_seg_limpio[
        base_seg_limpio['cosecha_agrupada'] == cosecha
    ]['cod_cliente'].unique()

    # Datos en el rango de 12 meses, solo de esos clientes
    df_rango = base_seg_limpio[
        (base_seg_limpio['cod_cliente'].isin(clientes_cosecha)) &
        (base_seg_limpio['periodo'] >= fecha_inicio) &
        (base_seg_limpio['periodo'] <= fecha_fin)
    ]

    # Peso total por cliente en los 12 meses
    peso_por_cliente = df_rango.groupby('cod_cliente')['peso'].sum()

    # --- Distribución del peso total ---
    print(f"\nClientes con peso en el rango: {len(peso_por_cliente):,}")
    print("\nDistribución del peso total (kg) por cliente:")
    desc = peso_por_cliente.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99])
    print(desc.round(1).to_string())

    # --- Detección de outliers (método IQR) ---
    q1, q3 = peso_por_cliente.quantile([.25, .75])
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr

    outliers = peso_por_cliente[(peso_por_cliente < lim_inf) | (peso_por_cliente > lim_sup)]
    print(f"\nLímites IQR -> inferior: {lim_inf:,.1f} | superior: {lim_sup:,.1f}")
    print(f"Nº de outliers: {len(outliers):,} ({len(outliers)/len(peso_por_cliente)*100:.1f}% de los clientes)")

    # --- Oficina de cada cliente outlier (última oficina registrada) ---
    oficina_cliente = (
        df_rango.sort_values('periodo')
        .groupby('cod_cliente')['des_oficina']
        .last()
    )

    df_outliers = (
        outliers.rename('peso_total')
        .to_frame()
        .join(oficina_cliente)
        .sort_values('peso_total', ascending=False)
    )

    print("\nDistribución de outliers por oficina:")
    resumen_oficina = (
        df_outliers.groupby('des_oficina')
        .agg(n_clientes=('peso_total', 'size'),
             peso_total=('peso_total', 'sum'),
             peso_prom=('peso_total', 'mean'))
        .sort_values('n_clientes', ascending=False)
    )
    print(resumen_oficina.round(1).to_string())

    print("\nTop 10 outliers (cliente, peso, oficina):")
    print(df_outliers.head(10).round(1).to_string())

    resultados_outliers[cosecha] = df_outliers


# =============================================================================
# INKA CHIPS -- poblacion y cosecha_grupo propios, tomados DIRECTO de
# cr_surtido_ideal (tabla de asignacion real del piloto), independiente de la
# poblacion accion de Galletas/Cereales. periodo_min_grupo = el min_periodo_cosecha
# mas temprano DE TODO EL GRUPO (no por cliente) -> asi el atribuible arranca
# desde que el primer cliente del bucket entro al piloto.
#
# MODO MENSUAL: la query llega hasta MES_CIERRE y el G03 sigue abierto (todo cliente que
# entre desde julio cae en cosecha_grupo = 2026-08-01), igual que en el matching.
#
# CAMBIO: el atribuible de Inka Chips ahora es el DIFERENCIAL YoY (este año
# menos el mismo mes del año pasado), no el volumen bruto -- pedido explicito
# del negocio. Para eso hace falta el AÑO ANTERIOR de cada cliente-mes, asi
# que se quita el filtro 'periodo >= periodo_min_grupo' de la SQL (que antes
# descartaba justo esas filas) y se trae el HISTORICO COMPLETO de 'ventas_cat'
# (acotado a 2025-01-01..MES_CIERRE, cubre de sobra los 12 meses atras de
# cualquier cosecha). El filtro a "periodo atribuible" se aplica
# despues, en pandas, una vez calculado el año anterior (celda siguiente).
# Cache renombrado (el shape de los datos cambio -- ya no es solo periodo
# atribuible, es historico completo -- para no reusar por error el cache viejo).
# =============================================================================

sql_inkachips = f"""
WITH cosecha AS (
  SELECT
    cod_cliente,
    MIN(periodo) AS min_periodo_cosecha
  FROM `acpe-prod-dlk-shrd-gld.delivery_comercial_dex.cr_surtido_ideal`
  WHERE des_estrategia= 'CMP Impulso'
    AND periodo BETWEEN '2026-02-01' AND '{MES_CIERRE_STR}'
  GROUP BY cod_cliente
),

cosecha_bucket AS (
  SELECT
    cod_cliente,
    min_periodo_cosecha,
    {CALENDARIO.sql_case('min_periodo_cosecha', MES_CIERRE)} AS cosecha_grupo
  FROM cosecha
),

periodo_min_por_grupo AS (
  SELECT
    cosecha_grupo,
    MIN(min_periodo_cosecha) AS periodo_min_grupo
  FROM cosecha_bucket
  GROUP BY cosecha_grupo
),

ventas_cat AS (
  SELECT
    periodo,
    cod_cliente_dex AS cod_cliente,
    des_categoria,
    SUM(num_peso_real) AS peso
  FROM `acpe-prod-dlk-shrd-gld.delivery_reporte_comercial.cr_venta_sell_out`
  WHERE cod_grupo_precio_dex = 'BO'
    AND periodo BETWEEN '2025-01-01' AND '{MES_CIERRE_STR}'
    AND des_marca = 'Inka Chips'   -- unico filtro: toda categoria donde vende esta marca
  GROUP BY periodo, cod_cliente_dex, des_categoria
)

SELECT
  cb.cosecha_grupo,
  pm.periodo_min_grupo,
  cb.cod_cliente,
  v.des_categoria,
  v.periodo,
  v.peso
FROM ventas_cat v
JOIN cosecha_bucket cb
  ON v.cod_cliente = cb.cod_cliente
JOIN periodo_min_por_grupo pm
  ON pm.cosecha_grupo = cb.cosecha_grupo
ORDER BY cb.cod_cliente, v.periodo
"""

inkachips_historico = ejecutar_grilla(sql_inkachips, nombre_cache='inkachips_historico_completo')
inkachips_historico['periodo'] = pd.to_datetime(inkachips_historico['periodo'])
inkachips_historico['cosecha_grupo'] = pd.to_datetime(inkachips_historico['cosecha_grupo'])
inkachips_historico['periodo_min_grupo'] = pd.to_datetime(inkachips_historico['periodo_min_grupo'])

print(f"Inka Chips historico completo: {len(inkachips_historico):,} filas cliente-categoria-mes | "
      f"{inkachips_historico.cod_cliente.nunique():,} clientes | "
      f"{inkachips_historico.peso.sum() / 1000:,.2f} t (bruto, incluye año anterior)")
