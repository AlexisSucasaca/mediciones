# flake8: noqa
# ruff: noqa
# Script de PROGRAMA: configuración, SQL, universo y series. Se ejecuta con exec() antes que la metodología de su familia.
# MANTENIDO A MANO: este archivo es la fuente de verdad (los notebooks originales, fuera de este paquete, quedan solo como referencia histórica).
# =============================================================================
# PROGRAMA CPB — configuración, universo, datos y series
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
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)

PROJECT_ID = "acpe-dev-uc-sandbox-aa"


def ejecutar_grilla(query: str, project_id: str = PROJECT_ID) -> pd.DataFrame:
    """Ejecuta SQL en BigQuery -> DataFrame (Decimal -> float). Propaga errores."""
    import decimal
    df = pd.read_gbq(query, project_id=project_id, dialect="standard", progress_bar_type="tqdm")
    for col in df.columns:
        if df[col].dtype == object:
            muestra = df[col].dropna()
            if len(muestra) and isinstance(muestra.iloc[0], decimal.Decimal):
                df[col] = df[col].astype(float)
    print(f"Filas: {len(df):,}")
    return df

# ── Ventana (numerador = actual; base = mismos meses un año atrás) ──────────
FECHA_INI_ACTUAL = "2026-01-01"
FECHA_FIN_ACTUAL = globals().get("FECHA_FIN_ACTUAL_OVERRIDE", "2026-09-01")   # el orquestador inyecta el mes

# ── Universo CPB ────────────────────────────────────────────────────────────
GRUPO_PRECIO         = "BO"
SEGMENTO_ESTRATEGICO = "Estrella"
SEGMENTOS            = ["1 Oro+", "2 Oro", "3 Plata"]
NEGOCIOS             = ["CP1"]          # definición cerrada: CP1 + categorías core (ver preflight 2.1)
seg_map              = {"1 Oro+": 1, "2 Oro": 2, "3 Plata": 3}

CLIENTES_CONTROL = [   # 14 DEX piloto (se filtran vía cr_cliente_alicorp.cod_cliente)
    "1000001534", "1000036414", "1000037029", "1000009451", "1000049464",
    "1000009493", "1000000460", "1000036569", "1000054782", "1000037030",
    "1000009069", "1000035697", "1000036567", "1000009338",
]

CATEGORIAS_CP1_CORE = [
    "Aceites", "Cereales", "Detergentes", "Galletas", "Jabones de Lavar",
    "Lavavajillas", "Pastas", "Salsas", "Shampoo", "Suavizantes", "Untables Domesticos",
]
CATEGORIAS_SIN_GC = [c for c in CATEGORIAS_CP1_CORE if c not in ("Galletas", "Cereales")]
VERSIONES = {
    "Con Galletas y Cereales": CATEGORIAS_CP1_CORE,   # 11 cat.
    "Sin Galletas y Cereales": CATEGORIAS_SIN_GC,     # 9 cat.
}

# ── Región: 6 valores de cr_cliente_alicorp -> 3 oficinas del reporte ───────
# (NS CHICO = Norte Chico + Sur Chico, por eso SUR CHICO va con Lima.)
REGIONES_ESPERADAS = ["PE01 - CM-LIMA / NS CHICO", "PE02 - CM-NORTE / ORIENTE", "PE03 - CM-SUR / CENTRO"]
MAPA_REGION = {
    "LIMA":      "PE01 - CM-LIMA / NS CHICO",
    "SUR CHICO": "PE01 - CM-LIMA / NS CHICO",
    "NORTE":     "PE02 - CM-NORTE / ORIENTE",
    "ORIENTE":   "PE02 - CM-NORTE / ORIENTE",
    "SUR":       "PE03 - CM-SUR / CENTRO",
    "CENTRO":    "PE03 - CM-SUR / CENTRO",
}

# ── Ejecución ───────────────────────────────────────────────────────────────
USAR_CACHE     = False   # True: reutiliza cache/ (se valida contra un COUNT(*) en BQ)
EXPORTAR_PARES = True    # hoja con el detalle de pares (versión con Galletas y Cereales)

# ── Derivados ───────────────────────────────────────────────────────────────
P_INI_ACTUAL = pd.Timestamp(FECHA_INI_ACTUAL).strftime("%Y-%m-01")
P_FIN_ACTUAL = pd.Timestamp(FECHA_FIN_ACTUAL).strftime("%Y-%m-01")
P_INI_BASE = (pd.Timestamp(P_INI_ACTUAL) - pd.DateOffset(years=1)).strftime("%Y-%m-01")
P_FIN_BASE = (pd.Timestamp(P_FIN_ACTUAL) - pd.DateOffset(years=1)).strftime("%Y-%m-01")

assert len(CLIENTES_CONTROL) == 14, "CPB debe tener 14 DEX piloto"
assert len(CATEGORIAS_CP1_CORE) == 11 and len(CATEGORIAS_SIN_GC) == 9
CATEGORIAS = CATEGORIAS_CP1_CORE   # nombre común con CPM (lo usa la metodología compartida)

print(f"Actual: {P_INI_ACTUAL} -> {P_FIN_ACTUAL} | Base: {P_INI_BASE} -> {P_FIN_BASE}")
print(f"Grupo precio {GRUPO_PRECIO} | negocios {NEGOCIOS} | {len(CATEGORIAS_CP1_CORE)} categorías | versiones: {list(VERSIONES)}")

# ── Paso 1 · metodología del piloto CMP aplicada a CPB (celda = mes t × categoría) ──
# Ventana móvil: para evaluar el mes t, la trayectoria de matching son los 12 meses [t-12, t-1].
# El flag Perfecto (cr_cliente_dex_historico) solo existe desde esta fecha: limita cuántos meses
# de la ventana se pueden verificar para declarar "control limpio" (ver meses_flag_verificados).
P_INI_FLAG = "2025-07-01"

# Recurrencia mínima (meses con compra > 0 dentro de la ventana de 12 meses), por categoría
MIN_MESES_POR_CATEGORIA = {"Galletas": 6, "Cereales": 4}   # valores acordados en el piloto CMP
MIN_MESES_DEFAULT = 6    # ⚠️ las otras 9 categorías NO están calibradas: revisar df_sens_min_meses (celda 2.5)


print(f"Ventana móvil de {VENTANA_MESES} meses | flag Perfecto disponible desde {P_INI_FLAG}")


_CC  = ",".join(f"'{c}'" for c in CLIENTES_CONTROL)
_CAT = ",".join(f"'{c}'" for c in CATEGORIAS_CP1_CORE)
_SEG = ",".join(f"'{s}'" for s in SEGMENTOS)
_NEG = ",".join(f"'{n}'" for n in NEGOCIOS)
_DIM = "acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional"
_DET = "acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente_detalle"

SQL_BASE = f"""
WITH sucursales_control AS (
  SELECT DISTINCT cod_sucursal
  FROM `{_DIM}.cr_cliente_alicorp`
  WHERE cod_cliente IN ({_CC})
),

dim_sucursal AS (
  SELECT DISTINCT cod_sucursal, nom_sucursal, des_region_sucursal
  FROM `{_DIM}.cr_cliente_alicorp`
),

clientes AS (
  SELECT DISTINCT
    a.periodo,
    a.cod_cliente_dex,
    a.cod_sucursal,
    b.nom_sucursal,
    b.des_region_sucursal,
    c.des_segmento_transaccional,
    CASE WHEN a.flg_cliente_perfecto = TRUE THEN 'Perfecto' ELSE 'No Perfecto' END AS grupo_control
  FROM `{_DIM}.cr_cliente_dex_historico` a
  LEFT JOIN dim_sucursal b ON a.cod_sucursal = b.cod_sucursal
  LEFT JOIN `{_DIM}.cr_cliente_dex` c ON a.cod_cliente_dex = c.cod_cliente_dex
  INNER JOIN sucursales_control sc ON a.cod_sucursal = sc.cod_sucursal
  WHERE a.periodo BETWEEN '{P_INI_ACTUAL}' AND '{P_FIN_ACTUAL}'
    AND a.cod_grupo_precio_dex = '{GRUPO_PRECIO}'
    AND c.des_segmento_estrategica = '{SEGMENTO_ESTRATEGICO}'
    AND c.des_segmento_transaccional IN ({_SEG})
)"""

SQL_CLIENTES = SQL_BASE + "\nSELECT * FROM clientes"

SQL_VENTAS = SQL_BASE + f""",

ids AS (SELECT DISTINCT cod_cliente_dex FROM clientes),

ventas_actual AS (
  SELECT v.periodo, v.id_interlocutor_origen AS cod_cliente_dex, v.des_categoria,
         SUM(v.num_peso_neto / 1000) AS volumen_actual_ton
  FROM `{_DET}` v
  INNER JOIN ids ON v.id_interlocutor_origen = ids.cod_cliente_dex
  WHERE v.periodo BETWEEN '{P_INI_ACTUAL}' AND '{P_FIN_ACTUAL}'
    AND v.cod_negocio IN ({_NEG})
    AND v.des_categoria IN ({_CAT})
  GROUP BY 1, 2, 3
),

ventas_base AS (
  SELECT DATE_ADD(v.periodo, INTERVAL 1 YEAR) AS periodo,
         v.id_interlocutor_origen AS cod_cliente_dex, v.des_categoria,
         SUM(v.num_peso_neto / 1000) AS volumen_base_ton
  FROM `{_DET}` v
  INNER JOIN ids ON v.id_interlocutor_origen = ids.cod_cliente_dex
  WHERE v.periodo BETWEEN '{P_INI_BASE}' AND '{P_FIN_BASE}'
    AND v.cod_negocio IN ({_NEG})
    AND v.des_categoria IN ({_CAT})
  GROUP BY 1, 2, 3
)

SELECT
  COALESCE(va.periodo, vb.periodo)                 AS periodo,
  COALESCE(va.cod_cliente_dex, vb.cod_cliente_dex) AS cod_cliente_dex,
  COALESCE(va.des_categoria, vb.des_categoria)     AS des_categoria,
  COALESCE(vb.volumen_base_ton, 0)                 AS volumen_base_ton,
  COALESCE(va.volumen_actual_ton, 0)               AS volumen_actual_ton
FROM ventas_actual va
FULL OUTER JOIN ventas_base vb
  ON va.periodo = vb.periodo AND va.cod_cliente_dex = vb.cod_cliente_dex
  AND va.des_categoria = vb.des_categoria
"""

# UB base: mismo query del notebook oficial (Sección 3bis), con ventana parametrizada
# Flag Perfecto por cliente × mes DESDE P_INI_FLAG (para verificar "control sin Perfecto en el último año").
# Query aparte: df_clientes (universo 2026 con filtros) NO se toca porque lo usa el motor oficial.
SQL_FLAG = SQL_BASE + f""",

ids AS (SELECT DISTINCT cod_cliente_dex FROM clientes)

SELECT DISTINCT a.periodo, a.cod_cliente_dex,
       CASE WHEN a.flg_cliente_perfecto = TRUE THEN 'Perfecto' ELSE 'No Perfecto' END AS grupo_control
FROM `{_DIM}.cr_cliente_dex_historico` a
INNER JOIN ids ON a.cod_cliente_dex = ids.cod_cliente_dex
WHERE a.periodo BETWEEN '{P_INI_FLAG}' AND '{P_FIN_ACTUAL}'
  AND a.cod_grupo_precio_dex = '{GRUPO_PRECIO}'
"""

# Serie mensual continua cliente × categoría (ene-25 → sep-26), mismos filtros que SQL_VENTAS.
# De aquí salen: trayectoria móvil [t-12, t-1], base = serie[t-12] y actual = serie[t].
SQL_HIST = SQL_BASE + f""",

ids AS (SELECT DISTINCT cod_cliente_dex FROM clientes)

SELECT v.periodo, v.id_interlocutor_origen AS cod_cliente_dex, v.des_categoria,
       SUM(v.num_peso_neto / 1000) AS volumen_ton
FROM `{_DET}` v
INNER JOIN ids ON v.id_interlocutor_origen = ids.cod_cliente_dex
WHERE v.periodo BETWEEN '{P_INI_BASE}' AND '{P_FIN_ACTUAL}'
  AND v.cod_negocio IN ({_NEG})
  AND v.des_categoria IN ({_CAT})
GROUP BY 1, 2, 3
"""
print("SQL listos: clientes, ventas, ub, flag, hist")

q_neg = SQL_BASE + f"""
, ids AS (SELECT DISTINCT cod_cliente_dex FROM clientes)
SELECT v.cod_negocio, ROUND(SUM(v.num_peso_neto / 1000), 1) AS ton
FROM `{_DET}` v
INNER JOIN ids ON v.id_interlocutor_origen = ids.cod_cliente_dex
WHERE v.periodo BETWEEN '{P_INI_ACTUAL}' AND '{P_FIN_ACTUAL}'
  AND v.cod_negocio IN ('CP1', 'CP2')
  AND v.des_categoria IN ({_CAT})
GROUP BY 1 ORDER BY 1
"""
neg = ejecutar_grilla(q_neg)
display(neg)
if "CP2" in set(neg["cod_negocio"]) and NEGOCIOS == ["CP1"]:
    t = neg.set_index("cod_negocio")["ton"]
    print(f"CP2 = {t.get('CP2', 0) / max(t.sum(), 1e-9):.1%} del volumen CP1+CP2 en las categorías core (excluido por definición).")

r = ejecutar_grilla("""
SELECT COUNT(*) AS filas,
       COUNT(DISTINCT CONCAT(CAST(cod_cliente AS STRING), '|', CAST(cod_familia AS STRING))) AS llaves
FROM `acpe-dev-uc-sandbox-aa.dev.ratios_uv_bain_2025`
""")
assert int(r.loc[0, "filas"]) == int(r.loc[0, "llaves"]), "ratios_uv_bain_2025 duplicado por cliente×familia: el UB base se inflaría."
print("Preflight OK")

def descargar(nombre: str, sql: str, dtypes=None) -> pd.DataFrame:
    """Descarga desde BQ; con USAR_CACHE reutiliza cache/ validando contra un COUNT(*) actual."""
    ruta = f"cache/cpb_{nombre}_{P_INI_ACTUAL}_{P_FIN_ACTUAL}.csv"
    if USAR_CACHE and os.path.exists(ruta):
        df = pd.read_csv(ruta, dtype=dtypes)
        n_bq = int(ejecutar_grilla(f"SELECT COUNT(*) AS n FROM ({sql})").loc[0, "n"])
        if n_bq == len(df):
            print(f"[{nombre}] cache verificado ({len(df):,} filas).")
            return df
        print(f"[{nombre}] ⚠️ cache desactualizado ({len(df):,} vs {n_bq:,} en BQ): se vuelve a descargar.")
    df = ejecutar_grilla(sql)
    os.makedirs("cache", exist_ok=True)
    df.to_csv(ruta, index=False)
    return df


_str = {"cod_cliente_dex": str, "cod_sucursal": str}
df_clientes = descargar("clientes", SQL_CLIENTES, _str)
df_ventas   = descargar("ventas",   SQL_VENTAS,   _str)
df_flag     = descargar("flag",          SQL_FLAG,     _str)
df_hist     = descargar("hist_ene25_sep26", SQL_HIST,  _str)

df_clientes["periodo"] = pd.to_datetime(df_clientes["periodo"])
df_clientes["cod_cliente_dex"] = df_clientes["cod_cliente_dex"].astype(str)
df_clientes["cod_sucursal"] = df_clientes["cod_sucursal"].astype(str)

df_ventas["periodo"] = pd.to_datetime(df_ventas["periodo"])
df_ventas["cod_cliente_dex"] = df_ventas["cod_cliente_dex"].astype(str)
for c in ["volumen_base_ton", "volumen_actual_ton"]:
    df_ventas[c] = df_ventas[c].astype(float)

# ── Validaciones ────────────────────────────────────────────────────────────
dup = df_clientes.duplicated(["periodo", "cod_cliente_dex"]).sum()
assert dup == 0, f"{dup} filas duplicadas cliente×mes en el universo (fan-out en cr_cliente_alicorp / cr_cliente_dex)."

meses = pd.date_range(P_INI_ACTUAL, P_FIN_ACTUAL, freq="MS")
res = (df_clientes.groupby(["periodo", "grupo_control"])["cod_cliente_dex"].nunique().unstack(fill_value=0)
       .reindex(meses).fillna(0).astype(int))
print("Clientes elegibles por mes y grupo:")
display(res)
assert (res.get("Perfecto", 0) > 0).all(), "Hay meses sin clientes Perfecto (¿flag no cargado en cr_cliente_dex_historico?)."
assert (res.get("No Perfecto", 0) > 0).all(), "Hay meses sin clientes No Perfecto."

print("Regiones (BQ -> reporte):")
_r = (df_clientes.assign(region=df_clientes["des_region_sucursal"].fillna("SIN REGION"))
      .groupby("region")["cod_cliente_dex"].nunique().rename("clientes").reset_index())
_r["region_reporte"] = _r["region"].replace(MAPA_REGION)
print(_r.to_string(index=False))
_extra = set(_r["region_reporte"]) - set(REGIONES_ESPERADAS)
if _extra:
    print(f"⚠️ Regiones sin mapear: {sorted(_extra)} -> completar MAPA_REGION.")

sin_cat = sorted(set(CATEGORIAS_CP1_CORE) - set(df_ventas["des_categoria"]))
if sin_cat:
    print(f"⚠️ Categorías sin ninguna fila de ventas: {sin_cat}")

df_region = df_clientes[["cod_sucursal", "des_region_sucursal"]].drop_duplicates()
assert df_region["cod_sucursal"].is_unique, "Una sucursal con más de una región."

# ── df_flag y df_hist (ventana móvil) ───────────────────────────────────────
df_flag["periodo"] = pd.to_datetime(df_flag["periodo"])
df_flag["cod_cliente_dex"] = df_flag["cod_cliente_dex"].astype(str)
# si un cliente-mes trae filas con flags distintos, manda Perfecto (conservador para el control)
df_flag = (df_flag.assign(_p=(df_flag["grupo_control"] == "Perfecto"))
           .groupby(["periodo", "cod_cliente_dex"], as_index=False)["_p"].max()
           .rename(columns={"_p": "es_perfecto"}))

df_hist["periodo"] = pd.to_datetime(df_hist["periodo"])
df_hist["cod_cliente_dex"] = df_hist["cod_cliente_dex"].astype(str)
df_hist["volumen_ton"] = df_hist["volumen_ton"].astype(float)

meses_all = pd.date_range(P_INI_BASE, P_FIN_ACTUAL, freq="MS")          # serie continua ene-25 .. sep-26
meses_flag = pd.date_range(P_INI_FLAG, P_FIN_ACTUAL, freq="MS")         # flag disponible jul-25 .. sep-26
assert df_hist["periodo"].between(meses_all[0], meses_all[-1]).all(), "df_hist trae meses fuera de ene-25..sep-26."
assert df_flag["periodo"].between(meses_flag[0], meses_flag[-1]).all(), "df_flag trae meses fuera de P_INI_FLAG..P_FIN_ACTUAL."

# Consistencia 1: flag 2026 de df_flag == grupo_control de df_clientes
_f26 = df_flag[df_flag["periodo"] >= pd.Timestamp(P_INI_ACTUAL)]
_m = df_clientes.assign(es_perfecto_cli=(df_clientes["grupo_control"] == "Perfecto")).merge(
    _f26, on=["periodo", "cod_cliente_dex"], how="left")
n_dif_flag = int((_m["es_perfecto_cli"] != _m["es_perfecto"].fillna(False)).sum())
assert n_dif_flag == 0, f"{n_dif_flag} cliente-mes con flag distinto entre df_clientes y df_flag."

# Consistencia 2: df_hist reproduce el actual y la base (t-12) de df_ventas
_k = ["periodo", "cod_cliente_dex", "des_categoria"]
_h26 = df_hist[df_hist["periodo"] >= pd.Timestamp(P_INI_ACTUAL)]
_ca = df_ventas[_k + ["volumen_actual_ton"]].merge(_h26, on=_k, how="outer").fillna(0.0)
_b = df_ventas.assign(periodo=df_ventas["periodo"] - pd.DateOffset(years=1))[_k + ["volumen_base_ton"]]
_hb = df_hist[df_hist["periodo"].isin(_b["periodo"].unique())]
_cb = _b.merge(_hb, on=_k, how="outer").fillna(0.0)
dif_act = float((_ca["volumen_actual_ton"] - _ca["volumen_ton"]).abs().max())
dif_base = float((_cb["volumen_base_ton"] - _cb["volumen_ton"]).abs().max())
assert dif_act < 1e-6 and dif_base < 1e-6, f"df_hist no reproduce df_ventas (actual {dif_act:.3g}, base {dif_base:.3g} ton)."
print(f"df_hist: {len(df_hist):,} filas | df_flag: {len(df_flag):,} filas, {df_flag['cod_cliente_dex'].nunique():,} clientes | "
      f"consistencia con df_clientes y df_ventas OK.")


meses_eval = pd.date_range(P_INI_ACTUAL, P_FIN_ACTUAL, freq="MS")
COLS_M = [f"m{i:02d}" for i in range(VENTANA_MESES)]
ATTR_COLS = ["des_oficina", "nom_compania", "des_segmento_transaccional"]   # nom_compania ≙ DEX (nom_sucursal)

# Atributos de estrato (último registro del cliente en la ventana evaluada)
_attrs = (df_clientes.sort_values("periodo", kind="mergesort").groupby("cod_cliente_dex").tail(1)
          .set_index("cod_cliente_dex")[["cod_sucursal", "nom_sucursal", "des_region_sucursal",
                                         "des_segmento_transaccional"]])
if (df_clientes.groupby("cod_cliente_dex")["cod_sucursal"].nunique() > 1).any():
    print("⚠️ Hay clientes que cambian de sucursal dentro de la ventana evaluada (se usa la última).")
df_uni = pd.DataFrame(index=_attrs.index)
df_uni["cod_sucursal"] = _attrs["cod_sucursal"]
df_uni["nom_compania"] = _attrs["nom_sucursal"]
df_uni["des_oficina"] = _attrs["des_region_sucursal"].fillna("Sin región asignada").replace(MAPA_REGION)
df_uni["des_segmento_transaccional"] = _attrs["des_segmento_transaccional"]

# ORDEN CANÓNICO: el muestreo (random_state) y los desempates del matching dependen del orden de las filas, y BigQuery
# no garantiza el orden de salida entre corridas. Con el universo ordenado por cliente, los mismos datos dan los mismos pares.
df_uni = df_uni.sort_index()

# FLAG: Perfecto por cliente × mes (jul-25 .. sep-26); sin fila en la tabla => False
FLAG = (df_flag.assign(_p=df_flag["es_perfecto"].astype(int))
        .pivot_table(index="cod_cliente_dex", columns="periodo", values="_p", aggfunc="max")
        .reindex(index=df_uni.index, columns=meses_flag).fillna(0).astype(bool))

# ELEG: cliente elegible en el mes t (tiene fila en el universo filtrado de ese mes)
ELEG = (df_clientes.assign(_e=1).pivot_table(index="cod_cliente_dex", columns="periodo", values="_e", aggfunc="max")
        .reindex(index=df_uni.index, columns=meses_eval).fillna(0).astype(bool))

# SERIES: matriz cliente × mes (ene-25 .. sep-26) en kg, por categoría
SERIES = {}
for _cat in CATEGORIAS_CP1_CORE:
    _h = df_hist[df_hist["des_categoria"] == _cat]
    if _h.empty:
        SERIES[_cat] = pd.DataFrame(0.0, index=df_uni.index, columns=meses_all)
    else:
        SERIES[_cat] = (_h.pivot_table(index="cod_cliente_dex", columns="periodo", values="volumen_ton", aggfunc="sum")
                          .reindex(index=df_uni.index, columns=meses_all).fillna(0.0).clip(lower=0) * 1000)

print(f"Clientes en el universo: {len(df_uni):,} | meses evaluados: {len(meses_eval)} | categorías: {len(SERIES)}")
print("Perfecto / elegibles por mes (flag):")
display(pd.DataFrame({"perfecto": FLAG[meses_eval].sum(), "elegibles": ELEG.sum(),
                      "meses_flag_verificados": [sum(m >= pd.Timestamp(P_INI_FLAG) for m in
                                                     pd.date_range(t - pd.DateOffset(months=VENTANA_MESES),
                                                                   t - pd.DateOffset(months=1), freq="MS"))
                                                 for t in meses_eval]}, index=meses_eval)
        .rename_axis("periodo"))
