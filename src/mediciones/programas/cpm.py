# flake8: noqa
# ruff: noqa
# Script de PROGRAMA: configuración, SQL, universo y series. Se ejecuta con exec() antes que la metodología de su familia.
# MANTENIDO A MANO: este archivo es la fuente de verdad (los notebooks originales, fuera de este paquete, quedan solo como referencia histórica).
# =============================================================================
# PROGRAMA CPM — configuración, universo, datos y series
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
import decimal
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)

PROJECT_ID = "acpe-dev-uc-sandbox-aa"


def ejecutar_grilla(query: str, project_id: str = PROJECT_ID) -> pd.DataFrame:
    """Ejecuta SQL en BigQuery -> DataFrame (Decimal -> float). Propaga errores."""
    df = pd.read_gbq(query, project_id=project_id, dialect="standard", progress_bar_type="tqdm")
    for col in df.columns:
        if df[col].dtype == object:
            muestra = df[col].dropna()
            if len(muestra) and isinstance(muestra.iloc[0], decimal.Decimal):
                df[col] = df[col].astype(float)
    print(f"Filas: {len(df):,}")
    return df

# ── Ventana ─────────────────────────────────────────────────────────────────
FECHA_INI_ACTUAL = "2026-01-01"   # primer mes del numerador
FECHA_FIN_ACTUAL = globals().get("FECHA_FIN_ACTUAL_OVERRIDE", "2026-09-01")   # último mes del numerador (base = mismos meses de 2025)

# ── Reportes a generar: "TOTAL" | "ANTIGUOS" | "NUEVOS" ─────────────────────
#   ANTIGUOS = clientes con base > 0 en la ventana; NUEVOS = base <= 0 y venta actual > 0.
#   Se calculan en pandas (no cuestan más BigQuery), pero cada uno genera su Excel.
REPORTES = ["TOTAL"]

# ── Modo de estandarización ─────────────────────────────────────────────────
#   "v14"               : réplica exacta de v14 (peso = share de base perfecta por segmento;
#                          si un segmento no tiene base no perfecta, cae del lado "no perfecto"
#                          pero NO del lado "perfecto" -> ver columna cobertura_estand_pct).
#   "segmentos_comunes" : solo segmentos con base > 0 en AMBOS lados, pesos renormalizados.
MODO_ESTAND = "v14"

# ── Región (etiquetas esperadas en cr_cliente_alicorp.des_region_sucursal) ──
REGIONES_ESPERADAS = [
    "PE01 - CM-LIMA / NS CHICO",
    "PE02 - CM-NORTE / ORIENTE",
    "PE03 - CM-SUR / CENTRO",
]
# Los 6 valores de des_region_sucursal se agrupan en las 3 oficinas del reporte.
# (NS CHICO = Norte Chico + Sur Chico, por eso SUR CHICO va con Lima.)
MAPA_REGION = {
    "LIMA":      "PE01 - CM-LIMA / NS CHICO",
    "SUR CHICO": "PE01 - CM-LIMA / NS CHICO",
    "NORTE":     "PE02 - CM-NORTE / ORIENTE",
    "ORIENTE":   "PE02 - CM-NORTE / ORIENTE",
    "SUR":       "PE03 - CM-SUR / CENTRO",
    "CENTRO":    "PE03 - CM-SUR / CENTRO",
}

# ── Universo CPM ────────────────────────────────────────────────────────────
GRUPO_PRECIO = "PM"
COD_NEGOCIO  = "CP1"
SEGMENTOS    = ["1 Oro+", "2 Oro", "3 Plata", "4 Bronce"]
CLIENTES_CONTROL = [   # 16 DEX
    "1000006093", "1000009069", "1000009338", "1000009451", "1000009493", "1000009536",
    "1000035697", "1000036414", "1000036567", "1000036569", "1000049464", "1000054782",
    "1000001534", "1000000460", "1000037030", "1000037029",
]
CATEGORIAS = [   # 14 categorías (10 de CPB sin Galletas + 4 exclusivas CPM)
    "Pasta Dental", "Jabones de Lavar", "Aceites", "Cereales", "Pastas",
    "Untables Domesticos", "Suavizantes", "Salsas", "Detergentes",
    "Lavavajillas JW", "Lavavajillas", "Shampoo", "Acondicionador", "Productos para Peinar",
]

# ── Derivados ───────────────────────────────────────────────────────────────
P_INI_ACTUAL = pd.Timestamp(FECHA_INI_ACTUAL).strftime("%Y-%m-01")
P_FIN_ACTUAL = pd.Timestamp(FECHA_FIN_ACTUAL).strftime("%Y-%m-01")
P_INI_BASE = (pd.Timestamp(P_INI_ACTUAL) - pd.DateOffset(years=1)).strftime("%Y-%m-01")
P_FIN_BASE = (pd.Timestamp(P_FIN_ACTUAL) - pd.DateOffset(years=1)).strftime("%Y-%m-01")

assert len(CLIENTES_CONTROL) == 16, "CPM debe tener 16 DEX"
assert len(CATEGORIAS) == 14, "CPM debe tener 14 categorías"
assert MODO_ESTAND in ("v14", "segmentos_comunes")
assert set(REPORTES) <= {"TOTAL", "ANTIGUOS", "NUEVOS"}

print(f"Actual: {P_INI_ACTUAL} -> {P_FIN_ACTUAL} | Base: {P_INI_BASE} -> {P_FIN_BASE}")
print(f"Reportes: {REPORTES} | Estandarización: {MODO_ESTAND}")

# ── Paso 2 · metodología del piloto CMP aplicada a CPM (celda de trabajo = mes t × categoría) ──
# Ventana móvil: para evaluar el mes t, la trayectoria de matching son los 12 meses [t-12, t-1].
# El flag Perfecto (cr_cliente_dex_historico) solo existe desde esta fecha: limita cuántos meses de la
# ventana se pueden verificar para declarar "control limpio" (ver meses_flag_verificados).
P_INI_FLAG = "2025-07-01"

# Recurrencia mínima (meses con compra > 0 dentro de la ventana de 12 meses), por categoría
MIN_MESES_POR_CATEGORIA = {"Cereales": 4}   # valor acordado en el piloto CMP
MIN_MESES_DEFAULT = 6    # ⚠️ las demás categorías (incluidas las 4 exclusivas de CPM) NO están calibradas: revisar df_sens_min_meses (celda 2.6)


print(f"Ventana móvil de {VENTANA_MESES} meses | flag Perfecto disponible desde {P_INI_FLAG}")


_CC  = ",".join(f"'{c}'" for c in CLIENTES_CONTROL)
_CAT = ",".join(f"'{c}'" for c in CATEGORIAS)
_SEG = ",".join(f"'{s}'" for s in SEGMENTOS)

_DIM = "acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional"

SQL_BASE = f"""
WITH sucursales_control AS (
  SELECT DISTINCT cod_sucursal
  FROM `{_DIM}.cr_cliente_alicorp`
  WHERE cod_cliente_actual IN ({_CC})
),

dim_sucursal AS (
  SELECT DISTINCT cod_sucursal, nom_sucursal, des_region_sucursal
  FROM `{_DIM}.cr_cliente_alicorp`
),

universo_actual AS (
  SELECT DISTINCT
    SAFE_CAST(TRIM(a.cod_cliente_dex) AS INT64) AS cod_cliente_dex_num,
    a.des_segmento_transaccional,
    b.nom_sucursal,
    b.des_region_sucursal
  FROM `{_DIM}.cr_cliente_dex` a
  INNER JOIN sucursales_control sc ON a.cod_sucursal = sc.cod_sucursal
  LEFT JOIN dim_sucursal b ON a.cod_sucursal = b.cod_sucursal
  WHERE a.cod_grupo_precio_dex = '{GRUPO_PRECIO}'
    AND a.des_segmento_transaccional IN ({_SEG})
),

flag_perfecto AS (
  SELECT DISTINCT
    h.periodo,
    SAFE_CAST(TRIM(h.cod_cliente_dex) AS INT64) AS cod_cliente_dex_num
  FROM `{_DIM}.cr_cliente_dex_historico` h
  INNER JOIN universo_actual u
    ON SAFE_CAST(TRIM(h.cod_cliente_dex) AS INT64) = u.cod_cliente_dex_num
  WHERE h.periodo BETWEEN '{P_INI_ACTUAL}' AND '{P_FIN_ACTUAL}'
    AND h.cod_grupo_precio_dex = '{GRUPO_PRECIO}'
    AND h.flg_cliente_perfecto = TRUE
)"""

SQL_RAW = SQL_BASE + f""",

ventas_actual AS (
  SELECT v.periodo,
    SAFE_CAST(TRIM(v.id_interlocutor_origen) AS INT64) AS cod_cliente_dex_num,
    v.des_categoria AS categoria,
    SUM(v.num_peso_neto / 1000) AS volumen_actual_ton
  FROM `acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente_detalle` v
  INNER JOIN universo_actual u
    ON SAFE_CAST(TRIM(v.id_interlocutor_origen) AS INT64) = u.cod_cliente_dex_num
  WHERE v.periodo BETWEEN '{P_INI_ACTUAL}' AND '{P_FIN_ACTUAL}'
    AND v.cod_negocio = '{COD_NEGOCIO}'
    AND v.des_categoria IN ({_CAT})
  GROUP BY 1, 2, 3
),

ventas_base_ton AS (
  SELECT DATE_ADD(v.periodo, INTERVAL 1 YEAR) AS periodo,
    SAFE_CAST(TRIM(v.id_interlocutor_origen) AS INT64) AS cod_cliente_dex_num,
    v.des_categoria AS categoria,
    SUM(v.num_peso_neto / 1000) AS volumen_base_ton
  FROM `acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente_detalle` v
  INNER JOIN universo_actual u
    ON SAFE_CAST(TRIM(v.id_interlocutor_origen) AS INT64) = u.cod_cliente_dex_num
  WHERE v.periodo BETWEEN '{P_INI_BASE}' AND '{P_FIN_BASE}'
    AND v.cod_negocio = '{COD_NEGOCIO}'
    AND v.des_categoria IN ({_CAT})
  GROUP BY 1, 2, 3
),

ventas_base_soles AS (
  SELECT DATE_ADD(v.periodo, INTERVAL 1 YEAR) AS periodo,
    SAFE_CAST(TRIM(v.id_interlocutor_origen) AS INT64) AS cod_cliente_dex_num,
    v.cod_cliente_sap, v.des_categoria AS categoria, v.cod_familia,
    SUM(v.mnt_bruto) AS venta_base_soles
  FROM `acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente_detalle` v
  INNER JOIN universo_actual u
    ON SAFE_CAST(TRIM(v.id_interlocutor_origen) AS INT64) = u.cod_cliente_dex_num
  WHERE v.periodo BETWEEN '{P_INI_BASE}' AND '{P_FIN_BASE}'
    AND v.cod_negocio = '{COD_NEGOCIO}'
    AND v.des_categoria IN ({_CAT})
  GROUP BY 1, 2, 3, 4, 5
),

ratios_ub AS (
  SELECT cod_cliente AS cod_cliente_sap, cod_familia,
    mnt_utilidad_bruta / NULLIF(mnt_venta_bruta, 0) AS ratio_ub
  FROM `acpe-dev-uc-sandbox-aa.dev.ratios_uv_bain_2025`
),

ub_base_cliente AS (
  SELECT vbs.periodo, vbs.cod_cliente_dex_num, vbs.categoria,
    SUM(vbs.venta_base_soles) AS venta_base_soles,
    SUM(vbs.venta_base_soles * r.ratio_ub) AS ub_base_soles,
    SUM(IF(r.ratio_ub IS NULL, vbs.venta_base_soles, 0)) AS venta_sin_ratio_soles
  FROM ventas_base_soles vbs
  LEFT JOIN ratios_ub r
    ON vbs.cod_cliente_sap = r.cod_cliente_sap AND vbs.cod_familia = r.cod_familia
  GROUP BY 1, 2, 3
),

claves AS (
  SELECT periodo, cod_cliente_dex_num, categoria FROM ventas_actual
  UNION DISTINCT
  SELECT periodo, cod_cliente_dex_num, categoria FROM ventas_base_ton
  UNION DISTINCT
  SELECT periodo, cod_cliente_dex_num, categoria FROM ub_base_cliente
)

SELECT
  k.periodo,
  k.cod_cliente_dex_num,
  k.categoria,
  u.des_segmento_transaccional AS seg,
  u.nom_sucursal               AS dex,
  u.des_region_sucursal        AS region,
  (fp.cod_cliente_dex_num IS NOT NULL) AS flg_perfecto,
  IFNULL(va.volumen_actual_ton, 0)      AS volumen_actual_ton,
  IFNULL(vb.volumen_base_ton, 0)        AS volumen_base_ton,
  IFNULL(ub.ub_base_soles, 0.0)         AS ub_base_soles,
  IFNULL(ub.venta_base_soles, 0.0)      AS venta_base_soles,
  IFNULL(ub.venta_sin_ratio_soles, 0.0) AS venta_sin_ratio_soles
FROM claves k
INNER JOIN universo_actual u ON k.cod_cliente_dex_num = u.cod_cliente_dex_num
LEFT JOIN flag_perfecto fp
  ON k.cod_cliente_dex_num = fp.cod_cliente_dex_num AND k.periodo = fp.periodo
LEFT JOIN ventas_actual va
  ON k.cod_cliente_dex_num = va.cod_cliente_dex_num AND k.periodo = va.periodo AND k.categoria = va.categoria
LEFT JOIN ventas_base_ton vb
  ON k.cod_cliente_dex_num = vb.cod_cliente_dex_num AND k.periodo = vb.periodo AND k.categoria = vb.categoria
LEFT JOIN ub_base_cliente ub
  ON k.cod_cliente_dex_num = ub.cod_cliente_dex_num AND k.periodo = ub.periodo AND k.categoria = ub.categoria
"""
print("SQL_RAW listo.")

# ── Queries adicionales para la metodología de matching (mismo universo y filtros que SQL_RAW) ──
# Universo con atributos de estrato (un registro por cliente)
SQL_UNI = SQL_BASE + """
SELECT cod_cliente_dex_num, des_segmento_transaccional AS seg, nom_sucursal AS dex, des_region_sucursal AS region
FROM universo_actual
"""

# Flag Perfecto (solo filas TRUE) DESDE P_INI_FLAG: para verificar "control sin Perfecto en el último año"
SQL_FLAG = SQL_BASE + f""",

flag_hist AS (
  SELECT DISTINCT h.periodo, SAFE_CAST(TRIM(h.cod_cliente_dex) AS INT64) AS cod_cliente_dex_num
  FROM `{_DIM}.cr_cliente_dex_historico` h
  INNER JOIN universo_actual u
    ON SAFE_CAST(TRIM(h.cod_cliente_dex) AS INT64) = u.cod_cliente_dex_num
  WHERE h.periodo BETWEEN '{P_INI_FLAG}' AND '{P_FIN_ACTUAL}'
    AND h.cod_grupo_precio_dex = '{GRUPO_PRECIO}'
    AND h.flg_cliente_perfecto = TRUE
)
SELECT periodo, cod_cliente_dex_num FROM flag_hist
"""

# Serie mensual continua cliente × categoría (ene-25 → sep-26): trayectoria móvil [t-12, t-1],
# base = serie[t-12] y actual = serie[t]
SQL_HIST = SQL_BASE + f""",

serie AS (
  SELECT v.periodo,
    SAFE_CAST(TRIM(v.id_interlocutor_origen) AS INT64) AS cod_cliente_dex_num,
    v.des_categoria AS categoria,
    SUM(v.num_peso_neto / 1000) AS volumen_ton
  FROM `acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente_detalle` v
  INNER JOIN universo_actual u
    ON SAFE_CAST(TRIM(v.id_interlocutor_origen) AS INT64) = u.cod_cliente_dex_num
  WHERE v.periodo BETWEEN '{P_INI_BASE}' AND '{P_FIN_ACTUAL}'
    AND v.cod_negocio = '{COD_NEGOCIO}'
    AND v.des_categoria IN ({_CAT})
  GROUP BY 1, 2, 3
)
SELECT periodo, cod_cliente_dex_num, categoria, volumen_ton FROM serie
"""
print("SQL listos: SQL_RAW, SQL_UNI, SQL_FLAG, SQL_HIST")

# (1) Perfectos por mes en la tabla de flag
pf = ejecutar_grilla(SQL_BASE + "\nSELECT periodo, COUNT(*) AS n_perfectos FROM flag_perfecto GROUP BY periodo ORDER BY periodo")
pf["periodo"] = pd.to_datetime(pf["periodo"])
meses_esperados = pd.date_range(P_INI_ACTUAL, P_FIN_ACTUAL, freq="MS")
pf = pf.set_index("periodo").reindex(meses_esperados).fillna(0).astype(int)
display(pf)
assert (pf["n_perfectos"] > 0).all(), \
    f"Hay meses SIN perfectos en cr_cliente_dex_historico: {list(pf.index[pf['n_perfectos'] == 0].strftime('%Y-%m'))}"

# (2) Universo: un registro por cliente
u = ejecutar_grilla(SQL_BASE + "\nSELECT COUNT(*) AS filas, COUNT(DISTINCT cod_cliente_dex_num) AS clientes FROM universo_actual")
print(u.to_string(index=False))
assert int(u.loc[0, "filas"]) == int(u.loc[0, "clientes"]), "Universo con clientes duplicados (revisar cr_cliente_alicorp)."

# (3) Unicidad de ratios UB
r = ejecutar_grilla("""
SELECT COUNT(*) AS filas,
       COUNT(DISTINCT CONCAT(CAST(cod_cliente AS STRING), '|', CAST(cod_familia AS STRING))) AS llaves
FROM `acpe-dev-uc-sandbox-aa.dev.ratios_uv_bain_2025`
""")
print(r.to_string(index=False))
assert int(r.loc[0, "filas"]) == int(r.loc[0, "llaves"]), "ratios_uv_bain_2025 duplicado por cliente×familia: el UB base se inflaría."
print("Preflight OK")

df_raw = ejecutar_grilla(SQL_RAW)
df_uni_raw  = ejecutar_grilla(SQL_UNI)
df_flag_raw = ejecutar_grilla(SQL_FLAG)
df_hist_raw = ejecutar_grilla(SQL_HIST)

def preparar_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Tipifica, normaliza etiquetas y precalcula columnas perfecto / no perfecto."""
    d = df.copy()
    d["periodo"] = pd.to_datetime(d["periodo"])
    for c in ["volumen_actual_ton", "volumen_base_ton", "ub_base_soles", "venta_base_soles", "venta_sin_ratio_soles"]:
        d[c] = pd.to_numeric(d[c]).astype(float).fillna(0.0)
    d["flg"] = d["flg_perfecto"].astype(bool)
    d["region"] = d["region"].fillna("SIN REGION").replace(MAPA_REGION)
    d["dex"] = d["dex"].fillna("SIN DEX")
    d["seg"] = d["seg"].fillna("SIN SEGMENTO")
    d["_g"] = "PM TOTAL"

    f = d["flg"]
    d["base_p"] = d["volumen_base_ton"].where(f, 0.0)
    d["act_p"]  = d["volumen_actual_ton"].where(f, 0.0)
    d["base_n"] = d["volumen_base_ton"].where(~f, 0.0)
    d["act_n"]  = d["volumen_actual_ton"].where(~f, 0.0)
    d["ub_p"]   = d["ub_base_soles"].where(f, 0.0)          # UB base solo de perfectos

    activo = (d["volumen_base_ton"] > 0) | (d["volumen_actual_ton"] > 0)
    d["cli_p"] = d["cod_cliente_dex_num"].where(f & activo)   # NaN si no aplica -> nunique lo ignora
    d["cli_n"] = d["cod_cliente_dex_num"].where(~f & activo)
    return d


d_all = preparar_raw(df_raw)

# ── Validaciones ────────────────────────────────────────────────────────────
print("Mapeo de región aplicado (BQ -> reporte):")
_cw = (df_raw.assign(region=df_raw["region"].fillna("SIN REGION"))
             .groupby("region")["cod_cliente_dex_num"].nunique().rename("clientes").reset_index())
_cw["region_reporte"] = _cw["region"].replace(MAPA_REGION)
print(_cw.to_string(index=False))

dup = d_all.duplicated(["cod_cliente_dex_num", "periodo", "categoria"]).sum()
assert dup == 0, f"{dup} filas duplicadas cliente×mes×categoría (fan-out en algún join)."

print("\nRegiones en la data:")
vc = d_all.groupby("region")["cod_cliente_dex_num"].nunique().rename("clientes")
print(vc.to_string())
extra = set(vc.index) - set(REGIONES_ESPERADAS)
faltan = set(REGIONES_ESPERADAS) - set(vc.index)
if extra or faltan:
    print(f"\n⚠️  Regiones fuera de lo esperado: {sorted(extra)} | sin data: {sorted(faltan)}"
          "\n    -> ajustar MAPA_REGION en la configuración y volver a correr desde esta celda.")

sin_cat = sorted(set(CATEGORIAS) - set(d_all["categoria"]))
if sin_cat:
    print(f"\n⚠️  Categorías sin ninguna fila (¿nombre distinto en BQ?, p. ej. tildes): {sin_cat}")

print("\nPerfectos activos y TN por mes:")
mes = d_all.groupby("periodo").agg(
    perfectos_activos=("cli_p", "nunique"),
    ton_base=("volumen_base_ton", "sum"),
    ton_actual=("volumen_actual_ton", "sum"),
).round(1)
display(mes)
assert (mes["perfectos_activos"] > 0).all(), "Algún mes no tiene perfectos con volumen."

venta_perf = d_all.loc[d_all["flg"], "venta_base_soles"].sum()
sin_ratio  = d_all.loc[d_all["flg"], "venta_sin_ratio_soles"].sum()
print(f"\nVenta base de perfectos sin ratio UB: {sin_ratio / venta_perf:.1%} "
      "(ese % no aporta UB base ni UB incremental)")

# ── Tipificación (nombres de columna que usan las celdas siguientes) ─────────
df_hist = df_hist_raw.rename(columns={"cod_cliente_dex_num": "cod_cliente_dex", "categoria": "des_categoria"}).copy()
df_hist["periodo"] = pd.to_datetime(df_hist["periodo"])
df_hist["cod_cliente_dex"] = df_hist["cod_cliente_dex"].astype("int64")
df_hist["volumen_ton"] = df_hist["volumen_ton"].astype(float)

df_flag = df_flag_raw.rename(columns={"cod_cliente_dex_num": "cod_cliente_dex"}).copy()
df_flag["periodo"] = pd.to_datetime(df_flag["periodo"])
df_flag["cod_cliente_dex"] = df_flag["cod_cliente_dex"].astype("int64")
df_flag = df_flag.drop_duplicates(["periodo", "cod_cliente_dex"])

meses_all = pd.date_range(P_INI_BASE, P_FIN_ACTUAL, freq="MS")           # serie continua ene-25 .. sep-26
meses_flag = pd.date_range(P_INI_FLAG, P_FIN_ACTUAL, freq="MS")          # flag disponible jul-25 .. sep-26
assert df_hist["periodo"].between(meses_all[0], meses_all[-1]).all(), "df_hist trae meses fuera de la ventana."
assert df_flag["periodo"].between(meses_flag[0], meses_flag[-1]).all(), "df_flag trae meses fuera de P_INI_FLAG..P_FIN_ACTUAL."

# Consistencia 1: flag 2026 de df_flag == flg de d_all (base de la sección 2.3)
_fk = df_flag[df_flag["periodo"] >= pd.Timestamp(P_INI_ACTUAL)][["periodo", "cod_cliente_dex"]].assign(_en_flag=True)
_m = d_all[["periodo", "cod_cliente_dex_num", "flg"]].rename(columns={"cod_cliente_dex_num": "cod_cliente_dex"}).merge(
    _fk, on=["periodo", "cod_cliente_dex"], how="left")
n_dif_flag = int((_m["_en_flag"].notna() != _m["flg"]).sum())
assert n_dif_flag == 0, f"{n_dif_flag} cliente-mes con flag distinto entre d_all y df_flag."

# Consistencia 2: df_hist reproduce el actual y la base (t-12) de d_all
_k = ["periodo", "cod_cliente_dex", "des_categoria"]
_a = d_all.rename(columns={"cod_cliente_dex_num": "cod_cliente_dex", "categoria": "des_categoria"})
_h26 = df_hist[df_hist["periodo"] >= pd.Timestamp(P_INI_ACTUAL)]
_ca = _a[_k + ["volumen_actual_ton"]].merge(_h26, on=_k, how="outer").fillna(0.0)
_b = _a.assign(periodo=_a["periodo"] - pd.DateOffset(years=1))[_k + ["volumen_base_ton"]]
_hb = df_hist[df_hist["periodo"].isin(_b["periodo"].unique())]
_cb = _b.merge(_hb, on=_k, how="outer").fillna(0.0)
dif_act = float((_ca["volumen_actual_ton"] - _ca["volumen_ton"]).abs().max())
dif_base = float((_cb["volumen_base_ton"] - _cb["volumen_ton"]).abs().max())
assert dif_act < 1e-6 and dif_base < 1e-6, f"df_hist no reproduce d_all (actual {dif_act:.3g}, base {dif_base:.3g} ton)."
print(f"df_hist: {len(df_hist):,} filas | df_flag: {len(df_flag):,} filas, {df_flag['cod_cliente_dex'].nunique():,} clientes | "
      f"consistencia con d_all OK (flag, actual y base).")

meses_eval = pd.date_range(P_INI_ACTUAL, P_FIN_ACTUAL, freq="MS")
COLS_M = [f"m{i:02d}" for i in range(VENTANA_MESES)]
ATTR_COLS = ["des_oficina", "nom_compania", "des_segmento_transaccional"]   # nom_compania ≙ DEX

# Atributos de estrato (universo estático: un registro por cliente)
_u = df_uni_raw.copy()
_u["cod_cliente_dex"] = _u["cod_cliente_dex_num"].astype("int64")
assert _u["cod_cliente_dex"].is_unique, "Universo con clientes duplicados."
df_uni = pd.DataFrame({
    "nom_compania": _u["dex"].fillna("SIN DEX").to_numpy(),
    "des_oficina": _u["region"].fillna("SIN REGION").replace(MAPA_REGION).to_numpy(),
    "des_segmento_transaccional": _u["seg"].fillna("SIN SEGMENTO").to_numpy()},
    index=pd.Index(_u["cod_cliente_dex"].to_numpy(), name="cod_cliente_dex"))

# ORDEN CANÓNICO: el muestreo (random_state) y los desempates del matching dependen del orden de las filas, y BigQuery
# no garantiza el orden de salida entre corridas. Con el universo ordenado por cliente, los mismos datos dan los mismos pares.
df_uni = df_uni.sort_index()

# FLAG: Perfecto por cliente × mes (jul-25 .. sep-26); sin fila en la tabla => False
FLAG = (df_flag.assign(_p=1).pivot_table(index="cod_cliente_dex", columns="periodo", values="_p", aggfunc="max")
        .reindex(index=df_uni.index, columns=meses_flag).fillna(0).astype(bool))

# ELEG: todo cliente del universo es elegible cada mes (el filtro de actividad lo hace el embudo de la celda 2.6)
ELEG = pd.DataFrame(True, index=df_uni.index, columns=meses_eval)

# SERIES: matriz cliente × mes (ene-25 .. sep-26) en kg, por categoría
SERIES = {}
for _cat in CATEGORIAS:
    _h = df_hist[df_hist["des_categoria"] == _cat]
    if _h.empty:
        SERIES[_cat] = pd.DataFrame(0.0, index=df_uni.index, columns=meses_all)
    else:
        SERIES[_cat] = (_h.pivot_table(index="cod_cliente_dex", columns="periodo", values="volumen_ton", aggfunc="sum")
                          .reindex(index=df_uni.index, columns=meses_all).fillna(0.0).clip(lower=0) * 1000)

print(f"Clientes en el universo: {len(df_uni):,} | meses evaluados: {len(meses_eval)} | categorías: {len(SERIES)}")
print("Perfectos por mes en la tabla de flag (revisar que NINGÚN mes, sobre todo el último, quede a medias):")
_vent = [sum(m >= pd.Timestamp(P_INI_FLAG) for m in pd.date_range(t - pd.DateOffset(months=VENTANA_MESES),
                                                                    t - pd.DateOffset(months=1), freq="MS"))
         for t in meses_eval]
display(pd.DataFrame({"perfectos_en_flag": FLAG.sum()}).rename_axis("periodo").T)
display(pd.DataFrame({"perfectos": FLAG[meses_eval].sum(), "universo": len(df_uni),
                      "meses_flag_verificados": _vent}, index=meses_eval).rename_axis("periodo"))
