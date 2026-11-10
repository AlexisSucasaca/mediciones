# Programa SINTÉTICO solo para probar el motor de punta a punta (cumple el mismo contrato de datos que cpb/cpm)
import os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")

FECHA_INI_ACTUAL = "2026-07-01"
FECHA_FIN_ACTUAL = globals().get("FECHA_FIN_ACTUAL_OVERRIDE", "2026-09-01")
P_INI_ACTUAL = pd.Timestamp(FECHA_INI_ACTUAL).strftime("%Y-%m-01")
P_FIN_ACTUAL = pd.Timestamp(FECHA_FIN_ACTUAL).strftime("%Y-%m-01")
VENTANA_MESES = 12
P_INI_FLAG = "2025-07-01"
CATEGORIAS = ["Galletas", "Cereales"]
MIN_MESES_POR_CATEGORIA = {"Galletas": 6, "Cereales": 4}
MIN_MESES_DEFAULT = 6
OUTLIER_METODO = "iqr_log"; OUTLIER_K = 1.5; OUTLIER_NIVEL = ["des_oficina"]
ATTR_COLS = ["des_oficina", "nom_compania", "des_segmento_transaccional"]

rng = np.random.default_rng(7)
N = 1500
ids = [f"{i:010d}" for i in rng.choice(10**8, N, replace=False)]
df_uni = pd.DataFrame({"cod_sucursal": rng.integers(1, 15, N).astype(str)}, index=pd.Index(ids, name="cod_cliente_dex"))
df_uni["nom_compania"] = "DEX " + df_uni["cod_sucursal"]
df_uni["des_oficina"] = rng.choice(["PE01 - CM-LIMA / NS CHICO", "PE02 - CM-NORTE / ORIENTE", "PE03 - CM-SUR / CENTRO"], N, p=[.5, .25, .25])
df_uni["des_segmento_transaccional"] = rng.choice(["1 Oro+", "2 Oro", "3 Plata"], N, p=[.2, .3, .5])
df_uni = df_uni.sort_index()

meses_eval = pd.date_range(P_INI_ACTUAL, P_FIN_ACTUAL, freq="MS")
meses_all = pd.date_range(pd.Timestamp(P_INI_ACTUAL) - pd.DateOffset(years=1), P_FIN_ACTUAL, freq="MS")
meses_flag = pd.date_range(P_INI_FLAG, P_FIN_ACTUAL, freq="MS")
COLS_M = [f"m{i:02d}" for i in range(VENTANA_MESES)]

es_perf = pd.Series(rng.random(N) < 0.22, index=df_uni.index)
FLAG = pd.DataFrame(False, index=df_uni.index, columns=meses_flag)
for m in meses_flag:
    if m >= pd.Timestamp("2026-06-01"):
        FLAG[m] = es_perf.to_numpy()
ELEG = pd.DataFrame(rng.random((N, len(meses_eval))) < 0.95, index=df_uni.index, columns=meses_eval)
ELEG[:] = True

SERIES = {}
for cat in CATEGORIAS:
    nivel = rng.lognormal(3.0, 0.7, N)
    compra = rng.random((N, len(meses_all))) < 0.88
    base = nivel[:, None] * rng.lognormal(0, 0.25, (N, len(meses_all)))
    X = pd.DataFrame(np.where(compra, base, 0.0), index=df_uni.index, columns=meses_all)
    efecto = es_perf.to_numpy()[:, None] & (meses_all >= pd.Timestamp("2026-07-01"))[None, :]
    SERIES[cat] = (X * np.where(efecto, 1.18, 1.0)).clip(lower=0)
print(f"[sintético] {N} clientes | meses_eval {len(meses_eval)} | categorías {CATEGORIAS}")
