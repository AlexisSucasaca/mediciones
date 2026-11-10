# flake8: noqa
# ruff: noqa
# Script de FAMILIA: se ejecuta con exec() en un espacio de nombres compartido con el script del programa
# (mediciones.ejecucion.ejecutar_familia). No es un módulo importable. MANTENIDO A MANO: este archivo es la fuente de verdad (los notebooks originales, fuera de este paquete, quedan solo como referencia histórica).
# =============================================================================
# METODOLOGÍA MENSUAL (CPB y CPM): pool → viabilidad → matching húngaro → panel DiD + bootstrap → base completa
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
def construir_pool(t, cat):
    """Pool de la celda (mes t × categoría): trayectoria móvil [t-12, t-1] en kg + embudo de limpieza."""
    min_meses = MIN_MESES_POR_CATEGORIA.get(cat, MIN_MESES_DEFAULT)
    meses_w = pd.date_range(t - pd.DateOffset(months=VENTANA_MESES), t - pd.DateOffset(months=1), freq="MS")
    M = SERIES[cat]
    W = M[meses_w].copy()
    W.columns = COLS_M
    actual = M[t]

    mw_flag = [m for m in meses_w if m >= pd.Timestamp(P_INI_FLAG)]
    n_perf_vent = FLAG[mw_flag].sum(axis=1) if mw_flag else pd.Series(0, index=df_uni.index)
    activo = ELEG[t] & ((W.sum(axis=1) > 0) | (actual > 0))

    pool = df_uni[ATTR_COLS].join(W)
    pool["base_kg"] = W["m00"]                       # = serie[t-12]
    pool["actual_kg"] = actual
    pool["peso_12m"] = W.sum(axis=1)
    pool["n_meses"] = (W > 0).sum(axis=1)
    pool["n_perf_vent"] = n_perf_vent
    pool["rol"] = np.where(FLAG[t], "AC", "CT")
    pool["_global"] = "Global"
    pool = pool[activo.to_numpy()]

    fn = []
    def _snap(cod, etapa, p):
        fn.append({"periodo": t, "categoria": cat, "cod": cod, "etapa": etapa,
                   "AC": int((p["rol"] == "AC").sum()), "CT": int((p["rol"] == "CT").sum())})

    _snap("e0", "0. Activos en la categoría", pool)
    pool = pool[~((pool["rol"] == "CT") & (pool["n_perf_vent"] > 0))]
    _snap("e1", "1. Control sin Perfecto en la ventana", pool)

    es_base0 = pool["base_kg"] == 0
    nu = pool[(pool["rol"] == "AC") & es_base0 & (pool["actual_kg"] > 0)]
    nuevos = pd.DataFrame({"periodo": t, "categoria": cat, "cod_cliente_dex": nu.index,
                           "des_oficina": nu["des_oficina"].to_numpy(), "nom_compania": nu["nom_compania"].to_numpy(),
                           "des_segmento_transaccional": nu["des_segmento_transaccional"].to_numpy(),
                           "toneladas_actual": (nu["actual_kg"] / 1000).to_numpy()})
    pool = pool[~es_base0]
    _snap("e2", "2. Antiguos (compraron en t-12)", pool)

    sens = [{"categoria": cat, "min_meses": k,
             "AC": int(((pool["rol"] == "AC") & (pool["n_meses"] >= k)).sum()),
             "CT": int(((pool["rol"] == "CT") & (pool["n_meses"] >= k)).sum())} for k in range(3, 9)]

    excl = []
    baja = pool[(pool["rol"] == "AC") & (pool["n_meses"] < min_meses)]
    excl.append(pd.DataFrame({"periodo": t, "categoria": cat, "cod_cliente_dex": baja.index,
                              "motivo": "baja_recurrencia", "base_ton": (baja["base_kg"] / 1000).to_numpy(),
                              "actual_ton": (baja["actual_kg"] / 1000).to_numpy()}))
    pool = pool[pool["n_meses"] >= min_meses]
    _snap("e3", f"3. Recurrencia >= {min_meses}", pool)

    outl = None
    if OUTLIER_METODO != "ninguno" and len(pool):
        v = np.log1p(pool["peso_12m"]) if OUTLIER_METODO == "iqr_log" else pool["peso_12m"]
        tmp = pool[OUTLIER_NIVEL].copy()
        tmp["_v"] = v.to_numpy()
        if OUTLIER_METODO in ("iqr_log", "iqr_raw"):
            q1 = tmp.groupby(OUTLIER_NIVEL)["_v"].transform(lambda s: s.quantile(.25))
            q3 = tmp.groupby(OUTLIER_NIVEL)["_v"].transform(lambda s: s.quantile(.75))
            lo, hi = q1 - OUTLIER_K * (q3 - q1), q3 + OUTLIER_K * (q3 - q1)
        else:
            hi = tmp.groupby(OUTLIER_NIVEL)["_v"].transform(lambda s: s.quantile(.99))
            lo = pd.Series(-np.inf, index=tmp.index)
        out = ((tmp["_v"] < lo) | (tmp["_v"] > hi)).to_numpy()
        _o = pool[OUTLIER_NIVEL].assign(_o=out.astype(int))
        outl = _o.groupby(OUTLIER_NIVEL).agg(n_pool=("_o", "size"), n_outlier=("_o", "sum")).reset_index()
        outl.insert(0, "periodo", t); outl.insert(1, "categoria", cat)
        ac_out = pool[out & (pool["rol"] == "AC").to_numpy()]
        excl.append(pd.DataFrame({"periodo": t, "categoria": cat, "cod_cliente_dex": ac_out.index,
                                  "motivo": "outlier", "base_ton": (ac_out["base_kg"] / 1000).to_numpy(),
                                  "actual_ton": (ac_out["actual_kg"] / 1000).to_numpy()}))
        pool = pool[~out]
    _snap("e4", "4. Sin outliers", pool)

    ac = pool[pool["rol"] == "AC"]
    tb = t - pd.DateOffset(months=12)
    diag = {"periodo": t, "categoria": cat, "AC": len(ac),
            "AC_con_meses_perfecto_en_ventana": int((ac["n_perf_vent"] > 0).sum()),
            "AC_perfecto_en_base": (int(FLAG.loc[ac.index, tb].sum()) if tb >= pd.Timestamp(P_INI_FLAG) else np.nan)}
    return {"pool": pool, "funnel": fn, "nuevos": nuevos, "excl": pd.concat(excl, ignore_index=True),
            "sens": sens, "outl": outl, "diag": diag}


# Meses a CALCULAR en esta corrida. El runner inyecta RECALCULAR_DESDE (el mes siguiente a lo ya publicado) para que un cierre normal calcule solo
# el mes nuevo: cada mes se empareja y se mide por separado, así que los meses cerrados no hacen falta (lo publicado no cambia y los pares fijos se
# conservan al guardar). Con validar_cerrados=True (o sin nada publicado) se calculan todos los meses y se mide la deriva de los cerrados.
_RECALC = globals().get('RECALCULAR_DESDE')
meses_calc = [t for t in meses_eval if _RECALC is None or t >= pd.Timestamp(_RECALC)]
print(f"Meses a calcular: {[t.strftime('%Y-%m') for t in meses_calc]}" + (f" (desde {pd.Timestamp(_RECALC).date()}; los cerrados no se recalculan)" if _RECALC else " (todos)"))

POOLS = {}
_f, _n, _e, _s, _o, _d = [], [], [], [], [], []
for t_ in meses_calc:
    for cat_ in CATEGORIAS:
        r_ = construir_pool(t_, cat_)
        POOLS[(t_, cat_)] = r_["pool"]
        _f += r_["funnel"]; _n.append(r_["nuevos"]); _e.append(r_["excl"]); _s += r_["sens"]; _d.append(r_["diag"])
        if r_["outl"] is not None:
            _o.append(r_["outl"])

df_embudo_limpieza = pd.DataFrame(_f)
df_nuevos = pd.concat(_n, ignore_index=True)
df_ac_excluidos = pd.concat(_e, ignore_index=True)
df_outliers = pd.concat(_o, ignore_index=True) if _o else pd.DataFrame()
df_diag_ventana = pd.DataFrame(_d)
df_sens_min_meses = pd.DataFrame(_s).groupby(["categoria", "min_meses"], as_index=False)[["AC", "CT"]].sum()
print(f"Celdas construidas: {len(POOLS)} (mes × categoría)")

# ── Resumen del embudo por categoría (unidad = cliente-mes, 8 meses acumulados) ──
_p = df_embudo_limpieza.pivot_table(index="categoria", columns="cod", values=["AC", "CT"], aggfunc="sum")
_nn = df_nuevos.groupby("categoria").size()
_ex = df_ac_excluidos.pivot_table(index="categoria", columns="motivo", values="cod_cliente_dex", aggfunc="count").fillna(0)
resumen_embudo = pd.DataFrame({
    "min_meses": pd.Series({c_: MIN_MESES_POR_CATEGORIA.get(c_, MIN_MESES_DEFAULT) for c_ in CATEGORIAS}),
    "AC_activos": _p[("AC", "e0")], "AC_nuevos": _nn, "AC_antiguos": _p[("AC", "e2")],
    "AC_baja_recurrencia": _ex.get("baja_recurrencia"), "AC_outlier": _ex.get("outlier"),
    "AC_matching": _p[("AC", "e4")],
    "CT_activos": _p[("CT", "e0")], "CT_limpio": _p[("CT", "e1")], "CT_antiguos": _p[("CT", "e2")],
    "CT_matching": _p[("CT", "e4")],
}).reindex(CATEGORIAS).fillna(0).astype(int, errors="ignore")
resumen_embudo["pct_AC_matching_de_antiguos"] = (100 * resumen_embudo["AC_matching"]
                                                  / resumen_embudo["AC_antiguos"].clip(lower=1)).round(1)
print("EMBUDO por categoría (cliente-mes acumulado ene–sep). AC_activos = nuevos + antiguos + sin compra en t ni t-12:")
display(resumen_embudo)

print("AC en matching por mes × categoría:")
display(df_embudo_limpieza[df_embudo_limpieza["cod"] == "e4"].pivot(index="categoria", columns="periodo", values="AC")
        .rename(columns=lambda d: d.strftime("%Y-%m")).reindex(CATEGORIAS))
print("CT en matching por mes × categoría:")
display(df_embudo_limpieza[df_embudo_limpieza["cod"] == "e4"].pivot(index="categoria", columns="periodo", values="CT")
        .rename(columns=lambda d: d.strftime("%Y-%m")).reindex(CATEGORIAS))

print("NUEVOS (base = 0 en t-12, compran en t): toneladas atribuibles por mes × categoría:")
display(df_nuevos.pivot_table(index="categoria", columns="periodo", values="toneladas_actual", aggfunc="sum", margins=True,
                              margins_name="TOTAL").round(2).rename(columns=lambda d: d.strftime("%Y-%m") if hasattr(d, "strftime") else d))

print("DIAGNÓSTICO de la ventana móvil — % de AC (en matching) cuya ventana ya incluye meses Perfecto (trayectoria 'contaminada'):")
_dg = df_diag_ventana.groupby("periodo")[["AC", "AC_con_meses_perfecto_en_ventana", "AC_perfecto_en_base"]].sum(min_count=1)
_dg["pct_AC_con_meses_perfecto_en_ventana"] = (100 * _dg["AC_con_meses_perfecto_en_ventana"] / _dg["AC"].clip(lower=1)).round(1)
_dg["pct_AC_perfecto_en_base"] = (100 * _dg["AC_perfecto_en_base"] / _dg["AC"].clip(lower=1)).round(1)
_dg.loc[_dg["AC_perfecto_en_base"].isna(), "pct_AC_perfecto_en_base"] = np.nan
display(_dg.rename(index=lambda d: d.strftime("%Y-%m")))

print("Sensibilidad a MIN_MESES: % de AC / CT antiguos (todos los meses) que sobrevive — para calibrar las categorías sin valor acordado:")
_b2 = df_embudo_limpieza[df_embudo_limpieza["cod"] == "e2"].groupby("categoria")[["AC", "CT"]].sum().add_suffix("_base")
_s2 = df_sens_min_meses.join(_b2, on="categoria")
_s2["AC_pct"] = (100 * _s2["AC"] / _s2["AC_base"].clip(lower=1)).round(1)
_s2["CT_pct"] = (100 * _s2["CT"] / _s2["CT_base"].clip(lower=1)).round(1)
df_sens_min_meses = _s2[["categoria", "min_meses", "AC", "CT", "AC_pct", "CT_pct"]]
display(df_sens_min_meses.pivot(index="categoria", columns="min_meses", values="AC_pct").reindex(CATEGORIAS).rename_axis(columns="AC %"))
display(df_sens_min_meses.pivot(index="categoria", columns="min_meses", values="CT_pct").reindex(CATEGORIAS).rename_axis(columns="CT %"))

NIVELES_VIAB = {"global": ["_global"], "oficina": ["des_oficina"], "dex": ["nom_compania"],
                "dex_segmento": ["nom_compania", "des_segmento_transaccional"]}


def conteos_cascada(pool, t, cat):
    ac, ct = pool[pool["rol"] == "AC"], pool[pool["rol"] == "CT"]
    filas = []
    for nivel, cols in NIVELES_VIAB.items():
        v = pd.DataFrame({"AC": ac.groupby(cols).size(), "CT": ct.groupby(cols).size()}).fillna(0).astype(int)
        v.index.names = cols
        v["ac_sin_control"] = (v["AC"] - v["CT"]).clip(lower=0)
        v = v.reset_index()
        v.insert(0, "periodo", t); v.insert(1, "categoria", cat); v.insert(2, "nivel", nivel)
        filas.append(v)
    return pd.concat(filas, ignore_index=True).drop(columns=["_global"], errors="ignore")


df_viab = pd.concat([conteos_cascada(p_, t_, c_) for (t_, c_), p_ in POOLS.items()], ignore_index=True)
df_viab = df_viab[df_viab["AC"] > 0].copy()
df_viab["ratio_CT_AC"] = (df_viab["CT"] / df_viab["AC"]).round(3)

print("Razón CT/AC global por mes × categoría:")
display(df_viab[df_viab["nivel"] == "global"].pivot(index="categoria", columns="periodo", values="ratio_CT_AC")
        .rename(columns=lambda d: d.strftime("%Y-%m")).reindex(CATEGORIAS))

print("% de AC que no tendría pareja 1:1 en ese nivel (acumulado ene–sep) — dónde empieza a trabajar la cascada:")
_g = df_viab.groupby(["categoria", "nivel"])[["AC", "ac_sin_control"]].sum()
_g["pct_ac_sin_control"] = (100 * _g["ac_sin_control"] / _g["AC"].clip(lower=1)).round(1)
display(_g["pct_ac_sin_control"].unstack("nivel")[["global", "oficina", "dex", "dex_segmento"]].reindex(CATEGORIAS))

# =============================================================================
# N1. MATCHING HÚNGARO POR CELDA — portado del piloto CMP (celda = mes t × categoría)
# =============================================================================
import os, zlib
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
from sklearn.metrics import pairwise_distances
import statsmodels.formula.api as smf
from mediciones.metodologia.atribucion import atribuir, estado_yoy   # atribución compartida (regla de 'nuevo' = REGLA_NUEVO, de config/programas.yaml)

N_JOBS = max(1, min(32, os.cpu_count() or 1))

# ── PARÁMETROS (valores del piloto, salvo lo indicado) ──────────────────────
                                  # (~0.004/mes) es más fino que la precisión del test con 12 puntos mensuales: casi ninguna celda sería equivalente.
                                  # Mira `horizonte_max_ok_meses` en df_resumen para ver hasta qué horizonte aguanta cada celda.

ESTRATOS_FULL = ["des_oficina", "nom_compania", "des_segmento_transaccional"]   # nom_compania ≙ DEX
MATCH_FULL    = ["des_oficina", "nom_compania", "des_segmento_transaccional"]
MUESTREAR, FRAC_MUESTREO, MIN_POR_CELDA = True, 0.60, 5
W_NIVEL, W_FORMA = 0.5, 0.5
RATIO_MIN, RATIO_MAX = 0.70, 1.40
PESOS_OVERRIDE, N_SEMILLAS_OVERRIDE = {}, {}            # claves (periodo, categoría): {'W_NIVEL':..,'W_FORMA':..} / n semillas
MAX_REUSO_CONTROL_POR_CELDA, MAX_REUSO_CONTROL_DEFAULT = {}, 1   # 1:1 estricto; 1:N solo si el embudo lo pide
VERBOSE_MATCHING = False

# ── PARES FIJOS (reproducibilidad) ──────────────────────────────────────────
# Los pares se guardan en un CSV y se reutilizan: las celdas (mes × categoría) que ya están en el archivo NO se vuelven a
# emparejar; solo se recalcula su diagnóstico (validez + ANCOVA) sobre los mismos pares. Solo se empareja lo que falta
# (por ejemplo, el mes nuevo). Para rehacer un mes: RECALCULAR_MESES = ["2026-09"]. Para empezar de cero: borrar el CSV.
RUTA_PARES_FIJOS = f"{PROGRAMA}_pares_fijos.csv"
USAR_PARES_FIJOS = True
RECALCULAR_MESES = []

NIVELES = [[len(MATCH_FULL) - k + 1, MATCH_FULL[:k], CALIPER_REL]
           for k in range(len(MATCH_FULL), 0, -1)][:NIVEL_MAX]
NIVELES[-1][2] = CALIPER_ULTIMO


# ── Semilla de matching: COPIA LITERAL del piloto (no se modifica la lógica) ──
from mediciones.nucleo.matching import _correr_semilla  # kernel idéntico en ambas familias (ver nucleo/matching.py)

# ── ANCOVA de paralelismo (misma regresión del piloto; un solo ajuste da p-valor e IC) ──
def _frame_largo(cod_accion, cod_control, ac_df, ct_df, cols_m):
    cod_accion, cod_control = list(cod_accion), list(cod_control)
    A = np.log1p(ac_df.loc[cod_accion, cols_m].to_numpy(float))
    C = np.log1p(ct_df.loc[cod_control, cols_m].to_numpy(float))
    T = len(cols_m)
    t = np.arange(T)
    dfl = pd.DataFrame({
        "cliente": np.concatenate([np.repeat(np.arange(len(A)), T), len(A) + np.repeat(np.arange(len(C)), T)]),
        "tiempo": np.concatenate([np.tile(t, len(A)), np.tile(t, len(C))]),
        "grupo": np.concatenate([np.ones(A.size), np.zeros(C.size)]),
        "valor": np.concatenate([A.ravel(), C.ravel()])})
    dfl["inter"] = dfl["tiempo"] * dfl["grupo"]
    return dfl


def test_ancova_paralelismo(cod_accion, cod_control, ac_df, ct_df, cols_m, margen_coef, alpha=ALPHA_TOST):
    """ANCOVA: log1p(peso) ~ tiempo + grupo + tiempo:grupo, SE clusterizados por cliente.
    Devuelve coef y p-valor de la interacción, su IC (1-2*alpha) y equivalencia TOST (IC dentro de ±margen)."""
    nan = {"coef": np.nan, "p_valor": np.nan, "ic_inf": np.nan, "ic_sup": np.nan, "ok": False}
    try:
        dfl = _frame_largo(cod_accion, cod_control, ac_df, ct_df, cols_m)
        mod = smf.ols("valor ~ tiempo + grupo + inter", data=dfl).fit(
            cov_type="cluster", cov_kwds={"groups": dfl["cliente"].to_numpy()})
        ic = mod.conf_int(alpha=2 * alpha)
        ic_inf, ic_sup = float(ic.loc["inter", 0]), float(ic.loc["inter", 1])
        coef = float(mod.params["inter"])
        return {"coef": coef, "p_valor": float(mod.pvalues["inter"]), "ic_inf": ic_inf, "ic_sup": ic_sup,
                "ok": bool(pd.notna(coef) and ic_inf > -margen_coef and ic_sup < margen_coef)}
    except Exception:
        return nan


def matching_celda(t, cat, pool):
    """Matching húngaro de una celda (mes t × categoría): mismo flujo del loop principal del piloto."""
    ac, ct = pool[pool["rol"] == "AC"], pool[pool["rol"] == "CT"]
    if len(ac) == 0 or len(ct) == 0:
        return None
    cols_m = COLS_M
    w_nivel = PESOS_OVERRIDE.get((t, cat), {}).get("W_NIVEL", W_NIVEL)
    w_forma = PESOS_OVERRIDE.get((t, cat), {}).get("W_FORMA", W_FORMA)
    n_sem = N_SEMILLAS_OVERRIDE.get((t, cat), N_SEMILLAS_DEFAULT)
    max_reuso = MAX_REUSO_CONTROL_POR_CELDA.get((t, cat), MAX_REUSO_CONTROL_DEFAULT)

    # cascada de estratos para el muestreo (baja de nivel si la mediana de celda < MIN_POR_CELDA)
    estratos_usar, nivel_estrato = ESTRATOS_FULL[:], 1
    for k in range(len(ESTRATOS_FULL), 0, -1):
        cand = ESTRATOS_FULL[:k]
        tam = ac.groupby([ac[c].astype(str) for c in cand]).size()
        estratos_usar, nivel_estrato = cand, len(ESTRATOS_FULL) - k + 1
        if len(tam) and tam.median() >= MIN_POR_CELDA:
            break

    # espacio de matching: nivel (log1p) + forma (diferencias mensuales)
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

    # precómputos independientes de la semilla
    ac_por_oficina_pre = dict(list(ac.groupby("des_oficina")))
    claves_unicas = []
    for _, claves_n, _ in NIVELES:
        if claves_n not in claves_unicas:
            claves_unicas.append(claves_n)
    ka_full_pre = {tuple(cl): ac[cl].astype(str).agg("|".join, axis=1) for cl in claves_unicas}
    kc_full_pre = {tuple(cl): ct[cl].astype(str).agg("|".join, axis=1) for cl in claves_unicas}

    res = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_correr_semilla)(
            seed, ac, ct, cols_m, estratos_usar, Lac_df, Lct_df, Fac_df, Fct_df,
            MED_N, MED_F, NIVELES, MUESTREAR, FRAC_MUESTREO,
            w_nivel, w_forma, CORR_UMBRAL, RATIO_MIN, RATIO_MAX,
            max_reuso=max_reuso, ac_por_oficina=ac_por_oficina_pre,
            ka_full_por_claves=ka_full_pre, kc_full_por_claves=kc_full_pre)
        for seed in range(1, n_sem + 1))
    res = [r for r in res if r is not None]
    if not res:
        return None
    rs_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in res])

    # búsqueda de la mejor semilla VÁLIDA que además sea EQUIVALENTE (TOST), como en el piloto
    val = [r for r in res if r["valida"]]
    semillas_probadas, best, anc = 0, None, None
    if val:
        candidatas = sorted(val, key=lambda r: abs(r["pend_rel"]))
        for cand in candidatas[:MAX_SEMILLAS_PROBAR_EQUIVALENCIA]:
            semillas_probadas += 1
            a_ = test_ancova_paralelismo(cand["_p"].cod_accion, cand["_p"].cod_control, ac, ct, cols_m,
                                         MARGEN_EQUIVALENCIA_PEND)
            if a_["ok"]:
                best, anc = cand, a_
                break
        if best is None:
            best = candidatas[0]
            anc = test_ancova_paralelismo(best["_p"].cod_accion, best["_p"].cod_control, ac, ct, cols_m,
                                          MARGEN_EQUIVALENCIA_PEND)
    else:
        best = min(res, key=lambda r: abs(r["pend_rel"]))
        anc = test_ancova_paralelismo(best["_p"].cod_accion, best["_p"].cod_control, ac, ct, cols_m,
                                      MARGEN_EQUIVALENCIA_PEND)

    amp = max(abs(anc["ic_inf"]), abs(anc["ic_sup"])) if pd.notna(anc["ic_inf"]) else np.nan
    resumen = {
        "periodo": t, "categoria": cat, "AC_limpio": len(ac), "CT_limpio": len(ct),
        "AC_muestra": best["n_muestra"], "seed": best["seed"], "pares": best["pares"],
        "controles_unicos": best.get("controles_unicos", best["pares"]),
        "max_reuso_observado": best.get("max_reuso_observado", 1),
        "cobertura_pct": best["cobertura_pct"], "corr": best["corr"], "ratio_vol": best["ratio_vol"],
        "pend_rel": best["pend_rel"], "valida": best["valida"], "nivel_estrato": nivel_estrato,
        "ancova_coef": round(anc["coef"], 5) if pd.notna(anc["coef"]) else np.nan,
        "ancova_p_valor": round(anc["p_valor"], 4) if pd.notna(anc["p_valor"]) else np.nan,
        "tost_ic_inf": round(anc["ic_inf"], 5) if pd.notna(anc["ic_inf"]) else np.nan,
        "tost_ic_sup": round(anc["ic_sup"], 5) if pd.notna(anc["ic_sup"]) else np.nan,
        "margen_equivalencia": round(MARGEN_EQUIVALENCIA_PEND, 5),
        # horizonte (meses) hasta el cual la celda seguiría siendo equivalente con el margen de 5%
        "horizonte_max_ok_meses": (round(np.log1p(MARGEN_ERROR_OBJETIVO) / amp, 1) if pd.notna(amp) and amp > 0 else np.nan),
        "semillas_validas": len(val), "semillas_probadas_equivalencia": semillas_probadas,
        "paralelismo_ok": bool(anc["ok"])}

    p_df = best["_p"].copy()
    p_df.insert(0, "periodo", t); p_df.insert(1, "des_categoria", cat)
    p_df["seed"], p_df["celda_valida"] = best["seed"], best["valida"]
    p_df["ofi_ac"], p_df["ofi_ct"] = ac.loc[p_df.cod_accion, "des_oficina"].to_numpy(), ct.loc[p_df.cod_control, "des_oficina"].to_numpy()
    p_df["comp_ac"], p_df["comp_ct"] = ac.loc[p_df.cod_accion, "nom_compania"].to_numpy(), ct.loc[p_df.cod_control, "nom_compania"].to_numpy()
    p_df["seg_ac"], p_df["seg_ct"] = (ac.loc[p_df.cod_accion, "des_segmento_transaccional"].to_numpy(),
                                      ct.loc[p_df.cod_control, "des_segmento_transaccional"].to_numpy())
    p_df["peso_ac"], p_df["peso_ct"] = ac.loc[p_df.cod_accion, "peso_12m"].to_numpy(), ct.loc[p_df.cod_control, "peso_12m"].to_numpy()
    sem = rs_df.assign(periodo=t, categoria=cat)
    return {"resumen": resumen, "semillas": sem, "pares": p_df,
            "pend": set(best["_pend"]), "muestra": set(best["_ac"].index)}


def _alinear_ids(serie, indice):
    """Deja los IDs del CSV con el mismo tipo que el índice del pool (numérico o texto)."""
    return pd.to_numeric(serie, errors="coerce") if indice.dtype.kind in "iu" else serie.astype(str)


def celda_desde_pares_fijos(t, cat, pool, pf):
    """Reconstruye una celda desde pares fijos: MISMOS pares, diagnóstico recalculado con las fórmulas de _correr_semilla."""
    ac, ct = pool[pool["rol"] == "AC"], pool[pool["rol"] == "CT"]
    pf = pf.copy()
    pf["cod_accion"], pf["cod_control"] = _alinear_ids(pf["cod_accion"], ac.index), _alinear_ids(pf["cod_control"], ct.index)
    n_orig = len(pf)
    pf = pf[pf["cod_accion"].isin(ac.index) & pf["cod_control"].isin(ct.index)].drop_duplicates("cod_accion")
    descartados = n_orig - len(pf)
    if len(pf) == 0:
        return None
    cols_m = COLS_M
    t_ac = ac.loc[pf["cod_accion"], cols_m].mean().to_numpy()
    t_ct = ct.loc[pf["cod_control"], cols_m].mean().to_numpy()
    d_ac, d_ct = np.diff(t_ac), np.diff(t_ct)
    corr = np.corrcoef(d_ac, d_ct)[0, 1] if d_ac.std() > 0 and d_ct.std() > 0 else np.nan
    ratio = d_ac.std() / d_ct.std() if d_ct.std() > 0 else np.inf
    brecha_t = t_ac - t_ct
    pend_rel = (linregress(np.arange(len(brecha_t)), brecha_t).slope / t_ac.mean()) if t_ac.mean() > 0 else np.inf
    valida = bool(pd.notna(corr) and corr >= CORR_UMBRAL and RATIO_MIN <= ratio <= RATIO_MAX)
    anc = test_ancova_paralelismo(pf["cod_accion"], pf["cod_control"], ac, ct, cols_m, MARGEN_EQUIVALENCIA_PEND)
    amp = max(abs(anc["ic_inf"]), abs(anc["ic_sup"])) if pd.notna(anc["ic_inf"]) else np.nan
    resumen = {
        "periodo": t, "categoria": cat, "AC_limpio": len(ac), "CT_limpio": len(ct),
        "AC_muestra": np.nan, "seed": np.nan, "pares": len(pf), "controles_unicos": pf["cod_control"].nunique(),
        "max_reuso_observado": int(pf["cod_control"].value_counts().max()),
        "cobertura_pct": np.nan, "corr": round(corr, 3) if pd.notna(corr) else np.nan,
        "ratio_vol": round(ratio, 3), "pend_rel": round(pend_rel, 5), "valida": valida, "nivel_estrato": np.nan,
        "ancova_coef": round(anc["coef"], 5) if pd.notna(anc["coef"]) else np.nan,
        "ancova_p_valor": round(anc["p_valor"], 4) if pd.notna(anc["p_valor"]) else np.nan,
        "tost_ic_inf": round(anc["ic_inf"], 5) if pd.notna(anc["ic_inf"]) else np.nan,
        "tost_ic_sup": round(anc["ic_sup"], 5) if pd.notna(anc["ic_sup"]) else np.nan,
        "margen_equivalencia": round(MARGEN_EQUIVALENCIA_PEND, 5),
        "horizonte_max_ok_meses": (round(np.log1p(MARGEN_ERROR_OBJETIVO) / amp, 1) if pd.notna(amp) and amp > 0 else np.nan),
        "semillas_validas": np.nan, "semillas_probadas_equivalencia": np.nan,
        "paralelismo_ok": bool(anc["ok"]), "origen": "pares_fijos", "pares_descartados": descartados}
    p_df = pd.DataFrame({"periodo": t, "des_categoria": cat, "cod_accion": pf["cod_accion"].to_numpy(),
                         "cod_control": pf["cod_control"].to_numpy()})
    for c_ in ("seed", "ronda_reuso", "dist_total", "dist_nivel", "dist_forma", "peso_pob"):
        p_df[c_] = pf[c_].to_numpy() if c_ in pf.columns else np.nan
    oa, oc = ac.loc[p_df.cod_accion, "des_oficina"].to_numpy(), ct.loc[p_df.cod_control, "des_oficina"].to_numpy()
    ca, cc = ac.loc[p_df.cod_accion, "nom_compania"].to_numpy(), ct.loc[p_df.cod_control, "nom_compania"].to_numpy()
    sa, sc = (ac.loc[p_df.cod_accion, "des_segmento_transaccional"].to_numpy(),
              ct.loc[p_df.cod_control, "des_segmento_transaccional"].to_numpy())
    nivel = pf["nivel_match"].to_numpy() if "nivel_match" in pf.columns else np.full(len(pf), np.nan)
    inferido = np.where((oa == oc) & (ca == cc) & (sa == sc), 1, np.where((oa == oc) & (ca == cc), 2, 3))
    p_df["nivel_match"] = np.where(pd.isna(nivel), inferido, nivel).astype(int)
    p_df["celda_valida"] = valida
    p_df["ofi_ac"], p_df["ofi_ct"], p_df["comp_ac"], p_df["comp_ct"], p_df["seg_ac"], p_df["seg_ct"] = oa, oc, ca, cc, sa, sc
    p_df["peso_ac"], p_df["peso_ct"] = ac.loc[p_df.cod_accion, "peso_12m"].to_numpy(), ct.loc[p_df.cod_control, "peso_12m"].to_numpy()
    return {"resumen": resumen, "semillas": None, "pares": p_df, "pend": set(), "muestra": set(p_df["cod_accion"]), "fijo": True}


# ── LOOP PRINCIPAL: una celda por (mes, categoría); reutiliza pares fijos y empareja solo lo que falta ──
_pf_all, _fijos = None, {}
if USAR_PARES_FIJOS and os.path.exists(RUTA_PARES_FIJOS):
    _pf_all = pd.read_csv(RUTA_PARES_FIJOS, dtype={"cod_accion": str, "cod_control": str}, parse_dates=["periodo"])
    _fijos = {k: g for k, g in _pf_all.groupby(["periodo", "categoria"])}
    print(f"Pares fijos cargados: {len(_pf_all):,} pares en {len(_fijos)} celdas ({RUTA_PARES_FIJOS})")
_recalc = {pd.Timestamp(m + "-01") for m in RECALCULAR_MESES}
resumen_out, semillas_out, matches_out, calc_out, MATCH_INFO = [], [], [], [], {}
_n_cel, _n_fijas, _n_nuevas, _descartados = len(POOLS), 0, 0, 0
for _i, ((t_, cat_), pool_) in enumerate(POOLS.items(), start=1):
    _es_fija = (t_, cat_) in _fijos and t_ not in _recalc
    if _es_fija:
        r_ = celda_desde_pares_fijos(t_, cat_, pool_, _fijos[(t_, cat_)])
        if r_ is None:      # ningún par fijo vale (IDs con otro formato, p. ej. sin ceros a la izquierda, o clientes fuera del pool): se re-empareja la celda
            print(f"  [AVISO] {t_.strftime('%Y-%m')} | {cat_}: los pares fijos no coinciden con los pools (¿formato de IDs?); se re-empareja la celda")
            _es_fija = False
            r_ = matching_celda(t_, cat_, pool_)
    else:
        r_ = matching_celda(t_, cat_, pool_)
    if r_ is None:
        print(f"  [SKIP] {t_.strftime('%Y-%m')} | {cat_}: sin acción o sin control tras la limpieza")
        continue
    if _es_fija:
        _n_fijas += 1
        _descartados += r_["resumen"]["pares_descartados"]
        if r_["resumen"]["pares_descartados"] > 0.05 * (r_["resumen"]["pares"] + r_["resumen"]["pares_descartados"]):
            print(f"  [AVISO] {t_.strftime('%Y-%m')} | {cat_}: {r_['resumen']['pares_descartados']} pares fijos ya no están en los pools (cambió el dato)")
    else:
        _n_nuevas += 1
        calc_out.append(r_["pares"])
    resumen_out.append(r_["resumen"])
    if r_["semillas"] is not None:
        semillas_out.append(r_["semillas"])
    matches_out.append(r_["pares"])
    MATCH_INFO[(t_, cat_)] = {"pend": r_["pend"], "muestra": r_["muestra"], "fijo": bool(r_.get("fijo", False))}
    if VERBOSE_MATCHING or _i % 11 == 0 or _i == _n_cel:
        _rr = r_["resumen"]
        print(f"  [{_i}/{_n_cel}] {t_.strftime('%Y-%m')} | {cat_}: pares={_rr['pares']:,} "
              f"valida={_rr['valida']} paralelismo_ok={_rr['paralelismo_ok']} ({'fijo' if _es_fija else 'calculado'})")
print(f"Celdas con pares fijos: {_n_fijas} | emparejadas ahora: {_n_nuevas} | pares fijos descartados por cambio de pools: {_descartados}")
if _pf_all is not None and len(_pf_all) and _descartados > 0.2 * len(_pf_all):
    print("  [AVISO] se descartó más del 20% de los pares fijos: revisa que los IDs del CSV tengan el mismo formato que los del universo (ceros a la izquierda, texto vs número).")

df_matches = pd.concat(matches_out, ignore_index=True)
df_resumen = pd.DataFrame(resumen_out)
df_semillas = pd.concat(semillas_out, ignore_index=True) if semillas_out else pd.DataFrame()

# ── Guardar los pares fijos: lo ya fijado se conserva tal cual; solo se agregan/reemplazan las celdas emparejadas ahora ──
if _n_nuevas > 0 or _pf_all is None:
    _cols_pf = ["periodo", "categoria", "cod_accion", "cod_control", "seed", "nivel_match", "ronda_reuso",
                "dist_total", "dist_nivel", "dist_forma", "peso_pob"]
    _nuevos_pf = (pd.concat(calc_out, ignore_index=True).rename(columns={"des_categoria": "categoria"}).reindex(columns=_cols_pf)
                  if calc_out else pd.DataFrame(columns=_cols_pf))
    if _pf_all is not None and len(_pf_all):
        _claves_nuevas = set(zip(_nuevos_pf["periodo"], _nuevos_pf["categoria"]))
        _viejos_pf = _pf_all[[(p, c) not in _claves_nuevas for p, c in zip(_pf_all["periodo"], _pf_all["categoria"])]]
        _nuevos_pf = pd.concat([_viejos_pf.reindex(columns=_cols_pf), _nuevos_pf], ignore_index=True)
    _tmp = RUTA_PARES_FIJOS + ".tmp"
    _nuevos_pf.sort_values(["periodo", "categoria", "cod_accion"]).to_csv(_tmp, index=False)
    os.replace(_tmp, RUTA_PARES_FIJOS)
    print(f"Pares fijos guardados en {RUTA_PARES_FIJOS}: {len(_nuevos_pf):,} pares")
# reportable = matching estadísticamente válido (corr + ratio_vol) Y paralelismo (ANCOVA/TOST) dentro del margen
df_resumen["reportable"] = df_resumen["valida"] & df_resumen["paralelismo_ok"]
REPORTABLE_MAP = df_resumen.set_index(["periodo", "categoria"])["reportable"].to_dict()

print(f"\nCeldas con matching: {len(df_resumen)}/{_n_cel} | válidas: {int(df_resumen.valida.sum())} | "
      f"paralelismo_ok: {int(df_resumen.paralelismo_ok.sum())} | REPORTABLES: {int(df_resumen.reportable.sum())}")
print(f"Pares totales: {len(df_matches):,} | cobertura media: {df_resumen.cobertura_pct.mean():.1f}% | "
      f"nivel de cascada de los pares:")
display(df_matches.groupby("nivel_match").size().rename("pares").to_frame())
_reu = df_matches.groupby(["periodo", "des_categoria", "cod_control"]).size()
print(f"Controles usados >1 vez dentro de una misma celda: {(_reu > 1).sum():,} (1:1 estricto => debe ser 0)")

print("\nREPORTABLE por categoría × mes:")
display(df_resumen.pivot(index="categoria", columns="periodo", values="reportable")
        .rename(columns=lambda d: d.strftime("%Y-%m")).reindex(CATEGORIAS))
print("Cobertura de matching (% de AC en el pool que tiene pareja):")
display(df_resumen.pivot(index="categoria", columns="periodo", values="cobertura_pct")
        .rename(columns=lambda d: d.strftime("%Y-%m")).reindex(CATEGORIAS))
print("ANCOVA/TOST — horizonte máximo (meses) hasta el que la celda seguiría siendo equivalente (margen 5%). "
      f"Con HORIZONTE_MESES_REPORTE={HORIZONTE_MESES_REPORTE}, pasa si es >= ese valor:")
display(df_resumen.pivot(index="categoria", columns="periodo", values="horizonte_max_ok_meses")
        .rename(columns=lambda d: d.strftime("%Y-%m")).reindex(CATEGORIAS))

# =============================================================================
# N2. PANEL POR PAR + BRECHA DiD + IC BOOTSTRAP
# =============================================================================
from mediciones.metodologia.inferencia import N_MIN_PARES   # piso técnico del piloto para publicar IC (30)
CAP_PCTL = 99.5           # winsorización de pesos mensuales (por celda), como el piloto
SOLO_REPORTABLES = False  # False = como el piloto (la columna `reportable` solo informa); True = vistas solo con celdas reportables

paneles, _capadas = [], 0
for (t_, cat_), g_ in df_matches.groupby(["periodo", "des_categoria"]):
    pool_ = POOLS[(t_, cat_)]
    a_ids, c_ids = g_["cod_accion"].to_numpy(), g_["cod_control"].to_numpy()
    A, C = pool_.loc[a_ids], pool_.loc[c_ids]
    vals = np.concatenate([A[COLS_M].to_numpy().ravel(), A["actual_kg"].to_numpy(),
                           C[COLS_M].to_numpy().ravel(), C["actual_kg"].to_numpy()])
    vals = vals[vals > 0]
    tope = float(np.quantile(vals, CAP_PCTL / 100)) if len(vals) else np.inf
    pre_a, post_a = A["base_kg"].to_numpy(), A["actual_kg"].to_numpy()
    pre_c, post_c = C["base_kg"].to_numpy(), C["actual_kg"].to_numpy()
    _capadas += int(((pre_a > tope) | (post_a > tope) | (pre_c > tope) | (post_c > tope)).sum())
    pre_a, post_a, pre_c, post_c = [np.minimum(x, tope) for x in (pre_a, post_a, pre_c, post_c)]
    paneles.append(pd.DataFrame({
        "periodo": t_, "des_categoria": cat_, "cod_accion": a_ids, "cod_control": c_ids,
        "par_id": [f"{t_.strftime('%Y-%m')}|{cat_}|{a}" for a in a_ids],
        "des_oficina": A["des_oficina"].to_numpy(), "nom_compania": A["nom_compania"].to_numpy(),
        "des_segmento_transaccional": A["des_segmento_transaccional"].to_numpy(),
        "pre_accion": pre_a, "post_accion": post_a, "pre_control": pre_c, "post_control": post_c,
        "did_kg": (post_a - pre_a) - (post_c - pre_c)}))
df_panel = pd.concat(paneles, ignore_index=True)
df_panel["periodo_str"] = df_panel["periodo"].dt.strftime("%Y-%m")
df_panel["reportable"] = [REPORTABLE_MAP.get((p, c), False) for p, c in zip(df_panel["periodo"], df_panel["des_categoria"])]
print(f"Panel de pares: {len(df_panel):,} filas | pares con algún peso topado (p{CAP_PCTL}): {_capadas:,} | "
      f"% de pares en celdas reportables: {100 * df_panel['reportable'].mean():.1f}%")


from mediciones.metodologia.inferencia import bootstrap_por_cliente as calcular_brechas_bootstrap, distribucion_brechas_mensual as calcular_distribucion_brechas   # método por cliente (ver inferencia.py)


# ── Núcleo: brecha mes × categoría (de aquí sale la banda IC para atribuir) ──
df_brechas = calcular_brechas_bootstrap(df_panel, ["des_categoria", "periodo_str"])
df_brechas["periodo"] = pd.to_datetime(df_brechas["periodo_str"])
df_brechas["reportable"] = [REPORTABLE_MAP.get((p, c), False) for p, c in zip(df_brechas["periodo"], df_brechas["des_categoria"])]
print(f"Brechas mes × categoría con IC: {len(df_brechas)} celdas (de {len(df_resumen)} con matching; "
      f"las que tienen < {N_MIN_PARES} pares no reportan IC)")
display(df_brechas.pivot(index="des_categoria", columns="periodo_str", values="brecha_pct").reindex(CATEGORIAS))
print("Amplitud del IC80 (pp):")
display(df_brechas.pivot(index="des_categoria", columns="periodo_str", values="amplitud_pp_80").reindex(CATEGORIAS))


# =============================================================================
# N3. TONELADAS ATRIBUIBLES IC80 + BASE COMPLETA (acción matcheado / sin pareja / nuevo / control espejo)
# =============================================================================
MES_BASE_OFFSET = pd.DateOffset(months=12)

# ── 1) Universo de acción: TODO cliente Perfecto elegible en t, con compra en t-12 o en t ──
_filas = []
for t_ in meses_calc:
    mask = (ELEG[t_] & FLAG[t_]).to_numpy()
    for cat_ in CATEGORIAS:
        M = SERIES[cat_]
        base_kg, act_kg = M[t_ - MES_BASE_OFFSET].to_numpy()[mask], M[t_].to_numpy()[mask]
        keep = (base_kg > 0) | (act_kg > 0)
        _filas.append(pd.DataFrame({"periodo": t_, "des_categoria": cat_,
                                    "cod_cliente": df_uni.index[mask][keep],
                                    "toneladas_ant": base_kg[keep] / 1000, "toneladas_actual": act_kg[keep] / 1000}))
accion = pd.concat(_filas, ignore_index=True)
accion["toneladas_variacion"] = accion["toneladas_actual"] - accion["toneladas_ant"]

# ── 2) Banda IC80 de la celda (mes × categoría) y atribuible ──
accion = accion.merge(df_brechas[["periodo", "des_categoria", "ic_pct_inf_80", "ic_pct_sup_80"]],
                      on=["periodo", "des_categoria"], how="left")
# [atribucion:inicio accion] estado, YoY, crecimiento acotado al IC80 y valor atribuible según la regla de 'nuevo' del programa (metodologia/atribucion.py)
_atr = atribuir(REGLA_NUEVO, accion["toneladas_ant"], accion["toneladas_actual"], accion["ic_pct_inf_80"], accion["ic_pct_sup_80"])
accion["flg_estado_yoy"] = _atr["flg_estado_yoy"]
accion["yoy_pct_raw"] = _atr["yoy_pct_raw"]
accion["crec_atrib_80_pct"] = _atr["crec_atrib_pct"]
accion["ton_atrib_80"] = _atr["ton_atrib"]
accion["flg_sin_banda_ic"] = _atr["flg_sin_banda_ic"]
accion["valor_atribuible"] = _atr["valor_atribuible"]
# [atribucion:fin accion]
sin_banda = accion["ic_pct_inf_80"].isna() | accion["ic_pct_sup_80"].isna()
es_ant = accion["flg_estado_yoy"] == "antiguo"
es_nue = ~es_ant
activo = accion["toneladas_actual"] > 0
accion["flg_bajo_esperado"] = np.where(es_ant & ~sin_banda, accion["yoy_pct_raw"] < accion["ic_pct_inf_80"], np.nan)
accion["flg_sobre_esperado"] = np.where(es_ant & ~sin_banda, accion["yoy_pct_raw"] > accion["ic_pct_sup_80"], np.nan)
accion["flg_tendencia"] = np.select([accion["toneladas_variacion"] > 0, accion["toneladas_variacion"] < 0], ["crece", "cae"], default="estable")

# ── 3) Rol y motivo (matcheado / sin pareja / nuevo) ──
_matched = set(zip(df_matches["periodo"], df_matches["des_categoria"], df_matches["cod_accion"]))
_excl = {(p, c, k): m for p, c, k, m in zip(df_ac_excluidos["periodo"], df_ac_excluidos["categoria"],
                                            df_ac_excluidos["cod_cliente_dex"], df_ac_excluidos["motivo"])}
def _rol_motivo(p, c, k, estado):
    if estado == "nuevo":
        return "accion_nuevo", None
    if (p, c, k) in _matched:
        return "accion_matcheado", None
    if (p, c, k) in _excl:
        return "accion_sin_pareja", _excl[(p, c, k)]
    info = MATCH_INFO.get((p, c))
    if info is None:
        return "accion_sin_pareja", "celda_sin_matching"
    if info.get("fijo"):
        return "accion_sin_pareja", "no_en_pares_fijos"
    if k in info["pend"]:
        return "accion_sin_pareja", "sin_control_en_cascada"
    if k in info["muestra"]:
        return "accion_sin_pareja", "otro"
    return "accion_sin_pareja", "fuera_de_muestra"
_rm = [_rol_motivo(p, c, k, e) for p, c, k, e in zip(accion["periodo"], accion["des_categoria"], accion["cod_cliente"], accion["flg_estado_yoy"])]
accion["rol"] = [r for r, _ in _rm]
accion["motivo_sin_pareja"] = [m for _, m in _rm]
accion = accion.join(df_uni[ATTR_COLS], on="cod_cliente")
accion["reportable"] = [REPORTABLE_MAP.get((p, c), False) for p, c in zip(accion["periodo"], accion["des_categoria"])]
accion["par_id"] = np.where(accion["rol"] == "accion_matcheado",
                            accion["periodo"].dt.strftime("%Y-%m") + "|" + accion["des_categoria"] + "|" + accion["cod_cliente"].astype(str), None)

# ── 4) Lado control: control espejo (el CT matcheado de cada par, con su base y actual reales) ──
ct_rows = []
for (t_, cat_), g_ in df_matches.groupby(["periodo", "des_categoria"]):
    P = POOLS[(t_, cat_)].loc[g_["cod_control"].to_numpy()]
    ct_rows.append(pd.DataFrame({
        "periodo": t_, "des_categoria": cat_, "cod_cliente": g_["cod_control"].to_numpy(),
        "toneladas_ant": P["base_kg"].to_numpy() / 1000, "toneladas_actual": P["actual_kg"].to_numpy() / 1000,
        "des_oficina": P["des_oficina"].to_numpy(), "nom_compania": P["nom_compania"].to_numpy(),
        "des_segmento_transaccional": P["des_segmento_transaccional"].to_numpy(),
        "par_id": [f"{t_.strftime('%Y-%m')}|{cat_}|{a}" for a in g_["cod_accion"]]}))
control = pd.concat(ct_rows, ignore_index=True)
control["rol"] = "control_espejo"
control["toneladas_variacion"] = control["toneladas_actual"] - control["toneladas_ant"]
control["yoy_pct_raw"] = np.where(control["toneladas_ant"] > 0, 100 * control["toneladas_variacion"] / control["toneladas_ant"], np.nan)
control["flg_estado_yoy"] = estado_yoy(REGLA_NUEVO, control["toneladas_ant"])       # [atribucion: control]
control["flg_tendencia"] = np.select([control["toneladas_variacion"] > 0, control["toneladas_variacion"] < 0], ["crece", "cae"], default="estable")
control["reportable"] = [REPORTABLE_MAP.get((p, c), False) for p, c in zip(control["periodo"], control["des_categoria"])]

# ── 5) Base completa ──
COLS_BASE_COMPLETA = [
    "cod_cliente", "des_categoria", "des_oficina", "nom_compania", "des_segmento_transaccional", "rol", "motivo_sin_pareja",
    "periodo", "toneladas_ant", "toneladas_actual", "toneladas_variacion", "yoy_pct_raw", "ic_pct_inf_80", "ic_pct_sup_80",
    "crec_atrib_80_pct", "ton_atrib_80", "valor_atribuible", "flg_estado_yoy", "flg_bajo_esperado", "flg_sobre_esperado",
    "flg_tendencia", "flg_sin_banda_ic", "reportable", "par_id"]
for _d in (accion, control):
    for c_ in COLS_BASE_COMPLETA:
        if c_ not in _d.columns:
            _d[c_] = np.nan
df_base_completa = (pd.concat([accion[COLS_BASE_COMPLETA], control[COLS_BASE_COMPLETA]], ignore_index=True)
                    .sort_values(["periodo", "des_categoria", "rol", "cod_cliente"]).reset_index(drop=True))
assert len(df_base_completa) < 1_000_000, "df_base_completa excede el límite de filas de una hoja de Excel."

print(f"df_base_completa: {len(df_base_completa):,} filas | {df_base_completa.cod_cliente.nunique():,} clientes únicos")
print("\nFilas y toneladas actuales por rol:")
display(df_base_completa.groupby("rol").agg(filas=("cod_cliente", "size"), toneladas_actual=("toneladas_actual", "sum"),
                                              valor_atribuible=("valor_atribuible", "sum")).round(1))
print("Acción sin pareja por motivo:")
display(df_base_completa[df_base_completa["rol"] == "accion_sin_pareja"].groupby("motivo_sin_pareja")
        .agg(filas=("cod_cliente", "size"), toneladas_actual=("toneladas_actual", "sum")).round(1))

_ac = df_base_completa[df_base_completa["rol"].str.startswith("accion")]
_res = (_ac.assign(ton_antiguos=_ac["valor_atribuible"].where(_ac["flg_estado_yoy"] == "antiguo", 0.0),
                   ton_nuevos=_ac["valor_atribuible"].where(_ac["flg_estado_yoy"] == "nuevo", 0.0))
        .groupby(_ac["periodo"].dt.strftime("%Y-%m"))[["ton_antiguos", "ton_nuevos"]].sum())
_res["total_atribuible"] = _res.sum(axis=1)
_res.loc["TOTAL"] = _res.sum()
print("TONELADAS ATRIBUIBLES por mes (todas las categorías): antiguos con IC80 + nuevos (todo):")
display(_res.round(1))
_sb = _ac[_ac["flg_sin_banda_ic"] == True]
print(f"Antiguos activos SIN banda IC (celda con < {N_MIN_PARES} pares o sin matching): {len(_sb):,} filas, "
      f"{_sb['toneladas_actual'].sum():,.1f} TN actuales -> valor_atribuible = NaN (no se imputa).")
