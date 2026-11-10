# COPIA LITERAL del código anterior (mensual), líneas 561-625 de familias/mensual/metodologia.py antes de extraerlo a metodologia/inferencia.py.
# No se edita: es la referencia de las pruebas diferenciales. Globales que necesita: np, pd, zlib, N_BOOT, SEED_BOOT, NIVELES_IC, N_MIN_PARES.
def calcular_brechas_bootstrap(panel_df, group_cols, cluster_col="cod_accion", n_boot=N_BOOT, seed=SEED_BOOT,
                               niveles_ic=NIVELES_IC, n_min=N_MIN_PARES, chunk=250):
    """Brecha DiD y sus IC por bootstrap de percentiles, remuestreando por cliente AC (cluster_col).
    Semilla determinista por grupo (la misma celda da el mismo IC en cualquier vista que la contenga)."""
    gcols = list(group_cols) if group_cols else ["_todo"]
    p = panel_df if group_cols else panel_df.assign(_todo="Global")
    res = []
    for keys, g in p.groupby(gcols, sort=True):
        n = len(g)
        if n < n_min:
            continue
        keys_t = keys if isinstance(keys, tuple) else (keys,)
        rng = np.random.default_rng([seed, zlib.crc32(repr(keys_t).encode())])
        cl = g.groupby(cluster_col, sort=True).agg(did=("did_kg", "sum"), base=("pre_accion", "sum"), npar=("did_kg", "size"))
        did_c, base_c, npar_c = cl["did"].to_numpy(), cl["base"].to_numpy(), cl["npar"].to_numpy()
        m = len(cl)
        did_obs, base_obs = did_c.sum(), base_c.sum()
        pct_obs = 100 * did_obs / base_obs if base_obs > 0 else np.nan
        d_b, b_b, n_b = [], [], []
        for s0 in range(0, n_boot, chunk):
            idx = rng.integers(0, m, size=(min(chunk, n_boot - s0), m))
            d_b.append(did_c[idx].sum(axis=1)); b_b.append(base_c[idx].sum(axis=1)); n_b.append(npar_c[idx].sum(axis=1))
        d_b, b_b, n_b = np.concatenate(d_b), np.concatenate(b_b), np.concatenate(n_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_b = np.where(b_b > 0, 100 * d_b / b_b, np.nan)
            kg_b = d_b / n_b
        fila = dict(zip(gcols, keys_t))
        fila.update({"n_pares": n, "n_clientes": m, "base_accion_kg": round(base_obs / n, 2),
                     "brecha_kg": round(did_obs / n, 3), "brecha_ton_pares": round(did_obs / 1000, 3),
                     "brecha_pct": round(pct_obs, 2),
                     "p_valor": round(2 * min((d_b <= 0).mean(), (d_b >= 0).mean()), 4),
                     "signo": "positivo" if did_obs > 0 else "negativo"})
        for nivel in niveles_ic:
            a = 1 - nivel / 100
            q = [100 * a / 2, 100 * (1 - a / 2)]
            lo_kg, hi_kg = np.percentile(kg_b, q)
            lo_t, hi_t = np.percentile(d_b / 1000, q)
            lo_p, hi_p = np.nanpercentile(pct_b, q)
            fila.update({f"ic_kg_inf_{nivel}": round(lo_kg, 3), f"ic_kg_sup_{nivel}": round(hi_kg, 3),
                         f"ic_ton_pares_inf_{nivel}": round(lo_t, 3), f"ic_ton_pares_sup_{nivel}": round(hi_t, 3),
                         f"ic_pct_inf_{nivel}": round(lo_p, 2), f"ic_pct_sup_{nivel}": round(hi_p, 2),
                         f"amplitud_pp_{nivel}": round(hi_p - lo_p, 2),
                         f"significativo_{nivel}": bool(lo_kg > 0 or hi_kg < 0)})
        res.append(fila)
    out = pd.DataFrame(res)
    return out.drop(columns=["_todo"], errors="ignore")

def calcular_distribucion_brechas(panel_df, group_cols, n_min=N_MIN_PARES):
    """Percentiles de la brecha individual por par (dispersión entre pares) por cada combinación de group_cols."""
    rows = []
    for keys, g in panel_df.groupby(list(group_cols), sort=True):
        if len(g) < n_min:
            continue
        keys_t = keys if isinstance(keys, tuple) else (keys,)
        d, b = g["did_kg"], g["pre_accion"]
        pct = (100 * d / b.where(b > 0)).dropna()
        fila = dict(zip(group_cols, keys_t))
        fila.update({"n_pares": len(g), "pct_pares_positivos": round(100 * (d > 0).mean(), 1),
                     "brecha_pct_media": round(100 * d.sum() / b.sum(), 2) if b.sum() > 0 else np.nan})
        for q in (10, 25, 50, 75, 90):
            fila[f"kg_p{q}"] = round(np.percentile(d, q), 3)
            fila[f"pct_p{q}"] = round(np.percentile(pct, q), 2) if len(pct) else np.nan
        rows.append(fila)
    return pd.DataFrame(rows)
