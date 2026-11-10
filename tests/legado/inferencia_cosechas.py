# COPIA LITERAL del código anterior (cosechas), líneas 1088-1219 de familias/cosechas/metodologia.py antes de extraerlo a metodologia/inferencia.py.
# No se edita: es la referencia de las pruebas diferenciales. Globales que necesita: np, pd, zlib, N_BOOT, SEED_BOOT, NIVELES_IC, N_MIN_PARES.
def calcular_brechas_bootstrap(panel_df, group_cols, n_boot=N_BOOT,
                                seed=SEED_BOOT, niveles_ic=NIVELES_IC,
                                n_min=30, n_min_dict=None):
    """
    Calcula la brecha DiD promedio y sus IC (bootstrap por par) para cada
    combinación de group_cols presente en panel_df (p.ej. cosecha×categoría×mes_rel,
    o cosecha×categoría×oficina×mes_rel).

    n_min: piso fijo (usado si no hay n_min_dict, o como fallback).
    n_min_dict: dict opcional {clave_sin_mes_rel: n_min_dinamico} — construido con
    n_minimo_requerido() a partir de la población real de cada celda (ver celda de
    'TAMAÑO DE MUESTRA MÍNIMO'). Se asume que 'mes_rel' es SIEMPRE el último elemento
    de group_cols, así que la clave de lookup es group_cols menos el último elemento.
    Con esto, una celda usa como piso su propio n mínimo estadístico, no uno genérico.
    """
    rng = np.random.default_rng(seed)
    res = []
    for keys, g in panel_df.groupby(group_cols):
        n = len(g)
        keys_t0 = keys if isinstance(keys, tuple) else (keys,)
        # Corte duro n<30, IGUAL que v6 (celda de distribución y del bootstrap de
        # diagnóstico). n_min_dict (n_minimo_requerido, 5%/95%) se guarda solo como
        # columna INFORMATIVA para las filas que sí sobreviven -- no reemplaza el
        # corte de v6, se agrega como vista adicional sobre lo que v6 ya calculaba.
        cumple_n_min = n >= 30
        if n < 30:
            continue
        if n_min_dict is not None:
            clave_pob = keys_t0[:-1] if len(keys_t0) > 1 else keys_t0
            n_min_efectivo = n_min_dict.get(clave_pob, n_min)
        else:
            n_min_efectivo = n_min
        did  = g.did_kg.to_numpy()
        base_ac = g.pre_accion.to_numpy()

        did_obs  = did.mean()
        base_obs = base_ac.mean()
        pct_obs  = 100 * did_obs / base_obs if base_obs > 0 else np.nan

        idx = rng.integers(0, n, size=(n_boot, n))       # remuestreo POR PAR
        did_b  = did[idx].mean(axis=1)
        base_b = base_ac[idx].mean(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_b = np.where(base_b > 0, 100 * did_b / base_b, np.nan)

        p_val = 2 * min((did_b <= 0).mean(), (did_b >= 0).mean())

        keys_t = keys if isinstance(keys, tuple) else (keys,)
        fila = dict(zip(group_cols, keys_t))
        fila.update({
            'periodo': g['periodo'].iloc[0],
            'n_pares': n,
            'base_accion_kg': round(base_obs, 2),
            'brecha_kg': round(did_obs, 3),
            'brecha_pct': round(pct_obs, 2),
            'p_valor': round(p_val, 4),
            'signo': 'positivo' if did_obs > 0 else 'negativo',
            'n_min_efectivo': n_min_efectivo, 'cumple_n_min': cumple_n_min})

        for nivel in niveles_ic:
            a = 1 - nivel / 100
            lo_kg, hi_kg = np.percentile(did_b, [100*a/2, 100*(1-a/2)])
            lo_p,  hi_p  = np.nanpercentile(pct_b, [100*a/2, 100*(1-a/2)])
            fila[f'ic_kg_inf_{nivel}']  = round(lo_kg, 3)
            fila[f'ic_kg_sup_{nivel}']  = round(hi_kg, 3)
            fila[f'ic_pct_inf_{nivel}'] = round(lo_p, 2)
            fila[f'ic_pct_sup_{nivel}'] = round(hi_p, 2)
            fila[f'amplitud_pp_{nivel}'] = round(hi_p - lo_p, 2)
            fila[f'significativo_{nivel}'] = bool(lo_kg > 0 or hi_kg < 0)

        # Compatibilidad hacia atrás: columnas "sin sufijo" ancladas al 95%
        fila['ic_kg_inf']     = fila['ic_kg_inf_95']
        fila['ic_kg_sup']     = fila['ic_kg_sup_95']
        fila['ic_pct_inf']    = fila['ic_pct_inf_95']
        fila['ic_pct_sup']    = fila['ic_pct_sup_95']
        fila['amplitud_pp']   = fila['amplitud_pp_95']
        fila['significativo'] = fila['significativo_95']

        res.append(fila)

    cols_orden = group_cols + ['n_pares'] if res else group_cols
    out = pd.DataFrame(res)
    if len(out):
        out = out.sort_values(group_cols).reset_index(drop=True)
    return out

def calcular_distribucion_brechas(panel_df, group_cols, n_min=30, n_min_dict=None):
    """Distribución (percentiles) de la brecha individual por par, para cada
    combinación de group_cols. Complementa a calcular_brechas_bootstrap (que da
    el promedio) mostrando la dispersión entre pares. Mismo criterio de n_min_dict.

    kg_p50 / pct_p50 (mediana) son el chequeo robusto contra outliers residuales:
    si brecha_pct (media, en calcular_brechas_bootstrap) y pct_p50 (aquí) divergen
    mucho para una misma celda, es señal de que todavía queda un outlier fuerte
    influyendo la media pese a la winsorización de PASO B."""
    dist_out = []
    for keys, g in panel_df.groupby(group_cols):
        keys_t0 = keys if isinstance(keys, tuple) else (keys,)
        # Corte duro n<30, IGUAL que calcular_brechas_bootstrap / v6.
        cumple_n_min = len(g) >= 30
        if len(g) < 30:
            continue
        if n_min_dict is not None:
            clave_pob = keys_t0[:-1] if len(keys_t0) > 1 else keys_t0
            n_min_efectivo = n_min_dict.get(clave_pob, n_min)
        else:
            n_min_efectivo = n_min
        d = g['did_kg']
        base_pos = g['pre_accion'].replace(0, np.nan)
        d_pct = 100 * d / base_pos

        keys_t = keys if isinstance(keys, tuple) else (keys,)
        fila = dict(zip(group_cols, keys_t))
        fila.update({
            'n_pares': len(g),
            'kg_min': round(d.min(), 2), 'kg_p10': round(d.quantile(.10), 2),
            'kg_p25': round(d.quantile(.25), 2), 'kg_p50': round(d.median(), 2),
            'kg_p75': round(d.quantile(.75), 2), 'kg_p90': round(d.quantile(.90), 2),
            'kg_max': round(d.max(), 2),
            'pct_pares_positivos': round(100 * (d > 0).mean(), 1),
            'pct_pares_negativos': round(100 * (d < 0).mean(), 1),
            'pct_p25': round(d_pct.quantile(.25), 1) if d_pct.notna().sum() > 5 else np.nan,
            'pct_p50': round(d_pct.median(), 1) if d_pct.notna().sum() > 5 else np.nan,
            'pct_p75': round(d_pct.quantile(.75), 1) if d_pct.notna().sum() > 5 else np.nan,
            'n_min_efectivo': n_min_efectivo, 'cumple_n_min': cumple_n_min,
        })
        dist_out.append(fila)
    out = pd.DataFrame(dist_out)
    if len(out):
        out = out.sort_values(group_cols).reset_index(drop=True)
    return out
