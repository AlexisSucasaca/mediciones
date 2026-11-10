"""Kernel del matching húngaro: una semilla (extraído de los notebooks; idéntico en la familia mensual y la de cosechas)."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
from sklearn.metrics import pairwise_distances


def _correr_semilla(seed, ac, ct, cols_m, estratos_usar, Lac_df, Lct_df, Fac_df, Fct_df,
                     MED_N, MED_F, NIVELES, MUESTREAR, FRAC_MUESTREO,
                     W_NIVEL, W_FORMA, CORR_UMBRAL, RATIO_MIN, RATIO_MAX,
                     max_reuso=1, ac_por_oficina=None,
                     ka_full_por_claves=None, kc_full_por_claves=None):
    # ac_por_oficina / ka_full_por_claves / kc_full_por_claves: PRECOMPUTADOS una
    # sola vez fuera del loop de semillas (no dependen del seed, solo el .sample()
    # de adentro si) -> evita recalcular lo mismo 50-150 veces por celda. Si vienen
    # en None (ej. alguien llama a esta funcion suelta, fuera del loop principal),
    # cae al calculo original -> mismo resultado, solo mas lento.
    # ---- MUESTREO (FIX 2 + FIX 3) --------------------------------
    if MUESTREAR:
        piezas = []
        fuente_oficina = ac_por_oficina if ac_por_oficina is not None else dict(list(ac.groupby('des_oficina')))
        for ofi, g_ac in fuente_oficina.items():
            tgt  = max(1, int(round(len(g_ac) * FRAC_MUESTREO)))
            frac = tgt / len(g_ac)

            g2 = g_ac.copy()
            g2['_e'] = g2[estratos_usar].astype(str).agg('|'.join, axis=1)
            tam_c = g2['_e'].value_counts()

            # asignación por mayor resto: proporcional exacta, sin min-1 ni trim
            raw   = tam_c * frac
            base  = np.floor(raw).astype(int)
            falta = tgt - int(base.sum())
            resto = (raw - base)
            if falta > 0:
                for e in resto.sort_values(ascending=False).index[:falta]:
                    base[e] += 1
            elif falta < 0:
                cand_baja = resto.sort_values().index
                quitados = 0
                for e in cand_baja:
                    if quitados >= -falta:
                        break
                    if base[e] > 0:
                        base[e] -= 1; quitados += 1

            sub = [gg.sample(n=int(base[e]), random_state=seed)
                   for e, gg in g2.groupby('_e') if base[e] > 0]
            if not sub:
                continue
            sub = pd.concat(sub).drop(columns='_e')
            sub['_peso_pob'] = len(g_ac) / len(sub)
            piezas.append(sub)
        ac_s = pd.concat(piezas) if piezas else ac.iloc[0:0]
    else:
        ac_s = ac.copy(); ac_s['_peso_pob'] = 1.0
    if len(ac_s) == 0:
        return None

    # ---- MATCHING EN CASCADA, con RONDAS para permitir reuso 1:N -------------
    # BUG previo: el contador de reuso funcionaba, pero un solo pase de la cascada
    # (3 niveles, el último con caliper laxo) agota TODO el control disponible en
    # una sola asignación óptima (linear_sum_assignment nunca repite DENTRO de una
    # misma llamada) -> nunca había una "segunda vez" en la que reusar un control.
    # Ahora cada RONDA es un pase completo de la cascada de estratos sobre la
    # acción aún sin pareja (pend), usando solo controles que no llegaron a su
    # tope de reuso. Con max_reuso=1 corre una sola ronda -> idéntico al 1:1 original.
    pend, pares_s = list(ac_s.index), []
    uso_control = {}
    ct_ids_all = list(ct.index)

    for ronda in range(max_reuso):
        if not pend:
            break
        disp = {cid for cid in ct_ids_all if uso_control.get(cid, 0) < max_reuso}
        if not disp:
            break
        for nivel, claves, cal in NIVELES:
            if not pend or not disp:
                break
            a_sub, c_sub = ac_s.loc[pend], ct.loc[sorted(disp)]
            if ka_full_por_claves is not None and kc_full_por_claves is not None:
                # subset por indice sobre la clave YA calculada -> nada de astype+join
                # de nuevo (eso es lo que se repetia identico 50-150 veces por celda,
                # sobre todo del lado control, que ni siquiera cambia con el seed)
                ka = ka_full_por_claves[tuple(claves)].loc[a_sub.index]
                kc = kc_full_por_claves[tuple(claves)].loc[c_sub.index]
            else:
                ka = a_sub[claves].astype(str).agg('|'.join, axis=1)
                kc = c_sub[claves].astype(str).agg('|'.join, axis=1)
            for blk in ka.unique():
                ia, ic = ka.index[ka == blk], kc.index[kc == blk]
                if len(ic) == 0:
                    continue
                Dn = pairwise_distances(Lac_df.loc[ia].to_numpy(),
                                        Lct_df.loc[ic].to_numpy()) / MED_N
                Df = pairwise_distances(Fac_df.loc[ia].to_numpy(),
                                        Fct_df.loc[ic].to_numpy()) / MED_F
                D = W_NIVEL * Dn + W_FORMA * Df
                for i, j in zip(*linear_sum_assignment(D)):
                    if float(D[i, j]) > cal:
                        continue
                    a_id, c_id = ia[i], ic[j]
                    pares_s.append({'nivel_match': nivel, 'ronda_reuso': ronda + 1,
                                    'cod_accion': a_id, 'cod_control': c_id,
                                    'dist_total': round(float(D[i, j]), 4),
                                    'dist_nivel': round(float(Dn[i, j]), 4),
                                    'dist_forma': round(float(Df[i, j]), 4),
                                    'peso_pob': ac_s.at[a_id, '_peso_pob']})
                    pend.remove(a_id)
                    uso_control[c_id] = uso_control.get(c_id, 0) + 1
                    disp.discard(c_id)

    if not pares_s:
        return None
    p_df = pd.DataFrame(pares_s)

    t_ac = ac.loc[p_df.cod_accion, cols_m].mean().to_numpy()
    t_ct = ct.loc[p_df.cod_control, cols_m].mean().to_numpy()
    d_ac, d_ct = np.diff(t_ac), np.diff(t_ct)
    corr  = (np.corrcoef(d_ac, d_ct)[0, 1]
             if d_ac.std() > 0 and d_ct.std() > 0 else np.nan)
    ratio = d_ac.std() / d_ct.std() if d_ct.std() > 0 else np.inf
    brecha = t_ac - t_ct
    pend_rel = (linregress(np.arange(len(brecha)), brecha).slope / t_ac.mean()
                if t_ac.mean() > 0 else np.inf)
    ok = (pd.notna(corr) and corr >= CORR_UMBRAL and RATIO_MIN <= ratio <= RATIO_MAX)

    return {
        'seed': seed, 'n_muestra': len(ac_s), 'pares': len(p_df),
        'cobertura_pct': round(100 * len(p_df) / len(ac_s), 1),
        'corr': round(corr, 3) if pd.notna(corr) else np.nan,
        'ratio_vol': round(ratio, 3), 'pend_rel': round(pend_rel, 5),
        'dist_med': round(p_df.dist_total.median(), 4), 'valida': ok,
        'controles_unicos': p_df.cod_control.nunique(),
        'max_reuso_observado': int(p_df.cod_control.value_counts().max()),
        '_p': p_df, '_ac': ac_s, '_pend': pend, '_tac': t_ac, '_tct': t_ct}
