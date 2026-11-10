"""
Los pares acción-espejo ya publicados en BigQuery (base_espejos) son la fuente de verdad. Antes de ejecutar un motor se comprueba que su estado local
(archivo de pares congelados) coincida con lo publicado; si falta y es restaurable, se reconstruye desde BigQuery. Si difiere, volver a emparejar cambiaría
meses ya publicados, así que la carga se bloquea.

Tipos de estado local:
  mensual   <programa>_pares_fijos.csv (periodo, categoria, cod_accion, cod_control, ...): se puede verificar y restaurar.
  cosechas  outputs/<programa>/df_matches.csv (cosecha, des_categoria, cod_accion, cod_control, ...): se puede verificar; no restaurar (el estado completo incluye más archivos).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _id10(s):
    return s.astype(str).str.zfill(10)


def publicados_normalizados(esp):
    """DataFrame de pares publicados con las columnas (periodo, cosecha, categoria, accion, control, nivel_match)."""
    if esp is None or len(esp) == 0:
        return pd.DataFrame(columns=['periodo', 'cosecha', 'categoria', 'accion', 'control', 'nivel_match'])
    d = pd.DataFrame({'periodo': pd.to_datetime(esp['periodo']), 'cosecha': pd.to_datetime(esp['cosecha_grupo']) if 'cosecha_grupo' in esp else pd.NaT,
                      'categoria': esp['des_categoria'], 'accion': _id10(esp['cod_cliente_accion']), 'control': _id10(esp['cod_cliente_espejo']),
                      'nivel_match': esp['nivel_match'] if 'nivel_match' in esp else pd.NA})
    return d.dropna(subset=['control']).drop_duplicates(['periodo', 'categoria', 'accion'])


def _leer_local(tipo, ruta):
    d = pd.read_csv(ruta, dtype=str)
    if tipo == 'mensual':
        return pd.DataFrame({'periodo': pd.to_datetime(d['periodo']), 'categoria': d['categoria'], 'accion': _id10(d['cod_accion']), 'control': _id10(d['cod_control'])})
    return pd.DataFrame({'cosecha': pd.to_datetime(d['cosecha']), 'categoria': d['des_categoria'], 'accion': _id10(d['cod_accion']), 'control': _id10(d['cod_control'])})


def excluir_cosechas(publicados, cosechas):
    """Quita de los pares publicados los de esas cosechas (las provisionales, como el G04: cambian de pares a propósito y no se verifican)."""
    if publicados is None or len(publicados) == 0 or not cosechas:
        return publicados
    coh = set(pd.to_datetime(list(cosechas)).to_period('M').to_timestamp())
    c = pd.to_datetime(publicados['cosecha'], errors='coerce').dt.to_period('M').dt.to_timestamp()
    return publicados[~c.isin(coh)]


def verificar(tipo, ruta, publicados):
    """Compara el estado local con los pares publicados. Estados: sin_publicado, ok, restaurable, faltante_no_restaurable, difiere, incompleto."""
    pub = publicados
    res = {'tipo': tipo, 'ruta': str(ruta), 'estado': None, 'n_publicados': int(len(pub)), 'n_locales': None, 'n_difieren': 0, 'n_faltan_local': 0}
    if len(pub) == 0:
        res['estado'] = 'sin_publicado'
        return res
    if not Path(ruta).exists():
        res['estado'] = 'restaurable' if tipo == 'mensual' else 'faltante_no_restaurable'
        return res
    loc = _leer_local(tipo, ruta)
    res['n_locales'] = int(len(loc))
    if tipo == 'mensual':
        k_loc, k_pub = ['periodo', 'categoria', 'accion'], ['periodo', 'categoria', 'accion']
        p = pub[k_pub + ['control']]
    else:
        pc = pub.dropna(subset=['cosecha'])
        k_loc, k_pub = ['cosecha', 'categoria', 'accion'], ['cosecha', 'categoria', 'accion']
        p = pc[k_pub + ['control']].drop_duplicates(k_pub)
    m = p.merge(loc.drop_duplicates(k_loc), on=k_pub, how='left', suffixes=('_pub', '_loc'))
    res['n_faltan_local'] = int(m['control_loc'].isna().sum())
    res['n_difieren'] = int((m['control_loc'].notna() & (m['control_loc'] != m['control_pub'])).sum())
    res['estado'] = 'difiere' if res['n_difieren'] else ('incompleto' if res['n_faltan_local'] else 'ok')
    return res


def restaurar_mensual(publicados, ruta):
    """Reconstruye el archivo de pares fijos desde lo publicado (las columnas de diagnóstico del emparejamiento quedan vacías)."""
    d = publicados.copy()
    out = pd.DataFrame({'periodo': d['periodo'].dt.strftime('%Y-%m-%d'), 'categoria': d['categoria'], 'cod_accion': d['accion'], 'cod_control': d['control'],
                        'seed': pd.NA, 'nivel_match': d['nivel_match'], 'ronda_reuso': pd.NA, 'dist_total': pd.NA, 'dist_nivel': pd.NA,
                        'dist_forma': pd.NA, 'peso_pob': pd.NA}).sort_values(['periodo', 'categoria', 'cod_accion'])
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ruta, index=False)
    return len(out)
