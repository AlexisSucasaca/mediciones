"""
Validación de una corrida YA HECHA (lee runs/<corrida>/<programa>_{ton,brecha,espejos}.parquet). No consulta BigQuery ni recalcula el motor.

    from mediciones.validacion import validar_corrida
    chequeos = validar_corrida(RAIZ, referencias={'surtido_motor': 'ruta/a/referencia_ton.parquet'})

Devuelve la tabla de chequeos (programa, grupo, chequeo, resultado, detalle) y escribe runs/<corrida>/validacion_AAAAMM.xlsx. Resultados:
  ✔ cumple · ✘ NO cumple (revisar antes de cargar) · ⚠ aviso (puede ser legítimo, mirar el detalle) · ℹ informativo.
Todas las identidades que se exigen (✘) se comprobaron antes sobre la salida real de surtido de septiembre (DiD, reglas por estado, cap por IC): un ✘ es un problema real.
Qué mide cada chequeo, y cómo leer la comparación con una referencia: docs/decisiones.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import normalizacion as nz
from .paridad import paridad_con_legacy
from .registro import cargar_registro

OK, MAL, AVISO, INFO = '✔', '✘', '⚠', 'ℹ'
ESTADOS_CLIENTE_CREADO = {'normal', 'no_existia_periodo_pasado', 'reactivado', 'sin_compra_ambos', 'cien_pct_atribuible_sin_control'}   # el último es Inka Chips (diferencial YoY, sin matching)
ESTADOS_ESTANDAR = {'antiguo', 'nuevo'}
TOL = 1e-6


class _Registro:
    def __init__(self, programa):
        self.programa, self.filas = programa, []

    def __call__(self, grupo, chequeo, resultado, detalle=''):
        self.filas.append(dict(programa=self.programa, grupo=grupo, chequeo=chequeo, resultado=resultado, detalle=detalle))


def _frac(mask):
    return float(np.mean(mask)) if len(mask) else 1.0


def _carpeta_corrida(raiz, corrida):
    runs = Path(raiz) / 'runs'
    if corrida:
        d = runs / corrida
        if not d.exists():
            hay = sorted(x.name for x in runs.glob('*') if x.is_dir() and any(x.glob('*_ton.parquet'))) if runs.exists() else []
            pista = ' (la línea base se crea al preparar lo reportado; si su paso falló, corrige los ✘ de ese paso)' if corrida.startswith('reportado_') else ''
            raise FileNotFoundError(f'No existe {d}{pista}. Corridas con parquet en {runs}: {hay or "ninguna"}')
        return d
    cand = sorted((p for p in runs.glob('*') if p.is_dir() and not p.name.startswith('reportado_') and any(p.glob('*_ton.parquet'))), key=lambda p: p.stat().st_mtime)   # la línea base reportada no es una corrida
    if not cand:
        raise FileNotFoundError(f'No hay corridas con bases en {runs}')
    return cand[-1]


def _leer_referencia(ref):
    ruta, hoja = (ref if isinstance(ref, (tuple, list)) else (ref, None))
    ruta = Path(ruta)
    if ruta.suffix == '.parquet':
        df = pd.read_parquet(ruta)
    elif ruta.suffix == '.csv':
        df = pd.read_csv(ruta, dtype={'cod_cliente': str})
    else:
        df = pd.read_excel(ruta, sheet_name=hoja or 0, dtype={'cod_cliente': str, 'cod_cliente_dex': str})
    if 'rol' in df.columns:                                       # hoja 'Base ton + UB atribuibles' de CPB/CPM: trae también los controles espejo, que no tienen atribuible
        df = df[df['rol'].astype(str).str.startswith('accion')]
    if 'valor_atribuible' not in df.columns and 'ton_atribuible' in df.columns:
        df = df.rename(columns={'ton_atribuible': 'valor_atribuible'})
    return df


# ── chequeos de una corrida de un programa ──────────────────────────────────────────────────────────────
def _integridad(r, ton, bre, esp, mes):
    d = ton.duplicated(['cod_cliente', 'des_categoria', 'periodo']).sum()
    r('Integridad', 'Llave única en toneladas (cliente × categoría × mes)', OK if d == 0 else MAL, f'{d:,} filas duplicadas' if d else f'{len(ton):,} filas')
    d = bre.duplicated(['par_id', 'periodo']).sum()
    r('Integridad', 'Llave única en brechas (par × mes)', OK if d == 0 else MAL, f'{d:,} filas duplicadas' if d else f'{len(bre):,} filas, {bre.par_id.nunique():,} pares')
    if esp is not None and len(esp):
        d = esp.duplicated(['cod_cliente_accion', 'des_categoria', 'periodo']).sum()
        r('Integridad', 'Llave única en espejos (cliente × categoría × mes)', OK if d == 0 else MAL, f'{d:,} duplicadas' if d else f'{len(esp):,} filas')
    nul = {c: int(ton[c].isna().sum()) for c in ('cod_cliente', 'des_categoria', 'periodo', 'toneladas_actual') if c in ton}
    r('Integridad', 'Sin nulos en las columnas clave de toneladas', OK if not any(nul.values()) else MAL, '' if not any(nul.values()) else str({k: v for k, v in nul.items() if v}))
    pmax, pmin = ton['periodo'].max(), ton['periodo'].min()
    r('Integridad', 'El último periodo es el mes de cierre', OK if pmax == mes else MAL, f'rango {pmin:%Y-%m} a {pmax:%Y-%m}; cierre {mes:%Y-%m}')
    neg = int((ton['toneladas_actual'] < 0).sum())
    r('Integridad', 'Toneladas actuales sin negativos', OK if neg == 0 else AVISO, f'{neg:,} filas negativas' if neg else '')


def _coherencia(r, ton, bre):
    nr = bre[~bre['flg_estimado_retroactivo'].fillna(False).astype(bool)].dropna(subset=['post_control', 'pre_control', 'post_accion', 'pre_accion', 'did_kg'])
    if len(nr):
        f = _frac(((nr.post_accion - nr.pre_accion) - (nr.post_control - nr.pre_control) - nr.did_kg).abs() < TOL)
        r('Coherencia', 'DiD = (post−pre acción) − (post−pre control)', OK if f >= 0.9999 else MAL, f'{f:.4%} de {len(nr):,} filas')
    g = bre.groupby(['cosecha_grupo', 'des_categoria', 'mes_rel'])['brecha_pct_celda'].nunique(dropna=True)
    malos = int((g > 1).sum())
    r('Coherencia', 'La brecha % es una sola por celda (cosecha × categoría × mes)', OK if malos == 0 else AVISO, f'{malos} celdas con más de un valor' if malos else f'{len(g)} celdas')
    if len(nr):
        k = ['cod_cliente', 'des_categoria', 'periodo']
        m = nr.assign(cod_cliente=nr['cod_cliente_accion_dex']).merge(ton[k + ['toneladas_ant', 'toneladas_actual']], on=k, how='left')
        sin = int(m['toneladas_actual'].isna().sum())
        mm = m.dropna(subset=['toneladas_actual'])
        if len(mm):
            f1, f2 = _frac((mm.pre_accion - mm.toneladas_ant * 1000).abs() < TOL), _frac((mm.post_accion - mm.toneladas_actual * 1000).abs() < TOL)
            r('Coherencia', 'La brecha usa las mismas toneladas que la base (pre/post acción)', OK if min(f1, f2) >= 0.99 else AVISO,
              f'pre {f1:.2%} · post {f2:.2%} coinciden (lo que difiere es la winsorización por celda); {sin:,} par-mes sin fila de toneladas')


def _atribucion(r, ton, regla_estado):
    est = set(ton['flg_estado_yoy'].dropna().unique())
    sub = ton[ton['des_categoria'].isin(ton.loc[ton['flg_estado_yoy'].notna(), 'des_categoria'].unique())]
    act, val = sub['toneladas_actual'], sub['valor_atribuible']
    est_s = sub['flg_estado_yoy']
    if est and est <= ESTADOS_CLIENTE_CREADO:
        for estado, esperado, texto in (('no_existia_periodo_pasado', act, 'atribuible = todo su volumen'), ('reactivado', 0.0, 'atribuible = 0'), ('sin_compra_ambos', 0.0, 'atribuible = 0')):
            m = est_s == estado
            if m.any():
                f = _frac((val[m] - (esperado[m] if isinstance(esperado, pd.Series) else esperado)).abs() < TOL)
                r('Atribución', f'Estado {estado}: {texto}', OK if f == 1.0 else MAL, f'{f:.4%} de {int(m.sum()):,} filas')
        n = sub[(est_s == 'normal') & (act > 0) & sub['ic_pct_inf_80'].notna() & sub['ic_pct_sup_80'].notna() & (sub['toneladas_ant'] > 0)]
        if len(n):
            c = 100 * n['valor_atribuible'] / n['toneladas_ant']
            f = _frac((c >= n['ic_pct_inf_80'] - 1e-6) & (c <= n['ic_pct_sup_80'] + 1e-6))
            r('Atribución', 'Estado normal: el crecimiento atribuido está dentro de la banda IC80', OK if f == 1.0 else MAL, f'{f:.4%} de {len(n):,} filas')
        m = (est_s == 'normal') & ~(act > 0)
        if m.any():
            r('Atribución', 'Estado normal sin compra actual: atribuible = 0', OK if _frac(val[m].abs() < TOL) == 1.0 else MAL, f'{int(m.sum()):,} filas')
    elif est and est <= ESTADOS_ESTANDAR:
        m = est_s == 'nuevo'
        if m.any():
            r('Atribución', 'Estado nuevo: atribuible = todo su volumen', OK if _frac((val[m] - act[m]).abs() < TOL) == 1.0 else MAL, f'{int(m.sum()):,} filas')
    else:
        r('Atribución', 'Reglas por estado', INFO, f'estados {sorted(est)} sin regla de validación propia (método propio del programa; regla de normalización: {regla_estado})')
    tot = ton.groupby('flg_estado_yoy', dropna=False)['valor_atribuible'].sum().round(2).to_dict()
    r('Atribución', 'Toneladas atribuibles por estado', INFO, str(tot))


def _totales(ton):
    t = (ton.groupby(['des_categoria', ton['periodo'].dt.strftime('%Y-%m')])
         .agg(filas=('cod_cliente', 'size'), clientes=('cod_cliente', 'nunique'), ton_ant=('toneladas_ant', 'sum'), ton_actual=('toneladas_actual', 'sum'),
              ton_atribuible=('valor_atribuible', 'sum')).round(3).reset_index().rename(columns={'periodo': 'mes'}))
    return t


def _contra_referencia(r, p, ton, ref, regla_estado):
    """Comparación fila a fila con una salida de referencia + diagnóstico de la causa si no coincide."""
    ref = ref.copy()
    ref['cod_cliente'] = ref['cod_cliente'].astype(str).str.zfill(10)
    ref['periodo'] = pd.to_datetime(ref['periodo'])
    k0 = ['cod_cliente', 'des_categoria', 'periodo']
    if ref.duplicated(k0).any():              # Inka Chips: varias filas por cliente-mes en el notebook; el motor las suma (toneladas y diferencial YoY son aditivos)
        adit = [c for c in ('toneladas_ant', 'toneladas_actual', 'valor_atribuible') if c in ref.columns]
        ref = ref.groupby(k0, as_index=False).agg({**{c: 'sum' for c in adit}, **{c: 'first' for c in ref.columns if c not in k0 + adit}})
    res = paridad_con_legacy(ton.assign(periodo=ton['periodo'].dt.date), ref, regla_estado=regla_estado)
    ok = res['coincide']
    r('Referencia', 'Paridad fila a fila (toneladas actuales y atribuible)', OK if ok else MAL,
      f"motor {res['suma_valor_atribuible_motor']:.3f} t · referencia {res['suma_valor_atribuible_legacy']:.3f} t · solo en motor {res['solo_en_motor']:,} · solo en referencia {res['solo_en_legacy']:,} · "
      f"filas con diferencia de atribuible {res['filas_con_dif_valor_atribuible']:,} · máx. dif. toneladas {res['max_dif_toneladas_actual']:.2e}")
    k = ['cod_cliente', 'des_categoria', 'periodo']
    cols = [c for c in ('toneladas_ant', 'toneladas_actual', 'ic_pct_inf_80', 'ic_pct_sup_80', 'valor_atribuible', 'ub_base_soles', 'ub_atribuible_soles') if c in ton and c in ref]
    m = ton[k + cols].merge(ref[k + cols], on=k, suffixes=('', '_ref'))
    dif = {c: int((~np.isclose(m[c].fillna(-9e9), m[c + '_ref'].fillna(-9e9), rtol=0, atol=1e-9)).sum()) for c in cols}
    r('Referencia', 'Filas comunes que difieren, por columna', INFO, f'{len(m):,} filas comunes: ' + ' · '.join(f'{c}: {n:,}' for c, n in dif.items()))
    entradas = dif.get('toneladas_ant', 0) + dif.get('toneladas_actual', 0)
    bandas = dif.get('ic_pct_inf_80', 0) + dif.get('ic_pct_sup_80', 0)
    if ok:
        causa = 'sin diferencias'
    elif res['solo_en_motor'] or res['solo_en_legacy']:
        causa = 'la población cambió (clientes o meses distintos): revisar el universo y el estado del matching'
    elif entradas:
        causa = 'cambiaron las TONELADAS de entrada (los datos de BigQuery se actualizaron desde que se generó la referencia); el atribuible cambia por arrastre'
    elif bandas:
        causa = 'las toneladas de entrada son iguales pero cambió la BANDA IC (bootstrap/matching): mirar semillas, datos del panel y el estado congelado'
    elif dif.get('valor_atribuible', 0):
        causa = 'entradas y bandas iguales pero distinto atribuible: diferencia en la REGLA de atribución'
    else:
        causa = 'indeterminada'
    r('Referencia', 'Causa más probable de la diferencia', INFO if ok else AVISO, causa)
    if 'ub_atribuible_soles' in dif:
        sin_ub = ton['ub_atribuible_soles'].isna().all()
        n_ub = dif['ub_atribuible_soles']
        total = f"UB atribuible: motor {ton['ub_atribuible_soles'].sum():,.0f} · referencia {ref['ub_atribuible_soles'].sum():,.0f} soles; filas comunes que difieren: {n_ub:,}"
        if sin_ub:
            r('Referencia', 'UB atribuible (motor vs referencia)', AVISO, 'el motor no calculó UB en esta corrida (¿calcular_ub=False?). ' + total)
        else:
            base = f" · de UB base: {dif['ub_base_soles']:,}" if 'ub_base_soles' in dif else ''
            r('Referencia', 'UB atribuible (motor vs referencia)', OK if n_ub == 0 else AVISO, total + base)
    a = ton.assign(mes=ton['periodo'].dt.strftime('%Y-%m')).groupby(['des_categoria', 'mes']).agg(ton_actual=('toneladas_actual', 'sum'), valor=('valor_atribuible', 'sum'), ub=('ub_atribuible_soles', 'sum'))
    ref_ub = ref['ub_atribuible_soles'] if 'ub_atribuible_soles' in ref else np.nan
    b = ref.assign(mes=ref['periodo'].dt.strftime('%Y-%m'), ub_atribuible_soles=ref_ub).groupby(['des_categoria', 'mes']).agg(ton_actual_ref=('toneladas_actual', 'sum'), valor_ref=('valor_atribuible', 'sum'), ub_ref=('ub_atribuible_soles', 'sum'))
    comp = a.join(b, how='outer').fillna(0).round(3)
    comp['dif_ton_actual'], comp['dif_valor'], comp['dif_ub'] = (comp.ton_actual - comp.ton_actual_ref).round(3), (comp.valor - comp.valor_ref).round(3), (comp.ub - comp.ub_ref).round(3)
    top = (m.assign(dif_valor=m['valor_atribuible'] - m['valor_atribuible_ref']).loc[lambda d: d.dif_valor.abs() > 1e-9]
           .sort_values('dif_valor', key=lambda s: s.abs(), ascending=False).head(50))
    return comp.reset_index(), top


def validar_corrida(raiz, corrida=None, programas=None, referencias=None, config=None, escribir=True, verbose=True):
    """
    Valida la última corrida (o `corrida`) de `runs/`. `referencias`: {programa: ruta | (ruta, hoja)} con la salida de referencia (parquet, csv o xlsx; hoja de toneladas).
    Devuelve el DataFrame de chequeos. No lanza excepciones por un ✘: la decisión de cargar o no es tuya (los ✘ deben revisarse antes de MODO='cargar').
    """
    raiz = Path(raiz)
    carpeta = _carpeta_corrida(raiz, corrida)
    aaaamm = re.search(r'(\d{4})(\d{2})', carpeta.name)                      # '202609_2026…' (corrida del motor) o 'reportado_202609' (línea base)
    mes = pd.Timestamp(f'{aaaamm.group(1)}-{aaaamm.group(2)}-01')
    familias, registro = cargar_registro(Path(config) if config else raiz / 'config' / 'programas.yaml')
    progs = programas or sorted({p.name.rsplit('_ton.parquet', 1)[0] for p in carpeta.glob('*_ton.parquet')})
    filas, hojas = [], {}
    for p in progs:
        r = _Registro(p)
        try:
            regla = familias[registro[p].familia].get('estado_yoy', 'estandar') if p in registro and registro[p].familia in familias else 'estandar'
            ton_raw, bre_raw = pd.read_parquet(carpeta / f'{p}_ton.parquet'), pd.read_parquet(carpeta / f'{p}_brecha.parquet')
            esp_raw = pd.read_parquet(carpeta / f'{p}_espejos.parquet') if (carpeta / f'{p}_espejos.parquet').exists() else None
            rep = mes.date()
            ton = nz.preparar_base_ton(ton_raw, p, rep, 'validacion', regla_estado=regla)
            bre = nz.preparar_base_brechas(bre_raw, p, rep, 'validacion')
            esp = nz.preparar_base_espejos(esp_raw, p, rep, 'validacion') if esp_raw is not None else None
            errores, _ = nz.validar(ton, bre, rep, esp=esp)                  # espera los periodos como fecha (así salen de la normalización)
            ton, bre = ton.assign(periodo=pd.to_datetime(ton['periodo'])), bre.assign(periodo=pd.to_datetime(bre['periodo']))
            esp = esp.assign(periodo=pd.to_datetime(esp['periodo'])) if esp is not None else None
            r('Integridad', 'Validación estándar del pipeline (llaves, espejos, kg, cobertura)', OK if not errores else MAL, '; '.join(errores))
            _integridad(r, ton, bre, esp, mes)
            _coherencia(r, ton, bre)
            _atribucion(r, ton, regla)
            hojas[f'{p[:20]} totales'] = _totales(ton)
            if referencias and p in referencias:
                comp, top = _contra_referencia(r, p, ton, _leer_referencia(referencias[p]), regla)
                hojas[f'{p[:20]} vs ref'], hojas[f'{p[:20]} top difs'] = comp, top
        except Exception as e:                                         # un programa que no se puede validar no tumba a los demás
            r('Integridad', 'No se pudo validar el programa', MAL, f'{type(e).__name__}: {e}')
        filas += r.filas
    out = pd.DataFrame(filas)
    if escribir and len(out):
        ruta = carpeta / f'validacion_{mes:%Y%m}.xlsx'
        with pd.ExcelWriter(ruta) as w:
            out.to_excel(w, sheet_name='Chequeos', index=False)
            for h, d in hojas.items():
                d.to_excel(w, sheet_name=h[:31], index=False)
        if verbose:
            print(f'[validación] {ruta}')
    if verbose and len(out):
        res = out.groupby(['programa', 'resultado']).size().unstack(fill_value=0)
        print(res.to_string())
        for _, f in out[out.resultado.isin([MAL, AVISO])].iterrows():
            print(f"  {f.resultado} [{f.programa}] {f.chequeo}: {f.detalle}")
    return out
