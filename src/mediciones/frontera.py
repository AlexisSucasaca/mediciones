"""
Frontera de meses cerrados: lo reportado no cambia; cada cierre solo publica meses nuevos.

Regla (una sola, la misma en el runner, en la vista de BigQuery y en las pruebas):

  * Cada cierre (`periodo_reporte` P) publica una "rebanada". Antes de publicar se calcula la FRONTERA F:
        F = último `periodo` publicado por cierres ANTERIORES a P (la línea base sembrada cuenta como un cierre),
            recortada al mes anterior a `provisional_desde` (meses que se pueden revisar, p. ej. Impulso oct-dic mientras se define el G4)
            y al mes anterior a `reabrir_desde` (reapertura explícita de meses cerrados, con motivo).
  * Se publica todo lo calculado con `periodo > F`. Lo de `periodo <= F` se recalcula solo para medir la deriva (validación): nunca se publica.
  * Lo VIGENTE de cada (`periodo`, cosecha) es la última versión de la rebanada MÁS RECIENTE que lo publicó (la cosecha solo existe en surtido/Champion).
    Sin meses provisionales, cada periodo lo publica un solo cierre (el primero en que fue nuevo) y queda fijo para siempre.
    Un mes provisional o reabierto se reemplaza COMPLETO por la rebanada más reciente (también desaparecen las filas que ya no salen).
"""
import pandas as pd


def _mes(x):
    return None if x is None or (not isinstance(x, str) and pd.isna(x)) else pd.Timestamp(x).to_period('M').to_timestamp()


def entradas_publicadas(publicado):
    """
    Mes de entrada de cada cliente según lo YA publicado: el panel solo tiene filas desde la entrada del cliente, así que su primer `periodo`
    publicado es su entrada. Devuelve {cod_cliente (10 dígitos): Timestamp}. Vacío si no hay nada publicado.
    Lo usa Champion para que un cliente que solo viene en la lista de clientes (sin hoja mensual) no cambie de entrada de un cierre a otro.
    """
    if publicado is None or len(publicado) == 0 or 'cod_cliente' not in publicado.columns:
        return {}
    d = publicado[['cod_cliente', 'periodo']].dropna()
    d = d.assign(cod_cliente=d['cod_cliente'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(10),
                 periodo=pd.to_datetime(d['periodo']).dt.to_period('M').dt.to_timestamp())
    return d.groupby('cod_cliente')['periodo'].min().to_dict()


def provisional_activo(periodo_reporte, provisional_desde=None, provisional_hasta=None):
    """Lo provisional rige mientras el cierre esté entre `provisional_desde` y `provisional_hasta` (este incluido; None = sin tope)."""
    if provisional_desde is None:
        return False
    p = _mes(periodo_reporte)
    return p >= _mes(provisional_desde) and (provisional_hasta is None or p <= _mes(provisional_hasta))


def calcular_frontera(publicado, periodo_reporte, provisional_desde=None, reabrir_desde=None, provisional_cosechas=None, provisional_hasta=None):
    """
    Frontera F (Timestamp de mes) o None si no hay nada publicado antes de este cierre (primera carga).
    `provisional_desde` SIN `provisional_cosechas` recorta la frontera (todo el programa es provisional desde ese mes). CON cosechas, la frontera
    no se recorta: lo provisional se republica solo para esas cosechas (ver `separar`).
    """
    p = _mes(periodo_reporte)
    f = None
    if publicado is not None and len(publicado):
        rep = pd.to_datetime(publicado['periodo_reporte'])
        previo = publicado.loc[rep < p, 'periodo']
        if len(previo):
            f = _mes(pd.to_datetime(previo).max())
    topes = [reabrir_desde]
    if not provisional_cosechas and provisional_activo(p, provisional_desde, provisional_hasta):
        topes.append(provisional_desde)
    for tope in topes:
        t = _mes(tope)
        if t is not None and f is not None:
            f = min(f, t - pd.offsets.MonthBegin(1))
    return f


def separar(df, frontera, provisional=None):
    """
    (publicar, cerradas): se publica lo posterior a F y, si `provisional` está activo ({'desde', 'cosechas'}), además las filas de esas cosechas
    desde ese mes (se republican completas en cada cierre). Sin frontera, todo se publica.
    """
    if df is None or len(df) == 0:
        return df, (df.iloc[0:0] if df is not None else df)
    per = pd.to_datetime(df['periodo'])
    m = (per > frontera) if frontera is not None else pd.Series(True, index=df.index)
    if provisional and provisional.get('cosechas') and 'cosecha_grupo' in df.columns:
        coh = {_mes(c) for c in provisional['cosechas']}
        cg = pd.to_datetime(df['cosecha_grupo'], errors='coerce').dt.to_period('M').dt.to_timestamp()
        m = m | ((per >= _mes(provisional['desde'])) & cg.isin(coh))
    return df.loc[m].copy(), df.loc[~m].copy()


def rebanada_existente(publicado, periodo_reporte):
    """True si este cierre ya publicó algo (volver a cargarlo exige reproceso)."""
    if publicado is None or len(publicado) == 0:
        return False
    return bool((pd.to_datetime(publicado['periodo_reporte']) == _mes(periodo_reporte)).any())


def cierre_posterior(publicado, periodo_reporte):
    """Último cierre publicado posterior a este (Timestamp) o None. Cargar un cierre más viejo que lo ya publicado no tiene sentido: se bloquea."""
    if publicado is None or len(publicado) == 0:
        return None
    rep = pd.to_datetime(publicado['periodo_reporte'])
    post = rep[rep > _mes(periodo_reporte)]
    return None if post.empty else post.max()


def _cosecha_txt(d):
    if 'cosecha_grupo' not in d.columns:
        return pd.Series('', index=d.index)
    return pd.to_datetime(d['cosecha_grupo'], errors='coerce').dt.strftime('%Y-%m').fillna('')


def vigente_local(df):
    """
    La regla de la vista vigente en pandas: última versión por (programa, periodo_reporte) y, por (programa, mes, cosecha), la rebanada más reciente.
    Por cosecha: una cosecha provisional (el G04) se reemplaza completa sin tocar las demás cosechas de ese mes.
    """
    if df is None or len(df) == 0:
        return df
    d = df.copy()
    rep = pd.to_datetime(d['periodo_reporte'])
    d = d[d['version'] == d.groupby(['programa', rep])['version'].transform('max')]
    rep = pd.to_datetime(d['periodo_reporte'])
    per = pd.to_datetime(d['periodo']).dt.to_period('M')
    ultima = rep.groupby([d['programa'], per, _cosecha_txt(d)]).transform('max')
    return d[rep == ultima]


def resumen(frontera, publicar, cerradas, deriva_resumen=None):
    """Lo que se imprime y se guarda en el log por tabla."""
    r = {'frontera': None if frontera is None else frontera.date().isoformat(),
         'filas_a_publicar': 0 if publicar is None else len(publicar),
         'meses_a_publicar': ([] if publicar is None or len(publicar) == 0
                              else sorted(pd.to_datetime(publicar['periodo']).dt.strftime('%Y-%m').unique().tolist())),
         'filas_cerradas_recalculadas': 0 if cerradas is None else len(cerradas)}
    if deriva_resumen:
        r.update({'filas_cerradas_con_deriva': deriva_resumen.get('filas_con_deriva', 0),
                  'filas_cerradas_que_no_estaban_publicadas': deriva_resumen.get('filas_nuevas', 0),
                  'filas_publicadas_que_ya_no_salen': deriva_resumen.get('filas_publicadas_ausentes_hoy', 0)})
    return r
