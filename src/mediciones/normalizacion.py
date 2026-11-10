"""Normalización al esquema único: flag YoY binario, valor atribuible y columnas técnicas; validación de las bases."""
import numpy as np
import pandas as pd

from .esquema import COLS_BRE, COLS_ESP, COLS_TON, COLS_UB, CLAVE, T_BRE, T_ESP, T_TON  # noqa: F401


def _fecha(serie):
    return pd.to_datetime(serie).dt.date


def _id10(serie):
    """Códigos de cliente a 10 caracteres (llave estable); los nulos se conservan."""
    return serie.where(serie.isna(), serie.astype(str).str.zfill(10))


def asignar_estado_yoy(ant, act):
    """antiguo = compró el mismo mes del año anterior; nuevo = toneladas nuevas; nulo = sin base ni compra."""
    estado = pd.Series(None, index=ant.index, dtype='object')
    estado[ant > 0] = 'antiguo'
    estado[~(ant > 0) & (act > 0)] = 'nuevo'
    return estado


def _tecnicas(d, programa, periodo_reporte, corrida_id, version, config_hash=None):
    d['programa'] = programa
    d['periodo_reporte'] = periodo_reporte
    d['corrida_id'] = corrida_id
    d['version'] = np.int64(version)
    d['config_hash'] = config_hash
    d['fecha_carga'] = pd.Timestamp.now(tz='UTC')
    return d


def preparar_base_ton(df, programa, periodo_reporte, corrida_id, version=1, config_hash=None, regla_estado='estandar'):
    """
    `regla_estado` (config/programas.yaml, `estado_yoy` de la familia):
      'estandar' -> antiguo/nuevo por toneladas_ant; el atribuible de los nuevos (ant == 0) es todo su volumen (CPB, CPM).
      'programa' -> se respetan `flg_estado_yoy` y `valor_atribuible` tal como los calculó el programa (surtido, Champion):
                    no_existia_periodo_pasado = todo; reactivado (existía, compró 0 ese mes del año pasado) = 0.
    """
    if regla_estado not in ('estandar', 'programa'):
        raise ValueError(f"regla_estado debe ser 'estandar' o 'programa' (recibí {regla_estado!r})")
    if regla_estado == 'programa':
        faltan = [c for c in ('flg_estado_yoy', 'valor_atribuible') if c not in df.columns]
        if faltan:
            raise ValueError(f"{programa}: con regla_estado='programa' el programa debe entregar {faltan}")
    d = df.copy()
    for c in COLS_TON:
        if c not in d.columns:
            d[c] = np.nan
    d['cod_cliente'] = d['cod_cliente'].astype(str).str.zfill(10)      # llave estable entre meses y programas
    d['cod_cliente_dex'] = d['cod_cliente_dex'].where(d['cod_cliente_dex'].notna(), d['cod_cliente']).astype(str).str.zfill(10)
    d['cosecha_grupo'] = _fecha(d['cosecha_grupo'])
    d['periodo'] = _fecha(d['periodo'])
    d['flg_cliente_perfecto'] = d['flg_cliente_perfecto'].astype('boolean')
    d['flg_estimado_retroactivo'] = d['flg_estimado_retroactivo'].fillna(False).astype(bool)

    if regla_estado == 'estandar':
        d['flg_estado_yoy'] = asignar_estado_yoy(d['toneladas_ant'], d['toneladas_actual'])
        nuevo = d['flg_estado_yoy'].eq('nuevo')
        d.loc[nuevo, 'valor_atribuible'] = d.loc[nuevo, 'toneladas_actual']
    d['kg_atribuibles'] = d['valor_atribuible'] * 1000
    _tecnicas(d, programa, periodo_reporte, corrida_id, version, config_hash)
    return d[COLS_TON]


def preparar_base_brechas(df, programa, periodo_reporte, corrida_id, version=1, config_hash=None):
    d = df.copy()
    for c in COLS_BRE:
        if c not in d.columns:
            d[c] = np.nan
    d['cod_cliente_accion_dex'] = d['cod_cliente_accion_dex'].astype(str).str.zfill(10)
    d['cod_cliente_control'] = _id10(d['cod_cliente_control'])
    d['cosecha_grupo'] = _fecha(d['cosecha_grupo'])
    d['periodo'] = _fecha(d['periodo'])
    d['flg_cliente_perfecto'] = d['flg_cliente_perfecto'].astype('boolean')
    d['reportable'] = d['reportable'].astype('boolean')
    d['flg_estimado_retroactivo'] = d['flg_estimado_retroactivo'].fillna(False).astype(bool)
    _tecnicas(d, programa, periodo_reporte, corrida_id, version, config_hash)
    return d[COLS_BRE]


def preparar_base_espejos(df, programa, periodo_reporte, corrida_id, version=1, config_hash=None):
    """Un registro por cliente de acción x categoría x mes, con su espejo (o sin espejo y el motivo)."""
    d = df.copy()
    for c in COLS_ESP:
        if c not in d.columns:
            d[c] = np.nan
    d['cod_cliente_accion'] = d['cod_cliente_accion'].astype(str).str.zfill(10)
    d['cod_cliente_espejo'] = _id10(d['cod_cliente_espejo'])
    d['cosecha_grupo'] = _fecha(d['cosecha_grupo'])
    d['periodo'] = _fecha(d['periodo'])
    d['flg_estimado_retroactivo'] = d['flg_estimado_retroactivo'].fillna(False).astype(bool)
    d['nivel_match'] = pd.to_numeric(d['nivel_match'], errors='coerce').astype('Int64')
    d['estado_espejo'] = np.where(d['cod_cliente_espejo'].notna(), 'con_espejo', 'sin_espejo')
    d.loc[d['estado_espejo'] == 'con_espejo', 'motivo_sin_espejo'] = np.nan
    _tecnicas(d, programa, periodo_reporte, corrida_id, version, config_hash)
    return d[COLS_ESP]


def validar(ton, bre, periodo_reporte, ton_original=None, esp=None):
    """Devuelve (errores, resumen). Los errores bloquean la carga."""
    errores, resumen = [], {}
    for nombre, d, tabla in (('ton', ton, T_TON), ('brechas', bre, T_BRE)):
        clave = CLAVE[tabla]
        dup = int(d.duplicated(clave).sum())
        if dup:
            errores.append(f'{nombre}: {dup:,} filas duplicadas en {clave}')
        nulos = [c for c in clave if d[c].isna().any()]
        if nulos:
            errores.append(f'{nombre}: nulos en la llave {nulos}')
        if d['periodo'].max() > periodo_reporte:
            errores.append(f'{nombre}: hay periodos posteriores al periodo_reporte')
    dif = (ton['kg_atribuibles'] - ton['valor_atribuible'] * 1000).abs().max()
    if dif > 1e-6:
        errores.append(f'ton: kg_atribuibles != valor_atribuible x 1000 (dif máx {dif})')
    k_ton = set(zip(ton['cod_cliente'], ton['des_categoria']))
    k_bre = set(zip(bre['cod_cliente_accion_dex'], bre['des_categoria']))
    huerfanos = len(k_bre - k_ton)
    if huerfanos:
        errores.append(f'brechas: {huerfanos:,} pares (cliente, categoría) sin fila en la base de toneladas')

    if esp is not None:
        dup = int(esp.duplicated(CLAVE[T_ESP]).sum())
        if dup:
            errores.append(f'espejos: {dup:,} filas duplicadas en {CLAVE[T_ESP]}')
        k_ton3 = set(zip(ton['cod_cliente'], ton['des_categoria'], ton['periodo']))
        # Los meses retroactivos pueden no tener fila de toneladas: la brecha retroactiva es una estimación por celda replicada a todos los pares de la
        # cosecha (también a clientes que aún no aparecían en el sell-out). Solo esos se eximen; fuera del retroactivo la fila de toneladas es obligatoria.
        retro = esp['flg_estimado_retroactivo'].fillna(False).astype(bool) if 'flg_estimado_retroactivo' in esp else pd.Series(False, index=esp.index)
        faltan = [k not in k_ton3 for k in zip(esp['cod_cliente_accion'], esp['des_categoria'], esp['periodo'])]
        faltan = pd.Series(faltan, index=esp.index)
        huerf = int((faltan & ~retro).sum())
        if huerf:
            errores.append(f'espejos: {huerf:,} clientes de acción sin fila en la base de toneladas')
        resumen['espejos_retroactivos_sin_toneladas'] = int((faltan & retro).sum())
        # todo par de la base de brechas debe tener su espejo registrado (y coincidir con el control del par si la brecha lo trae)
        m = bre.merge(esp[['cod_cliente_accion', 'des_categoria', 'periodo', 'cod_cliente_espejo']],
                      left_on=['cod_cliente_accion_dex', 'des_categoria', 'periodo'],
                      right_on=['cod_cliente_accion', 'des_categoria', 'periodo'], how='left')
        sin = int(m['cod_cliente_espejo'].isna().sum())
        if sin:
            errores.append(f'espejos: {sin:,} pares de la base de brechas sin espejo registrado')
        if 'cod_cliente_control' in m and m['cod_cliente_control'].notna().any():
            dist = int((m['cod_cliente_control'].notna() & (m['cod_cliente_control'] != m['cod_cliente_espejo'])).sum())
            if dist:
                errores.append(f'espejos: {dist:,} pares cuyo espejo difiere del control de la base de brechas')
        resumen['filas_espejos'] = len(esp)
        resumen['pct_con_espejo'] = round(100 * float((esp['estado_espejo'] == 'con_espejo').mean()), 1) if len(esp) else None
    resumen['filas_ton'], resumen['filas_brechas'] = len(ton), len(bre)
    resumen['clientes_ton'], resumen['pares'] = ton['cod_cliente'].nunique(), bre['par_id'].nunique()
    resumen['estado_yoy'] = ton['flg_estado_yoy'].value_counts(dropna=False).to_dict()
    resumen['valor_atribuible_tn_por_categoria'] = ton.groupby('des_categoria')['valor_atribuible'].sum().round(1).to_dict()
    resumen['valor_atribuible_tn_total'] = round(float(ton['valor_atribuible'].sum()), 1)
    if ton_original is not None and 'valor_atribuible' in ton_original:
        resumen['valor_atribuible_tn_total_segun_notebook'] = round(float(ton_original['valor_atribuible'].sum()), 1)
    resumen['filas_atribuible_mayor_a_compra (informativo)'] = int(
        ((ton['toneladas_actual'] > 0) & (ton['valor_atribuible'] > ton['toneladas_actual'] + 1e-9)).sum())
    return errores, resumen
