"""Línea base reportada: se lleva al esquema estándar como el motor, se carga a un dataset gemelo (sin tocar BigQuery en la prueba) y se compara con consultas válidas."""
import numpy as np
import pandas as pd
import pytest

from conftest import RAIZ
from mediciones import bigquery, reportado
from mediciones.esquema import DATASET, T_BRE, T_ESP, T_TON
from mediciones.validacion import MAL, validar_corrida
from test_champion_salida import _como_champion
from test_exportar_cosechas import _ns

CONFIG = RAIZ / 'config' / 'programas.yaml'


def _proyecto(tmp_path, programa, ton, bre, matches, sin_pareja, hoja_ton, hoja_bre):
    """Reproduce lo que tienes de un programa ya reportado: un Excel con las hojas de toneladas y brecha y, aparte, los pares en data/<programa>/outputs/."""
    carpeta = tmp_path / 'data' / programa / 'outputs' / reportado.CARPETA_OUTPUTS[programa]
    carpeta.mkdir(parents=True)
    matches.to_csv(carpeta / 'df_matches.csv', index=False)
    sin_pareja.to_csv(carpeta / 'df_sin_pareja.csv', index=False)
    xlsx = tmp_path / 'reportado.xlsx'
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({'x': [1]}).to_excel(w, sheet_name='Resumen', index=False)
        ton.to_excel(w, sheet_name=hoja_ton, index=False)
        bre.to_excel(w, sheet_name=hoja_bre, index=False)
    return xlsx


def _surtido(tmp_path):
    ns = _ns()
    return _proyecto(tmp_path, 'surtido_motor', ns['base_calculo_toneladas'], ns['base_calculo_brecha'], ns['df_matches'], ns['df_sin_pareja'],
                     'Base toneladas (cliente-mes)', 'Base brecha (par-mes)')


def test_surtido_reportado_llega_a_las_3_bases_estandar(tmp_path):
    xlsx = _surtido(tmp_path)
    raw = reportado.preparar_reportado(tmp_path, 'surtido_motor', xlsx, config=CONFIG, verbose=False)
    assert len(raw['ton']) == 36 and len(raw['bre']) == 18 and len(raw['esp']) > 0
    assert raw['ton'].duplicated(['cod_cliente', 'des_categoria', 'periodo']).sum() == 0
    assert raw['regla_estado'] == 'programa'


def test_champion_reportado_usa_la_salida_de_su_familia(tmp_path):
    ns = _como_champion()
    xlsx = _proyecto(tmp_path, 'champion_motor', ns['base_toneladas_export'], ns['base_calculo_brecha'].rename(columns={}), ns['df_matches'], ns['df_sin_pareja'],
                     'Base toneladas (cli-mes)', 'Base brecha (par-mes)')
    raw = reportado.preparar_reportado(tmp_path, 'champion_motor', xlsx, config=CONFIG, verbose=False)
    assert set(raw['ton'].des_categoria) == {'Galletas', 'Cereales', 'Inkachips'} and (raw['ton'].cod_cliente_dex == raw['ton'].cod_cliente).all()
    assert len(raw['esp']) > 0


def test_hoja_inexistente_y_pares_faltantes_dicen_que_falta(tmp_path):
    xlsx = _surtido(tmp_path)
    with pytest.raises(ValueError, match='Hojas del Excel'):
        reportado.preparar_reportado(tmp_path, 'surtido_motor', xlsx, config=CONFIG, hoja_ton='NoExiste')
    (tmp_path / 'data' / 'surtido_motor' / 'outputs' / reportado.CARPETA_OUTPUTS['surtido_motor'] / 'df_matches.csv').unlink()
    with pytest.raises(FileNotFoundError, match='df_matches.csv'):
        reportado.preparar_reportado(tmp_path, 'surtido_motor', xlsx, config=CONFIG)


def test_sin_cargar_solo_prepara_valida_y_guarda_una_copia_local_que_se_puede_validar(tmp_path):
    xlsx = _surtido(tmp_path)
    out = reportado.cargar_reportado(tmp_path, 'surtido_motor', xlsx, '2026-09-01', config=CONFIG, verbose=False)
    assert out['errores'] == [] and out['cargado'] is False and out['corrida_id'] == 'reportado_202609'
    assert (tmp_path / 'runs' / 'reportado_202609' / 'surtido_motor_ton.parquet').exists()
    v = validar_corrida(tmp_path, corrida='reportado_202609', config=CONFIG, verbose=False)
    assert not (v.resultado == MAL).any(), v[v.resultado == MAL][['chequeo', 'detalle']].to_string()


class _Cli:
    def __init__(self):
        self.consultas = []

    def query(self, q, **kw):
        self.consultas.append(q)
        return self

    def result(self):
        return self


def test_carga_al_dataset_gemelo_con_version_1_y_etiquetas_de_reportado(tmp_path, monkeypatch):
    xlsx = _surtido(tmp_path)
    cargas = []
    monkeypatch.setattr(bigquery, 'crear_dataset', lambda cli, ds, ubicacion='US': cargas.append(('dataset', ds)))
    monkeypatch.setattr(bigquery, 'contar_corrida', lambda cli, tabla, p, c, dataset=None: 0)
    monkeypatch.setattr(bigquery, 'cargar_df', lambda cli, df, tabla, dataset=None: cargas.append((tabla, dataset, df)) or len(df))
    out = reportado.cargar_reportado(tmp_path, 'surtido_motor', xlsx, '2026-09-01', config=CONFIG, cargar=True, cli=object(), verbose=False)
    assert out['cargado'] and cargas[0] == ('dataset', 'mediciones_reportado')
    tablas = {t: (ds, df) for t, ds, df in cargas[1:]}
    assert set(tablas) == {T_TON, T_BRE, T_ESP} and {ds for ds, _ in tablas.values()} == {'mediciones_reportado'}
    for t, (_, df) in tablas.items():
        assert (df.programa == 'surtido_motor').all() and (df.version == 1).all() and (df.corrida_id == 'reportado_202609').all() and (df.config_hash == 'reportado').all()
        assert (pd.to_datetime(df.periodo_reporte) == pd.Timestamp('2026-09-01')).all()


def test_no_carga_dos_veces_y_no_carga_si_hay_errores(tmp_path, monkeypatch):
    xlsx = _surtido(tmp_path)
    monkeypatch.setattr(bigquery, 'crear_dataset', lambda *a, **k: None)
    monkeypatch.setattr(bigquery, 'cargar_df', lambda *a, **k: 0)
    monkeypatch.setattr(bigquery, 'contar_corrida', lambda *a, **k: 5)
    with pytest.raises(RuntimeError, match='ya está cargada'):
        reportado.cargar_reportado(tmp_path, 'surtido_motor', xlsx, '2026-09-01', config=CONFIG, cargar=True, cli=object(), verbose=False)
    ns = _ns()
    ton = pd.concat([ns['base_calculo_toneladas'], ns['base_calculo_toneladas'].iloc[:1]], ignore_index=True)       # llave duplicada -> el exportador la rechaza
    (tmp_path / 'otro').mkdir()
    xlsx2 = _proyecto(tmp_path / 'otro', 'surtido_motor', ton, ns['base_calculo_brecha'], ns['df_matches'], ns['df_sin_pareja'], 'Base toneladas (cliente-mes)', 'Base brecha (par-mes)')
    with pytest.raises(AssertionError, match='llaves duplicadas'):
        reportado.cargar_reportado(tmp_path / 'otro', 'surtido_motor', xlsx2, '2026-09-01', config=CONFIG, verbose=False)


def test_el_dataset_gemelo_sale_del_ddl_de_produccion_y_nunca_toca_produccion():
    cli = _Cli()
    ref = bigquery.crear_dataset(cli, 'mediciones_reportado', ubicacion='southamerica-east1')
    ddl = cli.consultas[0]
    assert ref.endswith('.mediciones_reportado') and 'acpe-dev-uc-sandbox-aa.mediciones_reportado.base_ton_atribuibles' in ddl
    assert 'acpe-dev-uc-sandbox-aa.mediciones.' not in ddl and "DEFAULT 'southamerica-east1'" in ddl
    with pytest.raises(ValueError, match='producción'):
        bigquery.crear_dataset(cli, DATASET)


def test_las_consultas_de_comparacion_son_sql_valido_de_bigquery():
    sqlglot = pytest.importorskip('sqlglot')
    qs = reportado.sql_comparacion('mediciones_reportado', 'mediciones')
    assert set(qs) == {'ton_resumen', 'ton_top', 'ub_top', 'brecha_resumen', 'espejos_resumen'}
    for nombre, q in qs.items():
        assert '{rep}' not in q and '{mot}' not in q and '@p' in q, nombre
        assert 'mediciones_reportado.vw_' in q and 'sandbox-aa.mediciones.vw_' in q
        assert len(sqlglot.parse(q, read='bigquery')) == 1, nombre


# ── CPB / CPM: hojas 'Base ton + UB atribuibles' y 'Base brechas (par-mes)' (estructura del reportado) ──
CLIENTES = {'0000012372': 'accion_matcheado', '0000012625': 'accion_matcheado', '0000014450': 'accion_matcheado', '0000016351': 'accion_sin_pareja', '0000016461': 'accion_nuevo'}


def _mensual(tmp_path, sin_columna=None):
    filas, pares = [], []
    for mes in ('2026-01', '2026-02'):
        for cat in ('Aceites', 'Galletas'):
            celda = []
            for i, (cod, rol) in enumerate(CLIENTES.items()):
                nuevo = rol == 'accion_nuevo'
                ant = 0.0 if nuevo else 0.01 + 0.005 * i
                act = 0.01 if nuevo else ant * (1.10 - 0.4 * (i % 2))
                inf, sup = -6.09, 0.32
                yoy = np.nan if nuevo else 100 * (act - ant) / ant
                crec = np.nan if nuevo else float(np.clip(yoy, inf, sup))
                ton_atrib = np.nan if nuevo else ant * crec / 100
                valor = act if nuevo else ton_atrib
                par = f'{mes}|{cat}|{cod}' if rol == 'accion_matcheado' else None
                filas.append({'periodo': pd.Timestamp(mes + '-01'), 'cod_cliente': cod, 'des_categoria': cat, 'des_oficina': 'PE01 - CM-LIMA / NS CHICO', 'nom_compania': 'DISTRIBUIDORA FUERZADEX (LIMA)',
                              'des_segmento_transaccional': '2 Oro', 'rol': rol, 'motivo_sin_pareja': 'sin_control_comparable' if rol == 'accion_sin_pareja' else None,
                              'flg_estado_yoy': 'nuevo' if nuevo else 'antiguo', 'toneladas_ant': ant, 'toneladas_actual': act, 'toneladas_variacion': act - ant, 'yoy_pct_raw': yoy,
                              'ic_pct_inf_80': inf, 'ic_pct_sup_80': sup, 'crec_atrib_80_pct': crec, 'ton_atrib_80': ton_atrib, 'kg_atribuible': valor * 1000, 'ton_atribuible': valor,
                              'ub_base_soles': 10.0 + i, 'ub_por_ton_ref': 121.15, 'ub_atribuible_soles': round(valor * 1000, 6), 'flg_bajo_esperado': 0, 'flg_sobre_esperado': 1, 'flg_tendencia': 'crece',
                              'flg_sin_banda_ic': 0, 'reportable': True, 'par_id': par})
                if par:
                    pre_a, post_a, pre_c, post_c = ant * 1000, act * 1000, ant * 1000 * 1.1, ant * 1000 * 1.05
                    celda.append({'periodo': mes, 'des_categoria': cat, 'par_id': par, 'cod_accion': cod, 'cod_control': f'{1000 + i:010d}', 'des_oficina': 'PE01 - CM-LIMA / NS CHICO',
                                  'nom_compania': 'DISTRIBUIDORA FUERZADEX (LIMA)', 'des_segmento_transaccional': '2 Oro', 'pre_accion': pre_a, 'post_accion': post_a, 'pre_control': pre_c,
                                  'post_control': post_c, 'did_kg': (post_a - pre_a) - (post_c - pre_c), 'reportable': True})
            tot = 100 * sum(c['did_kg'] for c in celda) / sum(c['pre_accion'] for c in celda)
            pares += [dict(c, n_pares_celda=len(celda), brecha_pct_celda=round(tot, 2), ic_pct_inf_80_celda=-6.09, ic_pct_sup_80_celda=0.32) for c in celda]
    ton, bre = pd.DataFrame(filas), pd.DataFrame(pares)
    if sin_columna:
        ton = ton.drop(columns=[sin_columna])
    xlsx = tmp_path / 'cpb_v2_base_completa.xlsx'
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({'x': [1]}).to_excel(w, sheet_name='Resumen', index=False)
        ton.to_excel(w, sheet_name='Base ton + UB atribuibles', index=False)
        bre.to_excel(w, sheet_name='Base brechas (par-mes)', index=False)
    return xlsx, ton, bre


def test_cpb_reportado_llega_a_las_3_bases_estandar_con_sus_espejos(tmp_path):
    xlsx, ton, bre = _mensual(tmp_path)
    out = reportado.cargar_reportado(tmp_path, 'cpb', xlsx, '2026-02-01', config=CONFIG, verbose=False)
    assert out['errores'] == [], out['errores']
    d = tmp_path / 'runs' / 'reportado_202602'
    t, b, e = (pd.read_parquet(d / f'cpb_{x}.parquet') for x in ('ton', 'brecha', 'espejos'))
    assert len(t) == 20 and len(b) == 12 and len(e) == 20                                         # 5 clientes de acción × 2 meses × 2 categorías; 3 pares × 4 celdas
    m = t.assign(cod_cliente=t.cod_cliente.astype(str).str.zfill(10)).merge(
        ton.assign(periodo=pd.to_datetime(ton.periodo))[['cod_cliente', 'des_categoria', 'periodo', 'ton_atribuible']], on=['cod_cliente', 'des_categoria', 'periodo'])
    assert np.allclose(m.valor_atribuible.fillna(0), m.ton_atribuible.fillna(0))                  # ton_atribuible del reportado = valor_atribuible estándar
    assert e.loc[e.cod_cliente_accion == '0000012372', 'cod_cliente_espejo'].eq('0000001000').all()   # el espejo sale de cod_control de la hoja de brechas
    assert set(e.loc[e.cod_cliente_accion == '0000016351', 'motivo_sin_espejo']) == {'sin_control_comparable'}
    assert set(e.loc[e.cod_cliente_accion == '0000016461', 'motivo_sin_espejo']) == {'cliente_nuevo'}
    assert b.groupby(['periodo', 'des_categoria']).brecha_pct_celda.nunique().eq(1).all() and set(b.n_pares_celda) == {3}
    v = validar_corrida(tmp_path, corrida='reportado_202602', programas=['cpb'], config=CONFIG, verbose=False)
    assert not (v.resultado == MAL).any(), v[v.resultado == MAL][['chequeo', 'detalle']].to_string()


def test_cpb_reportado_sin_la_columna_de_atribuible_dice_cual_falta(tmp_path):
    xlsx, _, _ = _mensual(tmp_path, sin_columna='rol')
    with pytest.raises(ValueError, match="rol"):
        reportado.preparar_reportado(tmp_path, 'cpm', xlsx, config=CONFIG)


def test_la_ub_reportada_de_cpb_pasa_tal_cual_a_la_linea_base(tmp_path):
    xlsx, ton, _ = _mensual(tmp_path)
    raw = reportado.preparar_reportado(tmp_path, 'cpb', xlsx, config=CONFIG, verbose=False)
    k = ['cod_cliente', 'des_categoria', 'periodo']
    m = raw['ton'].merge(ton[ton.rol.str.startswith('accion')].assign(cod_cliente=lambda d: d.cod_cliente.str.zfill(10), periodo=lambda d: pd.to_datetime(d.periodo)),
                         on=k, suffixes=('', '_rep'))
    assert len(m) == len(raw['ton']) == 20
    for c in ('ub_base_soles', 'ub_por_ton_ref', 'ub_atribuible_soles'):
        assert np.allclose(m[c].astype(float), m[c + '_rep'].astype(float), equal_nan=True), c
    assert raw['ton'].ub_atribuible_soles.abs().sum() > 0                                      # la UB no quedó vacía
    out = reportado.cargar_reportado(tmp_path, 'cpb', xlsx, '2026-02-01', config=CONFIG, verbose=False)
    assert out['errores'] == []


def test_la_linea_base_local_nunca_se_toma_por_una_corrida_del_motor(tmp_path):
    """Con EJECUTAR = False el runner reutiliza la última corrida de runs/AAAAMM_*: la línea base reportada (reportado_AAAAMM) no puede ser candidata."""
    from mediciones import runner
    xlsx = _surtido(tmp_path)
    reportado.cargar_reportado(tmp_path, 'surtido_motor', xlsx, '2026-09-01', config=CONFIG, verbose=False)
    motor = tmp_path / 'runs' / '202609_20261015T205619_1a2807'
    motor.mkdir()
    for x in ('ton', 'brecha', 'espejos'):                                   # una "corrida del motor" distinguible: una fila menos
        d = pd.read_parquet(tmp_path / 'runs' / 'reportado_202609' / f'surtido_motor_{x}.parquet')
        d.iloc[:-1].to_parquet(motor / f'surtido_motor_{x}.parquet', index=False)
    ton, _, _ = runner._cargar_crudas(tmp_path, 'surtido_motor', pd.Timestamp('2026-09-01'))
    assert len(ton) == len(pd.read_parquet(tmp_path / 'runs' / 'reportado_202609' / 'surtido_motor_ton.parquet')) - 1
    v = validar_corrida(tmp_path, config=CONFIG, verbose=False)               # sin indicar corrida: elige la del motor, no la línea base
    assert v.empty is False and not (v.programa == 'x').any()


# ── Excel de Champion: 5 hojas (Base completa, toneladas, brecha, Sin pareja, Matches) ──
def _excel_champion(tmp_path, quitar_cosecha_grupo=False, ton=None, bre=None):
    ns = _como_champion()
    t = (ton if ton is not None else ns['base_toneladas_export']).copy()
    b = (bre if bre is not None else ns['base_calculo_brecha']).copy()
    if quitar_cosecha_grupo:
        t, b = t.drop(columns=['cosecha_grupo']), b.drop(columns=['cosecha_grupo'])
    xlsx = tmp_path / 'champion.xlsx'
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({'x': [1]}).to_excel(w, sheet_name='Base completa', index=False)
        t.to_excel(w, sheet_name='Base toneladas (cli-mes)', index=False)
        b.to_excel(w, sheet_name='Base brecha (par-mes)', index=False)
        ns['df_sin_pareja'].to_excel(w, sheet_name='Sin pareja (en muestra)', index=False)
        ns['df_matches'].to_excel(w, sheet_name='Matches', index=False)
    return xlsx, ns


def test_champion_toma_los_pares_de_las_hojas_de_su_propio_excel_sin_csv(tmp_path, capsys):
    xlsx, _ = _excel_champion(tmp_path)                                   # no hay data/champion_motor/outputs: no hace falta
    raw = reportado.preparar_reportado(tmp_path, 'champion_motor', xlsx, config=CONFIG)
    assert len(raw['esp']) > 0 and "hoja 'Matches' del Excel" in capsys.readouterr().out


def test_champion_con_un_excel_sin_cosecha_grupo_la_deduce_de_sus_pares(tmp_path):
    xlsx, ns = _excel_champion(tmp_path, quitar_cosecha_grupo=True)
    assert ns['df_matches']['cosecha'].nunique() == 1
    raw = reportado.preparar_reportado(tmp_path, 'champion_motor', xlsx, config=CONFIG, verbose=False)
    assert set(pd.to_datetime(raw['ton'].cosecha_grupo)) == {pd.Timestamp(ns['df_matches']['cosecha'].iloc[0])}
    assert len(raw['esp']) > 0


def test_surtido_sin_cosecha_grupo_falla_diciendo_programa_hoja_y_columnas(tmp_path):
    ns = _ns()
    m = ns['df_matches'].copy()
    m.loc[m.index[: len(m) // 2], 'cosecha'] = pd.Timestamp('2026-06-01')     # surtido tiene varias cosechas: no se puede deducir
    xlsx = _proyecto(tmp_path, 'surtido_motor', ns['base_calculo_toneladas'].drop(columns=['cosecha_grupo']), ns['base_calculo_brecha'], m, ns['df_sin_pareja'],
                     'Base toneladas (cliente-mes)', 'Base brecha (par-mes)')
    with pytest.raises(ValueError) as e:
        reportado.preparar_reportado(tmp_path, 'surtido_motor', xlsx, config=CONFIG)
    msg = str(e.value)
    assert '[surtido_motor]' in msg and 'Base toneladas (cliente-mes)' in msg and 'cosecha_grupo' in msg and 'Columnas que trae' in msg and 'varias cosechas' in msg


def test_una_ruta_explicita_de_pares_gana_a_las_hojas_del_excel(tmp_path, capsys):
    xlsx, ns = _excel_champion(tmp_path)
    csv = tmp_path / 'm.csv'
    ns['df_matches'].to_csv(csv, index=False)
    reportado.preparar_reportado(tmp_path, 'champion_motor', xlsx, config=CONFIG, matches=csv)
    salida = capsys.readouterr().out
    assert f'matches: CSV {csv}' in salida and "sin pareja: hoja 'Sin pareja (en muestra)'" in salida


# ── verificación previa de la comparación en BigQuery ──
class _CliVerif:
    """Cliente de BigQuery simulado: qué datasets existen, qué columnas tienen las vistas y cuántas filas hay por dataset y programa."""

    def __init__(self, datasets=(), columnas=None, filas=None, vistas=True):
        from mediciones import esquema as esq
        self.datasets, self.filas, self.vistas = set(datasets), filas or {}, vistas
        self.columnas = list(dict.fromkeys(esq.COLS_TON + esq.COLS_BRE + esq.COLS_ESP)) if columnas is None else list(columnas)

    def get_dataset(self, ref):
        from google.api_core.exceptions import NotFound
        if ref.split('.')[-1] not in self.datasets:
            raise NotFound(f'Dataset {ref} not found')
        return object()

    def get_table(self, ref):
        from google.api_core.exceptions import NotFound
        if not self.vistas:
            raise NotFound('vista')
        return type('T', (), {'schema': [type('F', (), {'name': c}) for c in self.columnas]})()

    def query(self, q, job_config=None):
        self.ultimo = q
        return self

    def result(self):
        ds = next(d for d in self.datasets if f'.{d}.' in self.ultimo)
        return [{'n': self.filas.get(ds, 0)}]


def test_la_comparacion_avisa_antes_de_consultar_que_falta_la_linea_base():
    pytest.importorskip('google.cloud.bigquery')
    cli = _CliVerif(datasets={'mediciones'}, filas={'mediciones': 10})
    with pytest.raises(RuntimeError) as e:
        reportado.comparar_en_bigquery(cli, 'mediciones_reportado', 'mediciones', 'surtido_motor', verbose=False)
    msg = str(e.value)
    assert 'surtido_motor' in msg and 'mediciones_reportado (reportado) no existe' in msg and 'CARGAR = True' in msg and 'mediciones (motor)' not in msg


def test_la_verificacion_distingue_cada_causa():
    pytest.importorskip('google.cloud.bigquery')
    from mediciones import esquema as esq
    ok = _CliVerif(datasets={'a', 'b'}, filas={'a': 5, 'b': 7})
    assert reportado.verificar_comparacion(ok, 'a', 'b', 'cpb') == []
    sin_filas = reportado.verificar_comparacion(_CliVerif(datasets={'a', 'b'}, filas={'a': 5, 'b': 0}), 'a', 'b', 'cpb')
    assert len(sin_filas) == 1 and "b existe pero no tiene filas de 'cpb'" in sin_filas[0] and "cargar_corrida_prueba(corrida, dataset='b'" in sin_filas[0]       # dataset gemelo
    produccion = reportado.verificar_comparacion(_CliVerif(datasets={'a', esq.DATASET}, filas={'a': 5, esq.DATASET: 0}), 'a', esq.DATASET, 'cpb')
    assert len(produccion) == 1 and "corre el orquestador con MODO = 'cargar'" in produccion[0]                                                                      # producción
    antigua = [c for c in dict.fromkeys(esq.COLS_TON + esq.COLS_BRE + esq.COLS_ESP) if c not in ('ic_pct_inf_80', 'ub_base_soles')]      # tabla creada con un DDL anterior
    p = reportado.verificar_comparacion(_CliVerif(datasets={'a', 'b'}, columnas=antigua, filas={'a': 5, 'b': 7}), 'a', 'b', 'cpb')
    assert len(p) == 2 and all("'ic_pct_inf_80'" in x and "'ub_base_soles'" in x and 'diagnosticar_esquema' in x and 'no renombra' in x for x in p)
    sin_vista = reportado.verificar_comparacion(_CliVerif(datasets={'a', 'b'}, vistas=False), 'a', 'b', 'cpb')
    assert len(sin_vista) == 6 and all('no existe' in x and 'ddl_mediciones.sql' in x for x in sin_vista)


# ── cargar_df ya no descarta columnas en silencio + diagnóstico de esquema ──
class _CliTablas:
    def __init__(self, tablas):
        self.tablas, self.cargado = tablas, None

    def get_table(self, ref):
        from google.api_core.exceptions import NotFound
        t = ref.split('.')[-1]
        if t not in self.tablas:
            raise NotFound(t)
        from google.cloud import bigquery as bq
        return type('T', (), {'schema': [bq.SchemaField(c, 'FLOAT64') for c in self.tablas[t]]})()

    def load_table_from_dataframe(self, df, ref, job_config=None):
        self.cargado = (ref, list(df.columns))
        return self

    def result(self):
        return self


def test_cargar_df_falla_si_la_tabla_no_tiene_una_columna_en_vez_de_perder_el_dato():
    pytest.importorskip('google.cloud.bigquery')
    cli = _CliTablas({'base_ton_atribuibles': ['periodo', 'ic_pct_inf', 'valor_atribuible']})        # tabla con el nombre antiguo
    df = pd.DataFrame({'periodo': [1], 'ic_pct_inf_80': [2.0], 'valor_atribuible': [3.0]})
    with pytest.raises(ValueError, match=r"ic_pct_inf_80.*diagnosticar_esquema"):
        bigquery.cargar_df(cli, df, 'base_ton_atribuibles', dataset='mediciones')
    assert cli.cargado is None                                                                          # no se cargó nada
    bigquery.cargar_df(cli, pd.DataFrame({'periodo': [1], 'valor_atribuible': [3.0]}), 'base_ton_atribuibles', dataset='mediciones')    # columnas que faltan en el df: siguen siendo NULL
    assert cli.cargado[1] == ['periodo', 'ic_pct_inf', 'valor_atribuible']


def test_diagnostico_de_esquema_sugiere_renombrar_o_agregar():
    pytest.importorskip('google.cloud.bigquery')
    from mediciones import esquema as esq
    ton = [c if c != 'ic_pct_inf_80' else 'ic_pct_inf' for c in esq.COLS_TON if c not in ('ub_base_soles', 'ub_por_ton_ref')]       # el caso real: nombre antiguo y sin UB base
    cli = _CliTablas({'base_ton_atribuibles': ton, 'base_brechas': list(esq.COLS_BRE), 'base_espejos': list(esq.COLS_ESP), 'log_metricas_ejecucion': list(esq.COLS_LOG)})
    d = bigquery.diagnosticar_esquema(cli, 'mediciones')
    assert set(d.tabla) == {'base_ton_atribuibles'}
    por_col = d.set_index('columna')
    assert por_col.loc['ic_pct_inf_80', 'sugerencia'].endswith('RENAME COLUMN ic_pct_inf TO ic_pct_inf_80;') and "trae 'ic_pct_inf'" in por_col.loc['ic_pct_inf_80', 'problema']
    assert 'ADD COLUMN IF NOT EXISTS ub_base_soles FLOAT64;' in por_col.loc['ub_base_soles', 'sugerencia']
    assert 'ic_pct_inf' not in set(d.columna)                                                          # el nombre antiguo no se reporta además como "sobra"
    ok = bigquery.diagnosticar_esquema(_CliTablas({'base_ton_atribuibles': list(esq.COLS_TON), 'base_brechas': list(esq.COLS_BRE), 'base_espejos': list(esq.COLS_ESP),
                                                  'log_metricas_ejecucion': list(esq.COLS_LOG)}), 'mediciones')
    assert ok.empty
    faltante = bigquery.diagnosticar_esquema(_CliTablas({}), 'mediciones')
    assert set(faltante.problema) == {'la tabla no existe'} and len(faltante) == 4


# ── actualizar_esquema / listar_corridas / borrar_corrida: sin borrar tablas ──
class _CliMutable:
    """Cliente simulado con tablas que cambian de verdad cuando se ejecutan los ALTER y el DDL (que crea las que faltan)."""

    def __init__(self, tablas, filas=None):
        self.tablas, self.filas, self.queries, self.ddl_ejecutado = {k: list(v) for k, v in tablas.items()}, filas or {}, [], False

    def get_table(self, ref):
        from google.api_core.exceptions import NotFound
        from google.cloud import bigquery as bq
        t = ref.split('.')[-1]
        if t not in self.tablas:
            raise NotFound(t)
        return type('T', (), {'schema': [bq.SchemaField(c, 'FLOAT64') for c in self.tablas[t]]})()

    def query(self, sql, job_config=None):
        import re
        from mediciones import esquema as esq
        self.queries.append(sql)
        self._sql = sql
        m = re.match(r'ALTER TABLE `[^`]*\.([a-z_]+)` ADD COLUMN IF NOT EXISTS (\w+) ', sql)
        if m:
            self.tablas[m.group(1)].append(m.group(2))
        m = re.match(r'ALTER TABLE `[^`]*\.([a-z_]+)` RENAME COLUMN (\w+) TO (\w+);', sql)
        if m:
            self.tablas[m.group(1)][self.tablas[m.group(1)].index(m.group(2))] = m.group(3)
        m = re.match(r'DROP TABLE IF EXISTS `[^`]*\.([a-z_]+)`', sql)
        if m:
            self.tablas.pop(m.group(1), None)
        if 'CREATE OR REPLACE VIEW' in sql:                                     # el DDL: crea las tablas que no existen y recrea las vistas
            self.ddl_ejecutado = True
            for t, cols in (('base_ton_atribuibles', esq.COLS_TON), ('base_brechas', esq.COLS_BRE), ('base_espejos', esq.COLS_ESP), ('log_metricas_ejecucion', esq.COLS_LOG)):
                self.tablas.setdefault(t, list(cols))
        return self

    def result(self):
        import re
        if self._sql.startswith('SELECT COUNT'):
            return [{'n': self.filas.get(re.search(r'\.([a-z_]+)`', self._sql).group(1), 0)}]
        return []

    def to_dataframe(self):
        return pd.DataFrame({'programa': ['cpb'], 'corrida_id': ['202609_x'], 'periodo_reporte': ['2026-09-01'], 'version': [1], 'filas': [10]})


def _tablas_antiguas():
    from mediciones import esquema as esq
    renombre = {'ic_pct_inf_80': 'ic_pct_inf', 'ic_pct_sup_80': 'ic_pct_sup'}
    ton = [renombre.get(c, c) for c in esq.COLS_TON if c not in ('ub_base_soles', 'ub_por_ton_ref')]       # el caso real: nombres antiguos y sin UB base
    return {'base_ton_atribuibles': ton, 'base_brechas': list(esq.COLS_BRE), 'base_espejos': list(esq.COLS_ESP), 'log_metricas_ejecucion': list(esq.COLS_LOG)}


def test_actualizar_esquema_sin_ejecutar_solo_muestra_el_plan():
    pytest.importorskip('google.cloud.bigquery')
    cli = _CliMutable(_tablas_antiguas())
    plan = bigquery.actualizar_esquema(cli, 'mediciones')
    assert cli.queries == [] and len(plan) == 4 and set(plan.tabla) == {'base_ton_atribuibles'}
    assert all(x.startswith('ALTER TABLE') and 'ADD COLUMN IF NOT EXISTS' in x for x in plan.sentencia)
    assert {a.replace('agregar ', '') for a in plan.accion} == {'ic_pct_inf_80', 'ic_pct_sup_80', 'ub_base_soles', 'ub_por_ton_ref'}


def test_actualizar_esquema_agregar_deja_el_esquema_al_dia_sin_tocar_las_columnas_antiguas():
    pytest.importorskip('google.cloud.bigquery')
    cli = _CliMutable(_tablas_antiguas())
    final = bigquery.actualizar_esquema(cli, 'mediciones', ejecutar=True)
    assert final.empty and cli.ddl_ejecutado                                       # al día, y las vistas se recrearon
    assert 'ic_pct_inf' in cli.tablas['base_ton_atribuibles'] and 'ic_pct_inf_80' in cli.tablas['base_ton_atribuibles']      # lo antiguo queda; lo nuevo existe
    assert not any(q.startswith('DROP') or q.startswith('DELETE') for q in cli.queries)                                     # nada se borra
    assert cli.queries.index(next(q for q in cli.queries if 'CREATE OR REPLACE VIEW' in q)) > cli.queries.index(next(q for q in cli.queries if q.startswith('ALTER')))


def test_actualizar_esquema_renombrar_conserva_los_valores_y_agrega_lo_que_no_tiene_antecesor():
    pytest.importorskip('google.cloud.bigquery')
    cli = _CliMutable(_tablas_antiguas())
    plan = bigquery.actualizar_esquema(cli, 'mediciones', modo='renombrar')
    sent = set(plan.sentencia)
    assert any('RENAME COLUMN ic_pct_inf TO ic_pct_inf_80' in x for x in sent) and any('RENAME COLUMN ic_pct_sup TO ic_pct_sup_80' in x for x in sent)
    assert any('ADD COLUMN IF NOT EXISTS ub_base_soles FLOAT64' in x for x in sent)
    assert not any('nan' in x.lower().split() or ' nan ' in x for x in sent)                    # regresión: nunca 'RENAME COLUMN nan ...'
    assert bigquery.actualizar_esquema(cli, 'mediciones', modo='renombrar', ejecutar=True).empty
    assert 'ic_pct_inf' not in cli.tablas['base_ton_atribuibles']
    with pytest.raises(ValueError, match='modo'):
        bigquery.actualizar_esquema(cli, 'mediciones', modo='otro')


def test_si_falta_una_tabla_entera_el_ddl_la_crea():
    pytest.importorskip('google.cloud.bigquery')
    t = _tablas_antiguas()
    del t['base_espejos']
    cli = _CliMutable(t)
    assert bigquery.actualizar_esquema(cli, 'mediciones', ejecutar=True).empty and 'base_espejos' in cli.tablas


def test_borrar_corrida_cuenta_primero_y_solo_borra_con_confirmacion():
    pytest.importorskip('google.cloud.bigquery')
    cli = _CliMutable(_tablas_antiguas(), filas={'base_ton_atribuibles': 480, 'base_brechas': 99, 'base_espejos': 400})
    assert bigquery.borrar_corrida(cli, '202609_x', 'mediciones') == {'base_ton_atribuibles': 480, 'base_brechas': 99, 'base_espejos': 400, 'log_metricas_ejecucion': 0}
    assert not any(q.startswith('DELETE') for q in cli.queries)                                    # sin confirmar: solo cuenta
    bigquery.borrar_corrida(cli, '202609_x', 'mediciones', programas=['cpb'], confirmar=True)
    borrados = [q for q in cli.queries if q.startswith('DELETE')]
    assert len(borrados) == 4 and all('WHERE corrida_id = @c AND programa IN UNNEST(@programas)' in q for q in borrados)
    assert not any(q.startswith('DROP') for q in cli.queries)                                      # nunca se borra una tabla
    assert list(bigquery.listar_corridas(cli, 'mediciones').columns) == ['programa', 'corrida_id', 'periodo_reporte', 'version', 'filas']


def test_las_columnas_del_log_del_esquema_son_las_que_escribe_el_motor():
    """Si fila_log cambia y COLS_LOG no, el diagnóstico dejaría pasar una tabla de log incompatible."""
    from mediciones import esquema as esq
    assert list(bigquery.fila_log('c', 'p', '2026-09-01', 'e', 'm', 1.0).keys()) == esq.COLS_LOG


# ── Siembra de la línea base en el dataset del motor (frontera de meses cerrados) ──
def test_sembrar_publica_lo_reportado_en_mediciones_como_el_cierre_con_version_1(tmp_path, monkeypatch):
    xlsx = _surtido(tmp_path)
    cargas, esquema = [], []
    monkeypatch.setattr(bigquery, 'actualizar_esquema', lambda cli, ejecutar=False, **k: esquema.append(ejecutar))
    monkeypatch.setattr(bigquery, 'contar_programa', lambda cli, tabla, p, dataset=None: 0)
    monkeypatch.setattr(bigquery, 'verificar_carga', lambda cli, df, tabla, dataset=None: cargas.append(('verificada', tabla, None)))
    monkeypatch.setattr(bigquery, 'cargar_df', lambda cli, df, tabla, dataset=None: cargas.append((tabla, dataset, df)) or len(df))
    previa = reportado.sembrar_linea_base(tmp_path, 'surtido_motor', xlsx, '2026-09-01', config=CONFIG, verbose=False)
    assert previa['cargado'] is False and set(previa['filas_a_sembrar']) == {T_TON, T_BRE, T_ESP} and not cargas     # sin cargar=True no toca nada
    out = reportado.sembrar_linea_base(tmp_path, 'surtido_motor', xlsx, '2026-09-01', config=CONFIG, cargar=True, cli=object(), verbose=False)
    assert out['cargado'] and out['dataset'] == 'mediciones' and esquema == [True]                                  # vistas al día antes de cargar
    assert [c[0] for c in cargas[:3]] == ['verificada'] * 3                                                          # las 3 se revisan antes de cargar la primera
    cargas = cargas[3:]
    assert {t for t, _, _ in cargas} == {T_TON, T_BRE, T_ESP} and {ds for _, ds, _ in cargas} == {None}              # dataset por defecto = el del motor
    for _, _, df in cargas:
        assert (df.version == 1).all() and (df.config_hash == 'reportado').all() and (pd.to_datetime(df.periodo_reporte) == pd.Timestamp('2026-09-01')).all()


def test_sembrar_se_niega_si_el_programa_ya_tiene_algo_publicado(tmp_path, monkeypatch):
    xlsx = _surtido(tmp_path)
    monkeypatch.setattr(bigquery, 'actualizar_esquema', lambda *a, **k: None)
    monkeypatch.setattr(bigquery, 'contar_programa', lambda *a, **k: 3)
    monkeypatch.setattr(bigquery, 'cargar_df', lambda *a, **k: pytest.fail('no debe cargar'))
    with pytest.raises(RuntimeError, match='solo se siembra una vez'):
        reportado.sembrar_linea_base(tmp_path, 'surtido_motor', xlsx, '2026-09-01', config=CONFIG, cargar=True, cli=object(), verbose=False)


def test_la_vista_vigente_y_sql_vigente_aplican_la_regla_de_frontera():
    ddl = (RAIZ / 'sql' / 'ddl_mediciones.sql').read_text(encoding='utf-8')
    for t in (T_TON, T_BRE, T_ESP):
        bloque = ddl[ddl.index(f'CREATE OR REPLACE VIEW `acpe-dev-uc-sandbox-aa.mediciones.vw_{t}_vigente`'):]
        bloque = bloque[:bloque.index(';')]
        assert 'MAX(version) OVER (PARTITION BY programa, periodo_reporte)' in bloque
        assert "MAX(periodo_reporte) OVER (PARTITION BY programa, DATE_TRUNC(periodo, MONTH), IFNULL(CAST(cosecha_grupo AS STRING), ''))" in bloque and 'periodo_reporte = _rep_max' in bloque
    q = bigquery.sql_vigente(T_TON, hasta='2026-10-01')
    i_filtro, i_version, i_mes = q.index("periodo_reporte <= DATE '2026-10-01'"), q.index('MAX(version)'), q.index('MAX(periodo_reporte)')
    assert i_mes < i_filtro < i_version                    # el corte a un cierre se aplica ANTES de elegir versión y rebanada (lo vigente a esa fecha)


class _Campo:
    def __init__(self, nombre, modo='NULLABLE'):
        self.name, self.mode = nombre, modo


class _CliEsquema:
    def __init__(self, campos, filas=0):
        self.campos, self.filas, self.borradas = campos, filas, []

    def get_table(self, ref):
        return type('T', (), {'schema': self.campos})()

    def query(self, q, **kw):
        return type('Q', (), {'result': lambda s: [{'n': self.filas}]})()

    def delete_table(self, ref, not_found_ok=False):
        self.borradas.append(ref)


def test_verificar_carga_detecta_obligatorias_vacias_y_columnas_sobrantes():
    df = pd.DataFrame({'programa': ['x'], 'valor': [1.0]})
    bigquery.verificar_carga(_CliEsquema([_Campo('programa', 'REQUIRED'), _Campo('valor')]), df, T_TON)        # ok
    with pytest.raises(ValueError, match="obligatorias.*'rol'"):                                               # el caso real: tabla de una versión anterior
        bigquery.verificar_carga(_CliEsquema([_Campo('programa', 'REQUIRED'), _Campo('valor'), _Campo('rol', 'REQUIRED')]), df, T_TON)
    with pytest.raises(ValueError, match='no tiene las columnas'):
        bigquery.verificar_carga(_CliEsquema([_Campo('programa')]), df, T_TON)


def test_sembrar_no_carga_ninguna_tabla_si_alguna_no_se_puede_cargar(tmp_path, monkeypatch):
    xlsx = _surtido(tmp_path)
    monkeypatch.setattr(bigquery, 'actualizar_esquema', lambda *a, **k: None)
    monkeypatch.setattr(bigquery, 'contar_programa', lambda *a, **k: 0)
    def verificar(cli, df, tabla, dataset=None):
        if tabla == T_ESP:
            raise ValueError('obligatorias vacías')
    monkeypatch.setattr(bigquery, 'verificar_carga', verificar)
    monkeypatch.setattr(bigquery, 'cargar_df', lambda *a, **k: pytest.fail('no debe cargar nada'))
    with pytest.raises(ValueError, match='obligatorias'):
        reportado.sembrar_linea_base(tmp_path, 'surtido_motor', xlsx, '2026-09-01', config=CONFIG, cargar=True, cli=object(), verbose=False)


def test_recrear_tablas_solo_si_estan_vacias(monkeypatch):
    llamadas = []
    monkeypatch.setattr(bigquery, 'actualizar_esquema', lambda cli, dataset=None, ejecutar=False, **k: llamadas.append((dataset, ejecutar)) or pd.DataFrame())
    con_filas = _CliEsquema([], filas=10)
    with pytest.raises(RuntimeError, match='no se recrean'):
        bigquery.recrear_tablas_vacias(con_filas, confirmar=True)
    assert con_filas.borradas == []
    vacia = _CliEsquema([], filas=0)
    assert bigquery.recrear_tablas_vacias(vacia) is None and vacia.borradas == []            # sin confirmar solo avisa
    bigquery.recrear_tablas_vacias(vacia, confirmar=True)
    assert len(vacia.borradas) == 3 and llamadas == [('mediciones', True)]


# ── Completar la UB de una línea base publicada sin UB (surtido / Champion) ──
def _ton_publicada(con_ub=False):
    from mediciones.esquema import COLS_TON
    d = pd.DataFrame({'programa': 'surtido_motor', 'cod_cliente': ['0000000001', '0000000002'], 'des_categoria': 'Galletas',
                      'periodo': [pd.Timestamp('2026-09-01').date()] * 2, 'toneladas_actual': [1.0, 2.0], 'toneladas_ant': [0.8, 1.5],
                      'valor_atribuible': [0.1, 0.3], 'flg_estado_yoy': 'normal', 'periodo_reporte': pd.Timestamp('2026-09-01').date(),
                      'version': 1, 'corrida_id': 'reportado_202609', 'config_hash': 'reportado'}).reindex(columns=COLS_TON)
    if con_ub:
        d['ub_atribuible_soles'] = 5.0
    return d


def _ub_falsa(cli, ton, negocios=None, meses_atras=0):
    per = sorted({pd.Timestamp(p).date() for p in ton['periodo']} | {(pd.Timestamp(p) - pd.DateOffset(months=12)).date() for p in ton['periodo']})
    return pd.DataFrame([dict(cod_cliente=c, des_categoria='Galletas', periodo=p, venta_actual_soles=100.0, venta_sin_ratio_soles=0.0,
                              ub_actual_soles=40.0, ton_detalle=1.0) for c in ton['cod_cliente'] for p in per])


def test_completar_ub_mantiene_filas_y_atribuible_y_sube_version(monkeypatch):
    from mediciones import ub
    cargas = []
    monkeypatch.setattr(bigquery, 'leer_rebanada', lambda cli, tabla, p, r, dataset=None: _ton_publicada())
    monkeypatch.setattr(ub, 'consulta_ub', _ub_falsa)
    monkeypatch.setattr(bigquery, 'verificar_carga', lambda cli, df, tabla, dataset=None: cargas.append('verificada'))
    monkeypatch.setattr(bigquery, 'cargar_df', lambda cli, df, tabla, dataset=None: cargas.append(df) or len(df))
    previa = reportado.completar_ub_linea_base(RAIZ, 'surtido_motor', cli=object(), verbose=False)
    assert previa['metodo_ub'] == 'notebook' and previa['cargado'] is False and cargas == []
    out = reportado.completar_ub_linea_base(RAIZ, 'surtido_motor', cli=object(), cargar=True, verbose=False)
    assert out['cargado'] and cargas[0] == 'verificada'
    df = cargas[1]
    assert (df.version == 2).all() and (df.corrida_id == 'reportado_202609_ub').all()
    assert df.valor_atribuible.tolist() == [0.1, 0.3] and df.ub_atribuible_soles.notna().all()     # atribuible intacto, UB completa
    assert (pd.to_datetime(df.periodo_reporte) == pd.Timestamp('2026-09-01')).all()                # mismo cierre


def test_completar_ub_se_niega_si_ya_hay_ub(monkeypatch):
    monkeypatch.setattr(bigquery, 'leer_rebanada', lambda *a, **k: _ton_publicada(con_ub=True))
    with pytest.raises(RuntimeError, match='ya tiene UB'):
        reportado.completar_ub_linea_base(RAIZ, 'surtido_motor', cli=object(), verbose=False)


def _ton_con_ub_inflada():
    d = _ton_publicada(con_ub=True)
    d['flg_estado_yoy'] = ['reactivado', 'normal']
    d['toneladas_ant'] = [0.0, 0.8]
    d['toneladas_actual'] = [2.0, 2.0]
    d['valor_atribuible'] = [0.0, 0.3]
    d['ub_atribuible_soles'] = [40.0, 5.0]               # el reactivado traía UB con 0 toneladas atribuibles (el error de la versión 2)
    d['version'] = 2
    d['corrida_id'] = 'reportado_202609_ub'
    return d


def test_reemplazar_republica_la_ub_corregida_como_version_siguiente(monkeypatch):
    from mediciones import ub
    cargas = []
    monkeypatch.setattr(bigquery, 'leer_rebanada', lambda *a, **k: _ton_con_ub_inflada())
    monkeypatch.setattr(ub, 'consulta_ub', _ub_falsa)
    monkeypatch.setattr(bigquery, 'verificar_carga', lambda *a, **k: None)
    monkeypatch.setattr(bigquery, 'cargar_df', lambda cli, df, tabla, dataset=None: cargas.append(df) or len(df))
    with pytest.raises(RuntimeError, match='reemplazar=True'):
        reportado.completar_ub_linea_base(RAIZ, 'surtido_motor', cli=object(), verbose=False)
    out = reportado.completar_ub_linea_base(RAIZ, 'surtido_motor', cli=object(), cargar=True, reemplazar=True, verbose=False)
    df = cargas[0]
    assert out['ub_publicada_antes'] == 45.0 and (df.version == 3).all() and (df.corrida_id == 'reportado_202609_ub3').all()
    assert df.valor_atribuible.tolist() == [0.0, 0.3]                                            # atribuible intacto
    reactivado = df[df.cod_cliente == '0000000001'].iloc[0]
    assert reactivado.ub_atribuible_soles == 0.0                                                 # sin toneladas atribuidas, sin UB atribuible
    assert out['cobertura']['filas_ub_con_atribuible_cero'] == 0


def test_la_cobertura_detecta_ub_sin_atribuible():
    from mediciones import ub
    d = _ton_publicada()
    d['valor_atribuible'] = [0.0, 0.3]
    u = _ub_falsa(None, d)
    con, cob = ub.aplicar_ub(d, u, metodo='actual')                                              # método 'actual': UB = ub_actual x valor/actual -> 0 si valor = 0
    assert cob['filas_ub_con_atribuible_cero'] == 0
