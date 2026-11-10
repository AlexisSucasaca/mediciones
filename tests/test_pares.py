import pandas as pd

from mediciones import pares


def _esp(periodos=('2026-08-01',), n=4, control_base=900000):
    f = []
    for per in periodos:
        for ct in ('Galletas', 'Cereales'):
            for i in range(1, n + 1):
                f.append({'periodo': pd.Timestamp(per), 'cosecha_grupo': pd.NaT, 'des_categoria': ct, 'cod_cliente_accion': f'{i:010d}',
                          'cod_cliente_espejo': f'{control_base + i:010d}', 'nivel_match': 1})
    return pd.DataFrame(f)


def _csv(ruta, esp):
    d = esp.rename(columns={'des_categoria': 'categoria', 'cod_cliente_accion': 'cod_accion', 'cod_cliente_espejo': 'cod_control'})
    d['periodo'] = pd.to_datetime(d['periodo']).dt.strftime('%Y-%m-%d')
    d[['periodo', 'categoria', 'cod_accion', 'cod_control']].to_csv(ruta, index=False)


def test_sin_publicado_es_primera_carga(tmp_path):
    v = pares.verificar('mensual', tmp_path / 'no_existe.csv', pares.publicados_normalizados(None))
    assert v['estado'] == 'sin_publicado'


def test_ok_difiere_e_incompleto(tmp_path):
    esp, ruta = _esp(('2026-07-01', '2026-08-01')), tmp_path / 'p.csv'
    pub = pares.publicados_normalizados(esp)
    _csv(ruta, esp)
    assert pares.verificar('mensual', ruta, pub)['estado'] == 'ok'
    mal = esp.copy(); mal.loc[mal.index[:3], 'cod_cliente_espejo'] = '0000000042'
    _csv(ruta, mal)
    v = pares.verificar('mensual', ruta, pub)
    assert v['estado'] == 'difiere' and v['n_difieren'] == 3
    _csv(ruta, esp.iloc[:-5])
    v = pares.verificar('mensual', ruta, pub)
    assert v['estado'] == 'incompleto' and v['n_faltan_local'] == 5
    _csv(ruta, pd.concat([esp, _esp(('2026-09-01',))]))                    # pares de meses aún no publicados no molestan
    assert pares.verificar('mensual', ruta, pub)['estado'] == 'ok'


def test_restaurar_desde_lo_publicado_deja_un_archivo_que_el_motor_puede_leer(tmp_path):
    esp, ruta = _esp(('2026-07-01', '2026-08-01')), tmp_path / 'sub' / 'cpb_pares_fijos.csv'
    pub = pares.publicados_normalizados(esp)
    assert pares.verificar('mensual', ruta, pub)['estado'] == 'restaurable'
    assert pares.restaurar_mensual(pub, ruta) == len(esp)
    # exactamente como lo lee la metodología mensual
    pf = pd.read_csv(ruta, dtype={'cod_accion': str, 'cod_control': str}, parse_dates=['periodo'])
    celdas = {k: g for k, g in pf.groupby(['periodo', 'categoria'])}
    assert len(celdas) == 4 and pf.cod_accion.str.len().eq(10).all()
    assert pares.verificar('mensual', ruta, pub)['estado'] == 'ok'


def test_cosechas_verifica_pero_no_restaura(tmp_path):
    esp = _esp(('2026-08-01',)); esp['cosecha_grupo'] = pd.Timestamp('2026-08-01')
    pub = pares.publicados_normalizados(esp)
    ruta = tmp_path / 'df_matches.csv'
    assert pares.verificar('cosechas', ruta, pub)['estado'] == 'faltante_no_restaurable'
    m = esp.rename(columns={'cosecha_grupo': 'cosecha', 'cod_cliente_accion': 'cod_accion', 'cod_cliente_espejo': 'cod_control'})
    m[['cosecha', 'des_categoria', 'cod_accion', 'cod_control']].to_csv(ruta, index=False)
    assert pares.verificar('cosechas', ruta, pub)['estado'] == 'ok'
    m.loc[m.index[0], 'cod_control'] = '0000000042'
    m[['cosecha', 'des_categoria', 'cod_accion', 'cod_control']].to_csv(ruta, index=False)
    assert pares.verificar('cosechas', ruta, pub)['estado'] == 'difiere'
