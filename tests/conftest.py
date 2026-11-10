import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from mediciones.frontera import vigente_local
import pytest

from mediciones.esquema import CLAVE, T_BRE, T_ESP, T_TON, VALOR

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def raiz_tmp(tmp_path):
    """Raíz de proyecto temporal con la config real."""
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'programas.yaml').write_text((RAIZ / 'config' / 'programas.yaml').read_text(encoding='utf-8'), encoding='utf-8')
    return tmp_path


def _h(cl, ct, per):
    return (int(cl) * 31 + len(ct) * 7 + int(pd.Timestamp(per).strftime('%Y%m'))) % 100


def construir_crudas(periodos, n_clientes=30, seed=0, delta=None, extra_periodo_clientes=None, sin_espejo_frac=0.0):
    """Bases crudas con el esquema estándar que deja un motor (antes de normalizar)."""
    ids = [f'{i:010d}' for i in range(1, n_clientes + 1)]
    filas, pares = [], []
    for cl in ids:
        for k, ct in enumerate(('Galletas', 'Cereales')):
            for per in periodos:
                # valores deterministas por (cliente, categoría, periodo): el mismo dato sale igual aunque cambie la lista de meses
                rng = np.random.default_rng([seed, int(cl), k, int(pd.Timestamp(per).strftime('%Y%m'))])
                ant = float(rng.uniform(0, 2)) if rng.random() > .15 else 0.0
                act = float(rng.uniform(0, 2.5)) if rng.random() > .2 else 0.0
                antiguo = ant > 0
                val = (ant * 0.1 if act > 0 else 0.0) if antiguo else act
                filas.append(dict(
                    cod_cliente=cl, cod_cliente_dex=cl, des_categoria=ct, des_oficina='Sur', des_segmento_transaccional='2 Oro',
                    nom_compania='DEX', flg_cliente_perfecto=np.nan, cosecha_grupo=np.nan, periodo=pd.Timestamp(per), mes_rel=np.nan,
                    toneladas_ant=ant, toneladas_actual=act, yoy_pct_raw=np.nan, ic_pct_inf_80=5., ic_pct_sup_80=15.,
                    flg_estado_yoy='x', crec_atrib_80_pct=10., ton_atrib_80=ant * .1, flg_estimado_retroactivo=False,
                    valor_atribuible=val))
                pares.append(dict(par_id=f'{per}|{ct}|{cl}', cod_cliente_accion_dex=cl, nom_compania_accion='DEX',
                                  flg_cliente_perfecto=np.nan, cosecha_grupo=np.nan, cosecha_label=None, des_categoria=ct,
                                  des_oficina='Sur', des_segmento_transaccional='2 Oro', periodo=pd.Timestamp(per), mes_rel=np.nan,
                                  pre_accion=ant * 1000, post_accion=act * 1000, pre_control=900., post_control=950.,
                                  did_kg=(act - ant) * 1000 - 50, flg_estimado_retroactivo=False, brecha_pct_celda=12., reportable=True))
    ton, bre = pd.DataFrame(filas), pd.DataFrame(pares)
    bre['cod_cliente_control'] = [f'{900000 + int(c):010d}' for c in bre.cod_cliente_accion_dex]
    if sin_espejo_frac:                                  # clientes de acción que no quedaron emparejados en esa celda
        sin = bre.apply(lambda r: _h(r.cod_cliente_accion_dex, r.des_categoria, r.periodo) < 100 * sin_espejo_frac, axis=1)
        bre = bre[~sin].reset_index(drop=True)
    if delta:                                              # delta: {(cod_cliente, categoria, periodo): nuevo_valor_atribuible}
        for (cl, ct, per), v in delta.items():
            ton.loc[(ton.cod_cliente == cl) & (ton.des_categoria == ct) & (ton.periodo == pd.Timestamp(per)), 'valor_atribuible'] = v
    return ton, bre


def construir_espejos(ton, bre):
    """Base de espejos cruda: cada fila de acción con el control de su par (o sin espejo y el motivo)."""
    e = (ton[['cod_cliente', 'des_categoria', 'periodo', 'cosecha_grupo', 'mes_rel', 'des_oficina', 'nom_compania',
              'des_segmento_transaccional', 'flg_estimado_retroactivo']].rename(columns={'cod_cliente': 'cod_cliente_accion'}))
    pares = bre[['cod_cliente_accion_dex', 'des_categoria', 'periodo', 'cod_cliente_control', 'par_id']].rename(
        columns={'cod_cliente_accion_dex': 'cod_cliente_accion', 'cod_cliente_control': 'cod_cliente_espejo'})
    e = e.merge(pares, on=['cod_cliente_accion', 'des_categoria', 'periodo'], how='left')
    e['motivo_sin_espejo'] = [None if pd.notna(x) else 'sin_pareja' for x in e.cod_cliente_espejo]
    e['des_oficina_espejo'], e['nom_compania_espejo'], e['des_segmento_espejo'] = 'Sur', 'DEX', '2 Oro'
    e['nivel_match'] = 1
    return e


class ServiciosMemoria:
    """Reemplazo en memoria de ServiciosBQ: guarda las tablas como lo haría BigQuery (append-only + vista vigente)."""

    def __init__(self):
        self.t = {T_TON: pd.DataFrame(), T_BRE: pd.DataFrame(), T_ESP: pd.DataFrame()}
        self.log = []
        self.cargas = []

    def _vigente(self, tabla, programa):
        d = self.t[tabla]
        if d.empty:
            return None
        d = d[d.programa == programa]
        if d.empty:
            return None
        return vigente_local(d)                      # misma regla que la vista vw_*_vigente (mediciones/frontera.py)

    def leer_publicado(self, tabla, programa):
        v = self._vigente(tabla, programa)
        return None if v is None else v[CLAVE[tabla] + [VALOR[tabla], 'periodo_reporte', 'version', 'config_hash']].copy()

    def leer_pares(self, programa):
        v = self._vigente(T_ESP, programa)
        if v is None:
            return pd.DataFrame(columns=['periodo', 'cosecha_grupo', 'des_categoria', 'cod_cliente_accion', 'cod_cliente_espejo', 'nivel_match'])
        v = v[v.cod_cliente_espejo.notna()]
        return v[['periodo', 'cosecha_grupo', 'des_categoria', 'cod_cliente_accion', 'cod_cliente_espejo', 'nivel_match']].copy()

    def ultima_version(self, tabla, programa, periodo_reporte):
        d = self.t[tabla]
        if d.empty:
            return None
        d = d[(d.programa == programa) & (d.periodo_reporte == periodo_reporte)]
        return None if d.empty else int(d.version.max())

    def cargar(self, df, tabla):
        self.t[tabla] = pd.concat([self.t[tabla], df], ignore_index=True)
        self.cargas.append((tabla, len(df), int(df.version.iloc[0])))
        return len(df)

    def consulta_ub(self, ton, negocios, meses_atras=0):
        self.meses_atras_ub = meses_atras
        df = pd.DataFrame({'periodo': ton.periodo, 'cod_cliente': ton.cod_cliente, 'des_categoria': ton.des_categoria,
                           'venta_actual_soles': ton.toneladas_actual * 9000, 'venta_sin_ratio_soles': 0.0,
                           'ub_actual_soles': ton.toneladas_actual * 2500, 'ton_detalle': ton.toneladas_actual})
        if meses_atras:                                                   # los meses de hace un año: la UB base (en la base real, de la misma venta)
            ant = pd.DataFrame({'periodo': (pd.to_datetime(ton.periodo) - pd.DateOffset(months=meses_atras)).dt.date, 'cod_cliente': ton.cod_cliente,
                                'des_categoria': ton.des_categoria, 'venta_actual_soles': ton.toneladas_ant * 9000, 'venta_sin_ratio_soles': 0.0,
                                'ub_actual_soles': ton.toneladas_ant * 2500, 'ton_detalle': ton.toneladas_ant})
            df = pd.concat([df, ant], ignore_index=True)
        return df

    def registrar_log(self, filas):
        self.log += filas

    def exportar(self, programa, mes, carpeta):
        return []
