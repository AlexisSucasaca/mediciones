"""
Atribución de toneladas: estado YoY del cliente-mes, crecimiento atribuible acotado a la banda del IC y valor atribuible.

Una sola función pública, `atribuir`, con un parámetro explícito `regla_nuevo` que declara cómo se trata al cliente que NO compró ese mes del año anterior
(`toneladas_ant == 0`) y compra ahora. Las dos reglas reproducen EXACTAMENTE lo que hacían los notebooks (ver tests/legado/ y test_atribucion.py):

  'sin_compra_mes_anterior'  (CPB, CPM)      antiguo | nuevo. Nuevo = toneladas_ant == 0 -> su atribuible es TODO su volumen.
  'cliente_creado'           (surtido, Champion)
                             normal | no_existia_periodo_pasado | reactivado | sin_compra_ambos. Entre los que no compraron el mes del año anterior:
                               * no_existia_periodo_pasado: el cliente se creó DESPUÉS de ese mes (fec_creacion > periodo_ant) -> atribuible = todo su volumen.
                               * reactivado: ya existía -> su atribuible es 0 (el crecimiento tope del IC se multiplica por toneladas_ant, que es 0).
                               * fec_creacion desconocida cuenta como reactivado (no se asume que no existía).

Qué banda usa cada llamada (IC 80 o 95 %) lo decide quien llama: aquí solo se acota a [ic_inf, ic_sup]. Unificar reglas (p. ej. una sola para todos) es una decisión de negocio que
cambia números; este módulo solo concentra el cálculo para que esa decisión sea un cambio de un parámetro y no de cuatro bloques de código.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGLAS_NUEVO = ('cliente_creado', 'sin_compra_mes_anterior')
ESTADOS = {'cliente_creado': ('normal', 'no_existia_periodo_pasado', 'reactivado', 'sin_compra_ambos'),
           'sin_compra_mes_anterior': ('antiguo', 'nuevo')}


def _serie(x, indice):
    return x if isinstance(x, pd.Series) else pd.Series(x, index=indice)


def estado_yoy(regla_nuevo, toneladas_ant, toneladas_actual=None, fec_creacion=None, periodo_ant=None):
    """Estado YoY de cada cliente-mes según la regla. Devuelve una Serie de texto con el índice de `toneladas_ant`."""
    _validar_regla(regla_nuevo)
    ant = toneladas_ant
    if regla_nuevo == 'sin_compra_mes_anterior':
        return pd.Series(np.where(ant == 0, 'nuevo', 'antiguo'), index=ant.index)
    if toneladas_actual is None or fec_creacion is None or periodo_ant is None:
        raise ValueError("regla 'cliente_creado' requiere toneladas_actual, fec_creacion y periodo_ant")
    act = toneladas_actual
    no_existia = (ant == 0) & (act > 0) & fec_creacion.notna() & (fec_creacion > periodo_ant)
    reactivado = (ant == 0) & (act > 0) & (~no_existia)
    sigue_cero = (ant == 0) & (act == 0)
    return pd.Series(np.select([no_existia, reactivado, sigue_cero], ['no_existia_periodo_pasado', 'reactivado', 'sin_compra_ambos'], default='normal'), index=ant.index)


def atribuir(regla_nuevo, toneladas_ant, toneladas_actual, ic_inf, ic_sup, *, fec_creacion=None, periodo_ant=None, estado=None, yoy_pct_raw=None):
    """
    Devuelve un DataFrame (mismo índice que `toneladas_ant`) con:
      flg_estado_yoy · yoy_pct_raw · crec_atrib_pct · ton_atrib · valor_atribuible · flg_sin_banda_ic

    `estado` y `yoy_pct_raw` se pueden pasar ya calculados (p. ej. las vistas ya traen el estado y el YoY de la base completa);
    si no, se calculan aquí con la regla. `ic_inf`/`ic_sup`: banda del IC de la celda (NaN = sin banda).
    """
    _validar_regla(regla_nuevo)
    idx = toneladas_ant.index
    ant, act = toneladas_ant, _serie(toneladas_actual, idx)
    ic_inf, ic_sup = _serie(ic_inf, idx), _serie(ic_sup, idx)
    if estado is None:
        estado = estado_yoy(regla_nuevo, ant, act, fec_creacion, periodo_ant)
    estado = _serie(estado, idx)
    activo = act > 0
    sin_banda = ic_inf.isna() | ic_sup.isna()

    if regla_nuevo == 'sin_compra_mes_anterior':
        yoy = _serie(yoy_pct_raw, idx) if yoy_pct_raw is not None else pd.Series(np.where(ant > 0, 100 * (act - ant) / ant, np.nan), index=idx)
        es_ant = estado == 'antiguo'
        es_nue = ~es_ant
        crec = yoy.clip(lower=ic_inf, upper=ic_sup)
        crec.loc[sin_banda | es_nue] = np.nan
        ton_atrib = ant * crec / 100
        valor = valor_atribuible(regla_nuevo, estado, act, ton_atrib)
        return pd.DataFrame({'flg_estado_yoy': estado, 'yoy_pct_raw': yoy, 'crec_atrib_pct': crec, 'ton_atrib': ton_atrib,
                             'valor_atribuible': valor, 'flg_sin_banda_ic': es_ant & activo & sin_banda}, index=idx)

    no_existia = estado == 'no_existia_periodo_pasado'
    reactivado = estado == 'reactivado'
    if yoy_pct_raw is None:
        yoy = pd.Series(np.where(ant > 0, 100 * (act - ant) / ant, np.where(reactivado, np.inf, np.where(no_existia, np.nan, 0.0))), index=idx)
    else:
        yoy = _serie(yoy_pct_raw, idx)
    crec = yoy.replace(np.inf, np.nan).clip(lower=ic_inf, upper=ic_sup)
    crec.loc[reactivado & ic_sup.notna()] = ic_sup[reactivado & ic_sup.notna()].clip(lower=0)      # reactivado: tope = IC superior (>= 0)...
    crec.loc[reactivado & ic_sup.isna()] = np.nan
    crec.loc[no_existia] = np.nan                                                                   # no_existia: nunca se acota (se atribuye todo abajo)
    ton_atrib = pd.Series(np.where(ant > 0, ant * crec / 100, 0.0), index=idx)                      # ...pero se multiplica por toneladas_ant (= 0): atribuible 0
    ton_atrib.loc[crec.isna()] = np.nan
    ton_atrib.loc[no_existia] = np.nan
    valor = valor_atribuible(regla_nuevo, estado, act, ton_atrib)
    return pd.DataFrame({'flg_estado_yoy': estado, 'yoy_pct_raw': yoy, 'crec_atrib_pct': crec, 'ton_atrib': ton_atrib,
                         'valor_atribuible': valor, 'flg_sin_banda_ic': False}, index=idx)


def valor_atribuible(regla_nuevo, estado, toneladas_actual, ton_atrib):
    """
    Atribuible final del cliente-mes a partir del estado, las toneladas actuales y las toneladas atribuibles ya acotadas al IC.
      cliente_creado:           no_existia_periodo_pasado = toneladas_actual; con compra = ton_atrib; sin compra = 0.
      sin_compra_mes_anterior:  nuevo = toneladas_actual; antiguo sin compra = 0 (caer a cero no es crecer menos de lo esperado); antiguo con compra = ton_atrib.
    """
    _validar_regla(regla_nuevo)
    activo = toneladas_actual > 0
    if regla_nuevo == 'cliente_creado':
        return np.select([estado == 'no_existia_periodo_pasado', activo], [toneladas_actual, ton_atrib], default=0.0)
    es_ant = estado == 'antiguo'
    return np.select([~es_ant, es_ant & ~activo, es_ant & activo], [toneladas_actual, 0.0, ton_atrib], default=np.nan)


def _validar_regla(regla):
    if regla not in REGLAS_NUEVO:
        raise ValueError(f'regla_nuevo debe ser una de {REGLAS_NUEVO} (recibí {regla!r})')
