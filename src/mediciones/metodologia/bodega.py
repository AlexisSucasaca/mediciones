"""
Exclusión de clientes con surtido de bodega (`surtido_ideal_bodeg`), CONGELADA POR CIERRE.

El script de surtido quita de la base de ventas a todo cliente que esté en `surtido_ideal_bodeg` (seis compañías DEX de Sur, Centro y Selva, desde 2026-04-01).
Esa tabla crece con cada periodo que se carga. Sin tope superior, el resultado de un cierre dependía de CUÁNDO se corría: los clientes cargados después del cierre
(por ejemplo los del mes siguiente) también quedaban excluidos y, como los pares están congelados, sus controles aparecían con volumen cero e inflaban el DiD.

Ahora la lista cuenta solo los periodos <= el cierre, igual que el resto de las consultas del script (todas acotadas a MES_CIERRE y con un snapshot de caché por cierre).
`PERIODO_MAX_BODEG` (en programas/surtido.py) permite fijar un corte anterior, p. ej. para reproducir un reportado generado antes de que se cargara el último periodo.
"""
import pandas as pd

TABLA_BODEG = 'acpe-dev-uc-sandbox-aa.dev.surtido_ideal_bodeg'
PERIODO_DESDE_BODEG = '2026-04-01'
COMPANIAS_BODEG = ('DEX APURIMAC S.A.C.', 'DEX MINALI SOCIEDAD ANONIMA CERRADA', 'ALMA DISTRIBUCIONES E.I.R.L.', 'RTDEX E.I.R.L.',
                   'DISTRIBUIDORA MADEX SOCIEDAD ANONIMA CERRADA', 'SELVA DEX S.A.C')


def periodo_max_bodega(mes_cierre, override=None):
    """Último periodo de `surtido_ideal_bodeg` que cuenta en este cierre ('AAAA-MM-DD'). Sin override es el propio cierre; un override solo puede ser anterior o igual."""
    cierre = pd.Timestamp(mes_cierre)
    if override is None:
        return cierre.strftime('%Y-%m-%d')
    p = pd.Timestamp(override)
    if p.day != 1:
        raise ValueError(f'PERIODO_MAX_BODEG debe ser el día 1 del mes (recibí {p.date()})')
    if p > cierre:
        raise ValueError(f'PERIODO_MAX_BODEG ({p.date()}) no puede ser posterior al cierre ({cierre.date()}): la lista está congelada al cierre, no mira el futuro')
    if p < pd.Timestamp(PERIODO_DESDE_BODEG):
        raise ValueError(f'PERIODO_MAX_BODEG ({p.date()}) es anterior al primer periodo de la tabla ({PERIODO_DESDE_BODEG}): la lista quedaría vacía')
    return p.strftime('%Y-%m-%d')


def sql_surtido_bodeg(periodo_max, periodo_desde=PERIODO_DESDE_BODEG, companias=COMPANIAS_BODEG, tabla=TABLA_BODEG):
    """Consulta de la lista de exclusión: clientes de esas compañías con surtido de bodega entre `periodo_desde` y `periodo_max`."""
    lista = ',\n                    '.join(f"'{c}'" for c in companias)
    return (f"\nSELECT customer_id cod_cliente_dex\nFROM `{tabla}`\nWHERE periodo >='{periodo_desde}'\nAND periodo <='{periodo_max}'\n"
            f"AND nom_compania IN ({lista})\nGROUP BY ALL;\n")
