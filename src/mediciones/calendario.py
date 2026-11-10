"""
Calendario de cosechas (surtido / Impulso): una sola fuente para los grupos, en vez de fechas escritas en varios lugares.

Cada grupo: meses de ALTAS (altas_desde..altas_hasta) y COSECHA agrupada (el mes desde el que se mide; los meses anteriores de quien entró antes
son retroactivos, con la brecha de los meses 0-2 de la cosecha). Un grupo con `preliminar: true` (el G04) tiene dos etapas:
  * PRELIMINAR (cierres anteriores a su cosecha): se ancla en `altas_desde` como si esa fuera su cosecha; las altas de cada mes se emparejan de forma
    incremental y se miden directo desde su entrada. En las salidas se rotula con su cosecha definitiva (misma identidad del grupo).
  * DEFINITIVO (desde el cierre de su cosecha): se descartan los pares preliminares, se empareja completo con la cosecha definitiva y se congela.

La fecha "interna" de un grupo en un cierre es la que usa la metodología (mes 0 del panel): el ancla en etapa preliminar, la cosecha en definitiva.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

POR_DEFECTO = [
    {'grupo': 'G01', 'nombre': 'GRUPO 01 (Feb-Abr)', 'altas_desde': '2026-02-01', 'altas_hasta': '2026-04-01', 'cosecha': '2026-04-01'},
    {'grupo': 'G02', 'nombre': 'GRUPO 02 (May-Jun)', 'altas_desde': '2026-05-01', 'altas_hasta': '2026-06-01', 'cosecha': '2026-06-01'},
    {'grupo': 'G03', 'nombre': 'GRUPO 03 (Jul-Ago)', 'altas_desde': '2026-07-01', 'altas_hasta': '2026-09-01', 'cosecha': '2026-08-01'},
    {'grupo': 'G04', 'nombre': 'GRUPO 04 (Oct-Dic)', 'altas_desde': '2026-10-01', 'altas_hasta': '2026-12-01', 'cosecha': '2026-12-01',
     'preliminar': True},
]


def _m(x):
    return pd.Timestamp(x).to_period('M').to_timestamp()


@dataclass(frozen=True)
class Grupo:
    grupo: str
    nombre: str
    altas_desde: pd.Timestamp
    altas_hasta: pd.Timestamp
    cosecha: pd.Timestamp
    preliminar: bool = False

    def preliminar_en(self, cierre):
        return self.preliminar and _m(cierre) < self.cosecha

    def interna(self, cierre):
        """Mes 0 que usa la metodología en este cierre."""
        return self.altas_desde if self.preliminar_en(cierre) else self.cosecha


class Calendario:
    def __init__(self, filas=None):
        filas = filas or POR_DEFECTO
        self.grupos = [Grupo(f['grupo'], f.get('nombre', f['grupo']), _m(f['altas_desde']), _m(f['altas_hasta']), _m(f['cosecha']),
                             bool(f.get('preliminar', False))) for f in filas]
        self.grupos.sort(key=lambda g: g.altas_desde)
        for a, b in zip(self.grupos, self.grupos[1:]):
            if b.altas_desde <= a.altas_hasta:
                raise ValueError(f'calendario de cosechas: las altas de {a.grupo} y {b.grupo} se superponen')
        for g in self.grupos:
            if not (g.altas_desde <= g.altas_hasta) or g.cosecha < g.altas_desde:
                raise ValueError(f'calendario de cosechas: {g.grupo} tiene fechas inconsistentes')

    # ── lo que usa el programa en un cierre ──
    def cosechas(self, cierre):
        """Fechas internas de todos los grupos que ya existen en este cierre (los que tienen altas hasta el cierre)."""
        return [g.interna(cierre) for g in self.grupos if g.altas_desde <= _m(cierre)]

    def nombres(self, cierre):
        """fecha interna -> nombre (en etapa preliminar se marca); incluye la cosecha definitiva para rotular las salidas."""
        out = {}
        for g in self.grupos:
            out[g.cosecha] = g.nombre
            if g.preliminar_en(cierre):
                out[g.altas_desde] = f'{g.nombre} preliminar'
        return out

    def abierta(self, cierre):
        """Fecha interna del grupo que recibe altas en este cierre (None si ninguno)."""
        c = _m(cierre)
        for g in self.grupos:
            if g.altas_desde <= c <= g.altas_hasta:
                return g.interna(cierre)
        return None

    def a_rehacer(self, cierre):
        """Grupos en etapa definitiva cuya cosecha ya llegó: se emparejan completos si aún no tienen pares definitivos."""
        return [g.cosecha for g in self.grupos if g.preliminar and _m(cierre) >= g.cosecha]

    def a_descartar(self, cierre):
        """Fechas internas preliminares cuyos pares congelados se descartan (el grupo ya pasó a definitivo)."""
        return [g.altas_desde for g in self.grupos if g.preliminar and _m(cierre) >= g.cosecha and g.altas_desde != g.cosecha]

    def alias(self, cierre):
        """Para las salidas: fecha interna preliminar -> cosecha definitiva (el grupo conserva su identidad)."""
        return {g.altas_desde: g.cosecha for g in self.grupos if g.preliminar_en(cierre) and g.altas_desde != g.cosecha}

    def asignar(self, min_periodo, cierre):
        """Mes de entrada al piloto -> fecha interna (texto 'AAAA-MM-DD'); NA si cae fuera de todo grupo."""
        mp = pd.to_datetime(min_periodo).dt.to_period('M').dt.to_timestamp()
        out = pd.Series(pd.NA, index=mp.index, dtype='object')
        for g in self.grupos:
            out[(mp >= g.altas_desde) & (mp <= g.altas_hasta)] = g.interna(cierre).strftime('%Y-%m-%d')
        return out

    def rangos_12m(self, cierre):
        """fecha interna (texto) -> (inicio, fin) de los 12 meses previos a la cosecha."""
        return {c.strftime('%Y-%m-%d'): (c - pd.DateOffset(months=12), c - pd.DateOffset(months=1)) for c in self.cosechas(cierre)}

    def sql_case(self, columna, cierre):
        """El mismo mapeo en SQL (para la consulta de Inka Chips)."""
        ramas = [f"      WHEN {columna} BETWEEN DATE '{g.altas_desde:%Y-%m-%d}' AND DATE '{g.altas_hasta:%Y-%m-%d}' "
                 f"THEN DATE '{g.interna(cierre):%Y-%m-%d}'   -- {g.grupo}" for g in self.grupos]
        return 'CASE\n' + '\n'.join(ramas) + '\n    END'

    def grupo_de(self, fecha_interna_o_cosecha):
        f = _m(fecha_interna_o_cosecha)
        for g in self.grupos:
            if f in (g.altas_desde, g.cosecha):
                return g
        return None
