"""
Tabla de equivalencias de códigos de cliente (cambios de código): una fila por cambio, acumulativa, nunca se borra.

Archivo `equivalencias_codigos.csv` en la carpeta de trabajo del programa (data/<programa>/), columnas:
    cod_cliente_nuevo, cod_cliente_antiguo, vigente_desde (AAAA-MM-01), origen (manual / legacy / geolocalizacion), observacion
Convención (la misma que la homologación por geolocalización de Champion): el historial se unifica en el código MÁS ANTIGUO, así un cliente que
cambia de código conserva su par y su serie. Las cadenas (A -> B -> C) se resuelven hasta el más antiguo; un ciclo es un error.
"""
import os

import pandas as pd

COLUMNAS = ['cod_cliente_nuevo', 'cod_cliente_antiguo', 'vigente_desde', 'origen', 'observacion']


def _z10(s):
    return s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(10)


def leer_equivalencias(ruta='equivalencias_codigos.csv'):
    """Devuelve {codigo_nuevo: codigo_mas_antiguo}. Si el archivo no existe, {} (no cambia nada)."""
    if not os.path.exists(ruta):
        return {}
    d = pd.read_csv(ruta, dtype=str)
    faltan = [c for c in ('cod_cliente_nuevo', 'cod_cliente_antiguo') if c not in d.columns]
    if faltan:
        raise ValueError(f'{ruta}: faltan las columnas {faltan} (se esperan {COLUMNAS})')
    d = d.dropna(subset=['cod_cliente_nuevo', 'cod_cliente_antiguo'])
    nuevo, antiguo = _z10(d['cod_cliente_nuevo']), _z10(d['cod_cliente_antiguo'])
    directo = {n: a for n, a in zip(nuevo, antiguo) if n != a}
    dup = nuevo[nuevo.duplicated()].unique()
    if len(dup) and any(len(set(antiguo[nuevo == x])) > 1 for x in dup):
        raise ValueError(f'{ruta}: un mismo código nuevo apunta a más de un antiguo: {sorted(dup)[:5]}')
    final = {}
    for n in directo:
        visto, x = {n}, directo[n]
        while x in directo:
            if x in visto:
                raise ValueError(f'{ruta}: ciclo de equivalencias que pasa por {x}')
            visto.add(x)
            x = directo[x]
        final[n] = x
    return final
