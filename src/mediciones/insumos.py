"""
Capa de insumos: toda consulta a BigQuery de un programa pasa por aquí, se guarda como snapshot y se registra con su hash.

Los programas leen con `pd.read_gbq`; la capa lo reemplaza mientras se ejecutan. Cada consulta se identifica por el hash de su SQL (+ proyecto), de modo que
**un cambio en la consulta nunca reutiliza un snapshot viejo** (el problema del caché CSV por nombre y mes). Cada resultado lleva un hash de contenido que no depende del orden de filas.

Modos:
  consultar   siempre consulta BigQuery y guarda un snapshot (lo habitual)
  reutilizar  usa el último snapshot de esa consulta si existe; si no, consulta y lo guarda
  reproducir  SOLO snapshots (ninguna consulta): falla si falta alguno. Con `contenidos` usa exactamente los de una corrida anterior
  desactivado no interviene (cada programa usa su propio acceso/caché)
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODOS = ('consultar', 'reutilizar', 'reproducir', 'desactivado')


class InsumoNoDisponible(RuntimeError):
    pass


def normalizar_sql(q):
    return ' '.join(str(q).split())


def clave_consulta(query, project_id=None):
    return hashlib.sha256((normalizar_sql(query) + '|' + str(project_id or '')).encode('utf-8')).hexdigest()[:16]


def hash_df(df):
    """Hash de contenido independiente del orden de filas: nombres de columna + nº de filas + suma (módulo 2^64) de los hashes de fila."""
    cab = '|'.join(map(str, df.columns)) + f'|{len(df)}'
    if len(df) == 0:
        return hashlib.sha256(cab.encode()).hexdigest()[:16]
    try:
        h = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype='uint64')
    except TypeError:                                    # columnas con objetos no hasheables
        h = pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy(dtype='uint64')
    return hashlib.sha256(f'{cab}|{int(h.sum(dtype=np.uint64))}'.encode()).hexdigest()[:16]


class CapaInsumos:
    def __init__(self, modo, directorio, contenidos=None):
        if modo not in MODOS:
            raise ValueError(f'modo de insumos inválido: {modo!r} (opciones: {MODOS})')
        self.modo, self.dir, self.contenidos = modo, Path(directorio), (contenidos or {})
        self.entradas = []

    # ── snapshots ──
    def _archivos(self, key):
        return sorted(self.dir.glob(f'{key}__*.parquet'), key=lambda p: p.stat().st_mtime) + \
               sorted(self.dir.glob(f'{key}__*.pkl'), key=lambda p: p.stat().st_mtime)

    def _guardar(self, key, contenido, df):
        self.dir.mkdir(parents=True, exist_ok=True)
        ruta = self.dir / f'{key}__{contenido}.parquet'
        if ruta.exists() or (self.dir / f'{key}__{contenido}.pkl').exists():
            return ruta
        dtypes = {c: str(t) for c, t in df.dtypes.items()}
        try:
            df.to_parquet(ruta, index=False)
            (self.dir / f'{key}__{contenido}.dtypes.json').write_text(json.dumps(dtypes), encoding='utf-8')
        except Exception:                                # columnas de tipo mixto que parquet no admite
            ruta = self.dir / f'{key}__{contenido}.pkl'
            df.to_pickle(ruta)
        return ruta

    def _leer_snapshot(self, ruta):
        if ruta.suffix == '.pkl':
            return pd.read_pickle(ruta)
        df = pd.read_parquet(ruta)
        side = ruta.with_suffix('.dtypes.json')
        if side.exists():                                # restaura los tipos del resultado original de BigQuery
            for c, t in json.loads(side.read_text(encoding='utf-8')).items():
                if c in df.columns and t != 'object' and str(df[c].dtype) != t:
                    with contextlib.suppress(Exception):
                        df[c] = df[c].astype(t)
        return df

    # ── lectura ──
    def envolver(self, original):
        def leer(query, *args, **kwargs):
            project_id = kwargs.get('project_id', args[0] if args else None)
            key = clave_consulta(query, project_id)
            t0 = time.time()
            snap = None
            if self.modo in ('reutilizar', 'reproducir'):
                pedido = self.contenidos.get(key)
                cand = self._archivos(key)
                if pedido:
                    cand = [p for p in cand if p.name.startswith(f'{key}__{pedido}.')]
                if cand:
                    snap = cand[-1]
                elif self.modo == 'reproducir':
                    raise InsumoNoDisponible(f'No hay snapshot de la consulta {key} en {self.dir}'
                                             + (f' con el contenido {pedido}' if pedido else '') + f'. SQL: {normalizar_sql(query)[:120]}...')
            if snap is not None:
                df, origen = self._leer_snapshot(snap), 'snapshot'
                contenido = snap.name.split('__')[1].split('.')[0]
            else:
                df, origen = (original or _consulta_real())(query, *args, **kwargs), 'bigquery'     # BigQuery solo se resuelve si hay que consultar
                contenido = hash_df(df)
                snap = self._guardar(key, contenido, df)
            self.entradas.append({'key': key, 'sql_preview': normalizar_sql(query)[:160], 'proyecto': project_id, 'origen': origen,
                                  'filas': int(len(df)), 'columnas': int(df.shape[1]), 'content_hash': contenido,
                                  'archivo': snap.name, 'segundos': round(time.time() - t0, 2)})
            return df
        return leer


def _consulta_real():
    """Solo se usa si pandas no trae `read_gbq` (versiones recientes): se consulta con pandas_gbq directamente (nunca con nuestro propio reemplazo)."""
    import pandas_gbq
    return pandas_gbq.read_gbq


@contextlib.contextmanager
def capa_insumos(modo, directorio, contenidos=None):
    """Mientras dura el bloque, `pd.read_gbq` pasa por la capa. Devuelve la capa (con `.entradas`)."""
    capa = CapaInsumos(modo, directorio, contenidos)
    if modo == 'desactivado':
        yield capa
        return
    previo, tenia = getattr(pd, 'read_gbq', None), hasattr(pd, 'read_gbq')
    pd.read_gbq = capa.envolver(previo)
    try:
        yield capa
    finally:
        if tenia:
            pd.read_gbq = previo
        else:
            delattr(pd, 'read_gbq')
