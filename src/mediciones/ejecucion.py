"""Ejecución de los motores: en proceso (familias) o vía notebook (motores aún no migrados)."""
from __future__ import annotations

import contextlib
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .insumos import capa_insumos

DIR_PAQUETE = Path(__file__).resolve().parent


@dataclass
class ResultadoFamilia:
    ton: pd.DataFrame
    bre: pd.DataFrame
    segundos: float
    esp: Optional[pd.DataFrame] = None       # base de espejos (cliente de acción x categoría x mes con su espejo)
    namespace: Optional[dict] = None     # variables intermedias (df_panel, df_base_completa...) para depurar
    insumos: list = field(default_factory=list)   # consultas a BigQuery de esta corrida (hash de SQL y de contenido)


def nueva_corrida_id(mes):
    return f"{mes:%Y%m}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"


@contextlib.contextmanager
def _en_directorio(ruta):
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    previo = os.getcwd()
    os.chdir(ruta)
    try:
        yield
    finally:
        os.chdir(previo)


def _display(*objs, **kw):
    try:
        from IPython.display import display
        display(*objs, **kw)
    except Exception:
        for o in objs:
            print(o)


def ejecutar_familia(familia, script_programa, programa, mes, workdir, familias_cfg, devolver_namespace=False,
                     insumos='consultar', dir_insumos=None, contenidos=None, scripts_familia=None, parametros=None):
    """
    Ejecuta en proceso: script del programa + scripts de su familia, en un espacio de nombres compartido (como las celdas
    de un notebook). Los scripts leen/escriben rutas relativas, por eso se ejecutan dentro de `workdir`.
    Las consultas a BigQuery pasan por la capa de insumos (snapshots con hash; ver mediciones.insumos). `insumos`: consultar | reutilizar | reproducir | desactivado.
    Devuelve las bases crudas (antes de normalizar), los insumos usados y, si se pide, el espacio de nombres para depurar.
    """
    os.environ.setdefault('MPLBACKEND', 'Agg')            # sin ventanas de gráficos
    cfg = familias_cfg[familia]
    ns = {'__name__': '__main__', 'PROGRAMA': programa, cfg['variable_mes']: pd.Timestamp(mes).strftime('%Y-%m-%d'),
          'display': _display}
    if 'regla_nuevo' in cfg:
        ns['REGLA_NUEVO'] = cfg['regla_nuevo']            # cómo se trata al cliente que no compró ese mes del año anterior (metodologia/atribucion.py)
    if parametros:
        ns.update(parametros)                             # del runner: CALENDARIO_COSECHAS (surtido), RECALCULAR_DESDE (CPB/CPM), ...
    scripts = [Path(script_programa)] + (list(map(Path, scripts_familia)) if scripts_familia else
                                         [DIR_PAQUETE / 'familias' / familia / f'{s}.py' for s in cfg['scripts']])
    dir_ins = Path(dir_insumos) if dir_insumos else Path(workdir).resolve() / 'insumos' / pd.Timestamp(mes).strftime('%Y%m')
    t0 = time.time()
    with _en_directorio(workdir), capa_insumos(insumos, dir_ins, contenidos) as capa:
        for s in scripts:
            exec(compile(s.read_text(encoding='utf-8'), str(s), 'exec'), ns)
    ton, bre = ns[cfg['salida_ton']], ns[cfg['salida_brecha']]
    esp = ns.get(cfg.get('salida_esp', ''))
    return ResultadoFamilia(ton.copy(), bre.copy(), time.time() - t0, None if esp is None else esp.copy(),
                            ns if devolver_namespace else None, list(capa.entradas))


def script_de_programa(prog):
    """`script` puede ser el nombre de un script del paquete (cpb) o una ruta a un .py (útil en pruebas)."""
    if str(prog.script).endswith('.py'):
        return Path(prog.script)
    return DIR_PAQUETE / 'programas' / f'{prog.script}.py'


# ── Motores que todavía son notebooks ───────────────────────────────────────────────────────────────
def inyectar_mes(nb, patrones, mes, programa=''):
    """Reemplaza el parámetro de mes en las celdas de código. Cada patrón debe coincidir EXACTAMENTE una vez."""
    mes_txt = mes.strftime('%Y-%m-%d')
    conteo = [0] * len(patrones)
    for celda in nb.cells:
        if celda.cell_type != 'code':
            continue
        src = celda.source
        for i, (rx, repl) in enumerate(patrones):
            src, n = re.subn(rx, repl.format(mes=mes_txt, programa=programa), src, flags=re.M)
            conteo[i] += n
        celda.source = src
    malos = [(patrones[i][0], n) for i, n in enumerate(conteo) if n != 1]
    if malos:
        raise RuntimeError(f'La inyección del mes debe coincidir una sola vez por patrón; obtuve: {malos}')
    return conteo
