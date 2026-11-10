"""
Trazabilidad: qué código, configuración, entorno e insumos produjo cada corrida.

`config_hash` = hash(código + configuración + entorno). Se guarda en las 3 bases de negocio (columna `config_hash`) y en el log; el detalle completo (con el
hash de los insumos de esa corrida) queda en runs/<corrida_id>/manifiesto_<programa>.json. Sirve para dos cosas: (1) saber exactamente con qué versión se produjo cada fila,
y (2) diagnosticar una deriva: si el `config_hash` publicado coincide con el actual, el cambio viene de los insumos; si no, viene del código, la configuración o el entorno.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from .ejecucion import DIR_PAQUETE, script_de_programa

LIBRERIAS = ['pandas', 'numpy', 'scipy', 'scikit-learn', 'statsmodels', 'joblib', 'pyarrow', 'pandas-gbq', 'google-cloud-bigquery', 'openpyxl']


def hash_texto(*partes) -> str:
    h = hashlib.sha256()
    for p in partes:
        h.update(str(p).encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()[:16]


def _hash_archivo(ruta) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()[:16]


def archivos_de_codigo(prog, familias_cfg):
    """Archivos que definen el comportamiento de un programa con motor en proceso."""
    fam = DIR_PAQUETE / 'familias' / prog.familia
    compartidos = sorted(p for p in (DIR_PAQUETE / 'metodologia').glob('*.py') if p.name != '__init__.py')       # núcleo compartido (atribución, ids...)
    return ([script_de_programa(prog)] + [fam / f'{s}.py' for s in familias_cfg[prog.familia]['scripts']] + compartidos
            + [DIR_PAQUETE / 'parametros.py', DIR_PAQUETE / 'nucleo' / 'matching.py', DIR_PAQUETE / 'grilla.py',
               DIR_PAQUETE / 'normalizacion.py', DIR_PAQUETE / 'ub.py'])      # normalización y UB también determinan los números publicados


def hash_codigo(prog, familias_cfg, raiz):
    """(hash, detalle). Motor en proceso: scripts + parámetros + núcleo. Motor notebook: las celdas de código del notebook (sin salidas)."""
    if prog.motor == 'familia':
        detalle = {str(a.name): _hash_archivo(a) for a in archivos_de_codigo(prog, familias_cfg)}
    else:
        nb = json.loads((Path(raiz) / prog.notebook).read_text(encoding='utf-8'))
        codigo = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        detalle = {Path(prog.notebook).name: hash_texto(codigo)}
    return hash_texto(json.dumps(detalle, sort_keys=True)), detalle


def hash_config(prog, familias_cfg):
    return hash_texto(json.dumps({'programa': asdict(prog), 'familia': familias_cfg[prog.familia]}, sort_keys=True, default=str))


def versiones_entorno():
    v = {'python': '.'.join(map(str, sys.version_info[:3]))}
    for lib in LIBRERIAS:
        try:
            v[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            v[lib] = None
    return v


def commit_git(raiz):
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=str(raiz), capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def construir_manifiesto(prog, familias_cfg, raiz, corrida_id, mes):
    """Manifiesto previo a la ejecución (los insumos se agregan después con `completar_con_insumos`)."""
    h_cod, detalle = hash_codigo(prog, familias_cfg, raiz)
    h_cfg = hash_config(prog, familias_cfg)
    ver = versiones_entorno()
    h_ent = hash_texto(json.dumps(ver, sort_keys=True))
    return {'corrida_id': corrida_id, 'programa': prog.nombre, 'mes': pd_fecha(mes), 'motor': prog.motor, 'familia': prog.familia,
            'config_hash': hash_texto(h_cod, h_cfg, h_ent), 'hash_codigo': h_cod, 'detalle_codigo': detalle, 'hash_config': h_cfg,
            'hash_entorno': h_ent, 'versiones': ver, 'commit_git': commit_git(raiz),
            'insumos': [], 'hash_insumos': None}


def pd_fecha(mes):
    return mes.strftime('%Y-%m-%d') if hasattr(mes, 'strftime') else str(mes)


def hash_insumos(entradas):
    return hash_texto(json.dumps(sorted((e['key'], e['content_hash']) for e in entradas)))


def completar_con_insumos(manifiesto, entradas):
    manifiesto['insumos'] = list(entradas)
    manifiesto['hash_insumos'] = hash_insumos(entradas) if entradas else None
    return manifiesto


def guardar_manifiesto(raiz, manifiesto):
    d = Path(raiz) / 'runs' / manifiesto['corrida_id']
    d.mkdir(parents=True, exist_ok=True)
    ruta = d / f"manifiesto_{manifiesto['programa']}.json"
    ruta.write_text(json.dumps(manifiesto, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    return ruta


def cargar_manifiesto(raiz, corrida_id, programa):
    ruta = Path(raiz) / 'runs' / corrida_id / f'manifiesto_{programa}.json'
    if not ruta.exists():
        raise FileNotFoundError(f'No existe el manifiesto de {programa} en la corrida {corrida_id}: {ruta}')
    return json.loads(ruta.read_text(encoding='utf-8'))
