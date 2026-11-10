"""Registro de familias y programas, cargado desde config/programas.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Programa:
    nombre: str
    familia: str
    motor: str                                   # 'familia' (en proceso) | 'notebook' (nbclient + Excel)
    script: Optional[str] = None                 # motor familia: nombre del script en mediciones/programas/
    notebook: Optional[str] = None               # motor notebook: ruta relativa a la raíz del proyecto
    patrones_mes: list = field(default_factory=list)   # [(regex, reemplazo con {mes}/{programa})]
    salida: Optional[str] = None                 # motor notebook: Excel que deja ({aaaamm} / {aaaa_mm})
    adaptador: Optional[str] = None              # 'estandar' | 'champion'
    negocios_ub: Optional[list] = None           # filtro cod_negocio de la venta para la UB
    metodo_ub: str = 'actual'                    # 'actual' (UB actual x valor/actual) | 'notebook' (UB base por tonelada; CPB y CPM): ver ub.py
    pares: Optional[dict] = None                 # estado local de pares congelados: {tipo: mensual|cosechas, ruta: ...} (ver pares.py)
    provisional_desde: Optional[str] = None      # 'AAAA-MM-01': desde ese mes los meses publicados se pueden revisar cada cierre (ver frontera.py); None = todo mes publicado queda fijo
    provisional_cosechas: Optional[list] = None  # cosechas (fecha de cosecha definitiva) a las que se limita lo provisional; None = todo el programa
    provisional_hasta: Optional[str] = None      # último cierre en que se republica lo provisional (después queda fijo); None = sin tope
    calendario_cosechas: Optional[list] = None   # grupos de cosecha (surtido): ver mediciones/calendario.py
    nota: str = ''


def cargar_registro(ruta_yaml):
    """Devuelve (familias, programas): dict de configuración de familias y dict[str, Programa]."""
    cfg = yaml.safe_load(Path(ruta_yaml).read_text(encoding='utf-8'))
    familias = cfg['familias']
    programas = {}
    for nombre, d in cfg['programas'].items():
        if d['familia'] not in familias:
            raise ValueError(f"{nombre}: familia desconocida {d['familia']!r}")
        if d['motor'] not in ('familia', 'notebook'):
            raise ValueError(f"{nombre}: motor debe ser 'familia' o 'notebook'")
        d = dict(d)
        d['patrones_mes'] = [tuple(p) for p in d.get('patrones_mes', [])]
        programas[nombre] = Programa(nombre=nombre, **d)
    return familias, programas
