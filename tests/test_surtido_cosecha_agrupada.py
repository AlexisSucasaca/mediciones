"""Celdas del notebook de surtido que habían quedado fuera del script (14: tipos del sell-out; 23: mes de entrada -> cosecha_agrupada)."""
from pathlib import Path

import numpy as np
import pandas as pd

RUTA = Path(__file__).parent.parent / 'src' / 'mediciones' / 'programas' / 'surtido.py'


def _region(nombre):
    L = RUTA.read_text(encoding='utf-8').split('\n')
    i = [k for k, l in enumerate(L) if l.startswith(f'# [{nombre}:inicio]')]
    j = [k for k, l in enumerate(L) if l.startswith(f'# [{nombre}:fin]')]
    assert len(i) == 1 and len(j) == 1, nombre
    return '\n'.join(L[i[0]:j[0] + 1])


def test_cosecha_agrupada_por_mes_de_entrada():
    from mediciones.calendario import Calendario
    meses = list(range(1, 13))
    for cierre, g04 in (('2026-10-01', '2026-10-01'), ('2026-11-01', '2026-10-01'), ('2026-12-01', '2026-12-01')):
        base = pd.DataFrame({'cod_cliente': [f'{m:010d}' for m in meses], 'min_periodo_cosecha': [f'2026-{m:02d}-01' for m in meses]})
        ns = {'base_seg_limpio': base, 'pd': pd, 'np': np, 'CALENDARIO': Calendario(), 'MES_CIERRE': pd.Timestamp(cierre)}
        exec(compile(_region('cosecha_agrupada'), 'cosecha_agrupada', 'exec'), ns)
        got = dict(zip(meses, ns['base_seg_limpio']['cosecha_agrupada']))
        esperado = {2: '2026-04-01', 3: '2026-04-01', 4: '2026-04-01', 5: '2026-06-01', 6: '2026-06-01',
                    7: '2026-08-01', 8: '2026-08-01', 9: '2026-08-01',              # G03: altas jul-set (cerró en setiembre)
                    10: g04, 11: g04, 12: g04}                                      # G04: anclado en oct (preliminar) y en dic desde su cosecha
        assert {m: got[m] for m in esperado} == esperado, cierre
        assert pd.isna(got[1])                                                # enero: previo al piloto, sin cosecha
        assert pd.api.types.is_datetime64_any_dtype(ns['base_seg_limpio']['min_periodo_cosecha'])


def test_sell_out_jun_tipos():
    ns = {'sell_out_jun': pd.DataFrame({'peso': ['1.5', 2], 'monto': [3, '4.25'], 'periodo': ['2026-01-01', '2026-02-01']}), 'pd': pd}
    exec(compile(_region('sell_out_jun'), 'sell_out_jun', 'exec'), ns)
    s = ns['sell_out_jun']
    assert s['peso'].dtype == float and s['monto'].dtype == float and pd.api.types.is_datetime64_any_dtype(s['periodo'])
