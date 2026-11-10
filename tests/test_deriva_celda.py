"""Deriva de meses cerrados medida por celda mes x categoría (x cosecha): la misma vara de la validación de setiembre."""
import pandas as pd

from mediciones.incremental import deriva_por_celda


def _ton(vals, clientes=('1', '2')):
    return pd.DataFrame({'cod_cliente': list(clientes), 'des_categoria': 'Galletas', 'periodo': pd.Timestamp('2026-08-01'), 'valor_atribuible': vals})


def test_toneladas_por_celda_ignora_altas_tardias_y_aplica_pct_y_minimo_absoluto():
    pub = _ton([10.0, 10.0])
    rec = pd.concat([_ton([10.05, 10.0]), _ton([5.0], clientes=('9',))], ignore_index=True)     # +0.25 % y un alta tardía (no cuenta)
    c = deriva_por_celda('base_ton_atribuibles', rec, pub)
    assert len(c) == 1 and round(c.dif.iloc[0], 6) == 0.05 and not c.fuera.iloc[0]
    c = deriva_por_celda('base_ton_atribuibles', _ton([10.5, 10.0]), pub)                       # +2.5 % -> fuera
    assert c.fuera.iloc[0]
    chica = deriva_por_celda('base_ton_atribuibles', _ton([0.0105, 0.0]), _ton([0.01, 0.0]))    # +5 % pero 0.5 kg -> dentro
    assert not chica.fuera.iloc[0]


def test_brechas_por_celda_y_cosecha():
    def b(v, coh):
        return pd.DataFrame({'par_id': ['p'], 'periodo': pd.Timestamp('2026-08-01'), 'des_categoria': 'Galletas',
                             'cosecha_grupo': pd.Timestamp(coh), 'brecha_pct_celda': v})
    pub = pd.concat([b(10.0, '2026-04-01'), b(20.0, '2026-06-01')])
    rec = pd.concat([b(10.2, '2026-04-01'), b(20.3, '2026-06-01')])
    c = deriva_por_celda('base_brechas', rec, pub)
    assert c.fuera.tolist() == [False, True] and c.cosecha_grupo.tolist() == ['2026-04', '2026-06']
    c = deriva_por_celda('base_brechas', b(10.0, '2026-04-01').iloc[0:0], pub)                  # sin recalculado (CPB/CPM con solo el mes nuevo)
    assert c.empty


def test_espejos_no_aplican():
    assert deriva_por_celda('base_espejos', _ton([1.0, 1.0]), _ton([1.0, 1.0])).empty
