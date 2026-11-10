"""Tabla acumulativa de cambios de código (Champion): cadenas al más antiguo, errores claros, sin archivo no cambia nada."""
import pandas as pd
import pytest

from mediciones.metodologia.equivalencias import leer_equivalencias


def _csv(tmp_path, filas):
    ruta = tmp_path / 'equivalencias_codigos.csv'
    pd.DataFrame(filas, columns=['cod_cliente_nuevo', 'cod_cliente_antiguo', 'vigente_desde', 'origen', 'observacion']).to_csv(ruta, index=False)
    return str(ruta)


def test_sin_archivo_no_cambia_nada(tmp_path):
    assert leer_equivalencias(str(tmp_path / 'no_existe.csv')) == {}


def test_cadenas_se_resuelven_al_mas_antiguo_y_se_normalizan_ids(tmp_path):
    ruta = _csv(tmp_path, [('3', '2', '2026-10-01', 'manual', ''), ('0000000002', '1', '2026-09-01', 'manual', '')])
    assert leer_equivalencias(ruta) == {'0000000003': '0000000001', '0000000002': '0000000001'}


def test_ciclo_y_nuevo_con_dos_antiguos_fallan(tmp_path):
    with pytest.raises(ValueError, match='ciclo'):
        leer_equivalencias(_csv(tmp_path, [('1', '2', '', '', ''), ('2', '1', '', '', '')]))
    with pytest.raises(ValueError, match='más de un antiguo'):
        leer_equivalencias(_csv(tmp_path, [('5', '1', '', '', ''), ('5', '2', '', '', '')]))


def test_champion_usa_la_tabla_de_equivalencias_y_el_legacy_flexible():
    from conftest import RAIZ
    src = (RAIZ / 'src' / 'mediciones' / 'programas' / 'champion.py').read_text(encoding='utf-8')
    i_eq, i_rep = src.index('leer_equivalencias(ARCHIVO_EQUIVALENCIAS)'), src.index("sell_out_fec['cod_cliente_dex'] = sell_out_fec['cod_cliente_dex'].replace(mapa_homologacion)")
    assert i_eq < i_rep, 'las equivalencias deben sumarse al mapa antes de homologar el sell-out'
    assert "glob('BD_Clientes_*Snack*.xlsx')" in src           # el archivo legacy se elige por el mes del nombre (ver test_champion_legacy.py)
