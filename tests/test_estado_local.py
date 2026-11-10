"""Una prueba (dry_run), una carga bloqueada o un error no dejan modificado el estado local; solo una carga real lo actualiza."""
import pandas as pd

from mediciones import estado_local
from conftest import ServiciosMemoria
from test_runner_memoria import AGO, SEP, _meses, correr, motor  # noqa: F401  (motor es un fixture)


def test_respaldar_y_restaurar_archivo_y_carpeta(tmp_path):
    archivo, carpeta, nuevo = tmp_path / 'p.csv', tmp_path / 'outputs' / 'X', tmp_path / 'no_existia.csv'
    archivo.write_text('antes')
    carpeta.mkdir(parents=True)
    (carpeta / 'df_matches.csv').write_text('pares antes')
    resp = estado_local.respaldar([archivo, carpeta, nuevo])
    archivo.write_text('despues')
    (carpeta / 'df_matches.csv').write_text('pares despues')
    (carpeta / 'estado_matching.json').write_text('{}')
    nuevo.write_text('creado en la corrida')
    estado_local.restaurar(resp)
    assert archivo.read_text() == 'antes' and (carpeta / 'df_matches.csv').read_text() == 'pares antes'
    assert not (carpeta / 'estado_matching.json').exists() and not nuevo.exists()
    assert not resp['tmp'].exists()


def test_rutas_de_estado_por_tipo(tmp_path):
    assert estado_local.rutas_de_estado(tmp_path, {'tipo': 'mensual', 'ruta': '{programa}_pares_fijos.csv'}, 'cpb') == [tmp_path / 'cpb_pares_fijos.csv']
    assert estado_local.rutas_de_estado(tmp_path, {'tipo': 'cosechas', 'ruta': 'outputs/S/df_matches.csv'}, 'x') == [tmp_path / 'outputs' / 'S']
    assert estado_local.rutas_de_estado(tmp_path, None, 'x') == []


def test_dry_run_no_deja_estado_y_cargar_si(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    ruta = raiz_tmp / 'data' / 'cpb' / 'cpb_pares_fijos.csv'
    antes = ruta.read_text()
    correr(raiz_tmp, SEP, svc, modo='dry_run')                                   # el motor escribe los pares de setiembre...
    assert ruta.read_text() == antes                                             # ...pero la prueba no los deja
    correr(raiz_tmp, SEP, svc, modo='dry_run')                                   # se puede repetir: sigue limpio
    assert ruta.read_text() == antes
    c = correr(raiz_tmp, SEP, svc, modo='cargar', tolerancia_celda={'ton_pct': 100})
    assert c.res['cpb'].get('cargado') and ruta.read_text() != antes             # la carga real sí actualiza el estado


def test_una_carga_bloqueada_tampoco_deja_estado(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    ruta = raiz_tmp / 'data' / 'cpb' / 'cpb_pares_fijos.csv'
    antes = ruta.read_text()
    c = correr(raiz_tmp, SEP, svc, modo='cargar')                                # bloqueada por deriva por celda
    assert c.resumen.estado.tolist() == ['bloqueado'] and ruta.read_text() == antes


def test_restaurar_con_la_carpeta_bloqueada_como_en_onedrive(tmp_path, monkeypatch):
    """OneDrive no deja borrar la carpeta: la restauración no la borra; vacía su contenido y copia el respaldo encima."""
    import os
    carpeta = tmp_path / 'outputs' / 'Champion Minimayorista - Medición'
    carpeta.mkdir(parents=True)
    (carpeta / 'df_matches.csv').write_text('pares setiembre')
    (carpeta / 'meta_pares_congelados.json').write_text('{"v": 1}')
    resp = estado_local.respaldar([carpeta])
    (carpeta / 'df_matches.csv').write_text('pares de la prueba')
    (carpeta / 'nuevo.csv').write_text('creado en la prueba')
    real_rmdir = os.rmdir
    monkeypatch.setattr(os, 'rmdir', lambda p, *a, **k: (_ for _ in ()).throw(PermissionError(5, 'Acceso denegado', str(p)))
                        if str(p) == str(carpeta) else real_rmdir(p, *a, **k))
    assert estado_local.restaurar(resp, verbose=False) == []
    assert (carpeta / 'df_matches.csv').read_text() == 'pares setiembre' and (carpeta / 'meta_pares_congelados.json').exists()
    assert not (carpeta / 'nuevo.csv').exists() and not resp['tmp'].exists()


def test_restaurar_reintenta_un_archivo_tomado_y_no_se_detiene_si_algo_falla(tmp_path, monkeypatch):
    import os
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    a.write_text('a0'); b.write_text('b0')
    resp = estado_local.respaldar([a, b])
    a.write_text('a1'); b.write_text('b1')
    monkeypatch.setattr(estado_local, 'ESPERA', 0.0)
    real_copy2, intentos = estado_local.shutil.copy2, {'a': 0}

    def copy2(src, dst, *x, **k):
        if str(dst) == str(a):
            intentos['a'] += 1
            if intentos['a'] < 3:
                raise PermissionError(32, 'archivo en uso', str(dst))     # tomado un momento: se reintenta
        if str(dst) == str(b):
            raise PermissionError(5, 'Acceso denegado', str(dst))         # bloqueado del todo
        return real_copy2(src, dst, *x, **k)
    monkeypatch.setattr(estado_local.shutil, 'copy2', copy2)
    fallos = estado_local.restaurar(resp, verbose=False)
    assert a.read_text() == 'a0' and intentos['a'] == 3                    # se restauró tras reintentar
    assert [f for f, _ in fallos] == [str(b)] and resp['tmp'].exists()     # b falló: se informa y el respaldo se conserva
