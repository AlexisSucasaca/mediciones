"""
Estado local de cada programa (pares fijos de CPB/CPM; carpeta outputs/ de surtido y Champion con el matching congelado y el estado del G04).

El motor actualiza ese estado al correr. Para que una PRUEBA no ensucie el cierre real, el runner lo respalda antes de correr el programa y lo
restaura al terminar, salvo que el programa se haya CARGADO de verdad: un `dry_run`, una carga bloqueada (deriva, cierre, pares) o un error dejan
el estado exactamente como estaba. Así se puede probar un mes las veces que haga falta y el cierre real parte limpio.

Windows/OneDrive: la restauración NUNCA borra la carpeta en sí (OneDrive suele bloquearla): vacía su contenido y vuelve a copiar el respaldo
encima, con reintentos si un archivo está tomado. Si algo no se puede restaurar, no se detiene ni borra el respaldo: avisa dónde quedó para
copiarlo a mano.
"""
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path

REINTENTOS, ESPERA = 6, 0.5


def rutas_de_estado(workdir, pares, programa):
    """Lo que el motor de ese programa reescribe. Mensual: el archivo de pares fijos. Cosechas: toda la carpeta del estado congelado."""
    if not pares or not pares.get('ruta'):
        return []
    r = Path(workdir) / pares['ruta'].format(programa=programa)
    return [r if pares.get('tipo') == 'mensual' else r.parent]


def respaldar(rutas):
    tmp = Path(tempfile.mkdtemp(prefix='mediciones_estado_'))
    items = []
    for i, r in enumerate(rutas):
        r = Path(r)
        if r.exists():
            dst = tmp / str(i)
            if r.is_dir():
                shutil.copytree(r, dst)
            else:
                shutil.copy2(r, dst)
            items.append((r, dst))
        else:
            items.append((r, None))
    return {'tmp': tmp, 'items': items}


def _con_reintentos(accion):
    ultimo = None
    for _ in range(REINTENTOS):
        try:
            return accion()
        except PermissionError as e:              # archivo tomado (OneDrive, Excel abierto, antivirus): se espera y se reintenta
            ultimo = e
            time.sleep(ESPERA)
    raise ultimo


def _borrar_archivo(f):
    def _b():
        try:
            os.chmod(f, stat.S_IWRITE)
        except OSError:
            pass
        os.remove(f)
    _con_reintentos(_b)


def _vaciar(carpeta):
    """Borra el CONTENIDO de la carpeta (no la carpeta). Las subcarpetas que no se puedan quitar quedan vacías (no importa)."""
    for raiz, dirs, archivos in os.walk(carpeta, topdown=False):
        for a in archivos:
            _borrar_archivo(os.path.join(raiz, a))
        for d in dirs:
            try:
                os.rmdir(os.path.join(raiz, d))
            except OSError:
                pass


def restaurar(respaldo, verbose=True):
    """
    Deja cada ruta como estaba: una carpeta recupera exactamente su contenido (sin borrar la carpeta), un archivo su versión anterior, y lo que
    no existía se quita. Nunca lanza error: devuelve la lista de rutas que no se pudieron restaurar (y en ese caso conserva el respaldo).
    """
    fallos = []
    for r, dst in respaldo['items']:
        try:
            if dst is None:                               # no existía antes de la corrida
                if r.is_dir():
                    _vaciar(r)
                    try:
                        os.rmdir(r)
                    except OSError:
                        pass                              # queda vacía: equivale a no tener estado
                elif r.exists():
                    _borrar_archivo(r)
            elif dst.is_dir():
                r.mkdir(parents=True, exist_ok=True)
                _vaciar(r)
                _con_reintentos(lambda: shutil.copytree(dst, r, dirs_exist_ok=True))   # copia encima de la carpeta existente
            else:
                r.parent.mkdir(parents=True, exist_ok=True)
                _con_reintentos(lambda: shutil.copy2(dst, r))     # sobrescribe sin borrar antes
        except Exception as e:
            fallos.append((str(r), f'{type(e).__name__}: {e}'))
    if fallos:
        if verbose:
            print(f"⚠ NO se pudo restaurar el estado local de: {[f for f, _ in fallos]}. El respaldo quedó en {respaldo['tmp']}: "
                  f"copia su contenido a esa(s) ruta(s) a mano antes de volver a correr. Detalle: {fallos}")
        return fallos
    descartar(respaldo)
    return []


def descartar(respaldo):
    shutil.rmtree(respaldo['tmp'], ignore_errors=True)
