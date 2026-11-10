"""
Runner: una corrida mensual completa.

  0. TRAZABILIDAD  manifiesto (código + configuración + entorno -> config_hash) y verificación de los pares congelados contra lo publicado.
  1. EJECUTAR      cada programa (en proceso con la metodología de su familia, o vía notebook si aún no se migró).
  2. NORMALIZAR    al esquema único y VALIDAR.
  3. INCREMENTAL   frente a lo ya publicado: solo llaves nuevas; si lo publicado cambia, se reporta como deriva y, sobre el umbral, se BLOQUEA la carga.
  4. UB            sobre las filas a cargar.
  5. CARGAR        (brechas, espejos y, al final, toneladas) y registrar el log y el manifiesto de la corrida.

Los accesos a BigQuery pasan por un objeto de servicios (ServiciosBQ), de modo que el runner se prueba con servicios en memoria.
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from . import bigquery, estado_local, frontera, incremental, normalizacion, pares, trazabilidad, ub
from .ejecucion import ejecutar_familia, nueva_corrida_id, script_de_programa
from .esquema import T_BRE, T_ESP, T_TON
from .registro import cargar_registro


# % máximo de filas ya publicadas que pueden cambiar al recalcular sin bloquear la carga (0 = ninguna). Los pares/espejos nunca deberían cambiar.
# Valores iniciales: calibrar con las primeras corridas reales.
# Bloqueo por deriva de los meses cerrados: por defecto se mide POR CELDA mes x categoría (incremental.TOLERANCIA_CELDA: 0.25 pp en brechas,
# 1 % y 1 kg en toneladas), que es el nivel en que se reporta. Los conteos por fila se informan; solo bloquean si se pasa umbral_deriva={tabla: %}.
UMBRAL_DERIVA_DEFECTO = {}
ESTADOS_PARES_QUE_BLOQUEAN = ('difiere', 'incompleto', 'faltante_no_restaurable')


class ServiciosBQ:
    """Acceso real a BigQuery (lo publicado, los pares, la UB, la carga y el log)."""

    def __init__(self, cli=None):
        self.cli = cli or bigquery.cliente_bq()

    def leer_publicado(self, tabla, programa):
        return bigquery.leer_publicado(self.cli, tabla, programa)

    def leer_pares(self, programa):
        return bigquery.leer_pares(self.cli, programa)

    def ultima_version(self, tabla, programa, periodo_reporte):
        return bigquery.ultima_version(self.cli, tabla, programa, periodo_reporte)

    def cargar(self, df, tabla):
        return bigquery.cargar_df(self.cli, df, tabla)

    def verificar_carga(self, df, tabla):
        return bigquery.verificar_carga(self.cli, df, tabla)

    def consulta_ub(self, ton, negocios, meses_atras=0):
        return ub.consulta_ub(self.cli, ton, negocios, meses_atras)

    def registrar_log(self, filas):
        bigquery.registrar_log(self.cli, filas)

    def exportar(self, programa, mes, carpeta):
        return bigquery.exportar_vigente(self.cli, programa, mes, carpeta)


@dataclass
class Corrida:
    corrida_id: str
    mes: pd.Timestamp
    modo: str
    res: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    resumen: Optional[pd.DataFrame] = None


def _say(verbose, *a):
    if verbose:
        print(*a)


def _guardar_crudas(raiz, corrida_id, p, ton, bre, esp=None):
    d = Path(raiz) / 'runs' / corrida_id
    d.mkdir(parents=True, exist_ok=True)
    ton.to_parquet(d / f'{p}_ton.parquet', index=False)
    bre.to_parquet(d / f'{p}_brecha.parquet', index=False)
    if esp is not None:
        esp.to_parquet(d / f'{p}_espejos.parquet', index=False)


def ubicar_fallo(exc):
    """
    Dónde se cayó un programa: último frame que está dentro de un script del motor (programas/ o familias/), con su línea y su texto; si el script es una portada
    de notebook, también la celda ('# ══ celda N del notebook'). Devuelve '' si el error no ocurrió en un script del motor.
    """
    marcos = [f for f in traceback.extract_tb(exc.__traceback__)
              if f.filename.endswith('.py') and any(d in f.filename.replace('\\', '/') for d in ('/programas/', '/familias/'))]
    if not marcos:
        return ''
    f = marcos[-1]
    celda = ''
    try:
        previas = Path(f.filename).read_text(encoding='utf-8').split('\n')[:f.lineno]
        marca = next((l for l in reversed(previas) if l.startswith('# ═') and 'del notebook' in l), None)
        celda = ' (' + marca.strip('# ═ ') + ')' if marca else ''
    except OSError:
        pass
    return f"{Path(f.filename).parent.name}/{Path(f.filename).name}, línea {f.lineno}{celda}: {(f.line or '').strip()[:200]}"


def guardar_traceback(exc, raiz, corrida_id, programa):
    """Guarda el traceback completo en runs/<corrida>/<programa>_traceback.txt y devuelve la ruta (o None si no se pudo)."""
    try:
        d = Path(raiz) / 'runs' / corrida_id
        d.mkdir(parents=True, exist_ok=True)
        ruta = d / f'{programa}_traceback.txt'
        ruta.write_text(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding='utf-8')
        return ruta
    except OSError:
        return None


def _cargar_crudas(raiz, p, mes):
    cand = sorted((Path(raiz) / 'runs').glob(f'{mes:%Y%m}_*/{p}_ton.parquet'))
    if not cand:
        raise FileNotFoundError(f'No hay una corrida guardada de {p} para {mes:%Y-%m} en runs/ (ejecutar=True para generarla)')
    ton = pd.read_parquet(cand[-1])
    bre = pd.read_parquet(str(cand[-1]).replace('_ton.parquet', '_brecha.parquet'))
    f_esp = Path(str(cand[-1]).replace('_ton.parquet', '_espejos.parquet'))
    return ton, bre, (pd.read_parquet(f_esp) if f_esp.exists() else None)


def diagnostico_deriva(pub, deriva, tabla, config_hash_actual):
    """Explica de dónde viene una deriva comparando el config_hash publicado con el actual."""
    if pub is None or deriva is None or len(deriva) == 0 or 'config_hash' not in pub.columns:
        return 'sin hash publicado: no se puede atribuir'
    clave = incremental.CLAVE[tabla]
    hs = set(deriva[clave].merge(pub[clave + ['config_hash']], on=clave, how='left')['config_hash'].dropna().unique())
    if not hs:
        return 'sin hash publicado: no se puede atribuir'
    if hs == {config_hash_actual}:
        return 'mismo código, configuración y entorno que en la publicación: la causa son los INSUMOS (BigQuery cambió retroactivamente)'
    return f'código, configuración o entorno distintos a los publicados (publicado: {sorted(hs)}, actual: {config_hash_actual})'


def correr_mes(mes, programas, *, modo='dry_run', ejecutar=True, reproceso=False, motivo='', calcular_ub=True,
               exportar_excel=True, raiz=None, servicios=None, verbose=True,
               insumos='consultar', reproducir_de=None, umbral_deriva=None, aceptar_deriva=False, motivo_deriva='',
               ignorar_pares=False, reabrir_desde=None, validar_cerrados=False, tolerancia_celda=None):
    """
    Ejecuta una corrida mensual. `servicios=None` intenta BigQuery (en dry_run, si falla, se asume primera carga).
    insumos: consultar | reutilizar | reproducir | desactivado (ver mediciones.insumos); `reproducir_de` = corrida cuyos insumos exactos se usan.
    umbral_deriva: {tabla: % máximo de filas publicadas que pueden cambiar}; sobre el umbral la carga se bloquea salvo `aceptar_deriva` (con `motivo_deriva`).
    """
    raiz = Path(raiz or os.environ.get('MEDICIONES_RAIZ', '.')).resolve()
    mes = pd.Timestamp(mes)
    if mes.day != 1:
        raise ValueError('mes debe ser el día 1')
    if modo not in ('dry_run', 'cargar'):
        raise ValueError("modo debe ser 'dry_run' o 'cargar'")
    if (reproceso or reabrir_desde) and not motivo:
        raise ValueError('reproceso y reabrir_desde exigen motivo')
    if aceptar_deriva and not motivo_deriva:
        raise ValueError('aceptar_deriva exige motivo_deriva')
    familias, registro = cargar_registro(raiz / 'config' / 'programas.yaml')
    desconocidos = [p for p in programas if p not in registro]
    if desconocidos:
        raise ValueError(f'Programas desconocidos: {desconocidos}. Disponibles: {sorted(registro)}')
    umbrales = {**UMBRAL_DERIVA_DEFECTO, **(umbral_deriva or {})}

    periodo_reporte = mes.date()
    respaldos = {}                       # estado local respaldado por programa (ver estado_local.py)
    co = Corrida(nueva_corrida_id(mes), mes, modo, {p: {} for p in programas})
    res, log = co.res, co.log
    _say(verbose, f'Corrida {co.corrida_id} | cierre {periodo_reporte} | programas {list(programas)} | modo {modo} | insumos {insumos}')

    # ── servicios (BigQuery) ──
    if servicios is None:
        try:
            servicios = ServiciosBQ()
        except Exception as e:
            if modo == 'cargar':
                raise
            _say(verbose, f'[aviso] sin conexión a BigQuery ({type(e).__name__}); se asume primera carga para la vista previa')

    def _publicado(tabla, p):
        if servicios is None:
            return None
        try:
            return servicios.leer_publicado(tabla, p)
        except Exception as e:
            if modo == 'cargar':
                raise
            _say(verbose, f'[aviso] no pude leer lo publicado de {tabla} ({type(e).__name__}); se asume primera carga')
            return None

    # ── Etapa 0: manifiesto y verificación de pares ──
    for p in programas:
        prog = registro[p]
        try:
            man = trazabilidad.construir_manifiesto(prog, familias, raiz, co.corrida_id, mes)
            res[p]['manifiesto'], res[p]['config_hash'] = man, man['config_hash']
        except Exception as e:
            res[p]['error'] = f'{type(e).__name__}: {e}'
            log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'ejecucion', 'falla', 1, res[p]['error'][:500]))
            _say(verbose, f'[{p}] FALLÓ al construir el manifiesto — {res[p]["error"][:300]}')
            continue
        _say(verbose, f"[{p}] config_hash {man['config_hash']} (código {man['hash_codigo']}, configuración {man['hash_config']}, entorno {man['hash_entorno']})")
        if prog.motor == 'familia' and prog.pares and servicios is not None and not ignorar_pares:
            try:
                ruta = raiz / 'data' / p / prog.pares['ruta'].format(programa=p)
                pub_pares = pares.publicados_normalizados(servicios.leer_pares(p))
                if prog.provisional_cosechas and frontera.provisional_activo(periodo_reporte, prog.provisional_desde, prog.provisional_hasta):
                    # una cosecha provisional (el G04) cambia de pares a propósito (preliminar -> definitivo): no se verifica contra lo publicado
                    pub_pares = pares.excluir_cosechas(pub_pares, prog.provisional_cosechas)
                v = pares.verificar(prog.pares['tipo'], ruta, pub_pares)
                if v['estado'] == 'restaurable':
                    n = pares.restaurar_mensual(pub_pares, ruta)
                    v['estado'], v['restaurados'] = 'restaurado', n
                    _say(verbose, f'[{p}] pares: el archivo local no existía; se restauraron {n:,} pares desde lo publicado en BigQuery ({ruta.name})')
                res[p]['pares'] = v
                log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'pares', 'estado', None, v['estado']))
                if v['estado'] in ESTADOS_PARES_QUE_BLOQUEAN:
                    msg = (f"pares locales {v['estado']} frente a lo publicado ({v['n_difieren']:,} distintos, {v['n_faltan_local']:,} faltan) en {ruta}: "
                           'volver a emparejar cambiaría meses ya publicados')
                    if modo == 'cargar':
                        res[p]['error'] = 'PARES: ' + msg
                        _say(verbose, f'[{p}] BLOQUEADO — {msg}')
                    else:
                        _say(verbose, f'[{p}] ⚠ {msg} (en modo cargar se bloquearía)')
                else:
                    _say(verbose, f"[{p}] pares: {v['estado']} ({v['n_publicados']:,} publicados)")
            except Exception as e:
                if modo == 'cargar':
                    raise
                _say(verbose, f'[{p}] [aviso] no pude verificar los pares ({type(e).__name__}: {e})')

    # ── Etapa 1: ejecutar ──
    inicio = time.time()
    for p in programas:
        if 'error' in res[p]:
            continue
        prog = registro[p]
        workdir = raiz / 'data' / p
        try:
            seg, esp_raw = None, None
            if prog.motor == 'familia':
                if ejecutar:
                    contenidos = None
                    if insumos == 'reproducir' and reproducir_de:
                        contenidos = {e['key']: e['content_hash'] for e in trazabilidad.cargar_manifiesto(raiz, reproducir_de, p)['insumos']}
                    params = {}
                    if prog.calendario_cosechas:
                        params['CALENDARIO_COSECHAS'] = prog.calendario_cosechas
                    if prog.script == 'champion':
                        # entrada de cada cliente según lo ya publicado: las altas que solo vienen en la lista de clientes no cambian de entrada entre cierres
                        params['ENTRADAS_PUBLICADAS'] = frontera.entradas_publicadas(_publicado(T_TON, p))
                    if prog.familia == 'mensual' and not validar_cerrados:
                        # CPB/CPM: cada mes se mide por separado -> se calcula solo lo posterior a lo ya publicado
                        _F0 = frontera.calcular_frontera(_publicado(T_TON, p), periodo_reporte, prog.provisional_desde, reabrir_desde,
                                                         provisional_cosechas=prog.provisional_cosechas, provisional_hasta=prog.provisional_hasta)
                        if _F0 is not None:
                            params['RECALCULAR_DESDE'] = (_F0 + pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d')
                            _say(verbose, f"[{p}] se calcula desde {params['RECALCULAR_DESDE']} (meses cerrados sin recalcular; "
                                          f"validar_cerrados=True para medir su deriva)")
                    if p not in respaldos:   # el estado local se restaura al final salvo que el programa se cargue de verdad (ver estado_local.py)
                        respaldos[p] = estado_local.respaldar(estado_local.rutas_de_estado(workdir, prog.pares, p))
                    r = ejecutar_familia(prog.familia, script_de_programa(prog), p, mes, workdir, familias,
                                         insumos=insumos, contenidos=contenidos, parametros=params)
                    ton_raw, bre_raw, esp_raw, seg = r.ton, r.bre, r.esp, r.segundos
                    trazabilidad.completar_con_insumos(res[p]['manifiesto'], r.insumos)
                    _guardar_crudas(raiz, co.corrida_id, p, ton_raw, bre_raw, esp_raw)
                    if r.insumos:
                        origen = pd.Series([e['origen'] for e in r.insumos]).value_counts().to_dict()
                        _say(verbose, f"[{p}] insumos: {len(r.insumos)} consultas {origen}; hash {res[p]['manifiesto']['hash_insumos']}")
                else:
                    ton_raw, bre_raw, esp_raw = _cargar_crudas(raiz, p, mes)
            else:
                raise ValueError(f"{p}: motor '{prog.motor}' no soportado (solo 'familia'; los motores-notebook se retiraron)")
            res[p].update(ton_raw=ton_raw, bre_raw=bre_raw, esp_raw=esp_raw, segundos=seg)
            if esp_raw is None:
                _say(verbose, f'[{p}] aviso: este motor no emite la base de espejos (solo los motores en proceso la emiten)')
            if seg is not None:
                log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'tiempo', 'segundos_motor', seg))
            _say(verbose, f'[{p}] motor {prog.motor}: ' + (f'{seg / 60:.1f} min' if seg else 'salida existente'))
        except Exception as e:
            res[p]['error'] = f'{type(e).__name__}: {e}'
            log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'ejecucion', 'falla', 1, res[p]['error'][:500]))
            _say(verbose, f'[{p}] FALLÓ — {res[p]["error"][:300]}')
            donde, ruta_tb = ubicar_fallo(e), guardar_traceback(e, raiz, co.corrida_id, p)
            if donde:
                _say(verbose, f'[{p}]   en {donde}')
            if ruta_tb:
                _say(verbose, f'[{p}]   traceback completo: {ruta_tb}')

    # ── Etapa 2: normalizar y validar (con el config_hash en cada fila) ──
    for p in programas:
        if 'ton_raw' not in res[p]:
            continue
        ch = res[p]['config_hash']
        ton = normalizacion.preparar_base_ton(res[p]['ton_raw'], p, periodo_reporte, co.corrida_id, config_hash=ch,
                                              regla_estado=familias[registro[p].familia].get('estado_yoy', 'estandar'))
        bre = normalizacion.preparar_base_brechas(res[p]['bre_raw'], p, periodo_reporte, co.corrida_id, config_hash=ch)
        esp = (normalizacion.preparar_base_espejos(res[p]['esp_raw'], p, periodo_reporte, co.corrida_id, config_hash=ch)
               if res[p].get('esp_raw') is not None else None)
        errores, resumen = normalizacion.validar(ton, bre, periodo_reporte, ton_original=res[p]['ton_raw'], esp=esp)
        res[p].update(ton=ton, bre=bre, esp=esp, errores=errores, resumen=resumen)
        _say(verbose, f'\n=== {p} ===')
        for k, v in resumen.items():
            _say(verbose, f'  {k}: {v}')
        for e in errores:
            _say(verbose, '  ERROR:', e)
        log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'qa', 'errores_validacion', len(errores)))
        for k in ('filas_ton', 'filas_brechas', 'valor_atribuible_tn_total'):
            log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'qa', k, resumen[k]))

    # ── Etapa 3: frontera de meses cerrados + deriva (README.md) ──
    # Se publica solo lo posterior a la frontera; lo cerrado se recalcula únicamente para medir la deriva (nunca se publica).
    for p in programas:
        if 'ton' not in res[p] or res[p]['errores']:
            continue
        res[p]['alertas'] = []
        pub_ton = _publicado(T_TON, p)
        _rp = registro[p]
        F = frontera.calcular_frontera(pub_ton, periodo_reporte, _rp.provisional_desde, reabrir_desde,
                                       provisional_cosechas=_rp.provisional_cosechas, provisional_hasta=_rp.provisional_hasta)
        prov = ({'desde': _rp.provisional_desde, 'cosechas': _rp.provisional_cosechas}
                if _rp.provisional_cosechas and frontera.provisional_activo(periodo_reporte, _rp.provisional_desde, _rp.provisional_hasta) else None)
        ya = frontera.rebanada_existente(pub_ton, periodo_reporte)
        res[p]['frontera'] = F
        posterior = frontera.cierre_posterior(pub_ton, periodo_reporte)
        if posterior is not None:
            msg = f'ya hay publicado un cierre posterior ({posterior.date()}): un cierre anterior no se puede cargar'
            if modo == 'cargar':
                res[p]['error'] = 'CIERRE: ' + msg
                _say(verbose, f'[{p}] BLOQUEADO — {msg}')
                continue
            _say(verbose, f'[{p}] ⚠ {msg} (en modo cargar se bloquearía)')
        _say(verbose, f"[{p}] frontera de meses cerrados: {F.date() if F is not None else 'ninguna (primera carga)'}"
                      + (f" · provisionales desde {_rp.provisional_desde}" + (f" (cosechas {_rp.provisional_cosechas})" if _rp.provisional_cosechas else '')
                         if _rp.provisional_desde and frontera.provisional_activo(periodo_reporte, _rp.provisional_desde, _rp.provisional_hasta) else '')
                      + (f" · reabierto desde {reabrir_desde}" if reabrir_desde else '')
                      + (' · este cierre YA está publicado' + (' (reproceso: se republica con versión nueva)' if reproceso else ': no se cargará nada') if ya else ''))
        for tabla, k in ((T_TON, 'ton'), (T_BRE, 'bre'), (T_ESP, 'esp')):
            df = res[p].get(k)
            if df is None:
                continue
            pub = pub_ton if tabla == T_TON else _publicado(tabla, p)
            publicar, cerradas = frontera.separar(df, F, prov)
            # validación: lo recalculado de los meses cerrados (o del cierre entero, si ya estaba publicado) contra lo publicado
            comparar = df if ya else cerradas
            pub_cmp = pub if (ya or pub is None or F is None) else frontera.separar(pub, F, prov)[1]   # lo publicado de los meses cerrados (sin lo provisional)
            _, r, deriva = incremental.comparar_con_publicado(comparar, pub_cmp, tabla)
            if ya and not reproceso:
                publicar = df.iloc[0:0]
            incr = {**frontera.resumen(F, publicar, cerradas, r), 'filas_nuevas': len(publicar),
                    'filas_ya_publicadas': r['filas_ya_publicadas'], 'filas_con_deriva': r['filas_con_deriva'],
                    'filas_publicadas_ausentes_hoy': r['filas_publicadas_ausentes_hoy'], 'suma_abs_deriva': r['suma_abs_deriva']}
            res[p][f'cargar_{k}'], res[p][f'incr_{k}'], res[p][f'deriva_{k}'] = publicar, incr, deriva
            for kk, v in incr.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'incremental', f'{tabla}.{kk}', v))
                else:
                    log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'incremental', f'{tabla}.{kk}', None, str(v)[:500]))
            _say(verbose, f'[{p}] {tabla}: {incr}')
            r = incr
            if len(deriva):
                diag = diagnostico_deriva(pub, deriva, tabla, res[p]['config_hash'])
                pct = 100 * r['filas_con_deriva'] / max(r['filas_ya_publicadas'], 1)
                _say(verbose, f'   ⚠ {len(deriva):,} llaves ya publicadas cambiaron al recalcular ({pct:.2f}% de lo publicado; no se aplican). Causa: {diag}')
                _say(verbose, deriva.head(5).to_string(index=False))
                log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'deriva', f'{tabla}.diagnostico', pct, diag[:200]))
                if tabla in umbrales and pct > umbrales[tabla]:
                    res[p]['alertas'].append(f'deriva en {tabla}: {pct:.2f}% de filas > {umbrales[tabla]}%')
            if tabla in (T_TON, T_BRE):
                cel = incremental.deriva_por_celda(tabla, comparar, pub_cmp, tolerancia_celda)
                res[p][f'celdas_{k}'] = cel
                fuera = cel[cel['fuera'].astype(bool)] if len(cel) else cel
                log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'deriva', f'{tabla}.celdas_fuera', len(fuera)))
                if len(cel):
                    _say(verbose, f"[{p}] {tabla}: {len(cel) - len(fuera)} de {len(cel)} celdas mes x categoría"
                                  f"{' x cosecha' if tabla == T_BRE else ''} dentro de tolerancia")
                if len(fuera):
                    _say(verbose, fuera.head(10).round({'publicado': 4, 'recalculado': 4, 'dif': 4, 'dif_pct': 2}).to_string(index=False))
                    _max = fuera['dif'].abs().max()
                    res[p]['alertas'].append(f"deriva en {tabla}: {len(fuera)} celdas mes x categoría fuera de tolerancia "
                                             f"(máx |dif| {_max:.3f}{' pp' if tabla == T_BRE else ' t'})")

    # bloqueos por deriva sobre el umbral (se resuelven al cargar)
    for p in programas:
        if res[p].get('alertas') and modo == 'cargar' and 'error' not in res[p]:
            if aceptar_deriva or reproceso:                       # un reproceso explícito (con motivo) es una decisión de republicar
                por_que = motivo_deriva if aceptar_deriva else f'reproceso: {motivo}'
                log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'deriva', 'aceptada', 1, por_que[:500]))
                _say(verbose, f'[{p}] deriva sobre el umbral ACEPTADA: {por_que}')
            else:
                res[p]['error'] = 'DERIVA: ' + '; '.join(res[p]['alertas']) + ' (usa aceptar_deriva con un motivo, o reproceso, si el cambio es legítimo)'
                _say(verbose, f'[{p}] BLOQUEADO — {res[p]["error"]}')
        elif res[p].get('alertas'):
            _say(verbose, f'[{p}] ⚠ {"; ".join(res[p]["alertas"])} (en modo cargar se bloquearía)')

    # ── Etapa 3b: UB ──
    if not calcular_ub:
        _say(verbose, 'UB desactivada: las columnas de UB quedan nulas')
    elif servicios is None:
        _say(verbose, '[aviso] sin BigQuery: no se puede calcular la UB (columnas nulas)')
    else:
        for p in programas:
            r = res[p]
            if 'cargar_ton' not in r or r['errores'] or 'error' in r or len(r['cargar_ton']) == 0:
                continue
            metodo_ub = registro[p].metodo_ub
            ub_df = servicios.consulta_ub(r['cargar_ton'], registro[p].negocios_ub, meses_atras=12 if metodo_ub == 'notebook' else 0)
            r['cargar_ton'], r['cobertura_ub'] = ub.aplicar_ub(r['cargar_ton'], ub_df, metodo=metodo_ub)
            cob = r['cobertura_ub']
            _say(verbose, f'[{p}] UB ({metodo_ub}): {cob}')
            for k, v in cob.items():
                log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'ub', k, v))
            if cob['pct_cobertura_ub'] < 90 or not 0.9 <= cob['ton_detalle_vs_base'] <= 1.1:
                _say(verbose, f'   ⚠ [{p}] cobertura de UB baja o toneladas del detalle DEX distintas a las del motor: revisar antes de cargar')

    # ── Etapa 4: cargar ──
    if modo == 'cargar':
        for p in programas:
            if 'cargar_ton' not in res[p] or res[p]['errores'] or 'error' in res[p]:
                continue
            # las 3 tablas se revisan contra el esquema real ANTES de cargar la primera: una tabla que no admite la carga no deja las otras a medias
            try:
                for tabla, k in ((T_BRE, 'bre'), (T_ESP, 'esp'), (T_TON, 'ton')):
                    d = res[p].get(f'cargar_{k}')
                    if d is not None and len(d) and hasattr(servicios, 'verificar_carga'):
                        servicios.verificar_carga(d.assign(version=1), tabla)
            except Exception as e:
                res[p]['error'] = f'CARGA: {e}'
                _say(verbose, f'[{p}] NO SE CARGA NADA — {e}')
                continue
            for tabla, k in ((T_BRE, 'bre'), (T_ESP, 'esp'), (T_TON, 'ton')):  # brechas, espejos y al final toneladas
                if f'cargar_{k}' not in res[p]:
                    continue
                df = res[p][f'cargar_{k}'].copy()
                ultima = servicios.ultima_version(tabla, p, periodo_reporte)
                version = (ultima or 0) + 1 if reproceso else (ultima or 1)
                if len(df) == 0:
                    _say(verbose, f'[{p}] {tabla}: nada nuevo que cargar')
                    continue
                df['version'] = version
                n = servicios.cargar(df, tabla)
                log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'carga', f'{tabla}.filas_cargadas', n))
                log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'carga', f'{tabla}.version', version))
                _say(verbose, f'[{p}] {tabla}: {n:,} filas cargadas (version {version})')
            if reproceso:
                log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'carga', 'reproceso', 1, motivo[:500]))
            res[p]['cargado'] = True
    else:
        _say(verbose, "DRY-RUN: no se escribió nada en BigQuery. Usa modo='cargar' para publicar.")

    # Estado local: solo queda actualizado si el programa se cargó de verdad. Prueba, bloqueo o error -> como estaba antes de la corrida.
    for p, resp in respaldos.items():
        if modo == 'cargar' and res[p].get('cargado'):
            estado_local.descartar(resp)
        else:
            fallos = estado_local.restaurar(resp, verbose=verbose)
            if fallos:
                res[p]['estado_no_restaurado'] = fallos
            else:
                _say(verbose, f"[{p}] estado local (pares / estado congelado) restaurado: esta corrida no lo deja modificado")

    # ── Etapa 5: manifiestos, resumen, log y exportes ──
    for p in programas:
        man = res[p].get('manifiesto')
        if man:
            trazabilidad.guardar_manifiesto(raiz, man)
            for k in ('config_hash', 'hash_codigo', 'hash_config', 'hash_entorno', 'hash_insumos', 'commit_git'):
                if man.get(k):
                    log.append(bigquery.fila_log(co.corrida_id, p, periodo_reporte, 'manifiesto', k, None, str(man[k])))
    filas = []
    for p in programas:
        r = res[p]
        estado = ('bloqueado' if str(r.get('error', '')).startswith(('DERIVA', 'PARES', 'CIERRE', 'CARGA')) else 'falló' if 'error' in r
                  else 'con errores' if r.get('errores') else 'ok')
        filas.append({
            'programa': p, 'estado': estado,
            'frontera': (r['frontera'].date().isoformat() if r.get('frontera') is not None else None),
            'meses_publicados': ', '.join(r.get('incr_ton', {}).get('meses_a_publicar', []) or []) or None,
            'minutos_motor': round(r['segundos'] / 60, 1) if r.get('segundos') else None,
            'ton_nuevas': r.get('incr_ton', {}).get('filas_nuevas'),
            'ton_deriva': r.get('incr_ton', {}).get('filas_con_deriva'),
            'brechas_nuevas': r.get('incr_bre', {}).get('filas_nuevas'),
            'brechas_deriva': r.get('incr_bre', {}).get('filas_con_deriva'),
            'espejos_nuevos': r.get('incr_esp', {}).get('filas_nuevas'),
            'espejos_deriva': r.get('incr_esp', {}).get('filas_con_deriva'),
            'valor_atrib_tn': r.get('resumen', {}).get('valor_atribuible_tn_total'),
            'config_hash': r.get('config_hash'),
            'alertas': '; '.join(r.get('alertas', [])) or None,
        })
    co.resumen = pd.DataFrame(filas)
    if modo == 'cargar' and servicios is not None:
        servicios.registrar_log(log)
        if exportar_excel:
            for p in programas:
                if 'cargar_ton' in res[p] and not res[p]['errores'] and 'error' not in res[p]:
                    _say(verbose, p, '->', servicios.exportar(p, mes, str(raiz / 'exportes')))
    return co


def main(argv=None):
    ap = argparse.ArgumentParser(description='Corrida mensual de mediciones')
    ap.add_argument('--mes', required=True, help='Cierre, día 1 (YYYY-MM-01)')
    ap.add_argument('--programas', nargs='+', required=True)
    ap.add_argument('--modo', default='dry_run', choices=['dry_run', 'cargar'])
    ap.add_argument('--sin-ejecutar', action='store_true', help='Reutiliza las salidas ya generadas este mes')
    ap.add_argument('--sin-ub', action='store_true')
    ap.add_argument('--reproceso', action='store_true')
    ap.add_argument('--motivo', default='')
    ap.add_argument('--insumos', default='consultar', choices=['consultar', 'reutilizar', 'reproducir', 'desactivado'])
    ap.add_argument('--reproducir-de', default=None, help='corrida_id cuyos insumos exactos se usan (con --insumos reproducir)')
    ap.add_argument('--aceptar-deriva', action='store_true', help='permite cargar aunque la deriva supere el umbral (exige --motivo-deriva)')
    ap.add_argument('--motivo-deriva', default='')
    ap.add_argument('--ignorar-pares', action='store_true', help='omite la verificación de pares congelados contra lo publicado')
    ap.add_argument('--validar-cerrados', action='store_true', help='CPB/CPM: recalcula también los meses cerrados para medir su deriva (más lento)')
    ap.add_argument('--reabrir-desde', default=None, help='AAAA-MM-01: reabre meses ya cerrados desde ese mes (exige --motivo); se republican en este cierre')
    ap.add_argument('--raiz', default=None)
    a = ap.parse_args(argv)
    co = correr_mes(a.mes, a.programas, modo=a.modo, ejecutar=not a.sin_ejecutar, calcular_ub=not a.sin_ub, reproceso=a.reproceso, motivo=a.motivo,
                    insumos=a.insumos, reproducir_de=a.reproducir_de, aceptar_deriva=a.aceptar_deriva, motivo_deriva=a.motivo_deriva,
                    ignorar_pares=a.ignorar_pares, reabrir_desde=a.reabrir_desde, validar_cerrados=a.validar_cerrados, raiz=a.raiz)
    print(co.resumen.to_string(index=False))


if __name__ == '__main__':
    main()
