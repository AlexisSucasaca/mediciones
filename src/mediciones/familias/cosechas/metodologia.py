# flake8: noqa
# ruff: noqa
# Script de FAMILIA: se ejecuta con exec() en un espacio de nombres compartido con el script del programa
# (mediciones.ejecucion.ejecutar_familia). No es un módulo importable. MANTENIDO A MANO: este archivo es la fuente de verdad (los notebooks originales, fuera de este paquete, quedan solo como referencia histórica).
# =============================================================================
# METODOLOGÍA DE COSECHAS (surtido y Champion): matching húngaro → panel DiD + bootstrap → atribuible IC80 → retroactivo → Inka Chips (diferencial YoY) → base de cálculo
# =============================================================================
from mediciones.parametros import (  # parámetros metodológicos comunes a las dos familias
    VENTANA_MESES,
    OUTLIER_METODO,
    OUTLIER_K,
    OUTLIER_NIVEL,
    USAR_LOG,
    CALIPER_REL,
    CALIPER_ULTIMO,
    NIVEL_MAX,
    CORR_UMBRAL,
    N_SEMILLAS_DEFAULT,
    MAX_SEMILLAS_PROBAR_EQUIVALENCIA,
    MARGEN_ERROR_OBJETIVO,
    HORIZONTE_MESES_REPORTE,
    MARGEN_EQUIVALENCIA_PEND,
    ALPHA_TOST,
    NIVELES_IC,
    N_BOOT,
    SEED_BOOT,
)
def leer_estado_matching():
    """Último cierre cuyos entrantes del G03 ya fueron matcheados (JSON), o FECHA_CORTE_BASE si no hay estado."""
    if os.path.exists(RUTA_ESTADO):
        with open(RUTA_ESTADO) as fh:
            return pd.Timestamp(json.load(fh)['ultimo_cierre_matcheado'])
    return FECHA_CORTE_BASE

def _leer_congelado(nombre, cols_fecha=(), cols_id=()):
    ruta = os.path.join(RUTA_SALIDA_MATCH, nombre)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. El modo mensual necesita el matching base ya guardado "
            f"(celda de guardado de la corrida v7). Si no lo tienes, usa REHACER_MATCHING=True una vez.")
    # los códigos se leen como string: si no, se pierden los ceros a la izquierda ('0004164673')
    df = pd.read_csv(ruta, dtype={c: str for c in cols_id}, low_memory=False)
    for c in cols_fecha:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def cargar_matching_congelado():
    """Lee de disco el matching ya cerrado (G01, G02 y G03-base + entrantes de cierres previos)."""
    return {
        'matches':    _leer_congelado('df_matches.csv', ('cosecha', 'min_per_ac'), ('cod_accion', 'cod_control')),
        'sin_pareja': _leer_congelado('df_sin_pareja.csv', ('cosecha', 'min_periodo_cosecha'), ('cod_accion',)),
        'funnel':     _leer_congelado('df_funnel.csv', ('cosecha',)),
        'resumen':    _leer_congelado('df_resumen.csv', ('cosecha',)),
        'semillas':   _leer_congelado('df_semillas.csv', ('cosecha',)),
        'viabilidad': _leer_congelado('df_viabilidad.csv', ('cosecha',)),
    }

# =============================================================================
# TAMAÑO DE MUESTRA MÍNIMO — reemplaza los umbrales fijos (n_min=30, MIN_CT_OFICINA=30)
# que no tenían relación con el tamaño real de la población. Fórmula estándar de
# muestreo para una proporción (p=0.5, caso conservador) CON corrección por
# población finita, porque acá la "población" (AC_limpio, CT_limpio, etc.) es
# conocida y relativamente chica -> ignorar la corrección sobreestima el n necesario.
# =============================================================================
from mediciones.metodologia.muestra import (CONFIANZA_OBJETIVO, Z_CONF, n_minimo_muestra, margen_error_real, n_minimo_requerido)   # compartido con Champion (antes copiado aquí, idéntico)


# =============================================================================
# PROYECTO ROMA — PIPELINE DE MATCHING v4 (corregido)
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
from sklearn.metrics import pairwise_distances
import statsmodels.formula.api as smf
from mediciones.metodologia.atribucion import atribuir, estado_yoy, valor_atribuible   # atribución compartida (regla de 'nuevo' = REGLA_NUEVO, de config/programas.yaml)

pd.set_option('display.width', 250)

# ─────────────────────────────────────────────────────────────────────────────
# TEST FORMAL DE TENDENCIAS PARALELAS (pre-cosecha), por celda cosecha×categoría
# ─────────────────────────────────────────────────────────────────────────────
def test_paralelismo(cod_accion, cod_control, ac_df, ct_df, cols_m):
    """
    H0: la pendiente pre-cosecha de AC y CT es la MISMA (paralelismo sí se cumple).
    Regresión: log1p(peso) ~ tiempo + grupo + tiempo:grupo
    Errores estándar clusterizados por cliente (cada cliente aporta 12 meses,
    autocorrelacionados entre sí -> sin clusterizar el p-valor sería artificialmente bajo).
    Se evalúa sobre el PAR FINAL matcheado (no sobre toda la población), porque es
    la comparación que efectivamente sustenta la brecha DiD reportada.
    Devuelve: p-valor de la interacción tiempo×grupo.
    """
    filas = []
    for cid in cod_accion:
        vals = np.log1p(ac_df.loc[cid, cols_m].to_numpy(float))
        for t, v in enumerate(vals):
            filas.append((f"AC_{cid}", t, 1, v))
    for cid in cod_control:
        vals = np.log1p(ct_df.loc[cid, cols_m].to_numpy(float))
        for t, v in enumerate(vals):
            filas.append((f"CT_{cid}", t, 0, v))
    dfl = pd.DataFrame(filas, columns=['cliente', 'tiempo', 'grupo', 'valor'])
    dfl['inter'] = dfl['tiempo'] * dfl['grupo']
    try:
        mod = smf.ols('valor ~ tiempo + grupo + inter', data=dfl).fit(
            cov_type='cluster', cov_kwds={'groups': dfl['cliente']})
        return float(mod.pvalues.get('inter', np.nan))
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# TEST DE EQUIVALENCIA (TOST) — reemplaza el test clásico de igualdad de pendientes.
# El test viejo (H0: pendientes iguales, no rechazar con p>0.10) tiene un problema
# real en celdas con muchos clientes: con más datos, el test tiene más poder para
# detectar CUALQUIER micro-diferencia de pendiente, así que rechaza "paralelismo"
# por divergencias sin relevancia de negocio -- no porque los grupos no sean
# comparables, sino porque el test es cada vez más exigente con más n.
# TOST invierte la pregunta: en vez de "¿puedo rechazar que son iguales?", pregunta
# "¿puedo afirmar que la diferencia está dentro de un margen que no importa?".
# Se acepta equivalencia si el IC (1 - 2*alpha) de la interacción tiempo×grupo cae
# ENTERO dentro de [-margen, +margen].
# ─────────────────────────────────────────────────────────────────────────────
def test_equivalencia_paralelismo(cod_accion, cod_control, ac_df, ct_df, cols_m,
                                   margen_coef, alpha=0.05):
    """
    Devuelve (coef, ic_inf, ic_sup, equivalencia_ok) para la pendiente diferencial
    (interacción tiempo×grupo) del PAR FINAL matcheado, sobre la ventana pre-cosecha.
    coef/ic están en escala log1p(peso) por mes (misma escala que el test viejo).
    """
    filas = []
    for cid in cod_accion:
        vals = np.log1p(ac_df.loc[cid, cols_m].to_numpy(float))
        for t, v in enumerate(vals):
            filas.append((f"AC_{cid}", t, 1, v))
    for cid in cod_control:
        vals = np.log1p(ct_df.loc[cid, cols_m].to_numpy(float))
        for t, v in enumerate(vals):
            filas.append((f"CT_{cid}", t, 0, v))
    dfl = pd.DataFrame(filas, columns=['cliente', 'tiempo', 'grupo', 'valor'])
    dfl['inter'] = dfl['tiempo'] * dfl['grupo']
    try:
        mod = smf.ols('valor ~ tiempo + grupo + inter', data=dfl).fit(
            cov_type='cluster', cov_kwds={'groups': dfl['cliente']})
        coef = float(mod.params.get('inter', np.nan))
        ic = mod.conf_int(alpha=2 * alpha)
        ic_inf, ic_sup = float(ic.loc['inter', 0]), float(ic.loc['inter', 1])
        ok = bool(pd.notna(coef) and ic_inf > -margen_coef and ic_sup < margen_coef)
        return coef, ic_inf, ic_sup, ok
    except Exception:
        return np.nan, np.nan, np.nan, False


# ─────────────────────────────────────────────────────────────────────────────
# MATCHING PARA UNA SEMILLA — extraído a función para poder paralelizar (joblib)
# Cada llamada es independiente entre sí (misma ac/ct, distinta submuestra por seed)
# -> ideal para ejecutar en paralelo entre los N_JOBS núcleos disponibles.
# ─────────────────────────────────────────────────────────────────────────────
from mediciones.nucleo.matching import _correr_semilla  # kernel idéntico en ambas familias (ver nucleo/matching.py)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 0 — PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

# Etiquetas de negocio para las cosechas agrupadas (mismo mapeo de la celda 22)

# MIN_MESES por categoría — PARAMETRIZABLE (dict). Corrida de comparación con
# Cereales=6 (igual que Galletas), para aislar el efecto de esa relajación del
# resto de cambios. El valor acordado con el negocio (Cereales=4) queda
# documentado en las instrucciones del proyecto para cuando se reactive.
EXCLUIR_NUEVOS = True

DROP_FILAS_NEG  = True
EXIGIR_POSITIVO = True


# Muestreo — depende SOLO de la población de acción (FIX 2)
MUESTREAR     = True
FRAC_MUESTREO = 0.60
ESTRATOS_FULL = ['min_periodo_cosecha', 'des_oficina',
                 'nom_compania', 'des_segmento_transaccional']
MIN_POR_CELDA = 5             # mediana de celda bajo esto -> baja nivel de cascada

# Matching — sin min_periodo_cosecha (los CT no lo tienen)
MATCH_FULL     = ['des_oficina', 'nom_compania', 'des_segmento_transaccional']

W_NIVEL  = 0.5                # cercanía (default)
W_FORMA  = 0.5                # paralelismo (default)

# ── MARGEN DE EQUIVALENCIA PARA PARALELISMO (TOST) ──────────────────────────
# En vez de testear "¿son las pendientes EXACTAMENTE iguales?", probamos "¿la
# diferencia de pendiente es lo bastante chica como para no mover la brecha de
# forma relevante?". El margen se define reutilizando el mismo estándar de
# negocio ya acordado (5% de margen de error, ver celda 'Tamaño de muestra
# mínimo'): es la pendiente mensual (en log1p) que, proyectada sobre
# HORIZONTE_MESES_REPORTE meses, movería la brecha en menos de ese 5%.
# *** PENDIENTE DE CONFIRMAR CON NEGOCIO: horizonte y % -- valores puestos por
# default, documentados para que se ajusten antes de reportar oficialmente. ***
                                        # (el TOST con OLS clusterizado es más caro que el
                                        # resto de la selección; no vale la pena correrlo
                                        # sobre las 50 semillas si las primeras ya alcanzan)

RATIO_MIN, RATIO_MAX = 0.70, 1.40

# ── Overrides puntuales — REACTIVADOS: G1-Cereales y G2-Galletas fallan
# 'valida' por ratio_vol fuera de rango (0.568 y 1.706 vs [0.70,1.40]) pese a
# tener corr razonable (0.769 y 0.758) -- mismo síntoma que Junio-Galletas ya
# documentado en las instrucciones del proyecto (ratio_vol fuera de rango con
# población balanceada). Se les da más peso a la forma de la trayectoria
# (W_FORMA=0.7, en vez de matchear casi solo por nivel/estrato) y más semillas
# (150 en vez de 50) para que la búsqueda tenga más chances de encontrar un
# control cuya volatilidad pre-cosecha se parezca más a la de acción.

# ── Reuso de control 1:N — DESACTIVADO en esta corrida de comparación ──
# 1:1 estricto en TODAS las cosechas (comportamiento original de v6), para aislar
# el efecto del resto de cambios sobre el resultado. El reuso 1:N en Grupo 01
# (justificado por escasez real de control, 96-98% del CT ya en uso en 1:1) queda
# documentado en las instrucciones del proyecto para cuando se reactive.

COL_TRATADO    = 'cosecha_agrupada'
GRAFICAR       = True

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — ROL A NIVEL CLIENTE
# ─────────────────────────────────────────────────────────────────────────────
df = base_seg_limpio.copy()
df['periodo']          = pd.to_datetime(df['periodo'])
df['cosecha_agrupada'] = pd.to_datetime(df['cosecha_agrupada'])
if 'min_periodo_cosecha' in df.columns:
    df['min_periodo_cosecha'] = pd.to_datetime(df['min_periodo_cosecha'])

cli_cosecha  = df.dropna(subset=[COL_TRATADO]).groupby('cod_cliente')[COL_TRATADO].min()
TRATADOS_ALL = set(cli_cosecha.index)

# FIX 1: "nuevo" = primera transacción REAL en toda la base
cli_primera_compra = df.groupby('cod_cliente')['periodo'].min()
print(f"[COBERTURA] La base arranca en {df['periodo'].min().date()}. "
      f"Un cliente que ya compraba antes de esa fecha se marcará como nuevo por error.")

cli_min_per = (df.dropna(subset=['min_periodo_cosecha'])
                 .groupby('cod_cliente')['min_periodo_cosecha'].min()
               if 'min_periodo_cosecha' in df.columns
               else pd.Series(dtype='datetime64[ns]'))

print(f"Clientes en base : {df['cod_cliente'].nunique():,}")
print(f"Acciones (todas) : {len(TRATADOS_ALL):,}")
print(f"Candidatos espejo: {df['cod_cliente'].nunique() - len(TRATADOS_ALL):,}")

ct_ids = set(df['cod_cliente'].unique()) - TRATADOS_ALL
ct_con_min = len(set(cli_min_per.index) & ct_ids)
print(f"\n[CHEQUEO] Controles con min_periodo_cosecha: {ct_con_min:,} de {len(ct_ids):,}")
print("  -> es atributo de ACCIÓN: sirve como ESTRATO, no como llave de matching."
      if ct_con_min < len(ct_ids) * 0.5 else
      "  -> los CT sí lo tienen: puedes moverlo a MATCH_FULL.")

print("\n[ESTRUCTURA] min_periodo_cosecha por cosecha_agrupada:")
print(df.dropna(subset=[COL_TRATADO]).groupby(COL_TRATADO)['min_periodo_cosecha']
        .agg(n_valores='nunique',
             valores=lambda s: [str(x.date()) for x in sorted(s.dropna().unique())])
        .to_string())

# Cascada de matching (FIX 4)
NIVELES = [[len(MATCH_FULL) - k + 1, MATCH_FULL[:k], CALIPER_REL]
           for k in range(len(MATCH_FULL), 0, -1)][:NIVEL_MAX]
NIVELES[-1][2] = CALIPER_ULTIMO

# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# MODO MENSUAL (parámetros en la celda de configuración)
#   REHACER_MATCHING=True  -> corre las 3 cosechas completas (comportamiento original).
#   REHACER_MATCHING=False -> G01, G02 y las parejas ya existentes del G03 quedan CONGELADAS
#     (se leen de disco). Solo se matchean los clientes del G03 que entraron al piloto después
#     del último cierre ya matcheado, contra controles que aún no están en ninguna pareja del G03.
# ─────────────────────────────────────────────────────────────────────────────
# [seleccion_cosechas:inicio] qué cosechas se emparejan en este cierre (calendario: ver mediciones/calendario.py)
#   * la cosecha ABIERTA (la que recibe altas este cierre): solo sus altas nuevas, de forma incremental (G03 hasta setiembre; G04 preliminar en oct-nov)
#   * cosechas a REHACER (el G04 al pasar a definitivo): emparejamiento completo, si todavía no tienen pares definitivos congelados
#   * pares preliminares a DESCARTAR (el G04 anclado en octubre, una vez que pasa a definitivo)
_REHACER_CAL = [pd.Timestamp(c) for c in globals().get('COSECHAS_REHACER', [])]
_DESCARTAR_CAL = [pd.Timestamp(c) for c in globals().get('COSECHAS_DESCARTAR', [])]
COSECHAS_REHACER_AHORA = []
if REHACER_MATCHING:
    CONGELADO = None
    ENTRANTES_G03 = set()
    COSECHAS_A_CORRER = list(COSECHAS)
else:
    CONGELADO = cargar_matching_congelado()
    _ya_definitivas = set(pd.to_datetime(CONGELADO['matches']['cosecha']).dropna().unique()) if len(CONGELADO['matches']) else set()
    COSECHAS_REHACER_AHORA = [c for c in _REHACER_CAL if c not in _ya_definitivas]
    _quitar = set(_DESCARTAR_CAL) | set(COSECHAS_REHACER_AHORA)
    if _quitar:
        for _k, _df in CONGELADO.items():
            if 'cosecha' in _df.columns and len(_df):
                CONGELADO[_k] = _df[~pd.to_datetime(_df['cosecha']).isin(list(_quitar))].reset_index(drop=True)
        print(f"  [CALENDARIO] se descartan del estado congelado las cosechas {sorted(c.strftime('%Y-%m') for c in _quitar)} "
              f"(preliminares que pasan a definitivo / cosechas que se emparejan completas)")
    ULTIMO_CIERRE_MATCHEADO = leer_estado_matching()
    ENTRANTES_G03 = ({c for c, f in cli_min_per.items()
                      if cli_cosecha.get(c, pd.NaT) == COSECHA_ABIERTA
                      and ULTIMO_CIERRE_MATCHEADO < f <= MES_CIERRE}
                     if COSECHA_ABIERTA is not None and COSECHA_ABIERTA not in COSECHAS_REHACER_AHORA else set())
    COSECHAS_A_CORRER = sorted(set(COSECHAS_REHACER_AHORA) | ({COSECHA_ABIERTA} if ENTRANTES_G03 else set()))
    print(f"[MODO MENSUAL] MES_CIERRE={MES_CIERRE.date()} | último cierre matcheado="
          f"{ULTIMO_CIERRE_MATCHEADO.date()} | entrantes de la cosecha abierta a matchear: {len(ENTRANTES_G03):,} | "
          f"cosechas completas: {[c.strftime('%Y-%m') for c in COSECHAS_REHACER_AHORA]}")
    if not COSECHAS_A_CORRER:
        print("  -> sin entrantes nuevos: se reutiliza TODO el matching congelado tal cual.")
# [seleccion_cosechas:fin]

matches_out, sinpar_out, funnel_out, resumen_out, semillas_out, pools_out, viabilidad_out = [], [], [], [], [], {}, []

for cosecha in COSECHAS_A_CORRER:
    for cat in CATEGORIAS:

        ini   = cosecha - relativedelta(months=VENTANA_MESES)
        fin   = cosecha - relativedelta(months=1)
        meses = pd.date_range(ini, fin, freq='MS')

        print("\n" + "=" * 110)
        print(f"COSECHA {cosecha.date()} [{GRUPO_COSECHA.get(cosecha, '?')}]  |  {cat}  |  "
              f"ventana {ini.date()} .. {fin.date()}")
        print("=" * 110)

        min_meses_celda = MIN_MESES_POR_CATEGORIA.get(cat, MIN_MESES_DEFAULT)
        w_nivel_celda = PESOS_OVERRIDE.get((cosecha, cat), {}).get('W_NIVEL', W_NIVEL)
        w_forma_celda = PESOS_OVERRIDE.get((cosecha, cat), {}).get('W_FORMA', W_FORMA)
        n_semillas_celda = N_SEMILLAS_OVERRIDE.get((cosecha, cat), N_SEMILLAS_DEFAULT)
        semillas_celda = list(range(1, n_semillas_celda + 1))
        max_reuso_celda = MAX_REUSO_CONTROL_POR_COSECHA.get(cosecha, MAX_REUSO_CONTROL_DEFAULT)
        if (cosecha, cat) in PESOS_OVERRIDE or (cosecha, cat) in N_SEMILLAS_OVERRIDE:
            print(f"  [OVERRIDE] W_NIVEL={w_nivel_celda} W_FORMA={w_forma_celda} "
                  f"N_SEMILLAS={n_semillas_celda} (celda con override específico)")
        if max_reuso_celda > 1:
            print(f"  [REUSO CONTROL] max_reuso_celda={max_reuso_celda} "
                  f"(un control puede matchear hasta {max_reuso_celda} veces en esta cosecha)")

        w = df[(df['periodo'] >= ini) & (df['periodo'] <= fin) &
               (df['des_categoria'] == cat)].copy()
        if w.empty:
            print("  [SKIP] sin datos"); continue

        n_neg = (w['peso'] < 0).sum()
        if DROP_FILAS_NEG and n_neg:
            print(f"  [NOTA] {n_neg:,} filas peso<0 eliminadas "
                  f"({w.loc[w.peso < 0, 'peso'].sum():,.1f} kg). Compra BRUTA "
                  f"-> misma regla en el post-período.")
            w = w[w['peso'] >= 0]

        wp  = w[w['peso'] > 0]
        agg = (w.groupby('cod_cliente').agg(peso_12m=('peso', 'sum'))
                 .join(wp.groupby('cod_cliente').agg(n_meses=('periodo', 'nunique')))
                 .fillna({'n_meses': 0}))

        tray = (w.pivot_table(index='cod_cliente', columns='periodo',
                              values='peso', aggfunc='sum')
                  .reindex(columns=meses).fillna(0.0).clip(lower=0))
        cols_m = [f"m{i:02d}" for i in range(len(meses))]
        tray.columns = cols_m

        attrs = (w.sort_values('periodo').groupby('cod_cliente').tail(1)
                   .set_index('cod_cliente')[['des_oficina', 'nom_compania',
                                              'des_segmento_transaccional']])

        pool = agg.join(tray).join(attrs)
        pool['min_periodo_cosecha'] = cli_min_per.reindex(pool.index)

        es_ac = pool.index.map(lambda i: cli_cosecha.get(i, pd.NaT) == cosecha)
        pool['rol'] = np.where(es_ac, 'AC',
                        np.where(pool.index.isin(TRATADOS_ALL), 'EXCL', 'CT'))
        # Trayectorias SIN limpiar: las parejas congeladas pueden contener clientes que hoy
        # un filtro de limpieza (outliers) sacaría -> las métricas de la unión no deben fallar por eso.
        pool_pre = pool.copy() if CONGELADO is not None else None

        # ── EMBUDO ───────────────────────────────────────────────────────────
        fn = []
        _snap = lambda e, p: fn.append({'etapa': e,
                                        'AC': int((p.rol == 'AC').sum()),
                                        'CT': int((p.rol == 'CT').sum())})
        _snap('0. Con datos en ventana', pool)
        pool = pool[pool['rol'] != 'EXCL'];  _snap('1. Sin acción de otra cosecha', pool)
        if EXCLUIR_NUEVOS:
            nuevos = pool['n_meses'] == 0    # no compró en los 12 meses pre-cosecha
            pool = pool[~nuevos]
        _snap('2. Sin clientes nuevos', pool)
        if EXIGIR_POSITIVO:
            pool = pool[pool['peso_12m'] > 0]
        _snap('3. Con volumen > 0', pool)
        pool = pool[pool['n_meses'] >= min_meses_celda]
        _snap(f'4. Recurrencia >= {min_meses_celda}', pool)

        if OUTLIER_METODO != 'ninguno' and len(pool):
            v = np.log1p(pool['peso_12m']) if OUTLIER_METODO == 'iqr_log' else pool['peso_12m']
            tmp = pool[OUTLIER_NIVEL].copy(); tmp['_v'] = v.values
            if OUTLIER_METODO in ('iqr_log', 'iqr_raw'):
                q1 = tmp.groupby(OUTLIER_NIVEL)['_v'].transform(lambda s: s.quantile(.25))
                q3 = tmp.groupby(OUTLIER_NIVEL)['_v'].transform(lambda s: s.quantile(.75))
                lo, hi = q1 - OUTLIER_K * (q3 - q1), q3 + OUTLIER_K * (q3 - q1)
            else:
                hi = tmp.groupby(OUTLIER_NIVEL)['_v'].transform(lambda s: s.quantile(.99))
                lo = pd.Series(-np.inf, index=tmp.index)
            out = (tmp['_v'] < lo) | (tmp['_v'] > hi)
            print("\n  Tasa de outlier por oficina (pool combinado AC+CT, en log):")
            print(pd.DataFrame({
                'n_pool'   : pool.groupby('des_oficina').size(),
                'n_outlier': out.groupby(pool['des_oficina']).sum(),
                'pct'      : (out.groupby(pool['des_oficina']).mean() * 100).round(1),
                'lim_sup_kg': (np.expm1(hi.groupby(pool['des_oficina']).first())
                               if OUTLIER_METODO == 'iqr_log'
                               else hi.groupby(pool['des_oficina']).first()).round(1),
            }).to_string().replace('\n', '\n  '))
            pool = pool[~out]
        _snap('5. Sin outliers', pool)

        fdf = pd.DataFrame(fn)
        fdf['AC_pct'] = (100 * fdf.AC / max(1, fdf.AC.iloc[0])).round(1)
        fdf['CT_pct'] = (100 * fdf.CT / max(1, fdf.CT.iloc[0])).round(1)
        print("\n  EMBUDO DE LIMPIEZA")
        print(fdf.to_string(index=False).replace('\n', '\n  '))
        for r in fn:
            funnel_out.append({'cosecha': cosecha, 'categoria': cat, **r})

        ac = pool[pool.rol == 'AC'].copy()
        ct = pool[pool.rol == 'CT'].copy()

        # ── MODO INCREMENTAL (solo G03 con REHACER_MATCHING=False) ───────────────
        # ac_ref/ct_ref = población limpia COMPLETA de la celda (para viabilidad y resumen).
        modo_incr = (CONGELADO is not None) and (cosecha == COSECHA_ABIERTA) and (cosecha not in COSECHAS_REHACER_AHORA)
        ac_ref, ct_ref = ac, ct
        if modo_incr:
            mt_c = CONGELADO['matches']
            mt_c = mt_c[(mt_c['cosecha'] == cosecha) & (mt_c['des_categoria'] == cat)]
            ctrl_usados = set(mt_c['cod_control'])
            ac = ac[ac.index.isin(ENTRANTES_G03)]
            ct = ct[~ct.index.isin(ctrl_usados)]
            print(f"  [INCREMENTAL] entrantes a matchear: {len(ac):,} | controles libres: {len(ct):,} "
                  f"de {len(ct_ref):,} ({len(ctrl_usados):,} ya usados en parejas congeladas)")
        if len(ac) == 0 or len(ct) == 0:
            print("  [SKIP] celda sin acción o sin control"); continue

        viab = pd.DataFrame({'AC': ac_ref.groupby('des_oficina').size(),
                             'CT': ct_ref.groupby('des_oficina').size()}).fillna(0).astype(int)
        viab['ratio_CT_AC'] = (viab.CT / viab.AC.replace(0, np.nan)).round(3)
        # Viabilidad = MIN_CT_OFICINA=30 fijo, IGUAL que v6 (piso fijo, no dinámico).
        # n_min_requerido (fórmula con corrección por población finita) se deja como
        # columna INFORMATIVA adicional -- no decide 'viable', solo da contexto extra
        # de qué tan lejos/cerca está esa oficina del estándar 5%/95% del proyecto.
        MIN_CT_OFICINA = 30
        viab['n_min_requerido'] = [n_minimo_requerido(a, c_) for a, c_ in zip(viab.AC, viab.CT)]
        viab['viable'] = viab.CT >= MIN_CT_OFICINA
        _inf = viab[~np.isfinite(viab['n_min_requerido'].astype(float))]
        if len(_inf):
            print(f"  [ALERTA] oficinas con <=1 cliente en AC o CT (n_min_requerido=inf, no viables):\n"
                  f"{_inf[['AC', 'CT']].to_string()}")
        print("\n  VIABILIDAD POR OFICINA")
        print(viab.to_string().replace('\n', '\n  '))
        for ofi, r in viab[~viab.viable].iterrows():
            print(f"  [NO VIABLE] {ofi}: {r.CT} controles para {r.AC:,} acciones. "
                  f"La muestra sí se toma; el problema saldrá en cobertura de matching.")

        # Guardamos la viabilidad de TODAS las oficinas (viables y no viables) —
        # esto es lo que sustenta, más adelante, si se puede reportar la brecha
        # partida por oficina o si hay que decir explícitamente "sin control suficiente".
        for ofi, r in viab.iterrows():
            viabilidad_out.append({
                'cosecha': cosecha, 'grupo': GRUPO_COSECHA.get(cosecha, '?'), 'categoria': cat,
                'des_oficina': ofi,
                'AC': int(r.AC), 'CT': int(r.CT),
                # n_minimo_muestra devuelve inf si una población tiene <=1 cliente (oficina con
                # 0-1 AC o CT): se conserva inf (=> "no alcanza", n_efectivo < inf) en vez de int(inf)
                'n_min_requerido': int(r.n_min_requerido) if np.isfinite(r.n_min_requerido) else np.inf,
                'ratio_CT_AC': float(r.ratio_CT_AC) if pd.notna(r.ratio_CT_AC) else np.nan,
                'viable': bool(r.viable)})

        # ── CASCADA DE ESTRATOS (FIX 5) ──────────────────────────────────────
        estratos_usar, nivel_estrato = ESTRATOS_FULL[:], 1
        for k in range(len(ESTRATOS_FULL), 0, -1):
            cand = ESTRATOS_FULL[:k]
            tam = ac.groupby([ac[c].astype(str) for c in cand]).size()
            estratos_usar = cand
            nivel_estrato = len(ESTRATOS_FULL) - k + 1
            if len(tam) and tam.median() >= MIN_POR_CELDA:
                break
        tam = ac.groupby([ac[c].astype(str) for c in estratos_usar]).size()
        print(f"\n  ESTRATOS (nivel {nivel_estrato}): {' × '.join(estratos_usar)}")
        print(f"    celdas={len(tam)} | mediana={tam.median():.0f} | "
              f"celdas<{MIN_POR_CELDA}: {(tam < MIN_POR_CELDA).sum()}")

        # ── ESPACIO DE MATCHING ──────────────────────────────────────────────
        f_log = (lambda M: np.log1p(M)) if USAR_LOG else (lambda M: M)
        Lac, Lct = f_log(ac[cols_m].to_numpy(float)), f_log(ct[cols_m].to_numpy(float))
        Fac, Fct = np.diff(Lac, axis=1), np.diff(Lct, axis=1)
        Lac_df, Lct_df = pd.DataFrame(Lac, index=ac.index), pd.DataFrame(Lct, index=ct.index)
        Fac_df, Fct_df = pd.DataFrame(Fac, index=ac.index), pd.DataFrame(Fct, index=ct.index)

        _r = np.random.default_rng(0)
        sa = _r.choice(len(ac), min(600, len(ac)), replace=False)
        sc = _r.choice(len(ct), min(600, len(ct)), replace=False)
        MED_N = float(np.median(pairwise_distances(Lac[sa], Lct[sc]))) or 1.0
        MED_F = float(np.median(pairwise_distances(Fac[sa], Fct[sc]))) or 1.0

        # ── LOOP DE SEMILLAS — PARALELIZADO CON JOBLIB (N_JOBS núcleos) ────────
        # Cada semilla es independiente -> se reparte entre los núcleos disponibles.
        # backend='threads' (no procesos): pairwise_distances/linear_sum_assignment son
        # C/Fortran y liberan el GIL, y así evitamos el costo de volver a serializar
        # ac/ct/Lac_df/... (varios miles de filas) en cada una de las 50 semillas.
        # Medido: ~15x más rápido que 'processes' para este tamaño de pool.
        # ── PRECOMPUTO por celda (cosecha x categoria) — una vez, no una vez por semilla ──
        # ac_por_oficina: el agrupamiento por oficina no depende del seed -> se arma
        # una vez y se comparte entre threads (memoria compartida, solo lectura, seguro).
        # ka_full/kc_full: la clave de estrato de cada cliente (ej. "oficina|compania|
        # segmento") tampoco depende del seed -> se calcula una vez por combinacion de
        # columnas usada en NIVELES, en vez de recalcularla con astype+join en cada una
        # de las N semillas (para celdas 1:1, el lado control se repetia identico).
        ac_por_oficina_pre = dict(list(ac.groupby('des_oficina')))
        claves_unicas = []
        for _, claves_n, _ in NIVELES:
            if claves_n not in claves_unicas:
                claves_unicas.append(claves_n)
        ka_full_pre = {tuple(cl): ac[cl].astype(str).agg('|'.join, axis=1) for cl in claves_unicas}
        kc_full_pre = {tuple(cl): ct[cl].astype(str).agg('|'.join, axis=1) for cl in claves_unicas}

        print(f"\n  ITERANDO {n_semillas_celda} SEMILLAS  (paralelo, N_JOBS={N_JOBS}, backend=threads)")

        resultados_semilla = Parallel(n_jobs=N_JOBS, prefer='threads')(
            delayed(_correr_semilla)(
                seed, ac, ct, cols_m, estratos_usar, Lac_df, Lct_df, Fac_df, Fct_df,
                MED_N, MED_F, NIVELES, MUESTREAR, FRAC_MUESTREO,
                w_nivel_celda, w_forma_celda, CORR_UMBRAL, RATIO_MIN, RATIO_MAX,
                max_reuso=max_reuso_celda,
                ac_por_oficina=ac_por_oficina_pre,
                ka_full_por_claves=ka_full_pre, kc_full_por_claves=kc_full_pre)
            for seed in semillas_celda)
        resultados_semilla = [r for r in resultados_semilla if r is not None]

        if not resultados_semilla:
            print("  [SKIP] ninguna semilla produjo pares"); continue

        rs_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                              for r in resultados_semilla])
        print(rs_df.to_string(index=False).replace('\n', '\n  '))
        for r in rs_df.to_dict('records'):
            semillas_out.append({'cosecha': cosecha, 'categoria': cat,
                                 'lote': 'incremental' if modo_incr else 'completo', **r})

        val = [r for r in resultados_semilla if r['valida']]

        # ── BÚSQUEDA DE LA MEJOR SEMILLA QUE TAMBIÉN SEA EQUIVALENTE (paralela) ──
        # Antes: se elegía la semilla por pend_rel (proxy barato, ya calculado en
        # cada semilla) y el test formal de paralelismo se corría UNA sola vez,
        # sobre esa ganadora, como formalidad posterior -- si fallaba, no se
        # probaba ninguna otra semilla candidata.
        # Ahora: se ordenan las semillas válidas por pend_rel (mejor proxy primero)
        # y se corre el TOST formal en cascada hasta encontrar la primera que
        # también sea equivalente -- es decir, se BUSCA de verdad el mejor par
        # comparable, no se testea solo a la que ganó por otro criterio.
        if modo_incr:
            # Entrantes: NO se gatea por corr/ratio/TOST del lote (n chico, no es la unidad que se
            # reporta). La semilla se elige por calidad de matching del lote (cobertura, luego distancia)
            # y validez/paralelismo se re-evalúan sobre la celda COMPLETA = pares congelados + nuevos.
            best = sorted(resultados_semilla, key=lambda r: (-r['cobertura_pct'], r['dist_med']))[0]
            semillas_probadas = 0
            ids_ac = [i for i in list(mt_c['cod_accion']) + list(best['_p'].cod_accion) if i in pool_pre.index]
            ids_ct = [i for i in list(mt_c['cod_control']) + list(best['_p'].cod_control) if i in pool_pre.index]
            n_perd = (len(mt_c) + len(best['_p'])) * 2 - len(ids_ac) - len(ids_ct)
            if n_perd:
                print(f"  [ALERTA] {n_perd} clientes de la unión sin trayectoria en la ventana -> excluidos de la métrica")
            t_ac_u = pool_pre.loc[ids_ac, cols_m].mean().to_numpy()
            t_ct_u = pool_pre.loc[ids_ct, cols_m].mean().to_numpy()
            d_ac_u, d_ct_u = np.diff(t_ac_u), np.diff(t_ct_u)
            corr_u = (np.corrcoef(d_ac_u, d_ct_u)[0, 1]
                      if d_ac_u.std() > 0 and d_ct_u.std() > 0 else np.nan)
            ratio_u = d_ac_u.std() / d_ct_u.std() if d_ct_u.std() > 0 else np.inf
            brecha_u = t_ac_u - t_ct_u
            pend_u = (linregress(np.arange(len(brecha_u)), brecha_u).slope / t_ac_u.mean()
                      if t_ac_u.mean() > 0 else np.inf)
            valida_u = bool(pd.notna(corr_u) and corr_u >= CORR_UMBRAL and RATIO_MIN <= ratio_u <= RATIO_MAX)
            equiv_coef, equiv_ic_inf, equiv_ic_sup, equivalencia_ok = test_equivalencia_paralelismo(
                ids_ac, ids_ct, pool_pre, pool_pre, cols_m,
                margen_coef=MARGEN_EQUIVALENCIA_PEND, alpha=ALPHA_TOST)
            _fila_c = CONGELADO['resumen'][(CONGELADO['resumen']['cosecha'] == cosecha) &
                                           (CONGELADO['resumen']['categoria'] == cat)]
            n_mue_c = int(_fila_c['AC_muestra'].iloc[0]) if len(_fila_c) else len(mt_c)
            n_pares_u = len(mt_c) + len(best['_p'])
            ct_u_all = list(mt_c['cod_control']) + list(best['_p'].cod_control)
            val = [best] if valida_u else []
            best = {**best, 'valida': valida_u,
                    'corr': round(corr_u, 3) if pd.notna(corr_u) else np.nan,
                    'ratio_vol': round(ratio_u, 3), 'pend_rel': round(pend_u, 5),
                    'pares': n_pares_u, 'n_muestra': n_mue_c + best['n_muestra'],
                    'cobertura_pct': round(100 * n_pares_u / max(1, n_mue_c + best['n_muestra']), 1),
                    'controles_unicos': len(set(ct_u_all)),
                    'max_reuso_observado': int(pd.Series(ct_u_all).value_counts().max()),
                    '_tac': t_ac_u, '_tct': t_ct_u}
            print(f"\n  >> UNIÓN G03 ({len(mt_c):,} congelados + {len(best['_p']):,} nuevos = {n_pares_u:,} pares) | "
                  f"corr={best['corr']} ratio={best['ratio_vol']} | valida={valida_u} | equivalencia_ok={equivalencia_ok}")
        else:
            equiv_coef = equiv_ic_inf = equiv_ic_sup = np.nan
            equivalencia_ok = False
            semillas_probadas = 0
            best = None

            if val:
                candidatas = sorted(val, key=lambda r: abs(r['pend_rel']))
                for cand in candidatas[:MAX_SEMILLAS_PROBAR_EQUIVALENCIA]:
                    semillas_probadas += 1
                    coef, ic_inf, ic_sup, ok = test_equivalencia_paralelismo(
                        cand['_p'].cod_accion, cand['_p'].cod_control, ac, ct, cols_m,
                        margen_coef=MARGEN_EQUIVALENCIA_PEND, alpha=ALPHA_TOST)
                    if ok:
                        best = cand
                        equiv_coef, equiv_ic_inf, equiv_ic_sup, equivalencia_ok = coef, ic_inf, ic_sup, ok
                        print(f"\n  >> SEMILLA {best['seed']} | corr={best['corr']} "
                              f"ratio={best['ratio_vol']} pend_rel={best['pend_rel']} "
                              f"pares={best['pares']:,} cobertura={best['cobertura_pct']}% "
                              f"| EQUIVALENTE tras probar {semillas_probadas} semilla(s) "
                              f"(coef={coef:.4f}, IC=[{ic_inf:.4f},{ic_sup:.4f}], "
                              f"margen=±{MARGEN_EQUIVALENCIA_PEND:.4f})")
                        break
                if best is None:
                    # Ninguna de las candidatas probadas fue equivalente -> nos quedamos
                    # con la mejor por pend_rel (igual que antes), pero documentado como
                    # NO equivalente, no se reporta como si hubiera pasado.
                    best = candidatas[0]
                    equiv_coef, equiv_ic_inf, equiv_ic_sup, equivalencia_ok = test_equivalencia_paralelismo(
                        best['_p'].cod_accion, best['_p'].cod_control, ac, ct, cols_m,
                        margen_coef=MARGEN_EQUIVALENCIA_PEND, alpha=ALPHA_TOST)
                    print(f"\n  >> SEMILLA {best['seed']} | corr={best['corr']} "
                          f"ratio={best['ratio_vol']} pend_rel={best['pend_rel']} "
                          f"pares={best['pares']:,} cobertura={best['cobertura_pct']}% "
                          f"| [NO EQUIVALENTE] ninguna de las {semillas_probadas} semilla(s) "
                          f"probadas (de {len(candidatas)} válidas) cayó dentro del margen "
                          f"±{MARGEN_EQUIVALENCIA_PEND:.4f} -> se documenta, no se descarta la celda.")
            else:
                best = min(resultados_semilla, key=lambda r: abs(r['pend_rel']))
                print(f"\n  [CELDA INCOMPARABLE] ninguna de {n_semillas_celda} semillas cumple "
                      f"corr>={CORR_UMBRAL} y ratio en [{RATIO_MIN},{RATIO_MAX}]. "
                      f"NO bajar umbrales: documentar exclusión.")

        if best.get('controles_unicos', best['pares']) < best['pares']:
            print(f"  [REUSO CONTROL EN USO] {best['pares']:,} pares usan solo "
                  f"{best['controles_unicos']:,} controles únicos (máx. reuso observado: "
                  f"{best['max_reuso_observado']}x) -> clusterizar SE por cod_control en el DiD.")

        for r in best['_p'].to_dict('records'):
            a_id, c_id = r['cod_accion'], r['cod_control']
            matches_out.append({
                'cosecha': cosecha, 'des_categoria': cat, 'seed': best['seed'],
                'celda_valida': best['valida'], **r,
                'ofi_ac': ac.at[a_id, 'des_oficina'],  'ofi_ct': ct.at[c_id, 'des_oficina'],
                'comp_ac': ac.at[a_id, 'nom_compania'], 'comp_ct': ct.at[c_id, 'nom_compania'],
                'seg_ac': ac.at[a_id, 'des_segmento_transaccional'],
                'seg_ct': ct.at[c_id, 'des_segmento_transaccional'],
                'min_per_ac': ac.at[a_id, 'min_periodo_cosecha'],
                'peso_ac': ac.at[a_id, 'peso_12m'],   'peso_ct': ct.at[c_id, 'peso_12m'],
                'nmeses_ac': ac.at[a_id, 'n_meses'],  'nmeses_ct': ct.at[c_id, 'n_meses']})

        for a_id in best['_pend']:
            sinpar_out.append({
                'cosecha': cosecha, 'des_categoria': cat, 'cod_accion': a_id,
                'des_oficina': ac.at[a_id, 'des_oficina'],
                'nom_compania': ac.at[a_id, 'nom_compania'],
                'des_segmento_transaccional': ac.at[a_id, 'des_segmento_transaccional'],
                'min_periodo_cosecha': ac.at[a_id, 'min_periodo_cosecha'],
                'peso_12m': ac.at[a_id, 'peso_12m'], 'n_meses': ac.at[a_id, 'n_meses']})

        # ── RESULTADO DE PARALELISMO (TOST) sobre la semilla finalmente elegida ──
        # Ya se calculó dentro de la búsqueda de arriba (equiv_coef/ic/ok) -- no se
        # vuelve a correr la regresión, solo se deja registrado para el resumen.
        paralelismo_ok = equivalencia_ok
        print(f"\n  [TEST EQUIVALENCIA] H0_equiv a rechazar: |pendiente_dif| >= "
              f"{MARGEN_EQUIVALENCIA_PEND:.4f} | coef interacción tiempo×grupo = "
              f"{equiv_coef:.4f}, IC{int(100*(1-2*ALPHA_TOST))}%=[{equiv_ic_inf:.4f},"
              f"{equiv_ic_sup:.4f}]" if pd.notna(equiv_coef) else "  [TEST EQUIVALENCIA] no calculable")
        print(f"    -> {'OK: diferencia de pendiente dentro del margen, tendencias equivalentes' if paralelismo_ok else '[ALERTA] diferencia de pendiente FUERA del margen tras buscar, revisar celda'} "
              f"(semillas válidas probadas: {semillas_probadas} de {len(val)})")

        resumen_out.append({
            'cosecha': cosecha, 'grupo': GRUPO_COSECHA.get(cosecha, '?'), 'categoria': cat,
            'AC_limpio': len(ac_ref), 'CT_limpio': len(ct_ref),
            'AC_muestra': best['n_muestra'], 'seed': best['seed'], 'pares': best['pares'],
            'controles_unicos': best.get('controles_unicos', best['pares']),
            'max_reuso_observado': best.get('max_reuso_observado', 1),
            'min_meses_usado': min_meses_celda,
            'cobertura_pct': best['cobertura_pct'], 'corr': best['corr'],
            'ratio_vol': best['ratio_vol'], 'pend_rel': best['pend_rel'],
            'valida': best['valida'], 'nivel_estrato': nivel_estrato,
            'equivalencia_coef': round(equiv_coef, 4) if pd.notna(equiv_coef) else np.nan,
            'equivalencia_ic_inf': round(equiv_ic_inf, 4) if pd.notna(equiv_ic_inf) else np.nan,
            'equivalencia_ic_sup': round(equiv_ic_sup, 4) if pd.notna(equiv_ic_sup) else np.nan,
            'margen_equivalencia': round(MARGEN_EQUIVALENCIA_PEND, 4),
            'semillas_validas': len(val), 'semillas_probadas_equivalencia': semillas_probadas,
            'paralelismo_ok': paralelismo_ok,
            'modo': 'incremental_g03' if modo_incr else 'completo',
            'pares_nuevos': len(best['_p'])})

        pools_out[(cosecha, cat)] = {'ac': ac, 'ct': ct, 'meses': meses,
                                     'ac_s': best['_ac'], 't_ac': best['_tac'],
                                     't_ct': best['_tct'], 'best': best}

df_matches     = pd.DataFrame(matches_out)
df_sin_pareja  = pd.DataFrame(sinpar_out)
df_funnel      = pd.DataFrame(funnel_out)
df_resumen     = pd.DataFrame(resumen_out)
# reportable = matching estadísticamente válido (corr + ratio_vol) Y equivalencia
# de tendencias (TOST) dentro del margen de negocio -- ver bloque de búsqueda de
# semilla más arriba: antes de llegar acá ya se intentó, en cascada, encontrar una
# semilla que fuera AMBAS cosas (no se testeó paralelismo solo en la ganadora por
# otro criterio). Si después de probar hasta MAX_SEMILLAS_PROBAR_EQUIVALENCIA
# candidatas ninguna es equivalente, la celda queda documentada como no reportable
# en vez de reportarse con una brecha que descansa en un supuesto no verificado.
if len(df_resumen):
    df_resumen['reportable'] = df_resumen['valida'] & df_resumen['paralelismo_ok']
df_semillas    = pd.DataFrame(semillas_out)
df_viabilidad  = pd.DataFrame(viabilidad_out)

# ── MODO MENSUAL: unir lo recién calculado (solo G03) con lo congelado ───────────────
if CONGELADO is not None:
    def _reemplazar_celdas(congelado, nuevo, col_cat):
        """Quita del congelado las celdas cosecha×categoría recalculadas y agrega las nuevas."""
        if nuevo is None or not len(nuevo):
            return congelado.copy()
        claves = set(zip(nuevo['cosecha'], nuevo[col_cat]))
        recalc = np.array([(c, k) in claves for c, k in zip(congelado['cosecha'], congelado[col_cat])])
        return pd.concat([congelado[~recalc], nuevo], ignore_index=True)

    df_resumen_lote = df_resumen.copy()
    df_matches    = pd.concat([CONGELADO['matches'], df_matches], ignore_index=True)
    df_sin_pareja = pd.concat([CONGELADO['sin_pareja'], df_sin_pareja], ignore_index=True)
    df_funnel     = _reemplazar_celdas(CONGELADO['funnel'], df_funnel, 'categoria')
    df_resumen    = _reemplazar_celdas(CONGELADO['resumen'], df_resumen, 'categoria')
    df_viabilidad = _reemplazar_celdas(CONGELADO['viabilidad'], df_viabilidad, 'categoria')
    _sem = CONGELADO['semillas'].copy()
    if 'lote' not in _sem.columns:
        _sem['lote'] = 'completo'
    df_semillas   = pd.concat([_sem, df_semillas], ignore_index=True)

    # la validez de la celda pasó a evaluarse sobre la UNIÓN -> se refleja en todas sus parejas
    for r in df_resumen_lote.itertuples():
        _m = (df_matches['cosecha'] == r.cosecha) & (df_matches['des_categoria'] == r.categoria)
        df_matches.loc[_m, 'celda_valida'] = r.valida
    df_resumen = df_resumen.sort_values(['cosecha', 'categoria']).reset_index(drop=True)
    df_resumen['reportable'] = df_resumen['valida'].astype(bool) & df_resumen['paralelismo_ok'].astype(bool)
    print(f"\n[MODO MENSUAL] pares totales tras unir: {len(df_matches):,} | "
          f"sin pareja: {len(df_sin_pareja):,} | celdas en resumen: {len(df_resumen)}")

print("\n" + "=" * 110); print("VIABILIDAD POR OFICINA — TODAS LAS CELDAS"); print("=" * 110)
print(df_viabilidad.to_string(index=False))
print(f"\nCeldas cosecha×categoría×oficina SIN control suficiente para {MARGEN_ERROR_OBJETIVO*100:.0f}% "
      f"de margen de error: {(~df_viabilidad.viable).sum():,} de {len(df_viabilidad):,}")

print("\n" + "=" * 110); print("RESUMEN GENERAL"); print("=" * 110)
print(df_resumen.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# TABLAS DE COMPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────
comp_pob_mue, comp_ac_ct = [], []

for (cosecha, cat), inf in pools_out.items():
    ac_pob, ac_mue, ct_pob = inf['ac'], inf['ac_s'], inf['ct']
    print("\n" + "=" * 110)
    print(f"COMPARACIONES — {cosecha.date()} | {cat}")
    print("=" * 110)

    # (A) Población vs Muestra
    for col in ESTRATOS_FULL:
        p = ac_pob[col].astype(str).value_counts(normalize=True).mul(100)
        m = ac_mue[col].astype(str).value_counts(normalize=True).mul(100)
        t = pd.DataFrame({'pob_%': p.round(1), 'muestra_%': m.round(1)}).fillna(0.0)
        t['dif_pp']    = (t['muestra_%'] - t['pob_%']).round(1)
        t['n_pob']     = ac_pob[col].astype(str).value_counts()
        t['n_muestra'] = ac_mue[col].astype(str).value_counts()
        t = t.fillna(0).sort_values('n_pob', ascending=False)
        t.index.name = col
        print(f"\n  [A] POBLACIÓN vs MUESTRA — {col}")
        print(t.to_string().replace('\n', '\n  '))
        for idx, r in t.iterrows():
            if abs(r['dif_pp']) > 5:
                print(f"     [ALERTA] {idx}: desvío {r['dif_pp']}pp")
        comp_pob_mue.append(t.assign(cosecha=cosecha, categoria=cat, variable=col))

    # (B) Acción vs Control (post-limpieza, pre-matching)
    for col in MATCH_FULL:
        a = ac_pob[col].astype(str).value_counts(normalize=True).mul(100)
        c = ct_pob[col].astype(str).value_counts(normalize=True).mul(100)
        t = pd.DataFrame({'accion_%': a.round(1), 'control_%': c.round(1)}).fillna(0.0)
        t['dif_pp']    = (t['control_%'] - t['accion_%']).round(1)
        t['n_accion']  = ac_pob[col].astype(str).value_counts()
        t['n_control'] = ct_pob[col].astype(str).value_counts()
        t = t.fillna(0).sort_values('n_accion', ascending=False)
        t.index.name = col
        print(f"\n  [B] ACCIÓN vs CONTROL — {col}")
        print(t.to_string().replace('\n', '\n  '))
        comp_ac_ct.append(t.assign(cosecha=cosecha, categoria=cat, variable=col))

df_comp_pob_mue = pd.concat(comp_pob_mue) if comp_pob_mue else pd.DataFrame()
df_comp_ac_ct   = pd.concat(comp_ac_ct)   if comp_ac_ct   else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO: trayectoria 12M por cosecha_agrupada
# ─────────────────────────────────────────────────────────────────────────────
if GRAFICAR and pools_out:
    for cosecha in COSECHAS:
        cells = [k for k in pools_out if k[0] == cosecha]
        if not cells:
            continue
        fig, axes = plt.subplots(1, len(cells), figsize=(7 * len(cells), 4.6), squeeze=False)
        for axi, key in enumerate(cells):
            inf, ax = pools_out[key], axes[0][axi]
            x = [m.strftime('%b-%y') for m in inf['meses']]
            ax.plot(x, inf['t_ac'], marker='o', lw=2.2, color='#1f3864', label='Acción')
            ax.plot(x, inf['t_ct'], marker='s', lw=2.2, ls='--', color='#c00000', label='Control')
            ax.fill_between(x, inf['t_ac'], inf['t_ct'], alpha=0.12, color='grey')
            r = inf['best']
            ax.set_title(f"{key[0].date()} | {key[1]}  "
                         f"[{'VÁLIDA' if r['valida'] else 'INCOMPARABLE'}]\n"
                         f"seed={r['seed']} corr={r['corr']} ratio={r['ratio_vol']} "
                         f"n={r['pares']:,} cob={r['cobertura_pct']}%",
                         fontsize=9.5, color='#1f3864' if r['valida'] else '#c00000')
            ax.set_ylabel('Peso promedio (kg)'); ax.legend(fontsize=8)
            ax.grid(alpha=0.25); ax.tick_params(axis='x', rotation=45, labelsize=8)
        fig.suptitle(f"Trayectoria pre-cosecha 12M — cosecha_agrupada {cosecha.date()}",
                     fontsize=12, weight='bold')
        plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNÓSTICO
# ─────────────────────────────────────────────────────────────────────────────
if len(df_matches):
    print("\nESTABILIDAD ENTRE SEMILLAS")
    print(df_semillas.groupby(['cosecha', 'categoria'])
          .agg(n_validas=('valida', 'sum'), pares_min=('pares', 'min'),
               pares_max=('pares', 'max'), corr_min=('corr', 'min'),
               corr_max=('corr', 'max'), cob_min=('cobertura_pct', 'min'),
               cob_max=('cobertura_pct', 'max')).to_string())

    bal = []
    for (c, k), g in df_matches.groupby(['cosecha', 'des_categoria']):
        for v in ['peso', 'nmeses']:
            a, b = g[f'{v}_ac'], g[f'{v}_ct']
            sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            bal.append({'cosecha': c, 'categoria': k, 'var': v,
                        'media_ac': round(a.mean(), 2), 'media_ct': round(b.mean(), 2),
                        'SMD': round((a.mean() - b.mean()) / sp if sp > 0 else 0, 3)})
    bdf = pd.DataFrame(bal); bdf['ok'] = bdf.SMD.abs() < 0.10
    print("\nBALANCE (|SMD| < 0.10)"); print(bdf.to_string(index=False))

    print("\nPARES POR NIVEL DE CASCADA")
    print(df_matches.groupby(['cosecha', 'des_categoria', 'nivel_match'])
          .size().rename('pares').to_string())

    print("\nCOBERTURA POR OFICINA (aquí sale el problema real de CM-SUR)")
    cob = pd.DataFrame({
        'matcheados': df_matches.groupby(['cosecha', 'des_categoria', 'ofi_ac']).size(),
        'sin_pareja': df_sin_pareja.groupby(['cosecha', 'des_categoria', 'des_oficina']).size()
    }).fillna(0).astype(int)
    cob['cobertura_%'] = (100 * cob.matcheados /
                          (cob.matcheados + cob.sin_pareja)).round(1)
    print(cob.to_string())

    print("\nSESGO DE SELECCIÓN (matcheados vs sin pareja)")
    ses = df_matches.groupby(['cosecha', 'des_categoria'])['peso_ac'].mean().rename('peso_match')
    ses = pd.concat([ses, df_sin_pareja.groupby(['cosecha', 'des_categoria'])['peso_12m']
                     .mean().rename('peso_sin_pareja')], axis=1).round(1)
    print(ses.to_string())

    reuso = df_matches.groupby('cod_control')['cosecha'].nunique()
    print(f"\nControles en >1 cosecha: {(reuso > 1).sum():,}/{len(reuso):,} "
          f"-> CLUSTERIZAR SE por cod_cliente en la regresión DiD")

    reuso_misma_celda = df_matches.groupby(['cosecha', 'des_categoria', 'cod_control']).size()
    reuso_misma_celda = reuso_misma_celda[reuso_misma_celda > 1]
    if len(reuso_misma_celda):
        print(f"\nControles reusados >1 vez DENTRO de la misma celda (MAX_REUSO_CONTROL): "
              f"{len(reuso_misma_celda):,} controles -> ídem, clusterizar SE por cod_control.")
        print(reuso_misma_celda.groupby(['cosecha', 'des_categoria']).agg(['count', 'max'])
              .to_string())

    inval = df_resumen[~df_resumen.valida]
    if len(inval):
        print("\n[CELDAS A EXCLUIR DEL REPORTE]")
        print(inval[['cosecha', 'categoria', 'corr', 'ratio_vol', 'pares']].to_string(index=False))

    print("\nTEST DE EQUIVALENCIA (TOST) POR CELDA — margen de negocio, no igualdad exacta")
    print(df_resumen[['cosecha', 'categoria', 'pares', 'equivalencia_coef',
                      'margen_equivalencia', 'semillas_validas',
                      'semillas_probadas_equivalencia', 'paralelismo_ok']]
          .to_string(index=False))
    no_parall = df_resumen[~df_resumen.paralelismo_ok]
    if len(no_parall):
        print("\n[ALERTA] Celdas que NO lograron equivalencia tras buscar en varias semillas "
              "(quedan fuera de 'reportable' -- ver df_resumen.reportable):")
        print(no_parall[['cosecha', 'categoria', 'equivalencia_coef', 'margen_equivalencia',
                         'semillas_probadas_equivalencia', 'semillas_validas']].to_string(index=False))

    print("\n" + "=" * 110); print("RESUMEN FINAL — ¿QUÉ CELDAS SON REPORTABLES?"); print("=" * 110)
    print(df_resumen[['grupo', 'categoria', 'pares', 'valida', 'paralelismo_ok', 'reportable']]
          .to_string(index=False))
    print(f"\nReportables: {df_resumen.reportable.sum()} de {len(df_resumen)} celdas.")

import os

# Ruta de salida relativa (portable entre Windows/Linux/Mac) -> antes apuntaba a una
# ruta de OneDrive de otra persona en Windows, que no existe en esta máquina Debian.
ruta_salida = os.path.join('outputs', 'Surtido Ideal - Medición')

# Crear carpeta si no existe
os.makedirs(ruta_salida, exist_ok=True)

# Respaldo del estado ANTERIOR a esta corrida (una vez por cierre) antes de sobrescribir
_ruta_resp = os.path.join(ruta_salida, f"respaldo_antes_de_{SUFIJO_CACHE}")
if not os.path.exists(_ruta_resp):
    os.makedirs(_ruta_resp)
    for _f in os.listdir(ruta_salida):
        if _f.endswith('.csv') or _f.endswith('.json'):
            shutil.copy2(os.path.join(ruta_salida, _f), _ruta_resp)

# Guardar cada DataFrame
df_matches.to_csv(os.path.join(ruta_salida, 'df_matches.csv'), index=False, encoding='utf-8')
df_sin_pareja.to_csv(os.path.join(ruta_salida, 'df_sin_pareja.csv'), index=False, encoding='utf-8')
df_funnel.to_csv(os.path.join(ruta_salida, 'df_funnel.csv'), index=False, encoding='utf-8')
df_resumen.to_csv(os.path.join(ruta_salida, 'df_resumen.csv'), index=False, encoding='utf-8')
df_semillas.to_csv(os.path.join(ruta_salida, 'df_semillas.csv'), index=False, encoding='utf-8')
df_viabilidad.to_csv(os.path.join(ruta_salida, 'df_viabilidad.csv'), index=False, encoding='utf-8')

print("✓ Archivos guardados en:")
print(f"  {os.path.join(ruta_salida, 'df_matches.csv')}")
print(f"  {os.path.join(ruta_salida, 'df_sin_pareja.csv')}")
print(f"  {os.path.join(ruta_salida, 'df_funnel.csv')}")
print(f"  {os.path.join(ruta_salida, 'df_resumen.csv')}")
print(f"  {os.path.join(ruta_salida, 'df_semillas.csv')}")
print(f"  {os.path.join(ruta_salida, 'df_viabilidad.csv')}")

# Estado para el próximo cierre: hasta aquí todos los entrantes del G03 ya tienen pareja (o quedaron sin pareja)
with open(RUTA_ESTADO, 'w') as _fh:
    json.dump({'ultimo_cierre_matcheado': MES_CIERRE_STR}, _fh)
print(f"  Estado actualizado: ultimo_cierre_matcheado = {MES_CIERRE_STR}")


# =============================================================================
# PASO A — TRANSACCIONAL DE ACCIÓN + CONTROL, MARCADA
# =============================================================================
import numpy as np, pandas as pd
from dateutil.relativedelta import relativedelta

ALPHA       = 0.05
MODO_PRE    = 'mismo_mes'     # 'mismo_mes' | 'promedio'
MESES_POST  = 120             # sin tope práctico: el corte lo define MES_CIERRE

# Llave larga: una fila por par, con su rol
llave = []
for r in df_matches.itertuples():
    # des_oficina del PAR = oficina del cliente de acción (así se armó el matching);
    # se la ponemos también al control para poder partir brechas por oficina más adelante.
    llave.append({'cod_cliente': r.cod_accion,  'rol': 'accion',
                  'cosecha_grupo': r.cosecha, 'des_categoria': r.des_categoria,
                  'des_oficina': r.ofi_ac,
                  'par_id': f"{r.cosecha.date()}|{r.des_categoria}|{r.cod_accion}"})
    llave.append({'cod_cliente': r.cod_control, 'rol': 'control',
                  'cosecha_grupo': r.cosecha, 'des_categoria': r.des_categoria,
                  'des_oficina': r.ofi_ac,
                  'par_id': f"{r.cosecha.date()}|{r.des_categoria}|{r.cod_accion}"})
df_llave = pd.DataFrame(llave)

print(f"Filas de llave: {len(df_llave):,}  (pares × 2 = {len(df_matches)*2:,})")
print(df_llave.groupby(['cosecha_grupo','des_categoria','rol']).size().unstack().to_string())

# Un control puede servir en varias cosechas -> el merge lo duplica a propósito
base = base_seg_limpio.copy()
base['periodo'] = pd.to_datetime(base['periodo'])

df_trx = base.merge(df_llave, on=['cod_cliente','des_categoria'], how='inner')
df_trx['flg_accion'] = (df_trx['rol'] == 'accion').astype(int)

print(f"\nTransaccional marcada: {len(df_trx):,} filas | "
      f"{df_trx.cod_cliente.nunique():,} clientes únicos")

# ── PASO B — PANEL MENSUAL POR PAR (CORREGIDO, con des_oficina) ────────────
paneles = []

for (cos, cat), g in df_trx.groupby(['cosecha_grupo','des_categoria']):
    ini_pre = cos - relativedelta(months=12)
    fin_pre = cos - relativedelta(months=1)

    # todos los pares de esta celda, fijos, sin importar si compraron o no
    # (incluye des_oficina: propiedad del PAR, no varía entre accion/control — viene de celda 44)
    pares_celda = df_llave[(df_llave.cosecha_grupo == cos) &
                           (df_llave.des_categoria == cat)][['par_id','rol','des_oficina']].drop_duplicates()
    n_pares_fijo = pares_celda.par_id.nunique()
    oficina_map = pares_celda.drop_duplicates('par_id').set_index('par_id')['des_oficina']

    px = (g.groupby(['par_id','rol','periodo'])['peso']
            .sum().clip(lower=0).reset_index())

    # ── Winsorización de outliers mensuales ──────────────────────────────
    # Diagnóstico: en Galletas-Abril, un solo cliente de control tuvo una compra
    # de 760 kg justo en el mes de cosecha (p99.5 de esa celda es ~36 kg) -> ese
    # ÚNICO valor infla la MEDIA del DiD ese mes (la mediana casi no se mueve).
    # Se topa cualquier peso mensual por encima del percentil 99.5 de su propia
    # celda (cosecha×categoría), aplicado por igual a TODOS los meses (pre y
    # post, acción y control) para no introducir un sesgo direccional.
    CAP_PCTL = 99.5
    tope_peso = px.loc[px.peso > 0, 'peso'].quantile(CAP_PCTL / 100)
    filas_capadas = px[px.peso > tope_peso]
    if len(filas_capadas):
        # identificar los clientes detrás de las filas capadas (para auditoría)
        clientes_capados = (
            g[g.peso > 0]
            .groupby(['cod_cliente', 'par_id', 'rol', 'periodo'])['peso'].sum()  # total real del mes
            .reset_index()
            .merge(filas_capadas[['par_id', 'rol', 'periodo']], on=['par_id', 'rol', 'periodo'])
            [['cod_cliente', 'rol', 'periodo', 'peso']]
            .sort_values('peso', ascending=False)
        )
        print(f"  [WINSORIZACION] {cos.date()} | {cat}: {len(filas_capadas)} fila(s) mes-par con "
              f"peso > p{CAP_PCTL} ({tope_peso:.1f} kg) -> capadas a ese tope. Clientes:")
        print(clientes_capados.head(10).to_string(index=False))
        px['peso'] = px['peso'].clip(upper=tope_peso)

    for k in range(MESES_POST):
        mes_post = cos + relativedelta(months=k)
        if mes_post > MES_CIERRE or mes_post > g['periodo'].max():
            break

        if MODO_PRE == 'mismo_mes':
            mes_base = mes_post - relativedelta(months=12)
            if mes_base < ini_pre:
                continue
            base_v = px[px.periodo == mes_base].set_index(['par_id','rol'])['peso']
        else:
            base_v = (px[(px.periodo >= ini_pre) & (px.periodo <= fin_pre)]
                        .groupby(['par_id','rol'])['peso'].sum() / 12)

        post_v = px[px.periodo == mes_post].set_index(['par_id','rol'])['peso']

        # REINDEX explícito sobre TODOS los pares × rol -> ausencia = 0, no ausencia de fila
        idx_completo = pd.MultiIndex.from_frame(pares_celda[['par_id','rol']])
        base_v = base_v.reindex(idx_completo, fill_value=0.0)
        post_v = post_v.reindex(idx_completo, fill_value=0.0)

        d = pd.DataFrame({'pre': base_v, 'post': post_v}).unstack('rol').fillna(0.0)
        d.columns = [f"{a}_{b}" for a, b in d.columns]
        for c in ['pre_accion','pre_control','post_accion','post_control']:
            if c not in d:
                d[c] = 0.0

        d['delta_accion']  = d.post_accion  - d.pre_accion
        d['delta_control'] = d.post_control - d.pre_control
        d['did_kg']        = d.delta_accion - d.delta_control
        d = d.assign(cosecha_grupo=cos, des_categoria=cat,
                     periodo=mes_post, mes_rel=k).reset_index()
        d['des_oficina'] = d['par_id'].map(oficina_map)

        assert d.par_id.nunique() == n_pares_fijo, \
            f"n_pares varía: {d.par_id.nunique()} vs {n_pares_fijo} esperado"

        paneles.append(d)

df_panel = pd.concat(paneles, ignore_index=True)
print(f"Panel: {len(df_panel):,} filas | pares por celda deben ser constantes -> chequeo:")
print(df_panel.groupby(['cosecha_grupo','des_categoria','mes_rel']).par_id.nunique().unstack().to_string())

# =============================================================================
# PASO C — BRECHA + IC BOOTSTRAP (función reutilizable)
# =============================================================================
# Antes esta lógica estaba duplicada (una vez por cada corte que se quisiera
# reportar: agregado, por oficina, MODO_PRE alterno...). Se extrae a función
# para poder reutilizarla con distintos group_cols sin repetir el bootstrap.

from mediciones.metodologia.inferencia import bootstrap_por_par as calcular_brechas_bootstrap, distribucion_brechas_cosechas as calcular_distribucion_brechas   # método por par (ver inferencia.py)


# ── n mínimo dinámico por celda (población real de esa cosecha×categoría) ──
n_min_dict_agregado = {
    (row.cosecha, row.categoria): n_minimo_requerido(row.AC_limpio, row.CT_limpio)
    for row in df_resumen.itertuples()
}
print("n mínimo requerido por celda (5% margen / 95% confianza):")
for k, v in n_min_dict_agregado.items():
    print(f"  {k[0].date()} | {k[1]}: n_min={v}")

# ── Cálculo agregado: cosecha × categoría × mes_rel (el nivel "oficial" del reporte) ──
df_brechas = calcular_brechas_bootstrap(
    df_panel, ['cosecha_grupo','des_categoria','mes_rel'], n_min_dict=n_min_dict_agregado)
df_distribucion = calcular_distribucion_brechas(
    df_panel, ['cosecha_grupo','des_categoria','mes_rel'], n_min_dict=n_min_dict_agregado)

# ── Propagar 'reportable' (validez del matching + paralelismo) a las brechas ──
reportable_map = {(row.cosecha, row.categoria): row.reportable for row in df_resumen.itertuples()}
df_brechas['reportable'] = df_brechas.apply(
    lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
df_distribucion['reportable'] = df_distribucion.apply(
    lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
print(f"\nFilas de df_brechas que SÍ son reportables (grupo comparable): "
      f"{df_brechas.reportable.sum()} de {len(df_brechas)}")

print("\n" + "="*110)
print("BRECHAS DiD — RESUMEN (IC 95% para lectura rápida; ver bandas completas abajo)")
print("="*110)
print(df_brechas[['cosecha_grupo','des_categoria','mes_rel','n_pares',
                  'brecha_kg','brecha_pct','ic_pct_inf','ic_pct_sup',
                  'p_valor','significativo']].to_string(index=False))

print("\n" + "="*110)
print("BRECHAS DiD — INTERVALOS DE CONFIANZA 80% / 90% / 95% (mismo bootstrap, 3 anchos)")
print("="*110)
cols_ic = ['cosecha_grupo','des_categoria','mes_rel','n_pares','brecha_pct']
for nivel in NIVELES_IC:
    cols_ic += [f'ic_pct_inf_{nivel}', f'ic_pct_sup_{nivel}']
print(df_brechas[cols_ic].to_string(index=False))

# Consolidado por cosecha × categoría (promedio de meses post)
agg_dict = dict(meses=('mes_rel','count'), n_pares=('n_pares','mean'),
                brecha_pct_prom=('brecha_pct','mean'))
for nivel in NIVELES_IC:
    agg_dict[f'meses_signif_{nivel}'] = (f'significativo_{nivel}', 'sum')

df_brecha_resumen = (df_brechas.groupby(['cosecha_grupo','des_categoria'])
    .agg(**agg_dict).round(2).reset_index())
for nivel in NIVELES_IC:
    df_brecha_resumen[f'pct_meses_signif_{nivel}'] = (
        100*df_brecha_resumen[f'meses_signif_{nivel}']/df_brecha_resumen.meses).round(1)

print("\nCONSOLIDADO (significancia a 80% / 90% / 95%)")
print(df_brecha_resumen.to_string(index=False))


# =============================================================================
# BRECHA ACUMULADA — 2 METODOS (pendiente de la reunion con Julio/Jessica)
# Ambos se apoyan en calcular_brechas_bootstrap (misma funcion de arriba) sobre
# df_panel: ninguno promedia brecha_pct ya calculado, agrupan filas par x mes_rel
# directamente en el mismo bootstrap -> el peso por "cantidad de clientes" sale
# solo de cuantos pares aportan cada mes, sin ponderacion manual encima de %.
#
# Metodo 1 (comparacion pareja entre cosechas): pool de mes_rel 0..M_COMUN, con
#   M_COMUN el mes_rel mas alto que las 3 cosechas alcanzan EN SIMULTANEO hoy
#   (dato-dependiente, no se hardcodea).
# Metodo 2 (maduracion, definicion de Julio): pool de mes_rel >= 2, solo para
#   cosechas que ya tienen al menos un mes_rel >= 2 disponible.
#
# NOTA sobre n_min_dict: calcular_brechas_bootstrap asume que 'mes_rel' es el
# ultimo elemento de group_cols para hacer el lookup del n minimo dinamico. Aqui
# group_cols = ['cosecha_grupo','des_categoria'] (sin mes_rel), asi que NO se le
# pasa n_min_dict_agregado (la clave no calzaria) -> corre con el piso fijo
# n_min=30, igual que el resto de cortes de v6.
# =============================================================================

M_COMUN = int(df_panel.groupby('cosecha_grupo')['mes_rel'].max().min())
print(f"M comun a las 3 cosechas (mes_rel mas alto que TODAS alcanzan hoy): {M_COMUN}")

# ── Metodo 1 — ventana [0, M_COMUN], igual para las 3 cosechas ──────────────
panel_metodo1 = df_panel[df_panel.mes_rel.between(0, M_COMUN)]
df_brecha_acum_m1 = calcular_brechas_bootstrap(panel_metodo1, ['cosecha_grupo', 'des_categoria'])
df_brecha_acum_m1['metodo'] = f'ponderado_hasta_mes_{M_COMUN}'
df_brecha_acum_m1['reportable'] = df_brecha_acum_m1.apply(
    lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
df_brecha_acum_m1 = df_brecha_acum_m1.drop(columns='periodo')  # no aplica: pool de varios meses

# ── Metodo 2 — ventana [2, +inf), solo cosechas con mes_rel maximo >= 2 ─────
max_mes_rel_cosecha = df_panel.groupby('cosecha_grupo')['mes_rel'].max()
cosechas_maduras = max_mes_rel_cosecha[max_mes_rel_cosecha >= 2].index
panel_metodo2 = df_panel[(df_panel.mes_rel >= 2) & (df_panel.cosecha_grupo.isin(cosechas_maduras))]

if len(panel_metodo2):
    df_brecha_acum_m2 = calcular_brechas_bootstrap(panel_metodo2, ['cosecha_grupo', 'des_categoria'])
    df_brecha_acum_m2['metodo'] = 'maduracion_desde_mes_2'
    df_brecha_acum_m2['reportable'] = df_brecha_acum_m2.apply(
        lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
    df_brecha_acum_m2 = df_brecha_acum_m2.drop(columns='periodo')
else:
    df_brecha_acum_m2 = pd.DataFrame()
    print("[SIN RESULTADOS] Ninguna cosecha alcanza todavia mes_rel >= 2 -> Metodo 2 "
          "queda vacio hasta que haya mas meses medidos.")

# ── Comparacion lado a lado -> responde el pedido de Julio: ¿cuanto difieren? ──
if len(df_brecha_acum_m2):
    comp_metodos = (
        df_brecha_acum_m1[['cosecha_grupo', 'des_categoria', 'n_pares', 'brecha_pct',
                            'ic_pct_inf', 'ic_pct_sup', 'reportable']]
        .rename(columns={'n_pares': 'n_pares_m1', 'brecha_pct': 'brecha_pct_m1',
                          'ic_pct_inf': 'ic_inf_m1', 'ic_pct_sup': 'ic_sup_m1'})
        .merge(df_brecha_acum_m2[['cosecha_grupo', 'des_categoria', 'n_pares', 'brecha_pct',
                                   'ic_pct_inf', 'ic_pct_sup']]
               .rename(columns={'n_pares': 'n_pares_m2', 'brecha_pct': 'brecha_pct_m2',
                                 'ic_pct_inf': 'ic_inf_m2', 'ic_pct_sup': 'ic_sup_m2'}),
               on=['cosecha_grupo', 'des_categoria'], how='outer'))
    comp_metodos['dif_pp'] = (comp_metodos['brecha_pct_m1'] - comp_metodos['brecha_pct_m2']).round(2)
    print("\n" + "=" * 100)
    print(f"METODO 1 (ponderado hasta mes {M_COMUN}) vs METODO 2 (maduracion desde mes 2)")
    print("=" * 100)
    print(comp_metodos.to_string(index=False))
else:
    comp_metodos = pd.DataFrame()

df_brecha_acumulada = pd.concat([df_brecha_acum_m1, df_brecha_acum_m2], ignore_index=True)
print(f"\ndf_brecha_acumulada: {len(df_brecha_acumulada):,} filas (Metodo 1 + Metodo 2)")


# =============================================================================
# CHEQUEO DE SENSIBILIDAD — misma lógica, pre-período = PROMEDIO 12M (no mismo mes)
# =============================================================================
# Antes esta celda repetía el Paso B/C completo y SOBREESCRIBÍA df_panel/df_brechas
# de la celda anterior -> lo que terminaba en el Excel dependía de cuál celda se
# corrió última, no de una decisión explícita. Ahora corre con MODO_PRE='promedio'
# pero guarda en df_panel_promedio / df_brechas_promedio, para comparar ambos
# enfoques sin que uno pise al otro. El reporte oficial usa 'mismo_mes' (celda 46).
MODO_PRE_SENSIBILIDAD = 'promedio'

paneles_prom = []
for (cos, cat), g in df_trx.groupby(['cosecha_grupo','des_categoria']):
    ini_pre = cos - relativedelta(months=12)
    fin_pre = cos - relativedelta(months=1)

    pares_celda = df_llave[(df_llave.cosecha_grupo == cos) &
                           (df_llave.des_categoria == cat)][['par_id','rol','des_oficina']].drop_duplicates()
    n_pares_fijo = pares_celda.par_id.nunique()
    oficina_map = pares_celda.drop_duplicates('par_id').set_index('par_id')['des_oficina']

    px = (g.groupby(['par_id','rol','periodo'])['peso']
            .sum().clip(lower=0).reset_index())

    for k in range(MESES_POST):
        mes_post = cos + relativedelta(months=k)
        if mes_post > MES_CIERRE or mes_post > g['periodo'].max():
            break

        base_v = (px[(px.periodo >= ini_pre) & (px.periodo <= fin_pre)]
                    .groupby(['par_id','rol'])['peso'].sum() / 12)
        post_v = px[px.periodo == mes_post].set_index(['par_id','rol'])['peso']

        idx_completo = pd.MultiIndex.from_frame(pares_celda[['par_id','rol']])
        base_v = base_v.reindex(idx_completo, fill_value=0.0)
        post_v = post_v.reindex(idx_completo, fill_value=0.0)

        d = pd.DataFrame({'pre': base_v, 'post': post_v}).unstack('rol').fillna(0.0)
        d.columns = [f"{a}_{b}" for a, b in d.columns]
        for c in ['pre_accion','pre_control','post_accion','post_control']:
            if c not in d:
                d[c] = 0.0

        d['delta_accion']  = d.post_accion  - d.pre_accion
        d['delta_control'] = d.post_control - d.pre_control
        d['did_kg']        = d.delta_accion - d.delta_control
        d = d.assign(cosecha_grupo=cos, des_categoria=cat,
                     periodo=mes_post, mes_rel=k).reset_index()
        d['des_oficina'] = d['par_id'].map(oficina_map)

        assert d.par_id.nunique() == n_pares_fijo, \
            f"n_pares varía: {d.par_id.nunique()} vs {n_pares_fijo} esperado"

        paneles_prom.append(d)

df_panel_promedio = pd.concat(paneles_prom, ignore_index=True)
df_brechas_promedio = calcular_brechas_bootstrap(
    df_panel_promedio, ['cosecha_grupo','des_categoria','mes_rel'])

# ── Comparación mismo_mes vs promedio: ¿cambia la conclusión de significancia? ──
comp = (df_brechas[['cosecha_grupo','des_categoria','mes_rel','brecha_pct','significativo']]
        .merge(df_brechas_promedio[['cosecha_grupo','des_categoria','mes_rel','brecha_pct','significativo']],
               on=['cosecha_grupo','des_categoria','mes_rel'], suffixes=('_mismo_mes','_promedio')))
comp['cambia_significancia'] = comp.significativo_mismo_mes != comp.significativo_promedio

print("\n" + "="*110)
print("SENSIBILIDAD: mismo_mes (oficial) vs promedio 12M — ¿cambia la conclusión?")
print("="*110)
print(comp.to_string(index=False))
if comp.cambia_significancia.any():
    print(f"\n[AVISO] {comp.cambia_significancia.sum()} celda(s) cambian de significativo/no-significativo "
          f"según el método de pre-período usado. Revisar antes de reportar.")
else:
    print("\nAmbos métodos coinciden en significancia para todas las celdas -> el resultado es robusto a esta decisión.")


# =============================================================================
# BRECHAS POR OFICINA — mismo bootstrap, partido por des_oficina, filtrado a:
#  (a) celdas cosecha×categoría que son 'reportable' a nivel agregado (si el grupo
#      completo no es comparable, partirlo por oficina no lo arregla), Y
#  (b) oficinas con MIN_CT_OFICINA=30 alcanzado (piso fijo, IGUAL que v6 -- no la
#      fórmula dinámica, que aquí queda solo como columna informativa en df_viabilidad).
# Este filtro por oficina SÍ excluye: replica lo que hacía el notebook original (v6)
# con el chequeo de viabilidad por oficina -> evidenciar que no hay control suficiente
# y, por eso mismo, NO calcular una brecha con él.
# =============================================================================

viab_ok = df_viabilidad.rename(columns={'cosecha': 'cosecha_grupo', 'categoria': 'des_categoria'})
viab_ok['reportable_padre'] = viab_ok.apply(
    lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
viab_ok = viab_ok[viab_ok.viable & viab_ok.reportable_padre][
    ['cosecha_grupo', 'des_categoria', 'des_oficina', 'grupo']]

print(f"Celdas cosecha×categoría×oficina que pasan AMBOS filtros (comparable + n mínimo): "
      f"{len(viab_ok):,} de {len(df_viabilidad):,}")

panel_oficina_ok = df_panel.merge(
    viab_ok, on=['cosecha_grupo', 'des_categoria', 'des_oficina'], how='inner')

print(f"Panel filtrado: {len(panel_oficina_ok):,} de {len(df_panel):,} filas "
      f"({100*len(panel_oficina_ok)/max(1,len(df_panel)):.1f}%)")

# n mínimo dinámico por oficina (ya viene calculado en df_viabilidad, celda del matching)
n_min_dict_oficina = {
    (row.cosecha, row.categoria, row.des_oficina): row.n_min_requerido
    for row in df_viabilidad.itertuples()
}

df_brechas_oficina = calcular_brechas_bootstrap(
    panel_oficina_ok, ['cosecha_grupo', 'des_categoria', 'des_oficina', 'mes_rel'],
    n_min_dict=n_min_dict_oficina)
df_distribucion_oficina = calcular_distribucion_brechas(
    panel_oficina_ok, ['cosecha_grupo', 'des_categoria', 'des_oficina', 'mes_rel'],
    n_min_dict=n_min_dict_oficina)

print("\n" + "=" * 110)
print("BRECHAS DiD POR OFICINA (solo celdas comparables y con n mínimo real alcanzado)")
print("=" * 110)
if len(df_brechas_oficina):
    print(df_brechas_oficina[['cosecha_grupo', 'des_categoria', 'des_oficina', 'mes_rel', 'n_pares',
                              'brecha_kg', 'brecha_pct', 'ic_pct_inf', 'ic_pct_sup',
                              'p_valor', 'significativo']].to_string(index=False))
else:
    print("[SIN RESULTADOS] Ninguna combinación cosecha×categoría×oficina pasó ambos filtros. "
          "Reportar la brecha SOLO a nivel cosecha×categoría (ver df_brechas) y documentar "
          "la limitación por oficina abajo.")

# ── Documentar explícitamente qué combinaciones NO se pueden reportar por oficina,
#    separando la razón: celda padre no comparable vs. oficina sin n mínimo ──
todas_combos = df_panel[['cosecha_grupo', 'des_categoria', 'des_oficina']].drop_duplicates()
df_sin_control_oficina = todas_combos.merge(
    viab_ok[['cosecha_grupo', 'des_categoria', 'des_oficina']],
    on=['cosecha_grupo', 'des_categoria', 'des_oficina'], how='left', indicator=True)
df_sin_control_oficina = (df_sin_control_oficina[df_sin_control_oficina._merge == 'left_only']
                          .drop(columns='_merge'))
df_sin_control_oficina['motivo'] = df_sin_control_oficina.apply(
    lambda r: 'celda padre no comparable (matching estadístico: corr/ratio_vol)'
    if not reportable_map.get((r.cosecha_grupo, r.des_categoria), False)
    else 'oficina sin n mínimo suficiente (evidencia falta de control, como en v6)', axis=1)

print(f"\ncosecha×categoría×oficina SIN brecha propia reportable: {len(df_sin_control_oficina):,}")
print(df_sin_control_oficina.to_string(index=False))


# =============================================================================
# EMBUDO MENSUAL ACCIÓN/CONTROL + NIVEL DE ERROR DE LA MUESTRA
# =============================================================================
# n_pares_matcheados : tamaño de la muestra pareada para esa celda (AC=CT, 1:1 por diseño de matching)
# n_accion_activo    : de esos pares, cuántos AC SÍ compraron ese mes puntual (post_accion > 0)
# n_control_activo   : de esos pares, cuántos CT SÍ compraron ese mes puntual (post_control > 0)
# n_efectivo         : el MENOR de los dos anteriores -> es el verdadero cuello de botella
#                      que sostiene la comparación de ese mes (no el total de pares matcheados)
# margen_error_XX_pct: fórmula de muestreo con corrección por población finita (ver
# n_minimo_muestra/margen_error_real, celda 'TAMAÑO DE MUESTRA MÍNIMO') — NO la versión
# simple sin corrección que se usaba antes. La población de referencia es AC_limpio /
# CT_limpio de esa celda (el universo del que salen los 'activos' de cada mes).
pob_lookup = {(row.cosecha, row.categoria): (row.AC_limpio, row.CT_limpio, row.grupo)
              for row in df_resumen.itertuples()}

embudo_rows = []
for (cos, cat, mrel), g in df_panel.groupby(['cosecha_grupo', 'des_categoria', 'mes_rel']):
    n_pares = g.par_id.nunique()
    n_ac_activo = int((g.post_accion > 0).sum())
    n_ct_activo = int((g.post_control > 0).sum())
    n_efectivo = min(n_ac_activo, n_ct_activo)
    ac_limpio_cel, ct_limpio_cel, grupo_cel = pob_lookup.get((cos, cat), (np.nan, np.nan, '?'))

    fila = {
        'cosecha_grupo': cos, 'grupo': grupo_cel, 'des_categoria': cat, 'mes_rel': mrel,
        'periodo': g.periodo.iloc[0],
        'n_pares_matcheados': n_pares,
        'n_accion_activo': n_ac_activo,
        'pct_accion_activo': round(100 * n_ac_activo / n_pares, 1) if n_pares else np.nan,
        'n_control_activo': n_ct_activo,
        'pct_control_activo': round(100 * n_ct_activo / n_pares, 1) if n_pares else np.nan,
        'n_efectivo': n_efectivo,
    }
    for nivel in [80, 90, 95]:
        e_ac = margen_error_real(n_ac_activo, ac_limpio_cel, confianza=nivel)
        e_ct = margen_error_real(n_ct_activo, ct_limpio_cel, confianza=nivel)
        e = np.nanmax([e_ac, e_ct])
        fila[f'margen_error_{nivel}_pct'] = e if pd.notna(e) else np.nan
    embudo_rows.append(fila)

df_embudo_mensual = pd.DataFrame(embudo_rows).sort_values(
    ['cosecha_grupo', 'des_categoria', 'mes_rel']).reset_index(drop=True)

# ── Contexto: población original ANTES del matching (de dónde partimos) ─────
pob = df_resumen.rename(columns={'cosecha': 'cosecha_grupo', 'categoria': 'des_categoria'})[
    ['cosecha_grupo', 'des_categoria', 'AC_limpio', 'CT_limpio', 'pares']]
df_embudo_mensual = df_embudo_mensual.merge(pob, on=['cosecha_grupo', 'des_categoria'], how='left')

# Guardado explícito -> mismo patrón que el resto del pipeline (carpeta 'Surtido Ideal - Medición')
ruta_salida_embudo = os.path.join('outputs', 'Surtido Ideal - Medición')
os.makedirs(ruta_salida_embudo, exist_ok=True)
df_embudo_mensual.to_csv(os.path.join(ruta_salida_embudo, 'df_embudo_mensual.csv'), index=False)
print(f"[GUARDADO] {os.path.join(ruta_salida_embudo, 'df_embudo_mensual.csv')}")

print("\n" + "=" * 110)
print("EMBUDO MENSUAL ACCIÓN/CONTROL — desde el mes de ejecución (mes_rel = 0)")
print("=" * 110)
cols_show = ['cosecha_grupo', 'des_categoria', 'mes_rel', 'AC_limpio', 'CT_limpio',
             'n_pares_matcheados', 'n_accion_activo', 'n_control_activo', 'n_efectivo',
             'margen_error_80_pct', 'margen_error_90_pct', 'margen_error_95_pct']
print(df_embudo_mensual[cols_show].to_string(index=False))

print("\n[LECTURA]")
print("  AC_limpio / CT_limpio   : población TOTAL disponible antes del matching (universo elegible).")
print("  n_pares_matcheados      : tamaño de la muestra pareada (fijo, AC=CT por diseño 1:1).")
print("  n_accion/control_activo : de esos pares, cuántos SÍ compraron ESE mes -> variable mes a mes.")
print("  n_efectivo              : el menor de los dos -> tamaño real que sostiene la brecha del mes.")
print("  margen_error_XX_pct     : con ese n_efectivo, qué tan preciso es el estimado a XX% de confianza.")

# ── Gráfico: evolución mensual de AC/CT activos + margen de error ───────────
for (cos, cat), g in df_embudo_mensual.groupby(['cosecha_grupo', 'des_categoria']):
    g = g.sort_values('mes_rel')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    ax1.plot(g.mes_rel, g.n_accion_activo, marker='o', color='#1f3864', label='Acción activo')
    ax1.plot(g.mes_rel, g.n_control_activo, marker='s', ls='--', color='#c00000', label='Control activo')
    ax1.axhline(g.n_pares_matcheados.iloc[0], color='grey', ls=':', lw=1, label='Pares matcheados (fijo)')
    ax1.set_title(f"{cos.date()} | {cat} — clientes activos por mes")
    ax1.set_xlabel('mes_rel'); ax1.set_ylabel('n° clientes'); ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

    for nivel, color in zip([80, 90, 95], ['#2e7d32', '#f9a825', '#c00000']):
        ax2.plot(g.mes_rel, g[f'margen_error_{nivel}_pct'], marker='o', label=f'{nivel}%', color=color)
    ax2.set_title(f"{cos.date()} | {cat} — margen de error (%) según n_efectivo")
    ax2.set_xlabel('mes_rel'); ax2.set_ylabel('margen de error (%)'); ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

    plt.tight_layout(); plt.show()


# =============================================================================
# EMBUDO MENSUAL POR OFICINA — mismo patrón que el embudo global (celda anterior),
# partido también por des_oficina. Evidencia mes a mes de por qué la brecha por
# oficina solo se puede reportar donde el control alcanza (ver celda 52 / df_viabilidad).
# =============================================================================
pob_lookup_oficina = {
    (row.cosecha, row.categoria, row.des_oficina): (row.AC, row.CT, row.n_min_requerido, row.viable)
    for row in df_viabilidad.itertuples()
}

embudo_ofi_rows = []
for (cos, cat, ofi, mrel), g in df_panel.groupby(['cosecha_grupo', 'des_categoria', 'des_oficina', 'mes_rel']):
    n_pares = g.par_id.nunique()
    n_ac_activo = int((g.post_accion > 0).sum())
    n_ct_activo = int((g.post_control > 0).sum())
    n_efectivo = min(n_ac_activo, n_ct_activo)
    ac_ofi, ct_ofi, n_min_ofi, viable_ofi = pob_lookup_oficina.get(
        (cos, cat, ofi), (np.nan, np.nan, np.nan, False))

    fila = {
        'cosecha_grupo': cos, 'des_categoria': cat, 'des_oficina': ofi, 'mes_rel': mrel,
        'periodo': g.periodo.iloc[0],
        'AC_oficina': ac_ofi, 'CT_oficina': ct_ofi,
        'n_min_requerido_oficina': n_min_ofi, 'viable_oficina': viable_ofi,
        'n_pares_matcheados': n_pares,
        'n_accion_activo': n_ac_activo,
        'pct_accion_activo': round(100 * n_ac_activo / n_pares, 1) if n_pares else np.nan,
        'n_control_activo': n_ct_activo,
        'pct_control_activo': round(100 * n_ct_activo / n_pares, 1) if n_pares else np.nan,
        'n_efectivo': n_efectivo,
    }
    for nivel in [80, 90, 95]:
        e_ac = margen_error_real(n_ac_activo, ac_ofi, confianza=nivel)
        e_ct = margen_error_real(n_ct_activo, ct_ofi, confianza=nivel)
        e = np.nanmax([e_ac, e_ct])
        fila[f'margen_error_{nivel}_pct'] = e if pd.notna(e) else np.nan
    embudo_ofi_rows.append(fila)

df_embudo_mensual_oficina = pd.DataFrame(embudo_ofi_rows).sort_values(
    ['cosecha_grupo', 'des_categoria', 'des_oficina', 'mes_rel']).reset_index(drop=True)

# Guardado — mismo patrón que el embudo global (misma carpeta, ya creada arriba)
df_embudo_mensual_oficina.to_csv(
    os.path.join(ruta_salida_embudo, 'df_embudo_mensual_oficina.csv'), index=False)
print(f"[GUARDADO] {os.path.join(ruta_salida_embudo, 'df_embudo_mensual_oficina.csv')}")

print("\n" + "=" * 110)
print("EMBUDO MENSUAL POR OFICINA — desde el mes de ejecución (mes_rel = 0)")
print("=" * 110)
cols_show_ofi = ['cosecha_grupo', 'des_categoria', 'des_oficina', 'mes_rel',
                 'AC_oficina', 'CT_oficina', 'n_min_requerido_oficina', 'viable_oficina',
                 'n_pares_matcheados', 'n_accion_activo', 'n_control_activo', 'n_efectivo',
                 'margen_error_80_pct', 'margen_error_90_pct', 'margen_error_95_pct']
print(df_embudo_mensual_oficina[cols_show_ofi].to_string(index=False))

print("\n[LECTURA]")
print("  AC_oficina / CT_oficina : población de esa oficina ANTES del matching (misma fuente que df_viabilidad).")
print("  n_min_requerido_oficina : n mínimo estadístico para esa oficina (5% margen / 95% confianza).")
print("  viable_oficina          : si CT_oficina >= n_min_requerido_oficina (snapshot inicial, sin mes).")
print("  n_accion/control_activo : de los pares de ESA oficina, cuántos SÍ compraron ese mes -> cae con el tiempo.")
print("  n_efectivo              : cuello de botella real mes a mes -> se degrada por oficina, incluso en")
print("                            oficinas marcadas 'viable' en el snapshot inicial (antes de perder activos).")

# ── Resumen: cuántas combinaciones oficina×mes quedan por debajo del n mínimo ──
bajo_n_min_ofi = df_embudo_mensual_oficina[
    df_embudo_mensual_oficina.n_efectivo < df_embudo_mensual_oficina.n_min_requerido_oficina]
print(f"\n[RESUMEN] combinaciones cosecha×categoría×oficina×mes con n_efectivo < n_min_requerido: "
      f"{len(bajo_n_min_ofi):,} de {len(df_embudo_mensual_oficina):,} "
      f"({100*len(bajo_n_min_ofi)/max(1,len(df_embudo_mensual_oficina)):.1f}%)")


import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(SEED_BOOT)

for cos in df_panel['cosecha_grupo'].unique():
    sub_cos = df_panel[df_panel.cosecha_grupo == cos]
    cats  = sorted(sub_cos['des_categoria'].unique())
    meses = sorted(sub_cos['mes_rel'].unique())

    fig, axes = plt.subplots(len(cats), len(meses),
                             figsize=(4.2*len(meses), 3.6*len(cats)), squeeze=False)

    for fi, cat in enumerate(cats):
        for ci, m in enumerate(meses):
            ax = axes[fi][ci]
            g = sub_cos[(sub_cos.des_categoria == cat) & (sub_cos.mes_rel == m)]
            n = len(g)
            if n < 30:
                ax.axis('off')
                continue

            did     = g['did_kg'].to_numpy()
            base_ac = g['pre_accion'].to_numpy()

            # las mismas 2000 simulaciones que ya corren en el Paso C
            idx = rng.integers(0, n, size=(N_BOOT, n))
            did_b  = did[idx].mean(axis=1)
            base_b = base_ac[idx].mean(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_b = np.where(base_b > 0, 100 * did_b / base_b, np.nan)
            pct_b = pct_b[~np.isnan(pct_b)]

            lo_80, hi_80 = np.percentile(pct_b, [10, 90])
            lo_95, hi_95 = np.percentile(pct_b, [2.5, 97.5])
            obs = 100 * did.mean() / base_ac.mean() if base_ac.mean() > 0 else np.nan

            ax.hist(pct_b, bins=40, color='#1f3864', alpha=0.75, edgecolor='white')
            ax.axvline(0,  color='red',     ls='--', lw=1.3, label='0')
            ax.axvline(obs, color='#c00000', ls='-',  lw=1.6, label='observado')
            # IC 80 en verde oscuro, línea continua
            ax.axvline(lo_80, color='#1b5e20', ls='-',  lw=1.2)
            ax.axvline(hi_80, color='#1b5e20', ls='-',  lw=1.2, label='IC 80%')
            # IC 95 en verde claro, línea punteada
            ax.axvline(lo_95, color='#2e7d32', ls=':',  lw=1.3)
            ax.axvline(hi_95, color='#2e7d32', ls=':',  lw=1.3, label='IC 95%')

            sig = 'SIG' if (lo_95 > 0 or hi_95 < 0) else 'ns'
            ax.set_title(f"{cat} | mes +{m}\nobs={obs:+.1f}% | "
                         f"IC80[{lo_80:+.1f}%,{hi_80:+.1f}%] IC95[{lo_95:+.1f}%,{hi_95:+.1f}%] | {sig}", 
                         fontsize=7.8)
            if fi == len(cats) - 1:
                ax.set_xlabel("brecha % (promedio simulado)")
            if ci == 0:
                ax.set_ylabel("n° simulaciones")
            ax.tick_params(labelsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Distribución de las {N_BOOT} simulaciones bootstrap — cosecha {cos.date()}\n"
                 f"(rojo = 0 | rojo oscuro = brecha observada | verde oscuro = IC 80% | verde claro punteado = IC 95%)",
                 fontsize=11, weight='bold')
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import os

# Crear carpeta de salida
OUTPUT_DIR = 'graficos_bootstrap'
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED_BOOT)

for cos in df_panel['cosecha_grupo'].unique():
    sub_cos = df_panel[df_panel.cosecha_grupo == cos]
    cats  = sorted(sub_cos['des_categoria'].unique())
    meses = sorted(sub_cos['mes_rel'].unique())

    fig, axes = plt.subplots(len(cats), len(meses),
                             figsize=(4.2*len(meses), 3.6*len(cats)), squeeze=False)

    for fi, cat in enumerate(cats):
        for ci, m in enumerate(meses):
            ax = axes[fi][ci]
            g = sub_cos[(sub_cos.des_categoria == cat) & (sub_cos.mes_rel == m)]
            n = len(g)
            if n < 30:
                ax.axis('off')
                continue

            did     = g['did_kg'].to_numpy()
            base_ac = g['pre_accion'].to_numpy()

            idx = rng.integers(0, n, size=(N_BOOT, n))
            did_b  = did[idx].mean(axis=1)
            base_b = base_ac[idx].mean(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_b = np.where(base_b > 0, 100 * did_b / base_b, np.nan)
            pct_b = pct_b[~np.isnan(pct_b)]

            lo_80, hi_80 = np.percentile(pct_b, [10, 90])
            lo_95, hi_95 = np.percentile(pct_b, [2.5, 97.5])
            obs = 100 * did.mean() / base_ac.mean() if base_ac.mean() > 0 else np.nan

            ax.hist(pct_b, bins=40, color='#1f3864', alpha=0.75, edgecolor='white')
            ax.axvline(0,  color='red',     ls='--', lw=1.3, label='0')
            ax.axvline(obs, color='#c00000', ls='-',  lw=1.6, label='observado')
            ax.axvline(lo_80, color='#1b5e20', ls='-',  lw=1.2)
            ax.axvline(hi_80, color='#1b5e20', ls='-',  lw=1.2, label='IC 80%')
            ax.axvline(lo_95, color='#2e7d32', ls=':',  lw=1.3)
            ax.axvline(hi_95, color='#2e7d32', ls=':',  lw=1.3, label='IC 95%')

            sig = 'SIG' if (lo_95 > 0 or hi_95 < 0) else 'ns'
            ax.set_title(f"{cat} | mes +{m}\nobs={obs:+.1f}% | "
                         f"IC80[{lo_80:+.1f}%,{hi_80:+.1f}%] IC95[{lo_95:+.1f}%,{hi_95:+.1f}%] | {sig}", 
                         fontsize=7.8)
            if fi == len(cats) - 1:
                ax.set_xlabel("brecha % (promedio simulado)")
            if ci == 0:
                ax.set_ylabel("n° simulaciones")
            ax.tick_params(labelsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Distribución de las {N_BOOT} simulaciones bootstrap — cosecha {cos.date()}\n"
                 f"(rojo = 0 | rojo oscuro = brecha observada | verde oscuro = IC 80% | verde claro punteado = IC 95%)",
                 fontsize=11, weight='bold')
    plt.tight_layout()
    
    # ═══════════════════════════════════════════════════════════════════
    # ── EXPORTAR A PNG Y PDF ──
    # ═══════════════════════════════════════════════════════════════════
    cosecha_str = cos.strftime('%Y-%m-%d')
    png_path = os.path.join(OUTPUT_DIR, f'bootstrap_{cosecha_str}.png')
    pdf_path = os.path.join(OUTPUT_DIR, f'bootstrap_{cosecha_str}.pdf')
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Cosecha {cosecha_str}:")
    print(f"  PNG (300 dpi): {png_path}")
    print(f"  PDF: {pdf_path}")
    
    plt.show()

print(f"\n✓ Todas las figuras guardadas en: {OUTPUT_DIR}/")
    

sql_cus = """
SELECT id_interlocutor_origen cod_cliente,
      nom_compania,
      fec_creacion,
      des_provincia,
      des_ciudad
FROM `acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente`
WHERE cod_boca_salida IN ('BO')
"""

cus_df = ejecutar_grilla(sql_cus, nombre_cache='cus_df')

cus_df['fec_creacion_mes'] = pd.to_datetime(cus_df['fec_creacion']).dt.to_period('M').dt.to_timestamp()

# =============================================================================
# CRECIMIENTO ATRIBUIBLE POR CLIENTE-CATEGORÍA (YoY vs brecha DiD)
# =============================================================================
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

CATEGORIAS = ['Galletas', 'Cereales']

# ── PASO 1: panel completo cliente × categoría × periodo, ausencia = 0 ──────
base = base_seg_limpio.copy()
base['periodo']          = pd.to_datetime(base['periodo'])
base['cosecha_agrupada']  = pd.to_datetime(base['cosecha_agrupada'])

piv = (base.groupby(['cod_cliente','des_categoria','periodo'])['peso']
            .sum().clip(lower=0).reset_index())

cli_cos = base.dropna(subset=['cosecha_agrupada']).groupby('cod_cliente')['cosecha_agrupada'].min()
print(f"Clientes con cosecha_agrupada: {cli_cos.nunique():,}  (tu número: 33,055 — verifica)")

periodos_all = pd.date_range(base['periodo'].min(), base['periodo'].max(), freq='MS')
print(f"Rango de periodos disponible: {periodos_all.min().date()} .. {periodos_all.max().date()}")

full_idx = pd.MultiIndex.from_product(
    [cli_cos.index, CATEGORIAS, periodos_all],
    names=['cod_cliente','des_categoria','periodo'])

panel = (piv.set_index(['cod_cliente','des_categoria','periodo'])['peso']
            .reindex(full_idx, fill_value=0.0)
            .reset_index()
            .sort_values(['cod_cliente','des_categoria','periodo']))

# lag 12 meses (año anterior, mismo mes) -> shift dentro de grupo, funciona porque
# periodos_all es una malla mensual completa sin huecos
panel['peso_ant'] = panel.groupby(['cod_cliente','des_categoria'])['peso'].shift(12)

# ── PASO 2: filtrar solo periodos POSTERIORES a la cosecha del cliente ──────
panel['cosecha_agrupada'] = panel['cod_cliente'].map(cli_cos)
panel = panel[panel['periodo'] >= panel['cosecha_agrupada']].copy()
panel = panel.dropna(subset=['peso_ant'])   # sin año anterior disponible -> no se puede medir YoY

panel['mes_rel'] = ((panel.periodo.dt.year  - panel.cosecha_agrupada.dt.year) * 12 +
                     (panel.periodo.dt.month - panel.cosecha_agrupada.dt.month))

print(f"\nFilas del panel (post-cosecha, con año anterior disponible): {len(panel):,}")
print(f"Rango de mes_rel: {panel.mes_rel.min()} a {panel.mes_rel.max()}")

# ── PASO 3: variación YoY, en toneladas y en % ───────────────────────────────
panel['toneladas_actual'] = panel['peso'] / 1000
panel['toneladas_ant']    = panel['peso_ant'] / 1000
panel['toneladas_variacion'] = panel['toneladas_actual'] - panel['toneladas_ant']

# ── PASO 4: traer el IC de la brecha DiD (celda 46) para poder comparar el YoY
#    individual de cada cliente contra la banda "normal" estimada por matching ──
# Usamos el IC 95% (columnas sin sufijo, ancladas al 95% dentro de df_brechas).
ic_cols = (df_brechas[['cosecha_grupo', 'des_categoria', 'mes_rel', 'ic_pct_inf', 'ic_pct_sup']]
           .rename(columns={'cosecha_grupo': 'cosecha_agrupada'}))

panel = panel.merge(ic_cols, on=['cosecha_agrupada', 'des_categoria', 'mes_rel'], how='left')

# sin_brecha: filas del panel para las que NO hay bootstrap calculado en df_brechas
# (p.ej. esa celda cosecha×categoría×mes_rel tuvo <30 pares y calcular_brechas_bootstrap
# la omitió) -> no se puede evaluar si el cliente superó el umbral, porque no hay umbral.
sin_brecha = panel['ic_pct_inf'].isna() | panel['ic_pct_sup'].isna()
print(f"\nFilas de panel SIN brecha DiD calculable (sin match en df_brechas): "
      f"{sin_brecha.sum():,} de {len(panel):,} ({100*sin_brecha.mean():.1f}%)")
if sin_brecha.mean() > 0.3:
    print("[AVISO] Más del 30% del panel no tiene banda de referencia -> revisar si el "
          "umbral n_min=30 de calcular_brechas_bootstrap está excluyendo demasiadas celdas.")


# =============================================================================
# AJUSTE: distinguir "reactivado" (existía, compró 0) de "no existía" (cliente
# nuevo en la base, fec_creacion posterior al periodo comparado)
# =============================================================================

# cus_df: una fila por cliente con su fecha de creación
cus = cus_df.copy()
cus['fec_creacion'] = pd.to_datetime(cus['fec_creacion'])
fec_creacion_map = cus.drop_duplicates('cod_cliente').set_index('cod_cliente')['fec_creacion']

dup = cus['cod_cliente'].duplicated().sum()
if dup:
    print(f"[ALERTA] {dup} cod_cliente duplicados en cus_df -> se tomó la primera fila de cada uno")

panel['fec_creacion'] = panel['cod_cliente'].map(fec_creacion_map)
sin_fec = panel['fec_creacion'].isna().sum()
print(f"Clientes en panel sin match en cus_df: {sin_fec:,} de {panel.cod_cliente.nunique():,} "
      f"-> se tratan como 'creación desconocida', no se asume nada")

# periodo comparado del año anterior = mismo mes, 12 meses antes
panel['periodo_ant'] = panel['periodo'] - pd.DateOffset(months=12)

# "no existía" = el cliente se creó DESPUÉS del periodo del año anterior
#   -> no es que se reactivó, es que simplemente no existía. No hay "antes" real.
# [atribucion:inicio principal] estado YoY y YoY crudo según la regla de 'nuevo' del programa (metodologia/atribucion.py)
# 'no existía' = el cliente se creó DESPUÉS del periodo comparado (no es que se reactivó, simplemente no existía);
# 'reactivado' real = existía antes del periodo comparado, pero compró 0 ese mes.
panel['flg_estado_yoy'] = estado_yoy(REGLA_NUEVO, panel['toneladas_ant'], panel['toneladas_actual'], panel['fec_creacion'], panel['periodo_ant'])
cond_no_existia = panel['flg_estado_yoy'] == 'no_existia_periodo_pasado'
cond_reactivado = panel['flg_estado_yoy'] == 'reactivado'
cond_sigue_cero = panel['flg_estado_yoy'] == 'sin_compra_ambos'

print(f"\nClientes-periodo 'no existía periodo pasado' (fec_creacion posterior): {cond_no_existia.sum():,}")
print(f"Clientes-periodo 'reactivado' (existía, compró 0 antes)              : {cond_reactivado.sum():,}")
print(f"Clientes-periodo sin compra en ambos años                            : {cond_sigue_cero.sum():,}")

_atr = atribuir(REGLA_NUEVO, panel['toneladas_ant'], panel['toneladas_actual'], panel['ic_pct_inf'], panel['ic_pct_sup'], estado=panel['flg_estado_yoy'])
panel['yoy_pct_raw'] = _atr['yoy_pct_raw']      # no_existia -> NaN explícito (no inf); reactivado -> inf
# [atribucion:fin principal]

yoy_para_clip = panel['yoy_pct_raw'].replace(np.inf, np.nan)

panel['flg_supero_umbral'] = (
    (yoy_para_clip > panel['ic_pct_sup']) | (yoy_para_clip < panel['ic_pct_inf']))
panel.loc[sin_brecha, 'flg_supero_umbral'] = np.nan
panel.loc[cond_no_existia, 'flg_supero_umbral'] = np.nan   # no aplica: no hay comparación válida

# 'flg_supero_umbral' mezcla ambas direcciones en un solo booleano -> no sirve para
# el punto 6 (detectar específicamente quién está POR DEBAJO). Se separan:
panel['flg_bajo_esperado']  = (yoy_para_clip < panel['ic_pct_inf'])
panel['flg_sobre_esperado'] = (yoy_para_clip > panel['ic_pct_sup'])
panel.loc[sin_brecha, ['flg_bajo_esperado', 'flg_sobre_esperado']] = np.nan
panel.loc[cond_no_existia, ['flg_bajo_esperado', 'flg_sobre_esperado']] = np.nan

# [atribucion:inicio principal_b] crecimiento atribuible acotado a la banda (reactivado: tope = IC superior; no_existia: nunca) y toneladas atribuibles
panel['crecimiento_atribuible_pct'] = _atr['crec_atrib_pct']
panel['toneladas_atribuibles'] = _atr['ton_atrib']      # NaN en no_existia (sin contrafactual) y donde no hay banda
# [atribucion:fin principal_b]

panel['flg_tendencia'] = np.select(
    [panel['toneladas_variacion'] > 0, panel['toneladas_variacion'] < 0],
    ['crece', 'cae'], default='estable')

# ── DF FINAL ──────────────────────────────────────────────────────────────
df_incremental = panel[[
    'cod_cliente','des_categoria','cosecha_agrupada','periodo','mes_rel',
    'fec_creacion','flg_estado_yoy',
    'toneladas_ant','toneladas_actual','toneladas_variacion',
    'yoy_pct_raw','ic_pct_inf','ic_pct_sup',
    'crecimiento_atribuible_pct','toneladas_atribuibles',
    'flg_supero_umbral','flg_bajo_esperado','flg_sobre_esperado','flg_tendencia'
]].reset_index(drop=True)

# NOTA (punto 3 del alcance): para clientes 'no_existia_periodo_pasado' (nuevos, sin
# contrafactual), 'toneladas_atribuibles' es NaN a propósito -> no hay banda con la que
# compararlos. Pero 'toneladas_actual' SÍ está poblada para ellos: esa es la métrica de
# volumen puro ("toda su compra del período") que se usa para estos clientes en el punto 3
# y en la vista de agosto del punto 6, en vez de una brecha que no se les puede calcular.

print(f"\ndf_incremental: {len(df_incremental):,} filas | "
      f"{df_incremental.cod_cliente.nunique():,} clientes únicos")

print("\nDISTRIBUCIÓN DE flg_estado_yoy")
print(df_incremental['flg_estado_yoy'].value_counts(normalize=True).mul(100).round(1).to_string())

print("\n[VERIFICACIÓN] toneladas_atribuibles debe ser NaN en TODAS las filas 'no_existia_periodo_pasado':")
chk = df_incremental[df_incremental.flg_estado_yoy == 'no_existia_periodo_pasado']
print(f"  filas: {len(chk):,} | con atribuible no-nulo (debe ser 0): "
      f"{chk['toneladas_atribuibles'].notna().sum():,}")

# caracteristicas del cliente
# [customer_feat:inicio] Las columnas de segmentación estratégica de surtido (flg_cliente_roma, cosecha_roma, es_roma, des_segmento_estrategica, flg_potential)
# solo existen en ese programa. Río abajo únicamente se usan oficina, segmento transaccional y compañía (distribuidor): esas son obligatorias; el resto va si existe.
_COLS_CUSTOMER_FEAT = ['cod_cliente', 'des_oficina', 'nom_compania', 'flg_cliente_roma', 'cosecha_roma', 'es_roma',
                       'des_segmento_transaccional', 'des_segmento_estrategica', 'flg_potential']
_faltan_cf = [c for c in ('cod_cliente', 'des_oficina', 'nom_compania', 'des_segmento_transaccional') if c not in base_seg_limpio.columns]
assert not _faltan_cf, f"base_seg_limpio no trae las columnas obligatorias de customer_feat: {_faltan_cf}"
customer_feat = base_seg_limpio[[c for c in _COLS_CUSTOMER_FEAT if c in base_seg_limpio.columns]].drop_duplicates()
# [customer_feat:fin]

# =============================================================================
# BASE COMPLETA — accion_matcheado + accion_sin_pareja + control_espejo (punto 5)
# =============================================================================

# ── 1) Lado ACCIÓN (matcheado + sin pareja): ya viene de df_incremental (celda 55-56),
#      que cubre a TODO cliente con cosecha_agrupada, tenga o no pareja en el matching.
match_keys = (df_matches[['cod_accion', 'des_categoria', 'cosecha']]
              .drop_duplicates()
              .rename(columns={'cod_accion': 'cod_cliente', 'cosecha': 'cosecha_agrupada'}))
match_keys['rol'] = 'accion_matcheado'

df_incremental_rol = df_incremental.merge(
    match_keys, on=['cod_cliente', 'des_categoria', 'cosecha_agrupada'], how='left')
df_incremental_rol['rol'] = df_incremental_rol['rol'].fillna('accion_sin_pareja')

ofi_map = customer_feat.drop_duplicates('cod_cliente').set_index('cod_cliente')['des_oficina']
df_incremental_rol['des_oficina'] = df_incremental_rol['cod_cliente'].map(ofi_map)

print("Distribución de rol dentro de df_incremental (lado acción):")
print(df_incremental_rol['rol'].value_counts().to_string())

# ── 2) Lado CONTROL: no tiene cosecha_agrupada propia -> se construye igual que el
#      panel de la celda 55, pero anclado a la cosecha_grupo del PAR al que sirve de
#      espejo (un control puede aparecer en varias cosechas; se usa la más temprana
#      para no duplicar su historia de toneladas -> ver "Controles en >1 cosecha",
#      diagnóstico de la celda del pipeline de matching).
control_keys = (df_llave[df_llave.rol == 'control']
                [['cod_cliente', 'des_categoria', 'cosecha_grupo', 'des_oficina']]
                .drop_duplicates())
ct_cosecha = control_keys.groupby('cod_cliente')['cosecha_grupo'].min()

# BUGFIX: antes esto era pd.MultiIndex.from_product([cod_cliente, CATEGORIAS, periodos_all]),
# que cruzaba TODOS los controles contra TODAS las categorias sin respetar en cual de ellas
# el cliente fue realmente matcheado como espejo (df_llave). Como `piv` si tiene el historial
# real de ese cliente en la OTRA categoria, el resultado no salia en 0 -> aparecia un
# control_espejo "fantasma" con datos reales de una categoria donde nunca fue control.
# Ahora solo se usan las combinaciones (cod_cliente, des_categoria) que existen de verdad
# en control_keys.
combos_ct = control_keys[['cod_cliente', 'des_categoria']].drop_duplicates()
ct_idx = pd.MultiIndex.from_frame(
    combos_ct.merge(pd.DataFrame({'periodo': periodos_all}), how='cross'))

panel_ct = (piv.set_index(['cod_cliente', 'des_categoria', 'periodo'])['peso']
              .reindex(ct_idx, fill_value=0.0)
              .reset_index()
              .sort_values(['cod_cliente', 'des_categoria', 'periodo']))

panel_ct['peso_ant'] = panel_ct.groupby(['cod_cliente', 'des_categoria'])['peso'].shift(12)
panel_ct['cosecha_grupo'] = panel_ct['cod_cliente'].map(ct_cosecha)
panel_ct = panel_ct[panel_ct['periodo'] >= panel_ct['cosecha_grupo']].copy()
panel_ct = panel_ct.dropna(subset=['peso_ant'])

panel_ct['mes_rel'] = ((panel_ct.periodo.dt.year - panel_ct.cosecha_grupo.dt.year) * 12 +
                        (panel_ct.periodo.dt.month - panel_ct.cosecha_grupo.dt.month))
panel_ct['toneladas_actual'] = panel_ct['peso'] / 1000
panel_ct['toneladas_ant'] = panel_ct['peso_ant'] / 1000
panel_ct['toneladas_variacion'] = panel_ct['toneladas_actual'] - panel_ct['toneladas_ant']
panel_ct['flg_tendencia'] = np.select(
    [panel_ct['toneladas_variacion'] > 0, panel_ct['toneladas_variacion'] < 0],
    ['crece', 'cae'], default='estable')

ofi_ct_map = control_keys.drop_duplicates('cod_cliente').set_index('cod_cliente')['des_oficina']
panel_ct['des_oficina'] = panel_ct['cod_cliente'].map(ofi_ct_map)
panel_ct['rol'] = 'control_espejo'

print(f"\nPanel de control: {len(panel_ct):,} filas | {panel_ct.cod_cliente.nunique():,} controles únicos")

# ── 3) Unificar esquema y concatenar ─────────────────────────────────────────
COLS_BASE_COMPLETA = [
    'cod_cliente', 'des_categoria', 'des_oficina', 'rol', 'cosecha_grupo', 'periodo', 'mes_rel',
    'toneladas_ant', 'toneladas_actual', 'toneladas_variacion',
    'yoy_pct_raw', 'ic_pct_inf', 'ic_pct_sup',
    'crecimiento_atribuible_pct', 'toneladas_atribuibles',
    'flg_estado_yoy', 'flg_bajo_esperado', 'flg_sobre_esperado', 'flg_tendencia']

accion_final = df_incremental_rol.rename(columns={'cosecha_agrupada': 'cosecha_grupo'})
for c in COLS_BASE_COMPLETA:
    if c not in accion_final.columns:
        accion_final[c] = np.nan
accion_final = accion_final[COLS_BASE_COMPLETA]

for c in COLS_BASE_COMPLETA:
    if c not in panel_ct.columns:
        panel_ct[c] = np.nan
control_final = panel_ct[COLS_BASE_COMPLETA]

df_base_completa = pd.concat([accion_final, control_final], ignore_index=True)

# ── Columna de referencia: reportable de la celda cosecha x categoria ───────
# Solo informativa -> NO filtra ni excluye ninguna fila de df_base_completa. Sirve
# para que quien lea el Excel sepa qué filas de accion/control quedan detras de
# una celda con matching valido + tendencias paralelas (ver hoja 'Test paralelismo'
# de metricas_piloto_cmp.xlsx) y cuales no.
df_base_completa['grupo_ref'] = df_base_completa['cosecha_grupo'].map(GRUPO_COSECHA)
df_base_completa = df_base_completa.merge(
    df_resumen[['grupo', 'categoria', 'reportable']].rename(
        columns={'grupo': 'grupo_ref', 'categoria': 'des_categoria'}),
    on=['grupo_ref', 'des_categoria'], how='left')
df_base_completa = df_base_completa.drop(columns='grupo_ref')

print(f"\ndf_base_completa: {len(df_base_completa):,} filas | "
      f"{df_base_completa.cod_cliente.nunique():,} clientes únicos")
print("\nDistribución por rol:")
print(df_base_completa['rol'].value_counts().to_string())
print("\nToneladas actuales totales por rol:")
print(df_base_completa.groupby('rol')['toneladas_actual'].sum().round(1).to_string())
print("\nFilas por celda reportable (columna de referencia, no filtra nada):")
print(df_base_completa.groupby(['cosecha_grupo', 'des_categoria', 'reportable']).size().to_string())


# =============================================================================
# FLAG CLIENTE PERFECTO / NO PERFECTO — se agrega como columna informativa a
# df_base_completa (no se calculan brechas partidas por este flag, solo se
# anota para que el negocio filtre/pivote la Base completa en Excel).
# =============================================================================
sql_cliente_perfecto = """
select periodo, cod_cliente_dex cod_cliente, flg_cliente_perfecto
from `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_cliente_dex_historico`
where periodo>='2026-01-01'
order by periodo
"""
df_cliente_perfecto = ejecutar_grilla(sql_cliente_perfecto, nombre_cache='cliente_perfecto_historico')

# BUGFIX: el alias implicito de la query (`cod_cliente_dex cod_cliente`, sin AS) no
# llega con ese nombre -> la columna vuelve como 'cod_cliente_dex' (su nombre original
# en BigQuery). Se normaliza el nombre explicitamente en vez de asumir el alias.
# _normalizar_ids ya corrio DENTRO de ejecutar_grilla sobre 'cod_cliente_dex' (tambien
# empieza con 'cod_' y contiene 'cliente'), asi que el dtype/str ya quedo bien -> el
# rename es solo de nombre, no hace falta volver a normalizar.
if 'cod_cliente' not in df_cliente_perfecto.columns and 'cod_cliente_dex' in df_cliente_perfecto.columns:
    df_cliente_perfecto = df_cliente_perfecto.rename(columns={'cod_cliente_dex': 'cod_cliente'})
elif 'cod_cliente' not in df_cliente_perfecto.columns:
    raise KeyError(f"Ni 'cod_cliente' ni 'cod_cliente_dex' estan en las columnas: "
                   f"{list(df_cliente_perfecto.columns)}")
# ejecutar_grilla ya aplica _normalizar_ids (cod_cliente) y parsea 'periodo' a
# datetime automaticamente (ver _PATRONES_FECHA en la celda de cache) -> no hace
# falta repetir esos casts aqui.

dup_cp = df_cliente_perfecto.duplicated(['cod_cliente', 'periodo']).sum()
if dup_cp:
    print(f"[ALERTA] {dup_cp:,} filas duplicadas en (cod_cliente, periodo) de "
          f"cr_cliente_dex_historico -> se conserva la primera de cada combinacion")
    df_cliente_perfecto = df_cliente_perfecto.drop_duplicates(['cod_cliente', 'periodo'])

df_base_completa = df_base_completa.merge(
    df_cliente_perfecto[['cod_cliente', 'periodo', 'flg_cliente_perfecto']],
    on=['cod_cliente', 'periodo'], how='left')

sin_match_cp = df_base_completa['flg_cliente_perfecto'].isna().sum()
print(f"\nflg_cliente_perfecto agregado a df_base_completa: "
      f"{len(df_base_completa) - sin_match_cp:,} de {len(df_base_completa):,} filas con match "
      f"({100 * (1 - sin_match_cp / len(df_base_completa)):.1f}%)")
print(df_base_completa.groupby(['des_categoria', 'flg_cliente_perfecto'], dropna=False)
      .size().rename('n_filas').to_string())


# =============================================================================
# ESTIMACION RETROACTIVA DE TONELADAS ATRIBUIBLES
# Hueco: meses entre el min_periodo_cosecha REAL del cliente (cuando entro de
# verdad al piloto) y cosecha_agrupada/cosecha_grupo (el mes ancla del bucket
# al que quedo asignado). Esos meses tienen toneladas observadas pero NO tienen
# brecha medida (mes_rel del pipeline arranca en 0 = cosecha_grupo). Se estima
# replicando la brecha del bucket, cliente por cliente y mes por mes -> es una
# APROXIMACION (no hay contrafactual propio para esos meses), no una medicion
# directa. Se marca explicito con flg_estimado_retroactivo para que nunca se
# lea como si fuera una brecha medida.
# =============================================================================

# ── 1) Brecha de referencia: pool mes_rel [0,2] por cosecha x categoria, en los
#      que se pueda (una cosecha con solo mes_rel=0 disponible hoy, ej. Grupo 03,
#      usa solo ese) -> mismo mecanismo de pooling que el Metodo 1/2 de arriba,
#      confirmado con el usuario ("tomaria incluso hasta m02, en los que se pueda").
panel_referencia_retro = df_panel[df_panel.mes_rel.between(0, 2)]
df_brecha_referencia_retro = calcular_brechas_bootstrap(
    panel_referencia_retro, ['cosecha_grupo', 'des_categoria'])
df_brecha_referencia_retro['reportable'] = df_brecha_referencia_retro.apply(
    lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)

print("Brecha de referencia para retroactivo (pool mes_rel 0-2, por cosecha x categoria):")
cols_ref_print = (['cosecha_grupo', 'des_categoria', 'n_pares', 'reportable', 'brecha_kg', 'brecha_pct']
                   + [f'ic_kg_inf_{n}' for n in NIVELES_IC] + [f'ic_kg_sup_{n}' for n in NIVELES_IC])
print(df_brecha_referencia_retro[cols_ref_print].to_string(index=False))

cols_ic_ref = [f'ic_kg_inf_{n}' for n in NIVELES_IC] + [f'ic_kg_sup_{n}' for n in NIVELES_IC]
ref_lookup = {
    (row.cosecha_grupo, row.des_categoria):
        {'brecha_kg': row.brecha_kg, 'reportable': row.reportable,
         **{c: getattr(row, c) for c in cols_ic_ref}}
    for row in df_brecha_referencia_retro.itertuples()
}

# ── 2) Poblacion objetivo: clientes accion con min_periodo_cosecha REAL anterior
#      a su cosecha_grupo (bucket) -> ahi esta el hueco a rellenar. Reusa
#      cli_min_per / cli_cosecha, ya calculados en el pipeline de matching.
cli_min_real = cli_min_per.reindex(cli_cosecha.index)
gap_clientes = pd.DataFrame({
    'cod_cliente': cli_cosecha.index,
    'cosecha_grupo': cli_cosecha.values,
    'min_periodo_cosecha_real': cli_min_real.values,
}).dropna(subset=['min_periodo_cosecha_real'])
gap_clientes = gap_clientes[gap_clientes.min_periodo_cosecha_real < gap_clientes.cosecha_grupo]

print(f"\nClientes accion con desfase real vs bucket (hueco a estimar): "
      f"{gap_clientes.cod_cliente.nunique():,} de {len(cli_cosecha):,} clientes accion")

# ── 3) Expandir a un mes por fila, entre el real y el bucket (exclusivo del
#      bucket -> ese mes ya lo mide el pipeline normal, no se duplica)
filas_gap = []
for r in gap_clientes.itertuples():
    meses_gap = pd.date_range(r.min_periodo_cosecha_real, r.cosecha_grupo,
                               freq='MS', inclusive='left')
    for m in meses_gap:
        filas_gap.append({'cod_cliente': r.cod_cliente, 'cosecha_grupo': r.cosecha_grupo,
                           'periodo': m})
df_meses_gap = pd.DataFrame(filas_gap)
print(f"Filas cliente-mes a estimar retroactivamente: {len(df_meses_gap):,}")

# ── 4) Cruzar con des_categoria (un cliente puede tener accion en Galletas,
#      Cereales o ambas -> sale de df_base_completa, lado accion) y con
#      toneladas_actual observadas ese mes (si las hay; puede no haber compra)
accion_cats = (df_base_completa[df_base_completa.rol.isin(['accion_matcheado', 'accion_sin_pareja'])]
               [['cod_cliente', 'des_categoria']].drop_duplicates())
df_retro = df_meses_gap.merge(accion_cats, on='cod_cliente', how='inner')

# BUGFIX: antes se tomaba toneladas_actual de df_base_completa, pero ese panel
# solo cubre periodo >= cosecha_agrupada (ver celda 'CRECIMIENTO ATRIBUIBLE POR
# CLIENTE-CATEGORIA') -> los meses retroactivos, anteriores a esa fecha por
# definicion, NUNCA encontraban match ahi y toneladas_actual_disponible salia
# False el 100% de las veces (confirmado en la validacion del Excel real). Se usa
# 'piv' (panel mensual crudo de base_seg_limpio, ya calculado en esa misma celda),
# que SI cubre los meses pre-cosecha.
toneladas_obs = piv.copy()
toneladas_obs['toneladas_actual'] = toneladas_obs['peso'] / 1000
toneladas_obs = (toneladas_obs[['cod_cliente', 'des_categoria', 'periodo', 'toneladas_actual']]
                  .drop_duplicates(['cod_cliente', 'des_categoria', 'periodo']))
df_retro = df_retro.merge(toneladas_obs, on=['cod_cliente', 'des_categoria', 'periodo'], how='left')
# aun con 'piv' puede no haber fila (cliente sin compra ese mes especifico) -> no
# se inventa un cero, se marca explicito con esta columna
df_retro['toneladas_actual_disponible'] = df_retro['toneladas_actual'].notna()

# ── 5) Aplicar la brecha de referencia de su cosecha x categoria (replicacion
#      plana, tal como lo planteo Julio -> NO se escala ni se interpola)
cols_a_pegar = ['brecha_kg', 'reportable'] + cols_ic_ref
for col in cols_a_pegar:
    df_retro[col] = df_retro.apply(
        lambda r: ref_lookup.get((r.cosecha_grupo, r.des_categoria), {}).get(col, np.nan), axis=1)

df_retro = df_retro.rename(columns={'brecha_kg': 'toneladas_atribuibles_kg_estimadas'})
df_retro['flg_estimado_retroactivo'] = True

sin_referencia = df_retro['toneladas_atribuibles_kg_estimadas'].isna().sum()
if sin_referencia:
    print(f"[ALERTA] {sin_referencia:,} filas sin brecha de referencia disponible "
          f"(cosecha x categoria sin datos en mes_rel 0-2 todavia) -> quedan sin estimar, "
          f"no se les asigna un valor por defecto.")

print(f"\ndf_retro: {len(df_retro):,} filas cliente-mes estimadas | "
      f"{df_retro.cod_cliente.nunique():,} clientes | "
      f"toneladas atribuibles estimadas totales: "
      f"{df_retro['toneladas_atribuibles_kg_estimadas'].sum() / 1000:,.1f} t")
print(df_retro.groupby(['cosecha_grupo', 'des_categoria', 'reportable'])
      ['toneladas_atribuibles_kg_estimadas'].agg(['count', 'sum']).round(1).to_string())


# =============================================================================
# TONELADAS ATRIBUIBLES (IC80) RETROACTIVAS — misma metodologia EXACTA que la
# celda "CRECIMIENTO ATRIBUIBLE POR CLIENTE-CATEGORIA" de arriba (yoy_pct_raw
# capado a la banda IC80 x toneladas_ant; 'nuevos' = volumen puro sin capar),
# aplicada a los meses del hueco retroactivo (antes de cosecha_agrupada).
#
# 'Nuevo' se verifica IGUAL que en esa celda -> no es solo "no compro el año
# pasado" (peso_ant==0): eso separa CANDIDATOS. fec_creacion decide si es
# 'no_existia_periodo_pasado' (sin contrafactual, nunca atribuible) o
# 'reactivado' (existia antes, compro 0 ese mes -> SI atribuible, capado al
# IC80 superior). Sin esa distincion se sobre-cuenta 'nuevos' con clientes que
# en realidad ya existian.
# =============================================================================

# Reusa df_meses_gap / accion_cats (cliente-mes del hueco retroactivo, armados
# en la celda anterior) -> misma poblacion, sin recalcular.
retro_pob = df_meses_gap.merge(accion_cats, on='cod_cliente', how='inner')
retro_pob['periodo_ant'] = retro_pob['periodo'] - pd.DateOffset(months=12)

# peso actual y peso del año anterior (12 meses antes), desde 'piv' (panel
# crudo de base_seg_limpio, ya calculado en la celda de crecimiento atribuible)
piv_actual = piv.rename(columns={'peso': 'peso_actual_r'})
retro_pob = retro_pob.merge(piv_actual, on=['cod_cliente', 'des_categoria', 'periodo'], how='left')
retro_pob['peso_actual_r'] = retro_pob['peso_actual_r'].fillna(0.0)

piv_ant = piv.rename(columns={'periodo': 'periodo_ant', 'peso': 'peso_ant_r'})
retro_pob = retro_pob.merge(piv_ant, on=['cod_cliente', 'des_categoria', 'periodo_ant'], how='left')
retro_pob['peso_ant_r'] = retro_pob['peso_ant_r'].fillna(0.0)

retro_pob['toneladas_actual'] = retro_pob['peso_actual_r'] / 1000
retro_pob['toneladas_ant'] = retro_pob['peso_ant_r'] / 1000
retro_pob['toneladas_variacion'] = retro_pob['toneladas_actual'] - retro_pob['toneladas_ant']

# fec_creacion -> reusa fec_creacion_map, ya armado en la celda de ajuste nuevos/reactivados
retro_pob['fec_creacion'] = retro_pob['cod_cliente'].map(fec_creacion_map)
sin_fec_r = retro_pob['fec_creacion'].isna().sum()
if sin_fec_r:
    print(f"Clientes en el hueco retroactivo sin match en cus_df: {sin_fec_r:,} de "
          f"{retro_pob.cod_cliente.nunique():,} -> se tratan como 'creacion desconocida', no se asume nada")

# [atribucion:inicio retro] estado YoY según la regla de 'nuevo' del programa (metodologia/atribucion.py)
retro_pob['flg_estado_yoy'] = estado_yoy(REGLA_NUEVO, retro_pob['toneladas_ant'], retro_pob['toneladas_actual'], retro_pob['fec_creacion'], retro_pob['periodo_ant'])
# [atribucion:fin retro]

print("\nDistribucion flg_estado_yoy (hueco retroactivo):")
print(retro_pob['flg_estado_yoy'].value_counts(normalize=True).mul(100).round(1).to_string())

# banda IC80 de referencia: pool mes_rel 0-2 por cosecha x categoria (misma
# ventana ya acordada para el retroactivo de brecha, arriba en este notebook)
banda80 = (df_brechas[df_brechas.mes_rel.between(0, 2)]
           .groupby(['cosecha_grupo', 'des_categoria'])[['ic_pct_inf_80', 'ic_pct_sup_80']]
           .mean().reset_index())
retro_pob = retro_pob.merge(banda80, on=['cosecha_grupo', 'des_categoria'], how='left')

# [atribucion:inicio retro_b] YoY, crecimiento atribuible (IC80 promedio) y toneladas atribuibles del hueco retroactivo
_atr_r = atribuir(REGLA_NUEVO, retro_pob['toneladas_ant'], retro_pob['toneladas_actual'], retro_pob['ic_pct_inf_80'], retro_pob['ic_pct_sup_80'], estado=retro_pob['flg_estado_yoy'])
retro_pob['yoy_pct_raw'] = _atr_r['yoy_pct_raw']
retro_pob['crecimiento_atribuible_pct'] = _atr_r['crec_atrib_pct']
retro_pob['toneladas_atribuibles_80'] = _atr_r['ton_atrib']
# [atribucion:fin retro_b]
retro_pob['flg_estimado_retroactivo'] = True

# ── Resumen: antiguos (capados IC80) + nuevos (volumen puro), por grupo x mes ──
# BUGFIX: la formula real tambien exige 'es_activo' (peso_actual_r > 0) para
# contar como 'antiguos' -- un cliente que existia pero compro CERO este mes
# queda FUERA del atribuible (no se le asigna una caida capada al IC80
# inferior). Sin este filtro se sobre-castigaba el atribuible de esos meses
# (confirmado comparando contra los meses medidos ya validados exacto).
es_activo_r = retro_pob['peso_actual_r'] > 0
antiguos_r = retro_pob[(retro_pob.flg_estado_yoy != 'no_existia_periodo_pasado') & es_activo_r]
nuevos_r = retro_pob[retro_pob.flg_estado_yoy == 'no_existia_periodo_pasado']

resumen_ton_retro = (
    antiguos_r.groupby(['cosecha_grupo', 'des_categoria', 'periodo'])['toneladas_atribuibles_80']
    .sum().rename('ton_atribuibles_antiguos_80')
    .to_frame()
    .join(nuevos_r.groupby(['cosecha_grupo', 'des_categoria', 'periodo'])['toneladas_actual']
          .sum().rename('ton_nuevos_volumen_puro'), how='outer')
    .fillna(0.0)
    .reset_index())
resumen_ton_retro['ton_atribuibles_total_80'] = (
    resumen_ton_retro['ton_atribuibles_antiguos_80'] + resumen_ton_retro['ton_nuevos_volumen_puro'])

print("\nToneladas atribuibles (IC80) RETROACTIVAS, por cosecha x categoria x mes:")
print(resumen_ton_retro.round(2).to_string(index=False))


# =============================================================================
# HELPERS DE EXPORTACION A EXCEL — estilos + escribir_hoja(), usados por la
# 'Base de calculo' y las 'Vistas Excel' mas abajo. Ya no arma un Excel propio
# aqui -- los unicos 2 exportables del notebook son esos dos.
# =============================================================================
from mediciones.excel_estilos import (Workbook, Font, PatternFill, Alignment, Border, Side, get_column_letter, CellIsRule, FONT_NAME, HEADER_FILL, HEADER_FONT, BODY_FONT, ALERT_FILL, OK_FILL, THIN, BORDER, UMBRAL_ESTILO_CELDA, escribir_hoja)   # compartido con Champion (antes copiado aquí, idéntico)


# =============================================================================
# VISTAS EXCEL (pedido: global / categoria / oficina x categoria /
# oficina x segmento x categoria, brecha + toneladas, mes a mes) — PREP.
# des_segmento_transaccional nunca se habia mergeado a df_panel ni a Base
# completa -> se trae de customer_feat (celda de caracteristicas del cliente).
# =============================================================================

oficina_map_cf = customer_feat.drop_duplicates('cod_cliente').set_index('cod_cliente')['des_oficina']
segmento_map = customer_feat.drop_duplicates('cod_cliente').set_index('cod_cliente')['des_segmento_transaccional']

# par_id -> cod_cliente del lado ACCION (el segmento del par = segmento de quien
# recibio el piloto, igual que ya se hace con des_oficina en la celda del PASO A)
par_accion_map = df_llave[df_llave.rol == 'accion'].drop_duplicates('par_id').set_index('par_id')['cod_cliente']

df_panel_seg = df_panel.copy()
df_panel_seg['cod_cliente_accion'] = df_panel_seg['par_id'].map(par_accion_map)
df_panel_seg['des_segmento_transaccional'] = df_panel_seg['cod_cliente_accion'].map(segmento_map)

sin_seg = df_panel_seg['des_segmento_transaccional'].isna().sum()
print(f"Pares sin segmento transaccional (sin match en customer_feat): {sin_seg:,} de {len(df_panel_seg):,}")

# Base completa (lado accion) + IC80 (Base completa solo trae las columnas 'sin
# sufijo', ancladas al 95% -- ver calcular_brechas_bootstrap) + segmento. La
# formula real de toneladas atribuibles IC80 (validada contra el slide) usa
# yoy_pct_raw capado a ic_pct_inf_80/sup_80, NO 'crecimiento_atribuible_pct'
# (esa columna de Base completa esta anclada al 95%, es otra cosa).
ic80_cols = df_brechas[['cosecha_grupo', 'des_categoria', 'mes_rel', 'ic_pct_inf_80', 'ic_pct_sup_80']]
base_vistas = df_base_completa[df_base_completa.rol.isin(['accion_matcheado', 'accion_sin_pareja'])].copy()
base_vistas = base_vistas.merge(ic80_cols, on=['cosecha_grupo', 'des_categoria', 'mes_rel'], how='left')
base_vistas['des_segmento_transaccional'] = base_vistas['cod_cliente'].map(segmento_map)

# [atribucion:inicio vistas] el estado y el YoY ya vienen de la base completa; solo se acota a la banda IC80
_atr_v = atribuir(REGLA_NUEVO, base_vistas['toneladas_ant'], base_vistas['toneladas_actual'], base_vistas['ic_pct_inf_80'], base_vistas['ic_pct_sup_80'],
                  estado=base_vistas['flg_estado_yoy'], yoy_pct_raw=base_vistas['yoy_pct_raw'])
base_vistas['crec_atrib_80'] = _atr_v['crec_atrib_pct']
base_vistas['ton_atrib_80'] = _atr_v['ton_atrib']
# [atribucion:fin vistas]

# retro_pob (toneladas retroactivas, celda de arriba): agregar oficina y
# segmento por cliente (el df solo traia cod_cliente, cosecha_grupo, periodo)
retro_pob['des_oficina'] = retro_pob['cod_cliente'].map(oficina_map_cf)
retro_pob['des_segmento_transaccional'] = retro_pob['cod_cliente'].map(segmento_map)

# meses calendario retroactivos reales por cosecha (dato-dependiente, no hardcodeado)
periodos_retro_por_grupo = (
    df_meses_gap.groupby('cosecha_grupo')['periodo'].apply(lambda s: sorted(s.unique())).to_dict())

print(f"\nGrupos con hueco retroactivo: "
      f"{ {k.date(): [p.date() for p in v] for k, v in periodos_retro_por_grupo.items()} }")


# =============================================================================
# VISTAS EXCEL — BRECHA: Global / Categoria / Oficina x Categoria /
# Oficina x Segmento x Categoria. Cada vista trae AMBAS columnas de tiempo
# (periodo calendario Y mes_rel) en la misma hoja (formato largo/tidy, no
# matriz) -- asi se puede pivotear en Excel de cualquiera de las dos formas.
# Incluye el estimado retroactivo (flg_estimado_retroactivo=True), replicando
# la referencia pooled de mes_rel 0-2 -- MISMO mecanismo ya usado arriba para
# la brecha retroactiva a nivel cosecha x categoria, ahora tambien disponible
# a nivel oficina x segmento x categoria (antes no existia a ese grano).
# =============================================================================

COLS_IC = (['n_pares', 'brecha_kg', 'brecha_pct', 'p_valor'] +
           [f'ic_pct_inf_{n}' for n in NIVELES_IC] + [f'ic_pct_sup_{n}' for n in NIVELES_IC] +
           [f'significativo_{n}' for n in NIVELES_IC])

def construir_vista_brecha(extra_cols, nombre_vista):
    group_cols_medido = ['cosecha_grupo'] + extra_cols + ['mes_rel']
    group_cols_ref = ['cosecha_grupo'] + extra_cols

    panel_v = df_panel_seg
    if 'des_segmento_transaccional' in extra_cols:
        panel_v = panel_v.dropna(subset=['des_segmento_transaccional'])

    medido = calcular_brechas_bootstrap(panel_v, group_cols_medido)
    if 'des_categoria' in extra_cols:
        medido['reportable'] = medido.apply(
            lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)
    else:
        medido['reportable'] = np.nan
    medido['flg_estimado_retroactivo'] = False
    medido = medido.drop(columns=['periodo'], errors='ignore')
    medido['periodo'] = medido.apply(
        lambda r: r.cosecha_grupo + pd.DateOffset(months=int(r.mes_rel)), axis=1)

    ref = calcular_brechas_bootstrap(panel_v[panel_v.mes_rel.between(0, 2)], group_cols_ref)
    ref = ref.drop(columns=['periodo', 'mes_rel'], errors='ignore')

    filas_retro = []
    for r in ref.itertuples(index=False):
        clave = {c: getattr(r, c) for c in group_cols_ref}
        for p in periodos_retro_por_grupo.get(clave['cosecha_grupo'], []):
            mrel = (p.year - clave['cosecha_grupo'].year) * 12 + (p.month - clave['cosecha_grupo'].month)
            fila = dict(clave)
            fila['periodo'] = p
            fila['mes_rel'] = mrel
            for col in COLS_IC:
                fila[col] = getattr(r, col, np.nan)
            fila['reportable'] = (reportable_map.get((clave['cosecha_grupo'], clave.get('des_categoria')), False)
                                   if 'des_categoria' in extra_cols else np.nan)
            fila['flg_estimado_retroactivo'] = True
            filas_retro.append(fila)
    retro_df = pd.DataFrame(filas_retro)

    out = pd.concat([medido, retro_df], ignore_index=True, sort=False)
    out['cosecha_label'] = out['cosecha_grupo'].map(GRUPO_COSECHA)
    out = out.sort_values(extra_cols + ['cosecha_grupo', 'mes_rel']).reset_index(drop=True)
    print(f"Vista brecha '{nombre_vista}': {len(out):,} filas ({int(out.flg_estimado_retroactivo.sum())} retroactivas)")
    return out

vista_brecha_global = construir_vista_brecha([], 'Global')
vista_brecha_categoria = construir_vista_brecha(['des_categoria'], 'Categoria')
vista_brecha_oficina = construir_vista_brecha(['des_oficina', 'des_categoria'], 'Oficina x Categoria')
vista_brecha_of_seg = construir_vista_brecha(
    ['des_oficina', 'des_segmento_transaccional', 'des_categoria'], 'Oficina x Segmento x Categoria')


# =============================================================================
# VISTAS EXCEL — TONELADAS ATRIBUIBLES (IC80): mismos 4 niveles que la brecha,
# mismo formato largo/tidy con periodo + mes_rel juntos. Antiguos (capados
# IC80) + nuevos (volumen puro) -- formula real validada contra el slide.
# =============================================================================

def construir_vista_toneladas(extra_cols, nombre_vista):
    group_cols = ['cosecha_grupo'] + extra_cols + ['periodo']

    bv = base_vistas
    rp = retro_pob
    if 'des_segmento_transaccional' in extra_cols:
        bv = bv.dropna(subset=['des_segmento_transaccional'])
        rp = rp.dropna(subset=['des_segmento_transaccional'])

    def _resumen(df_ant_col, df_nue_col, df_in, col_ant, col_nue):
        # BUGFIX: 'antiguos' tambien exige es_activo (toneladas_actual > 0) --
        # un cliente que existia pero compro CERO este mes queda FUERA del
        # atribuible, no se le asigna una caida capada al IC80 inferior.
        es_activo = df_in['toneladas_actual'] > 0
        ant = df_in[(df_in.flg_estado_yoy != 'no_existia_periodo_pasado') & es_activo]
        nue = df_in[df_in.flg_estado_yoy == 'no_existia_periodo_pasado']
        g_ant = ant.groupby(group_cols)[col_ant].sum().rename('ton_atrib_antiguos_80')
        g_nue = nue.groupby(group_cols)[col_nue].sum().rename('ton_nuevos_puro')
        out = pd.concat([g_ant, g_nue], axis=1).fillna(0.0).reset_index()
        out['ton_atrib_total_80'] = out['ton_atrib_antiguos_80'] + out['ton_nuevos_puro']
        out['mes_rel'] = out.apply(
            lambda r: (r.periodo.year - r.cosecha_grupo.year) * 12 + (r.periodo.month - r.cosecha_grupo.month), axis=1)
        return out

    medido = _resumen(None, None, bv, 'ton_atrib_80', 'toneladas_actual')
    medido['flg_estimado_retroactivo'] = False

    retro = _resumen(None, None, rp, 'toneladas_atribuibles_80', 'toneladas_actual')
    retro['flg_estimado_retroactivo'] = True

    out = pd.concat([medido, retro], ignore_index=True, sort=False)
    out['cosecha_label'] = out['cosecha_grupo'].map(GRUPO_COSECHA)
    out = out.sort_values(extra_cols + ['cosecha_grupo', 'mes_rel']).reset_index(drop=True)
    print(f"Vista toneladas '{nombre_vista}': {len(out):,} filas ({int(out.flg_estimado_retroactivo.sum())} retroactivas)")
    return out

vista_ton_global = construir_vista_toneladas([], 'Global')
vista_ton_categoria = construir_vista_toneladas(['des_categoria'], 'Categoria')
vista_ton_oficina = construir_vista_toneladas(['des_oficina', 'des_categoria'], 'Oficina x Categoria')
vista_ton_of_seg = construir_vista_toneladas(
    ['des_oficina', 'des_segmento_transaccional', 'des_categoria'], 'Oficina x Segmento x Categoria')


# =============================================================================
# PAPAS FRITAS / INKA CHIPS -- SIN grupo control, atribuible = DIFERENCIAL YoY
# (este mes vs el mismo mes del año pasado, pedido explicito del negocio: si
# compro 10 el año pasado y 30 este año, el atribuible es 20; si compro 0 el
# año pasado, el atribuible es 30 -- ambos casos son solo la resta, sin piso
# en 0: si el cliente compro MENOS que el año pasado, el diferencial sale
# negativo y se reporta tal cual, no se fuerza a 0).
#
# Poblacion y cosecha_grupo vienen de cr_surtido_ideal (celda anterior,
# 'inkachips_historico'), NO de cli_cosecha (Galletas/Cereales). Categoria SIN
# recodificar salvo el agrupamiento a 'Inkachips' pedido explicito (mas abajo).
# =============================================================================

inkachips_historico = inkachips_historico.copy()

# ── Año anterior: mismo cliente-categoria, periodo - 12 meses -> self-merge
# sobre el HISTORICO completo (por eso la query de la celda anterior ya no
# filtra por periodo_min_grupo: sin esas filas de "año pasado" no habria con
# que calcular el diferencial).
inkachips_historico['periodo_ant'] = inkachips_historico['periodo'] - pd.DateOffset(months=12)
peso_ant_lookup = inkachips_historico[['cod_cliente', 'des_categoria', 'periodo', 'peso']].rename(
    columns={'periodo': 'periodo_ant', 'peso': 'peso_ant'})
inkachips_historico = inkachips_historico.merge(
    peso_ant_lookup, on=['cod_cliente', 'des_categoria', 'periodo_ant'], how='left')
inkachips_historico['peso_ant'] = inkachips_historico['peso_ant'].fillna(0.0)

# ── Filtrar a las filas ATRIBUIBLES: periodo >= periodo_min_grupo (el resto
# del historico traido solo sirve de referencia para el año anterior, no debe
# quedar como fila propia -- mismo criterio que antes, ahora aplicado en pandas).
papas = inkachips_historico[
    inkachips_historico['periodo'] >= inkachips_historico['periodo_min_grupo']
].copy()

papas['toneladas_actual'] = papas['peso'] / 1000
papas['toneladas_ant'] = papas['peso_ant'] / 1000
papas['valor_atribuible'] = papas['toneladas_actual'] - papas['toneladas_ant']  # diferencial YoY, con signo

# Agrupar estas 4 categorias reales de venta como una sola 'Inkachips' (pedido
# explicito del negocio) -- el resto de des_categoria que pueda traer la marca
# se deja tal cual viene (sin recodificar).
CATEGORIAS_INKACHIPS = ['Golosinas y Snacks SSCC', 'Papas Fritas', 'Platano Frito', 'Vegetales']
papas['des_categoria'] = papas['des_categoria'].where(
    ~papas['des_categoria'].isin(CATEGORIAS_INKACHIPS), 'Inkachips')

# Diagnostico informativo (NO filtra poblacion): overlap con los clientes accion
# de Galletas/Cereales, para dejar trazado si son la misma base o no.
rol_gc_map = (df_base_completa[df_base_completa.rol.isin(['accion_matcheado', 'accion_sin_pareja'])]
              .drop_duplicates('cod_cliente').set_index('cod_cliente')['rol'])
papas['rol_gc'] = papas['cod_cliente'].map(rol_gc_map)  # NaN = no esta en accion Galletas/Cereales

n_total_inka = papas['cod_cliente'].nunique()
n_solo_inka = papas.loc[papas['rol_gc'].isna(), 'cod_cliente'].nunique()
print(f"Inka Chips -- clientes con compra atribuible: {n_total_inka:,} | "
      f"de ellos, {n_solo_inka:,} NO estan en la poblacion accion de Galletas/Cereales "
      f"(poblacion propia via cr_surtido_ideal, no se filtran por eso)")

vista_papas = (papas.groupby(['cosecha_grupo', 'periodo'])['valor_atribuible'].sum()).rename(
    'toneladas_papas_fritas').reset_index()
vista_papas['cosecha_label'] = vista_papas['cosecha_grupo'].map(GRUPO_COSECHA)
vista_papas = vista_papas.sort_values(['cosecha_grupo', 'periodo']).reset_index(drop=True)

print(f"\nToneladas atribuibles Inka Chips (diferencial YoY, sin capado ni IC): "
      f"{vista_papas.toneladas_papas_fritas.sum():,.2f} t")
print(vista_papas.to_string(index=False))

# ── Diagnostico: desglose por des_categoria real (ya con Inkachips agrupado) ──
diag_cat = (papas.groupby(['cosecha_grupo', 'periodo', 'des_categoria'])['valor_atribuible'].sum()).rename('t').reset_index()
diag_cat['cosecha_label'] = diag_cat['cosecha_grupo'].map(GRUPO_COSECHA)
print("\nDesglose por des_categoria (diferencial YoY):")
print(diag_cat.pivot_table(index=['cosecha_label', 'periodo'], columns='des_categoria', values='t').round(2).to_string())

# ── Se coloca como TOTAL: se suma al TOTAL de toneladas de la vista Global ──
# (no reemplaza a Cereales/Galletas, se agrega como componente adicional,
# igual que "nuevos" se agrega a "antiguos" en las demas vistas)
vista_ton_global_con_papas = vista_ton_global.merge(
    vista_papas[['cosecha_grupo', 'periodo', 'toneladas_papas_fritas']],
    on=['cosecha_grupo', 'periodo'], how='left')
vista_ton_global_con_papas['toneladas_papas_fritas'] = vista_ton_global_con_papas['toneladas_papas_fritas'].fillna(0.0)
vista_ton_global_con_papas['ton_atrib_total_80'] = (
    vista_ton_global_con_papas['ton_atrib_total_80'] + vista_ton_global_con_papas['toneladas_papas_fritas'])

print(f"\nTon Global CON Inka Chips sumado al TOTAL: "
      f"{vista_ton_global_con_papas['toneladas_papas_fritas'].sum():,.2f} t agregadas")


# =============================================================================
# BASE DE CALCULO v3 — el atribuible ya sale directo en la base, para que una
# tabla dinamica simple (Suma / Promedio) sea el numero final, sin formulas
# aparte ni riesgo de repetir el bug de toneladas_actual==0.
# =============================================================================

# ── Base TONELADAS ───────────────────────────────────────────────────────────
nom_compania_map = customer_feat.drop_duplicates('cod_cliente').set_index('cod_cliente')['nom_compania']  # distribuidor (con el que se arma dex_oficina)
cols_comunes_ton = ['cod_cliente', 'cod_cliente_dex', 'des_categoria', 'des_oficina',
                     'des_segmento_transaccional', 'nom_compania', 'flg_cliente_perfecto',
                     'cosecha_grupo', 'periodo', 'mes_rel', 'toneladas_ant', 'toneladas_actual',
                     'yoy_pct_raw', 'ic_pct_inf_80', 'ic_pct_sup_80', 'flg_estado_yoy',
                     'crec_atrib_80_pct', 'ton_atrib_80', 'flg_estimado_retroactivo']

base_ton_medido = base_vistas.rename(columns={'crec_atrib_80': 'crec_atrib_80_pct'}).copy()
base_ton_medido['flg_estimado_retroactivo'] = False
base_ton_medido['cod_cliente_dex'] = base_ton_medido['cod_cliente']
base_ton_medido['nom_compania'] = base_ton_medido['cod_cliente'].map(nom_compania_map)

base_ton_retro = retro_pob.rename(columns={
    'crecimiento_atribuible_pct': 'crec_atrib_80_pct',
    'toneladas_atribuibles_80': 'ton_atrib_80'}).copy()
base_ton_retro['mes_rel'] = base_ton_retro.apply(
    lambda r: (r.periodo.year - r.cosecha_grupo.year) * 12 + (r.periodo.month - r.cosecha_grupo.month), axis=1)
base_ton_retro['cod_cliente_dex'] = base_ton_retro['cod_cliente']
base_ton_retro['nom_compania'] = base_ton_retro['cod_cliente'].map(nom_compania_map)
if 'flg_cliente_perfecto' not in base_ton_retro.columns:
    base_ton_retro = base_ton_retro.merge(
        df_cliente_perfecto[['cod_cliente', 'periodo', 'flg_cliente_perfecto']],
        on=['cod_cliente', 'periodo'], how='left')

base_calculo_toneladas = pd.concat([
    base_ton_medido[cols_comunes_ton], base_ton_retro[cols_comunes_ton]
], ignore_index=True).sort_values(['cod_cliente', 'des_categoria', 'periodo']).reset_index(drop=True)

# valor_atribuible: YA resuelve antiguos+nuevos fila por fila -> Suma directa =
# el numero final, sin IF ni distincion manual. RECUERDA: toneladas_actual==0
# excluye al cliente del atribuible (no es lo mismo caer a cero que crecer
# menos de lo esperado -- ver bug corregido antes) -> se deja en 0, no NaN,
# para que sumarlo no rompa nada.
# [atribucion:inicio valor]
base_calculo_toneladas['valor_atribuible'] = valor_atribuible(REGLA_NUEVO, base_calculo_toneladas['flg_estado_yoy'], base_calculo_toneladas['toneladas_actual'],
                                                              base_calculo_toneladas['ton_atrib_80'])
# [atribucion:fin valor]

# ── Papas Fritas / Inka Chips: mismas columnas, 100% atribuible ────────────
# (sin control, sin capado IC -> se agrega como filas mas de la MISMA tabla,
# asi un Suma de 'valor_atribuible' agrupado por cosecha_grupo x periodo ya
# la incluye automaticamente en el total, sin sheet aparte que acordarse de sumar)
# BUGFIX: filtrar solo por 'periodo' (sin cosecha_grupo) tomaba la UNION de
# periodos de los 3 grupos -> un cliente de G02/G03 heredaba Feb/Mar (que le
# corresponden solo a G01) porque ese valor de periodo existia en ALGUN grupo,
# no en el suyo. Se filtra por la combinacion (cosecha_grupo, periodo) real ->
# cada grupo arranca desde su propio mes -2/-1 (fecha de despliegue), igual
# que Galletas/Cereales.
periodos_validos_por_grupo = vista_ton_global_con_papas[['cosecha_grupo', 'periodo']].drop_duplicates()
papas_base = papas.merge(periodos_validos_por_grupo, on=['cosecha_grupo', 'periodo'], how='inner').copy()
papas_base['cod_cliente_dex'] = papas_base['cod_cliente']
# des_categoria NO se hardcodea: 'papas' trae la categoria real ('Inkachips' agrupada, ver celda anterior)
papas_base['des_oficina'] = papas_base['cod_cliente'].map(oficina_map_cf)
papas_base['des_segmento_transaccional'] = papas_base['cod_cliente'].map(segmento_map)
papas_base['nom_compania'] = papas_base['cod_cliente'].map(nom_compania_map)
papas_base['flg_cliente_perfecto'] = papas_base.merge(
    df_cliente_perfecto[['cod_cliente', 'periodo', 'flg_cliente_perfecto']],
    on=['cod_cliente', 'periodo'], how='left')['flg_cliente_perfecto']
papas_base['mes_rel'] = papas_base.apply(
    lambda r: (r.periodo.year - r.cosecha_grupo.year) * 12 + (r.periodo.month - r.cosecha_grupo.month), axis=1)
# toneladas_ant, toneladas_actual y valor_atribuible YA vienen de 'papas' (diferencial YoY con signo):
# no se sobrescriben aqui.
papas_base['yoy_pct_raw'] = np.nan
papas_base['ic_pct_inf_80'] = np.nan
papas_base['ic_pct_sup_80'] = np.nan
papas_base['flg_estado_yoy'] = 'cien_pct_atribuible_sin_control'
papas_base['crec_atrib_80_pct'] = np.where(
    papas_base['toneladas_ant'] > 0, 100 * papas_base['valor_atribuible'] / papas_base['toneladas_ant'], np.nan)
papas_base['ton_atrib_80'] = np.nan  # no aplica capado -- ver valor_atribuible
papas_base['flg_estimado_retroactivo'] = False

cols_papas = cols_comunes_ton + ['valor_atribuible']
base_calculo_toneladas = pd.concat([
    base_calculo_toneladas, papas_base[cols_papas]
], ignore_index=True).sort_values(['cod_cliente', 'des_categoria', 'periodo']).reset_index(drop=True)

print(f"Papas Fritas/Inka Chips agregado a base_calculo_toneladas: {len(papas_base):,} filas, "
      f"{papas_base.valor_atribuible.sum():,.2f} t")

print(f"base_calculo_toneladas: {len(base_calculo_toneladas):,} filas "
      f"({base_calculo_toneladas.cod_cliente.nunique():,} clientes) -> Suma de 'valor_atribuible' "
      f"agrupado como sea (grupo, categoria, oficina, segmento, mes) = el numero final.")

# ── Base BRECHA (grano par x mes_rel), CON retroactivo incluido ─────────────
cols_brecha = ['par_id', 'cod_cliente_accion_dex', 'nom_compania_accion', 'flg_cliente_perfecto', 'cosecha_grupo',
               'des_categoria', 'des_oficina', 'des_segmento_transaccional',
               'periodo', 'mes_rel', 'pre_accion', 'post_accion', 'pre_control', 'post_control',
               'did_kg', 'flg_estimado_retroactivo']

base_brecha_medido = df_panel_seg.copy()
base_brecha_medido['cod_cliente_accion_dex'] = base_brecha_medido['cod_cliente_accion']
base_brecha_medido['nom_compania_accion'] = base_brecha_medido['cod_cliente_accion'].map(nom_compania_map)
base_brecha_medido['flg_estimado_retroactivo'] = False
base_brecha_medido = base_brecha_medido.merge(
    df_cliente_perfecto[['cod_cliente', 'periodo', 'flg_cliente_perfecto']]
    .rename(columns={'cod_cliente': 'cod_cliente_accion_dex'}),
    on=['cod_cliente_accion_dex', 'periodo'], how='left')

ref_pool = (df_panel_seg[df_panel_seg.mes_rel.between(0, 2)]
            .groupby(['cosecha_grupo', 'des_categoria'])
            .agg(pre_accion_ref=('pre_accion', 'mean'), did_kg_ref=('did_kg', 'mean'))
            .reset_index())

filas_brecha_retro = []
for r in ref_pool.itertuples(index=False):
    pares_grupo = df_panel_seg[(df_panel_seg.cosecha_grupo == r.cosecha_grupo) &
                                (df_panel_seg.des_categoria == r.des_categoria)][
        ['par_id', 'cod_cliente_accion', 'des_oficina', 'des_segmento_transaccional']
    ].drop_duplicates('par_id')
    for p in periodos_retro_por_grupo.get(r.cosecha_grupo, []):
        mrel = (p.year - r.cosecha_grupo.year) * 12 + (p.month - r.cosecha_grupo.month)
        tmp = pares_grupo.copy()
        tmp['cosecha_grupo'] = r.cosecha_grupo
        tmp['des_categoria'] = r.des_categoria
        tmp['periodo'] = p
        tmp['mes_rel'] = mrel
        tmp['pre_accion'] = r.pre_accion_ref
        tmp['post_accion'] = np.nan
        tmp['pre_control'] = np.nan
        tmp['post_control'] = np.nan
        tmp['did_kg'] = r.did_kg_ref
        tmp['cod_cliente_accion_dex'] = tmp['cod_cliente_accion']
        tmp['nom_compania_accion'] = tmp['cod_cliente_accion'].map(nom_compania_map)
        tmp['flg_estimado_retroactivo'] = True
        filas_brecha_retro.append(tmp)

base_brecha_retro = (pd.concat(filas_brecha_retro, ignore_index=True) if filas_brecha_retro
                      else pd.DataFrame(columns=cols_brecha))
if len(base_brecha_retro):
    base_brecha_retro = base_brecha_retro.merge(
        df_cliente_perfecto[['cod_cliente', 'periodo', 'flg_cliente_perfecto']]
        .rename(columns={'cod_cliente': 'cod_cliente_accion_dex'}),
        on=['cod_cliente_accion_dex', 'periodo'], how='left')

cols_brecha_disp = [c for c in cols_brecha if c in base_brecha_medido.columns and
                     c in (base_brecha_retro.columns if len(base_brecha_retro) else base_brecha_medido.columns)]

base_calculo_brecha = pd.concat([
    base_brecha_medido[cols_brecha_disp], base_brecha_retro[cols_brecha_disp]
], ignore_index=True)
base_calculo_brecha['cosecha_label'] = base_calculo_brecha['cosecha_grupo'].map(GRUPO_COSECHA)

# brecha_pct YA calculada a nivel de celda (cosecha_grupo x categoria x mes_rel),
# repetida en cada fila de esa celda -> un simple PROMEDIO en tabla dinamica
# reproduce el TOTAL ponderado por n_pares solo, porque promediar una columna
# con el mismo valor repetido n_pares veces por grupo ES ponderar por n_pares.
# Medido: se trae DIRECTO de df_brechas (ya validado), no se recalcula aparte,
# para blindar contra cualquier divergencia de metodologia.
brecha_celda = df_brechas[['cosecha_grupo', 'des_categoria', 'mes_rel', 'brecha_pct', 'reportable']].rename(
    columns={'brecha_pct': 'brecha_pct_celda'})
base_calculo_brecha = base_calculo_brecha.merge(
    brecha_celda, on=['cosecha_grupo', 'des_categoria', 'mes_rel'], how='left')

# retroactivo: el valor ya es uniforme por fila (did_kg/pre_accion son la misma
# referencia repetida) -> se calcula directo, no hace falta merge
mask_retro = base_calculo_brecha['flg_estimado_retroactivo']
base_calculo_brecha.loc[mask_retro, 'brecha_pct_celda'] = (
    100 * base_calculo_brecha.loc[mask_retro, 'did_kg'] / base_calculo_brecha.loc[mask_retro, 'pre_accion'])
base_calculo_brecha.loc[mask_retro, 'reportable'] = base_calculo_brecha.loc[mask_retro].apply(
    lambda r: reportable_map.get((r.cosecha_grupo, r.des_categoria), False), axis=1)

base_calculo_brecha = base_calculo_brecha.sort_values(
    ['des_categoria', 'cosecha_grupo', 'periodo', 'par_id']).reset_index(drop=True)

print(f"\nbase_calculo_brecha: {len(base_calculo_brecha):,} filas "
      f"({base_calculo_brecha.par_id.nunique():,} pares, "
      f"{int(base_calculo_brecha.flg_estimado_retroactivo.sum()):,} filas retroactivas) -> "
      f"Promedio de 'brecha_pct_celda' agrupado por lo que sea = el TOTAL ponderado por n_pares, "
      f"sin SUMPRODUCT.")
