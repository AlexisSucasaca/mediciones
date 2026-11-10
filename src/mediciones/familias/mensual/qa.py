# flake8: noqa
# ruff: noqa
# Script de FAMILIA: se ejecuta con exec() en un espacio de nombres compartido con el script del programa
# (mediciones.ejecucion.ejecutar_familia). No es un módulo importable. MANTENIDO A MANO: este archivo es la fuente de verdad (los notebooks originales, fuera de este paquete, quedan solo como referencia histórica).
# =============================================================================
# QA DE LA CORRIDA (chequeos de identidad sobre la base completa)
# =============================================================================
# ── Chequeos ──
chk = []
def _chk(nombre, dif, tol=1e-6):
    chk.append({"chequeo": nombre, "max_dif": float(dif), "estado": "OK" if abs(dif) <= tol else "REVISAR"})

_reuso = df_matches.groupby(["periodo", "des_categoria", "cod_control"]).size()
if MAX_REUSO_CONTROL_DEFAULT == 1 and not MAX_REUSO_CONTROL_POR_CELDA:
    _chk("1:1 estricto: ningún control repetido dentro de una celda", (_reuso > 1).sum())
_chk("Pares del panel = pares del matching", len(df_panel) - len(df_matches))
_chk("TN de nuevos en base completa = TN de nuevos del embudo",
     df_base_completa.loc[df_base_completa["rol"] == "accion_nuevo", "toneladas_actual"].sum() - df_nuevos["toneladas_actual"].sum())
_ac_sum = df_base_completa[df_base_completa["rol"].str.startswith("accion")].groupby(["periodo", "des_categoria"])["toneladas_actual"].sum()
# solo los meses CALCULADOS en esta corrida (con RECALCULAR_DESDE, un cierre normal calcula solo el mes nuevo)
_ref = {(t_, c_): SERIES[c_][t_][(ELEG[t_] & FLAG[t_]).to_numpy()].sum() / 1000 for t_ in globals().get('meses_calc', meses_eval) for c_ in CATEGORIAS}
_chk("TN actuales de acción en base completa = serie de ventas (mes × categoría)",
     max(abs(_ac_sum.get(k, 0.0) - v) for k, v in _ref.items()) if _ref else 0.0)
df_chequeos = pd.DataFrame(chk)
print("\nCHEQUEOS:")
display(df_chequeos)
assert (df_chequeos["estado"] == "OK").all(), "Hay chequeos en REVISAR: ver df_chequeos."
