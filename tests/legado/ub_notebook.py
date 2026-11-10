# COPIA LITERAL de la UB del notebook de CPB (idéntica en CPM): líneas 49-60 de la celda 23 de legacy/CPB_Brechas_ene_sep2026.ipynb.
# Entradas: `accion` con periodo, des_categoria, flg_estado_yoy, toneladas_ant, toneladas_actual, valor_atribuible y ub_base_soles (ya con fillna(0)).
# No se edita: es la referencia de las pruebas diferenciales de ub.py.
_ref = (accion[(accion["flg_estado_yoy"] == "antiguo") & (accion["toneladas_ant"] > 0)].groupby(["periodo", "des_categoria"])
        .agg(_u=("ub_base_soles", "sum"), _t=("toneladas_ant", "sum")))
_ref["ub_por_ton_ref"] = _ref["_u"] / _ref["_t"]
accion = accion.merge(_ref[["ub_por_ton_ref"]].reset_index(), on=["periodo", "des_categoria"], how="left")
_ant_m = accion["flg_estado_yoy"] == "antiguo"
_nue_m = accion["flg_estado_yoy"] == "nuevo"
_act_m = accion["toneladas_actual"] > 0
_ub_por_ton = np.where(accion["toneladas_ant"] > 0, accion["ub_base_soles"] / accion["toneladas_ant"], np.nan)   # S/ por TN base del cliente
accion["ub_atrib_80_soles"] = np.select(
    [_ant_m & _act_m, _ant_m, _nue_m],
    [accion["valor_atribuible"] * _ub_por_ton, 0.0, accion["toneladas_actual"] * accion["ub_por_ton_ref"].fillna(0.0)],
    default=np.nan)
