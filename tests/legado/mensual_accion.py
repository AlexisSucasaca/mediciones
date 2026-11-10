# COPIA LITERAL del código anterior (mensual, acción), líneas 655-673 de src/mediciones/familias/mensual/metodologia.py antes de extraerlo a metodologia/atribucion.py.
# No se edita: es la referencia de las pruebas diferenciales.
accion["flg_estado_yoy"] = np.where(accion["toneladas_ant"] == 0, "nuevo", "antiguo")
accion["toneladas_variacion"] = accion["toneladas_actual"] - accion["toneladas_ant"]
accion["yoy_pct_raw"] = np.where(accion["toneladas_ant"] > 0, 100 * accion["toneladas_variacion"] / accion["toneladas_ant"], np.nan)

# ── 2) Banda IC80 de la celda (mes × categoría) y atribuible ──
accion = accion.merge(df_brechas[["periodo", "des_categoria", "ic_pct_inf_80", "ic_pct_sup_80"]],
                      on=["periodo", "des_categoria"], how="left")
sin_banda = accion["ic_pct_inf_80"].isna() | accion["ic_pct_sup_80"].isna()
es_ant = accion["flg_estado_yoy"] == "antiguo"
es_nue = ~es_ant
activo = accion["toneladas_actual"] > 0

accion["crec_atrib_80_pct"] = accion["yoy_pct_raw"].clip(lower=accion["ic_pct_inf_80"], upper=accion["ic_pct_sup_80"])
accion.loc[sin_banda | es_nue, "crec_atrib_80_pct"] = np.nan
accion["ton_atrib_80"] = accion["toneladas_ant"] * accion["crec_atrib_80_pct"] / 100
accion["flg_sin_banda_ic"] = es_ant & activo & sin_banda
accion["valor_atribuible"] = np.select(
    [es_nue, es_ant & ~activo, es_ant & activo],
    [accion["toneladas_actual"], 0.0, accion["ton_atrib_80"]], default=np.nan)
