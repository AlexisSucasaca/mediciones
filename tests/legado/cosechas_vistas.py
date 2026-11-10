# COPIA LITERAL del código anterior (cosechas, vistas), líneas 2502-2511 de src/mediciones/familias/cosechas/metodologia.py antes de extraerlo a metodologia/atribucion.py.
# No se edita: es la referencia de las pruebas diferenciales.
base_vistas['crec_atrib_80'] = base_vistas['yoy_pct_raw'].clip(
    lower=base_vistas['ic_pct_inf_80'], upper=base_vistas['ic_pct_sup_80'])
base_vistas.loc[base_vistas.flg_estado_yoy == 'reactivado', 'crec_atrib_80'] = \
    base_vistas.loc[base_vistas.flg_estado_yoy == 'reactivado', 'ic_pct_sup_80'].clip(lower=0)
base_vistas.loc[base_vistas.flg_estado_yoy == 'no_existia_periodo_pasado', 'crec_atrib_80'] = np.nan

base_vistas['ton_atrib_80'] = np.where(
    base_vistas['toneladas_ant'] > 0, base_vistas['toneladas_ant'] * base_vistas['crec_atrib_80'] / 100, 0.0)
base_vistas.loc[base_vistas['crec_atrib_80'].isna(), 'ton_atrib_80'] = np.nan
base_vistas.loc[base_vistas.flg_estado_yoy == 'no_existia_periodo_pasado', 'ton_atrib_80'] = np.nan
