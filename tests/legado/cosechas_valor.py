# COPIA LITERAL del código anterior (cosechas, valor_atribuible final), líneas 2766-2771 de src/mediciones/familias/cosechas/metodologia.py antes de extraerlo a metodologia/atribucion.py.
# No se edita: es la referencia de las pruebas diferenciales.
es_nuevo = base_calculo_toneladas['flg_estado_yoy'] == 'no_existia_periodo_pasado'
es_activo = base_calculo_toneladas['toneladas_actual'] > 0
base_calculo_toneladas['valor_atribuible'] = np.select(
    [es_nuevo, es_activo],
    [base_calculo_toneladas['toneladas_actual'], base_calculo_toneladas['ton_atrib_80']],
    default=0.0)
