# COPIA LITERAL del código anterior (cosechas, hueco retroactivo), líneas 2310-2350 de src/mediciones/familias/cosechas/metodologia.py antes de extraerlo a metodologia/atribucion.py.
# No se edita: es la referencia de las pruebas diferenciales.
cond_no_existia_r = (
    (retro_pob['peso_ant_r'] == 0) & (retro_pob['peso_actual_r'] > 0) &
    retro_pob['fec_creacion'].notna() & (retro_pob['fec_creacion'] > retro_pob['periodo_ant']))
cond_reactivado_r = (retro_pob['peso_ant_r'] == 0) & (retro_pob['peso_actual_r'] > 0) & (~cond_no_existia_r)
cond_sigue_cero_r = (retro_pob['peso_ant_r'] == 0) & (retro_pob['peso_actual_r'] == 0)

retro_pob['flg_estado_yoy'] = np.select(
    [cond_no_existia_r, cond_reactivado_r, cond_sigue_cero_r],
    ['no_existia_periodo_pasado', 'reactivado', 'sin_compra_ambos'], default='normal')

print("\nDistribucion flg_estado_yoy (hueco retroactivo):")
print(retro_pob['flg_estado_yoy'].value_counts(normalize=True).mul(100).round(1).to_string())

retro_pob['yoy_pct_raw'] = np.where(
    retro_pob['peso_ant_r'] > 0, 100 * retro_pob['toneladas_variacion'] / retro_pob['toneladas_ant'],
    np.where(cond_reactivado_r, np.inf, np.where(cond_no_existia_r, np.nan, 0.0)))

# banda IC80 de referencia: pool mes_rel 0-2 por cosecha x categoria (misma
# ventana ya acordada para el retroactivo de brecha, arriba en este notebook)
banda80 = (df_brechas[df_brechas.mes_rel.between(0, 2)]
           .groupby(['cosecha_grupo', 'des_categoria'])[['ic_pct_inf_80', 'ic_pct_sup_80']]
           .mean().reset_index())
retro_pob = retro_pob.merge(banda80, on=['cosecha_grupo', 'des_categoria'], how='left')

yoy_para_clip_r = retro_pob['yoy_pct_raw'].replace(np.inf, np.nan)
retro_pob['crecimiento_atribuible_pct'] = yoy_para_clip_r.clip(
    lower=retro_pob['ic_pct_inf_80'], upper=retro_pob['ic_pct_sup_80'])

# reactivados (SI atribuible, con tope en ic_pct_sup_80) -- igual que la celda original
retro_pob.loc[cond_reactivado_r & retro_pob['ic_pct_sup_80'].notna(), 'crecimiento_atribuible_pct'] = \
    retro_pob.loc[cond_reactivado_r & retro_pob['ic_pct_sup_80'].notna(), 'ic_pct_sup_80'].clip(lower=0)
retro_pob.loc[cond_reactivado_r & retro_pob['ic_pct_sup_80'].isna(), 'crecimiento_atribuible_pct'] = np.nan

# no_existia (NUNCA atribuible, por definicion del negocio)
retro_pob.loc[cond_no_existia_r, 'crecimiento_atribuible_pct'] = np.nan

retro_pob['toneladas_atribuibles_80'] = np.where(
    retro_pob['toneladas_ant'] > 0,
    retro_pob['toneladas_ant'] * retro_pob['crecimiento_atribuible_pct'] / 100, 0.0)
retro_pob.loc[retro_pob['crecimiento_atribuible_pct'].isna(), 'toneladas_atribuibles_80'] = np.nan
retro_pob.loc[cond_no_existia_r, 'toneladas_atribuibles_80'] = np.nan
