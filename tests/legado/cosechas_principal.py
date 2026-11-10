# COPIA LITERAL del código anterior (cosechas, panel principal), líneas 1908-1969 de src/mediciones/familias/cosechas/metodologia.py antes de extraerlo a metodologia/atribucion.py.
# No se edita: es la referencia de las pruebas diferenciales. Se omiten 11 líneas de banderas (ver abajo).
cond_no_existia = (
    (panel['peso_ant'] == 0) & (panel['peso'] > 0) &
    panel['fec_creacion'].notna() &
    (panel['fec_creacion'] > panel['periodo_ant'])
)

# "reactivado" real = existía antes del periodo comparado, pero compró 0 ese mes
cond_reactivado = (
    (panel['peso_ant'] == 0) & (panel['peso'] > 0) &
    (~cond_no_existia)
)

cond_sigue_cero = (panel['peso_ant'] == 0) & (panel['peso'] == 0)

print(f"\nClientes-periodo 'no existía periodo pasado' (fec_creacion posterior): {cond_no_existia.sum():,}")
print(f"Clientes-periodo 'reactivado' (existía, compró 0 antes)              : {cond_reactivado.sum():,}")
print(f"Clientes-periodo sin compra en ambos años                            : {cond_sigue_cero.sum():,}")

panel['flg_estado_yoy'] = np.select(
    [cond_no_existia, cond_reactivado, cond_sigue_cero],
    ['no_existia_periodo_pasado', 'reactivado', 'sin_compra_ambos'],
    default='normal')

# ── yoy_pct_raw: igual que antes, pero 'no_existia' queda sin % calculable ──
panel['yoy_pct_raw'] = np.where(
    panel['peso_ant'] > 0,
    100 * panel['toneladas_variacion'] / panel['toneladas_ant'],
    np.where(cond_reactivado, np.inf,
    np.where(cond_no_existia, np.nan, 0.0)))   # no_existia -> NaN explícito, no inf

# ── crecimiento_atribuible_pct: 'no_existia' NUNCA tiene atribuible ────────
yoy_para_clip = panel['yoy_pct_raw'].replace(np.inf, np.nan)

# [banderas flg_supero_umbral / flg_bajo_esperado / flg_sobre_esperado OMITIDAS de esta copia: no son parte de la atribución y asignan NaN a columnas booleanas, lo que pandas>=3 rechaza]

panel['crecimiento_atribuible_pct'] = yoy_para_clip.clip(
    lower=panel['ic_pct_inf'], upper=panel['ic_pct_sup'])

# reactivados (SÍ atribuible, con tope en ic_pct_sup)
panel.loc[cond_reactivado & panel['ic_pct_sup'].notna(), 'crecimiento_atribuible_pct'] = \
    panel.loc[cond_reactivado & panel['ic_pct_sup'].notna(), 'ic_pct_sup'].clip(lower=0)
panel.loc[cond_reactivado & panel['ic_pct_sup'].isna(), 'crecimiento_atribuible_pct'] = np.nan

# no_existia (NUNCA atribuible, por definición del negocio)
panel.loc[cond_no_existia, 'crecimiento_atribuible_pct'] = np.nan

panel['toneladas_atribuibles'] = np.where(
    panel['toneladas_ant'] > 0,
    panel['toneladas_ant'] * panel['crecimiento_atribuible_pct'] / 100,
    0.0)
panel.loc[panel['crecimiento_atribuible_pct'].isna(), 'toneladas_atribuibles'] = np.nan
panel.loc[cond_no_existia, 'toneladas_atribuibles'] = np.nan   # explícito, aunque ya quede así arriba
