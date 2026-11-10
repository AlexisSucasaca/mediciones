"""Estilos y escritura de hojas Excel de las salidas (surtido y Champion). Antes estaba copiado, idéntico, en los dos scripts."""
import pandas as pd

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import CellIsRule

FONT_NAME   = 'Arial'
HEADER_FILL = PatternFill('solid', start_color='1F3864', end_color='1F3864')
HEADER_FONT = Font(name=FONT_NAME, bold=True, color='FFFFFF', size=10)
BODY_FONT   = Font(name=FONT_NAME, size=10)
ALERT_FILL  = PatternFill('solid', start_color='FFC7CE', end_color='FFC7CE')
OK_FILL     = PatternFill('solid', start_color='C6EFCE', end_color='C6EFCE')
THIN = Side(style='thin', color='D9D9D9')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


UMBRAL_ESTILO_CELDA = 20_000  # por encima de esto, se salta fuente/borde por celda
# (ese loop celda-a-celda, con cientos de miles de filas x columnas, fue justo lo que
# colgo el kernel en la corrida anterior -> KeyboardInterrupt en 'Base completa',
# 322k filas x 21 columnas = 6.7M celdas de fuente+borde individuales)

def escribir_hoja(wb, nombre, df, freeze='A2', pct_cols=None, alert_col=None, alert_val=False):
    ws = wb.create_sheet(nombre)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime('%Y-%m-%d')

    ws.append(list(df.columns))
    # itertuples (sin index, sin nombre) es mucho mas rapido que iterrows() para
    # hojas grandes -> iterrows reconstruye una Series con label por cada fila,
    # itertuples entrega tuplas simples.
    for row in df.itertuples(index=False, name=None):
        ws.append(row)

    for j, col in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=j)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

    n_filas = len(df)
    if n_filas <= UMBRAL_ESTILO_CELDA:
        for i in range(2, n_filas + 2):
            for j in range(1, len(df.columns) + 1):
                cell = ws.cell(row=i, column=j)
                cell.font = BODY_FONT
                cell.border = BORDER
    else:
        # Hoja grande: aplicar fuente/borde celda por celda es O(filas x columnas), y
        # openpyxl vuelve a resolver/hashear el estilo compartido en CADA asignacion ->
        # con cientos de miles de filas eso fue lo que colgo el kernel la vez anterior.
        # Se prioriza terminar la exportacion: se pierde el borde/fuente por celda, se
        # conserva encabezado, ancho de columna, freeze panes y el resaltado condicional.
        print(f"  [PERFORMANCE] '{nombre}': {n_filas:,} filas > {UMBRAL_ESTILO_CELDA:,} "
              f"-> se omite fuente/borde por celda para no colgar la exportacion "
              f"(encabezado, anchos, freeze y resaltado condicional SI se aplican)")

    if pct_cols:
        for c in pct_cols:
            if c in df.columns:
                j = df.columns.get_loc(c) + 1
                for i in range(2, n_filas + 2):
                    ws.cell(row=i, column=j).number_format = '0.0"%"'

    if alert_col and alert_col in df.columns:
        j = get_column_letter(df.columns.get_loc(alert_col) + 1)
        rng = f"{j}2:{j}{n_filas + 1}"
        ws.conditional_formatting.add(
            rng, CellIsRule(operator='equal', formula=[str(alert_val)], fill=ALERT_FILL))
        ws.conditional_formatting.add(
            rng, CellIsRule(operator='equal', formula=[str(not alert_val)], fill=OK_FILL))

    for j, col in enumerate(df.columns, start=1):
        # Ancho de columna: vectorizado con pandas (.astype(str).str.len()) en vez del
        # list comprehension con str(v) fila por fila -> tambien pesaba en hojas
        # grandes. Sobre una MUESTRA cuando la hoja es grande (no hace falta recorrer
        # 322k valores para estimar un ancho de columna razonable).
        serie = df[col]
        if n_filas > UMBRAL_ESTILO_CELDA:
            serie = serie.sample(min(5000, n_filas), random_state=0)
        max_len_valores = serie.astype(str).str.len().max() if n_filas else 0
        max_len = max(len(str(col)), int(max_len_valores) if pd.notna(max_len_valores) else 0)
        ws.column_dimensions[get_column_letter(j)].width = min(max(max_len + 2, 10), 32)

    ws.freeze_panes = freeze
    return ws
