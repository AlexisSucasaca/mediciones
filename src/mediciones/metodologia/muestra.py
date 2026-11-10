"""Tamaño de muestra con corrección por población finita (estándar acordado: 5% de margen de error, 95% de confianza).
Compartido por la familia de cosechas (surtido) y Champion: antes estaba copiado, idéntico, en los dos scripts."""
import numpy as np
import pandas as pd

from mediciones.parametros import MARGEN_ERROR_OBJETIVO

CONFIANZA_OBJETIVO = 95
Z_CONF = {80: 1.2816, 90: 1.6449, 95: 1.9600}


def n_minimo_muestra(N, margen_error=MARGEN_ERROR_OBJETIVO, confianza=CONFIANZA_OBJETIVO, p=0.5):
    """n mínimo para estimar una proporción con el margen de error y confianza dados,
    de una población de tamaño N (corrección por población finita incluida)."""
    if N is None or N <= 1 or pd.isna(N):
        return np.inf
    z = Z_CONF[confianza]
    n0 = (z ** 2 * p * (1 - p)) / (margen_error ** 2)
    n = n0 / (1 + (n0 - 1) / N)
    return int(np.ceil(n))


def margen_error_real(n, N, confianza=CONFIANZA_OBJETIVO, p=0.5):
    """Margen de error REAL logrado con una muestra n de una población N
    (inversa de n_minimo_muestra -> para reportar, no solo para filtrar)."""
    if n is None or N is None or n <= 0 or N <= 0 or pd.isna(n) or pd.isna(N) or n > N:
        return np.nan
    z = Z_CONF[confianza]
    se = np.sqrt(p * (1 - p) / n) * np.sqrt(max(N - n, 0) / (N - 1)) if N > 1 else np.nan
    return round(100 * z * se, 2)


def n_minimo_requerido(poblacion_ac, poblacion_ct, margen_error=MARGEN_ERROR_OBJETIVO,
                        confianza=CONFIANZA_OBJETIVO):
    """El n mínimo requerido es el MAYOR entre lo que exige la población de acción
    y lo que exige la de control -> el lado más exigente gobierna."""
    return max(n_minimo_muestra(poblacion_ac, margen_error, confianza),
               n_minimo_muestra(poblacion_ct, margen_error, confianza))
