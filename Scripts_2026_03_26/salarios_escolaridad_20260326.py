import pandas as pd
import numpy as np
import os
from datetime import datetime

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN, DEFLACTOR Y SALARIO MÍNIMO
# ----------------------------------------------------------------------

# INPC Histórico Trimestral (Base 2018=100)
# Se añadió el T4 de 2025 (142.465) y una proyección base para T1 2026
inpc_historico_trimestral = [
    # 2005 a 2024...
    58.821, 59.137, 59.404, 60.118, 60.913, 60.928, 61.517, 62.575,
    63.460, 63.464, 63.745, 65.185, 66.108, 66.683, 67.630, 69.130,
    70.026, 70.544, 70.996, 71.812, 73.328, 73.249, 73.805, 74.751,
    75.576, 75.422, 75.745, 77.266, 78.807, 78.579, 79.451, 80.662,
    81.866, 82.345, 82.450, 83.813, 85.365, 85.296, 85.841, 87.338,
    88.050, 87.874, 88.195, 89.383, 90.445, 90.201, 90.753, 92.382,
    95.047, 95.735, 96.637, 98.477, 100.087, 100.025, 101.133, 102.957,
    103.929, 104.115, 104.556, 106.067, 107.538, 107.029, 108.307, 109.473,
    111.497, 113.048, 114.577, 117.114, 119.580, 121.832, 124.326, 126.495,
    128.538, 128.844, 130.126, 132.100, 133.767, 134.339, 136.032, 137.400,
    # 2025 (T1 a T4)
    138.743, 140.012, 140.948, 142.465,
    # 2026 (T1 aprox con datos de enero/febrero)
    144.150
]

DEFLACTOR_DICT = {}
idx = 0
for y in range(2005, 2027):
    for q in range(1, 5):
        if idx < len(inpc_historico_trimestral):
            DEFLACTOR_DICT[(y, q)] = inpc_historico_trimestral[idx]
            idx += 1

# Vector de Salario Mínimo General (SMG) Diario por año
SALARIO_MINIMO_DICT = {
    2005: 45.40, 2006: 47.16, 2007: 49.00, 2008: 51.05, 2009: 53.40,
    2010: 55.84, 2011: 58.13, 2012: 60.60, 2013: 63.12, 2014: 65.58,
    2015: 68.28, 2016: 73.04, 2017: 80.04, 2018: 88.36, 2019: 102.68,
    2020: 123.22, 2021: 141.70, 2022: 172.87, 2023: 207.44, 2024: 248.93,
    2025: 278.80, 2026: 315.04
}

EDUC_GRUPOS = {
    'Sin_Escolaridad': [0],
    'Preescolar': [1],          # Opcional, pero ayuda a cerrar la suma
    'Primaria': [2],
    'Secundaria': [3],
    'Preparatoria': [4],
    'Normal': [5],              # Opcional
    'Carrera_Tecnica': [6],     # Opcional
    'Profesional': [7],
    'Posgrado': [8, 9],
    'No_Especificada': [99]
}

# ----------------------------------------------------------------------
# 2. FUNCIONES AUXILIARES
# ----------------------------------------------------------------------

def weighted_average(df, value_col, weight_col):
    df_filtered = df.dropna(subset=[value_col, weight_col])
    df_filtered = df_filtered[df_filtered[value_col] > 0]
    if df_filtered.empty or df_filtered[weight_col].sum() == 0:
        return np.nan
    return np.average(df_filtered[value_col], weights=df_filtered[weight_col])

def weighted_std(df, value_col, weight_col, mean_val):
    """Calcula la desviación estándar ponderada de una variable."""
    df_filtered = df.dropna(subset=[value_col, weight_col])
    df_filtered = df_filtered[df_filtered[value_col] > 0]
    if df_filtered.empty or df_filtered[weight_col].sum() == 0 or pd.isna(mean_val):
        return np.nan
    variance = np.average((df_filtered[value_col] - mean_val)**2, weights=df_filtered[weight_col])
    return np.sqrt(variance)

def safe_pct(num, den):
    return (num / den) * 100 if den else np.nan

def obtener_nombre_archivo(year, quarter, file_format='dta'):
    year_short = str(year)[-2:]
    base_name = None
    if year <= 2018: base_name = f"SDEMT{quarter}{year_short}".upper()
    elif year == 2019: base_name = f"sdemt{quarter}{year_short}".lower()
    elif (year == 2020 and quarter >= 3) or year in [2021, 2022]: base_name = f"ENOEN_SDEMT{quarter}{year_short}".upper()
    elif year >= 2023: base_name = f"ENOE_SDEMT{quarter}{year_short}".upper()
    dir_name = f"ENOE_{year}_{quarter}"
    return os.path.join("Data/ENOE_dta", dir_name, f"{base_name}.{file_format}") if base_name else None

# ----------------------------------------------------------------------
# 3. PROCESAMIENTO
# ----------------------------------------------------------------------

def procesar_trimestre_nacional(year, quarter):
    file_path = obtener_nombre_archivo(year, quarter)
    PONDERATOR = 'fac_tri' if ((year == 2020 and quarter >= 3) or year >= 2021) else 'fac'
    
    if not file_path or not os.path.exists(file_path): return None

    try:
        df = pd.read_stata(file_path, convert_categoricals=False)
        df.columns = df.columns.str.lower()
    except Exception:
        return None

    req_cols = [PONDERATOR, 'r_def', 'c_res', 'eda', 'clase1', 'clase2',
                'emp_ppal', 'ingocup', 'ing_x_hrs', 'cs_p13_1']
    for col in req_cols:
        if col not in df.columns: df[col] = np.nan if col in ['ingocup', 'ing_x_hrs'] else 0

    df[PONDERATOR] = pd.to_numeric(df[PONDERATOR], errors='coerce').fillna(0)
    df['r_def'] = df['r_def'].astype(str).str.strip()
    df['cs_p13_1'] = pd.to_numeric(df['cs_p13_1'], errors='coerce').fillna(99).astype(int)
    
    # Filtro base
    df = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3]))]
    
    # --- TABLA 1: DEMOGRÁFICOS Y EMPLEO ---
    data = {'year': year, 'quarter': quarter}
    
    pob_total = df[PONDERATOR].sum()
    df_15 = df[df['eda'].between(15, 98)]
    pob_15 = df_15[PONDERATOR].sum()
    
    df_pea = df_15[df_15['clase1'] == 1]
    pea_total = df_pea[PONDERATOR].sum()
    
    df_ocupada = df_15[df_15['clase2'] == 1]
    df_ocup_ing = df_ocupada[df_ocupada['ingocup'] > 0]
    ocup_ing_total = df_ocup_ing[PONDERATOR].sum()
    
    formal_total = df_ocup_ing[df_ocup_ing['emp_ppal'] == 2][PONDERATOR].sum()
    informal_total = df_ocup_ing[df_ocup_ing['emp_ppal'] == 1][PONDERATOR].sum()
    
    data.update({
        'pob_total': pob_total,
        'pob_15ymas': pob_15,
        'pct_15ymas_total': safe_pct(pob_15, pob_total),
        'pea_total': pea_total,
        'pct_pea_15ymas': safe_pct(pea_total, pob_15),
        'ocup_ing_pos_total': ocup_ing_total,
        'pct_ocup_ing_pos_pea': safe_pct(ocup_ing_total, pea_total),
        'formal_total': formal_total,
        'pct_formal_ocup_ing': safe_pct(formal_total, ocup_ing_total),
        'informal_total': informal_total,
        'pct_informal_ocup_ing': safe_pct(informal_total, ocup_ing_total),
    })

    # ------------------------------------------------------------------
    # TABLA 2A: EDUCACIÓN EN TODA LA PEA (Incluye ingresos $0)
    # ------------------------------------------------------------------
    # Aseguramos que los desempleados o no remunerados tengan ingreso 0 en vez de NaN
    df_pea = df_pea.copy()
    df_pea['ingocup_zero'] = df_pea['ingocup'].fillna(0)
    df_pea['ing_x_hrs_zero'] = df_pea['ing_x_hrs'].fillna(0)

    for label, codes in EDUC_GRUPOS.items():
        df_edu_pea = df_pea[df_pea['cs_p13_1'].isin(codes)]
        pop_edu = df_edu_pea[PONDERATOR].sum()
        
        if pop_edu > 0:
            avg_ing_pea = np.average(df_edu_pea['ingocup_zero'], weights=df_edu_pea[PONDERATOR])
            avg_hrs_pea = np.average(df_edu_pea['ing_x_hrs_zero'], weights=df_edu_pea[PONDERATOR])
            variance_pea = np.average((df_edu_pea['ingocup_zero'] - avg_ing_pea)**2, weights=df_edu_pea[PONDERATOR])
            std_ing_pea = np.sqrt(variance_pea)
        else:
            avg_ing_pea = np.nan; avg_hrs_pea = np.nan; std_ing_pea = np.nan
            
        data.update({
            f'pop_pea_edu_{label}': pop_edu,
            f'pct_pea_edu_{label}': safe_pct(pop_edu, pea_total),
            f'ing_mensual_pea_{label}': avg_ing_pea,
            f'ing_hora_pea_{label}': avg_hrs_pea,
            f'std_ing_pea_{label}': std_ing_pea
        })

    # ------------------------------------------------------------------
    # TABLA 2B: EDUCACIÓN SÓLO EN OCUPADOS CON INGRESOS > 0
    # ------------------------------------------------------------------
    for label, codes in EDUC_GRUPOS.items():
        df_edu_ing = df_ocup_ing[df_ocup_ing['cs_p13_1'].isin(codes)]
        pop_edu = df_edu_ing[PONDERATOR].sum()
        
        avg_ing_pos = weighted_average(df_edu_ing, 'ingocup', PONDERATOR)
        
        data.update({
            f'pop_ing_edu_{label}': pop_edu,
            f'pct_ing_edu_{label}': safe_pct(pop_edu, ocup_ing_total),
            f'ing_mensual_ing_{label}': avg_ing_pos,
            f'ing_hora_ing_{label}': weighted_average(df_edu_ing, 'ing_x_hrs', PONDERATOR),
            f'std_ing_ing_{label}': weighted_std(df_edu_ing, 'ingocup', PONDERATOR, avg_ing_pos)
        })

    # --- SALARIO MÍNIMO ---
    sm_diario = SALARIO_MINIMO_DICT.get(year, 248.93)
    sm_mensual = sm_diario * 30.4
    
    df_informal = df_ocup_ing[df_ocup_ing['emp_ppal'] == 1]
    informal_menos_1sm = df_informal[df_informal['ingocup'] < sm_mensual][PONDERATOR].sum()
    
    total_menos_3sm = df_ocup_ing[df_ocup_ing['ingocup'] < (3 * sm_mensual)][PONDERATOR].sum()
    
    data.update({
        'sm_diario_oficial': sm_diario,
        'sm_mensual_estimado': sm_mensual,
        'informal_menos_1sm': informal_menos_1sm,
        'pct_informal_menos_1sm': safe_pct(informal_menos_1sm, informal_total),
        'total_menos_3sm': total_menos_3sm,
        'pct_total_menos_3sm': safe_pct(total_menos_3sm, ocup_ing_total)
    })

    return pd.Series(data)

# ----------------------------------------------------------------------
# 4. EJECUCIÓN
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # Define tus rangos manualmente para probar
    inicio = (2005, 1)
    fin = (2025, 4)
    
    periodos = []
    y, q = inicio
    while (y, q) <= fin:
        if not (y == 2020 and q in [1, 2]):
            periodos.append((y, q))
        q += 1
        if q > 4:
            q = 1; y += 1

    resultados = []
    for y, q in periodos:
        print(f"Procesando {y} Q{q}...")
        res = procesar_trimestre_nacional(y, q)
        if res is not None: resultados.append(res)

    if resultados:
        df_final = pd.DataFrame(resultados)
        
        # --- DEFLACTAR COLUMNAS MONETARIAS ---
        df_final['deflactor_inpc'] = df_final.apply(lambda row: DEFLACTOR_DICT.get((row['year'], row['quarter']), np.nan), axis=1)
        
        cols_monetarias = [c for c in df_final.columns if 'ing_' in c or 'sm_' in c or 'std_' in c]
        for col in cols_monetarias:
            df_final[f"def_{col}"] = (df_final[col] / df_final['deflactor_inpc']) * 100

        output_file = f"ENOE_Series_Tiempo_{datetime.now().strftime('%Y%m%d')}.csv"
        df_final.to_csv(output_file, index=False)
        print(f"\n¡Éxito! Base guardada en {output_file}")