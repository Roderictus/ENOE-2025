import pandas as pd
import numpy as np
import os
from collections import defaultdict
from datetime import datetime

# ----------------------------------------------------------------------
# CONFIGURATION & MAPPINGS
# ----------------------------------------------------------------------

# Mapping from PDF Question cs_p13_1
EDUC_MAP = {
    0: 'Ninguno',
    1: 'Preescolar',
    2: 'Primaria',
    3: 'Secundaria',
    4: 'Preparatoria',
    5: 'Normal',
    6: 'CarreraTec',
    7: 'Profesional',
    8: 'Maestria',
    9: 'Doctorado',
    99: 'NoSabe'
}

ENTIDADES = {
    1: 'Aguascalientes', 2: 'Baja California', 3: 'Baja California Sur', 4: 'Campeche',
    5: 'Coahuila', 6: 'Colima', 7: 'Chiapas', 8: 'Chihuahua', 9: 'Ciudad de Mexico',
    10: 'Durango', 11: 'Guanajuato', 12: 'Guerrero', 13: 'Hidalgo', 14: 'Jalisco',
    15: 'Mexico', 16: 'Michoacan', 17: 'Morelos', 18: 'Nayarit', 19: 'Nuevo Leon',
    20: 'Oaxaca', 21: 'Puebla', 22: 'Queretaro', 23: 'Quintana Roo', 24: 'San Luis Potosi',
    25: 'Sinaloa', 26: 'Sonora', 27: 'Tabasco', 28: 'Tamaulipas', 29: 'Tlaxcala',
    30: 'Veracruz', 31: 'Yucatan', 32: 'Zacatecas'
}

def weighted_average(df, value_col, weight_col):
    """Calculates weighted average handling NaNs and zeros appropriately."""
    df_filtered = df.dropna(subset=[value_col, weight_col])
    
    # Strictly for income, we usually want > 0 if analyzing wages, 
    # but the filter is often applied before calling this function.
    if df_filtered.empty or df_filtered[weight_col].sum() == 0:
        return np.nan
    return np.average(df_filtered[value_col], weights=df_filtered[weight_col])

def safe_pct(num, den):
    return (num / den) * 100 if den else np.nan

def obtener_nombre_archivo(year, quarter, file_format='dta'):
    # (Your existing file path logic remains identical here)
    year_short = str(year)[-2:]
    base_name = None
    if year <= 2018:
        base_name = f"SDEMT{quarter}{year_short}".upper()
    elif year == 2019:
        base_name = f"sdemt{quarter}{year_short}".lower()
    elif (year == 2020 and quarter >= 3) or year in [2021, 2022]:
        base_name = f"ENOEN_SDEMT{quarter}{year_short}".upper()
    elif year >= 2023:
        base_name = f"ENOE_SDEMT{quarter}{year_short}".upper()
    
    dir_name = f"ENOE_{year}_{quarter}"
    file_name = f"{base_name}.{file_format}" if base_name else None
    return os.path.join("Data/ENOE_dta", dir_name, file_name) if file_name else None

def generar_periodos(inicio, fin):
    """
    Generates all (year, quarter) tuples between start and end (inclusive).
    Example: (2005, 1) to (2005, 4) -> [(2005,1), (2005,2), (2005,3), (2005,4)]
    """
    y_ini, q_ini = inicio
    y_fin, q_fin = fin

    if not (1 <= q_ini <= 4 and 1 <= q_fin <= 4):
        raise ValueError("Quarter must be between 1 and 4.")
    if (y_ini, q_ini) > (y_fin, q_fin):
        raise ValueError("Start period must be before or equal to end period.")

    periodos = []
    year, quarter = y_ini, q_ini
    while (year, quarter) <= (y_fin, q_fin):
        periodos.append((year, quarter))
        if quarter == 4:
            year += 1
            quarter = 1
        else:
            quarter += 1
    return periodos

# ----------------------------------------------------------------------
# PROCESSING LOGIC
# ----------------------------------------------------------------------

def procesar_trimestre_enoe(year, quarter):
    # Setup
    file_path = obtener_nombre_archivo(year, quarter)
    is_fac_tri = ((year == 2020 and quarter >= 3) or (year >= 2021))
    PONDERATOR = 'fac_tri' if is_fac_tri else 'fac'
    
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        df = pd.read_stata(file_path, convert_categoricals=False)
        df.columns = df.columns.str.lower()
    except Exception as e:
        print(f"Error reading {year} Q{quarter}: {e}")
        return None

    # Required Columns (Added cs_p13_1)
    req_cols = [PONDERATOR, 'r_def', 'c_res', 'ent', 'sex', 'eda', 
                'clase1', 'clase2', 'ingocup', 'cs_p13_1']
    
    # Ensure columns exist (fill missing with 0 or NaN)
    for col in req_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    # Conversions
    df[PONDERATOR] = pd.to_numeric(df[PONDERATOR], errors='coerce').fillna(0)
    df['ingocup'] = pd.to_numeric(df['ingocup'], errors='coerce') # Keep NaNs for now
    df['cs_p13_1'] = pd.to_numeric(df['cs_p13_1'], errors='coerce').fillna(99).astype(int)
    
    # Base Filters
    df = df[(df['r_def'].astype(str).str.strip() == '0.0') & (df['c_res'].isin([1, 3]))]
    df_15 = df[df['eda'].between(15, 98)].copy()
    
    if df_15.empty: return None

    # Subsets
    df_pea = df_15[df_15['clase1'] == 1]
    
    # --- POSITIVE INCOME POPULATION (The focus of your request) ---
    # Filter: PEA who have income > 0 (excludes unpaid workers and unemployed)
    df_w_income = df_pea[df_pea['ingocup'] > 0].copy()
    
    # Split by Sex (1=Men, 2=Women)
    df_w_income_h = df_w_income[df_w_income['sex'] == 1]
    df_w_income_m = df_w_income[df_w_income['sex'] == 2]

    # Aggregates
    pea_total = df_pea[PONDERATOR].sum()
    pop_w_inc_total = df_w_income[PONDERATOR].sum()
    pop_w_inc_h = df_w_income_h[PONDERATOR].sum()
    pop_w_inc_m = df_w_income_m[PONDERATOR].sum()

    data = {
        'year': year, 'quarter': quarter,
        
        # 1. Total Counts
        'pea_total': pea_total,
        'pop_ing_positivo_total': pop_w_inc_total,
        'pop_ing_positivo_hombres': pop_w_inc_h,
        'pop_ing_positivo_mujeres': pop_w_inc_m,
        
        # 2. Proportions (Income Earners / Total PEA)
        'prop_ing_positivo_vs_pea': safe_pct(pop_w_inc_total, pea_total),
        
        # 3. Average Monthly Salary (Weighted)
        'ingreso_prom_mensual_total': weighted_average(df_w_income, 'ingocup', PONDERATOR),
        'ingreso_prom_mensual_hombres': weighted_average(df_w_income_h, 'ingocup', PONDERATOR),
        'ingreso_prom_mensual_mujeres': weighted_average(df_w_income_m, 'ingocup', PONDERATOR),
    }

    # 4. Education & Wage Loop (Using cs_p13_1)
    # We calculate: Total People with that degree (from those with income) AND their Avg Wage
    for code, label in EDUC_MAP.items():
        if code == 99: continue # Skip unknown for cleaner tables
        
        # Filter for this specific education level
        df_edu = df_w_income[df_w_income['cs_p13_1'] == code]
        df_edu_h = df_w_income_h[df_w_income_h['cs_p13_1'] == code]
        df_edu_m = df_w_income_m[df_w_income_m['cs_p13_1'] == code]
        
        # Counts (People with Income > 0 in this educ level)
        data[f'pop_{label}_total'] = df_edu[PONDERATOR].sum()
        data[f'pop_{label}_hombres'] = df_edu_h[PONDERATOR].sum()
        data[f'pop_{label}_mujeres'] = df_edu_m[PONDERATOR].sum()
        
        # Wages (Avg Income for this educ level)
        data[f'wage_{label}_total'] = weighted_average(df_edu, 'ingocup', PONDERATOR)
        data[f'wage_{label}_hombres'] = weighted_average(df_edu_h, 'ingocup', PONDERATOR)
        data[f'wage_{label}_mujeres'] = weighted_average(df_edu_m, 'ingocup', PONDERATOR)

    return pd.Series(data)

# ----------------------------------------------------------------------
# EXECUTION
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Define period list manually or use your input function
    inicio = (2005, 1)
    fin = (2025, 4)
    periodos = generar_periodos(inicio, fin)
    
    results = []
    print(f"Processing {len(periodos)} quarters...")
    
    for y, q in periodos:
        print(f"Processing {y} Q{q}")
        res = procesar_trimestre_enoe(y, q)
        if res is not None:
            results.append(res)
            
    if results:
        df_final = pd.DataFrame(results)
        df_final.to_csv('ENOE_Education_Wages_National.csv', index=False)
        print("Done! Saved to ENOE_Education_Wages_National.csv")