import pandas as pd
import numpy as np
import os
from datetime import datetime

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN Y DICCIONARIOS
# ----------------------------------------------------------------------

# Mapeo de la variable cs_p13_1 (Niveles educativos detallados)
EDUC_MAP = {
    0: 'Ninguno', 1: 'Preescolar', 2: 'Primaria', 3: 'Secundaria',
    4: 'Preparatoria', 5: 'Normal', 6: 'CarreraTec', 7: 'Profesional',
    8: 'Maestria', 9: 'Doctorado', 99: 'NoSabe'
}

# Niveles educativos agregados (variable niv_ins)
NIVELES_EDUC_AGREGADOS = {
    1: 'prim_inc', 2: 'prim_comp', 3: 'secundaria', 4: 'sup_y_mas'
}

# INPC Histórico Trimestral (Base 2018=100)
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
    138.743, 140.012, 140.948
]

# Construcción robusta del diccionario del deflactor
# Mapea (año, trimestre) -> INPC. Si se pide 2025 T4, regresará None/NaN
DEFLACTOR_DICT = {}
idx = 0
for y in range(2005, 2027):
    for q in range(1, 5):
        if idx < len(inpc_historico_trimestral):
            DEFLACTOR_DICT[(y, q)] = inpc_historico_trimestral[idx]
            idx += 1


# ----------------------------------------------------------------------
# 2. FUNCIONES AUXILIARES
# ----------------------------------------------------------------------

def weighted_average(df, value_col, weight_col):
    """Calcula el promedio ponderado excluyendo ceros e inválidos según la variable."""
    df_filtered = df.dropna(subset=[value_col, weight_col])
    
    if value_col in ['ingocup', 'ing_x_hrs', 'anios_esc', 'hrsocup']:
        if value_col in ['ingocup', 'ing_x_hrs']:
            df_filtered = df_filtered[df_filtered[value_col] > 0].copy()
        else:
            df_filtered = df_filtered[df_filtered[value_col] < 99].copy()
            
    if df_filtered.empty or df_filtered[weight_col].sum() == 0:
        return np.nan
    return np.average(df_filtered[value_col], weights=df_filtered[weight_col])

def safe_pct(num, den):
    return (num / den) * 100 if den else np.nan

def obtener_nombre_archivo(year, quarter, file_format='dta'):
    """Determina la ruta del archivo SDEMT según el trimestre."""
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

def pedir_rango_trimestral():
    """Pide al usuario el rango de años y trimestres."""
    while True:
        try:
            print("\n--- Definición del Rango de la Serie de Tiempo ---")
            start_year = int(input("Ingrese el AÑO de inicio (e.g., 2005): "))
            start_quarter = int(input("Ingrese el TRIMESTRE de inicio (1 a 4): "))
            end_year = int(input("Ingrese el AÑO final (e.g., 2025): "))
            end_quarter = int(input("Ingrese el TRIMESTRE final (1 a 4): "))

            if not (1 <= start_quarter <= 4 and 1 <= end_quarter <= 4):
                raise ValueError("El trimestre debe ser entre 1 y 4.")
            break
        except ValueError as e:
            print(f"Entrada inválida: {e}")

    periodos = []
    y, q = start_year, start_quarter
    while (y, q) <= (end_year, end_quarter):
        if y == 2020 and q in [1, 2]:
            print(f"Aviso: Saltando {y} T{q} (No disponible por pandemia).")
        else:
            periodos.append((y, q))
        q += 1
        if q > 4:
            q = 1
            y += 1
    return periodos

# ----------------------------------------------------------------------
# 3. LÓGICA DE PROCESAMIENTO NACIONAL
# ----------------------------------------------------------------------

def procesar_trimestre_nacional(year, quarter):
    file_path = obtener_nombre_archivo(year, quarter)
    is_fac_tri = ((year == 2020 and quarter >= 3) or (year >= 2021))
    PONDERATOR = 'fac_tri' if is_fac_tri else 'fac'
    
    if not file_path or not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        return None

    try:
        df = pd.read_stata(file_path, convert_categoricals=False)
        df.columns = df.columns.str.lower()
    except Exception as e:
        print(f"Error leyendo {year} Q{quarter}: {e}")
        return None

    # Columnas requeridas
    req_cols = [PONDERATOR, 'r_def', 'c_res', 'sex', 'eda', 'clase1', 'clase2',
                'pos_ocu', 'emp_ppal', 'ingocup', 'ing_x_hrs', 'hrsocup', 
                'niv_ins', 'anios_esc', 'seg_soc', 'cs_p13_1']
    
    for col in req_cols:
        if col not in df.columns:
            df[col] = np.nan if col in ['ingocup', 'ing_x_hrs'] else 0

    # Limpieza de tipos
    df[PONDERATOR] = pd.to_numeric(df[PONDERATOR], errors='coerce').fillna(0)
    df['r_def'] = df['r_def'].astype(str).str.strip()
    df['cs_p13_1'] = pd.to_numeric(df['cs_p13_1'], errors='coerce').fillna(99).astype(int)
    
    # Filtros base
    df = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3]))]
    df_15 = df[df['eda'].between(15, 98)].copy()
    
    if df_15.empty: return None

    # Subconjuntos
    df_pea = df_15[df_15['clase1'] == 1]
    df_ocupada = df_15[df_15['clase2'] == 1]
    df_ocupada_h = df_ocupada[df_ocupada['sex'] == 1]
    df_ocupada_m = df_ocupada[df_ocupada['sex'] == 2]
    
    df_ocup_ing = df_ocupada[df_ocupada['ingocup'] > 0]
    df_ocup_ing_h = df_ocup_ing[df_ocup_ing['sex'] == 1]
    df_ocup_ing_m = df_ocup_ing[df_ocup_ing['sex'] == 2]

    # Diccionario de resultados
    data = {'year': year, 'quarter': quarter}
    
    # Básicos
    pob_15_total = df_15[PONDERATOR].sum()
    data['pob_15ymas_total'] = pob_15_total
    data['pea_total'] = df_pea[PONDERATOR].sum()
    data['ocupada_total'] = df_ocupada[PONDERATOR].sum()
    data['desocupada_total'] = df_15[df_15['clase2'] == 2][PONDERATOR].sum()

    # Masa e Ingresos Generales
    data['masa_salarial_total'] = (df_ocup_ing['ingocup'] * df_ocup_ing[PONDERATOR]).sum()
    data['ing_mensual_hombres'] = weighted_average(df_ocupada_h, 'ingocup', PONDERATOR)
    data['ing_mensual_mujeres'] = weighted_average(df_ocupada_m, 'ingocup', PONDERATOR)
    data['ing_hora_hombres'] = weighted_average(df_ocupada_h, 'ing_x_hrs', PONDERATOR)
    data['ing_hora_mujeres'] = weighted_average(df_ocupada_m, 'ing_x_hrs', PONDERATOR)

    # Formalidad e Informalidad
    formal_total = df_ocupada[df_ocupada['emp_ppal'] == 2][PONDERATOR].sum()
    informal_total = df_ocupada[df_ocupada['emp_ppal'] == 1][PONDERATOR].sum()
    data['formal_total'] = formal_total
    data['informal_total'] = informal_total
    data['pct_formal_total'] = safe_pct(formal_total, pob_15_total)

    # --- Bloque 1: Educación Detallada (cs_p13_1) ---
    for code, label in EDUC_MAP.items():
        if code == 99: continue
        df_edu = df_ocup_ing[df_ocup_ing['cs_p13_1'] == code]
        df_edu_h = df_edu[df_edu['sex'] == 1]
        df_edu_m = df_edu[df_edu['sex'] == 2]
        
        data[f'pop_edu_{label}_total'] = df_edu[PONDERATOR].sum()
        data[f'wage_edu_{label}_total'] = weighted_average(df_edu, 'ingocup', PONDERATOR)
        data[f'wage_edu_{label}_hombres'] = weighted_average(df_edu_h, 'ingocup', PONDERATOR)
        data[f'wage_edu_{label}_mujeres'] = weighted_average(df_edu_m, 'ingocup', PONDERATOR)

    # --- Bloque 2: Educación Agregada (niv_ins) y Horas ---
    WEEKS_PER_MONTH = 52 / 12
    df_ocup_rem = df_ocupada[(df_ocupada['pos_ocu'] != 4) & (df_ocupada['ingocup'] > 0)]
    
    for codigo, etiqueta in NIVELES_EDUC_AGREGADOS.items():
        # Ingresos por nivel agregado
        df_nivel = df_ocupada[df_ocupada['niv_ins'] == codigo]
        data[f'ing_{etiqueta}_hombres'] = weighted_average(df_nivel[df_nivel['sex'] == 1], 'ingocup', PONDERATOR)
        data[f'ing_{etiqueta}_mujeres'] = weighted_average(df_nivel[df_nivel['sex'] == 2], 'ingocup', PONDERATOR)
        
        # Formalidad e Informalidad por nivel agregado
        df_rem_nivel = df_ocup_rem[df_ocup_rem['niv_ins'] == codigo]
        den_total = df_rem_nivel[PONDERATOR].sum()
        formal_e = df_rem_nivel[df_rem_nivel['emp_ppal'] == 2][PONDERATOR].sum()
        informal_e = df_rem_nivel[df_rem_nivel['emp_ppal'] == 1][PONDERATOR].sum()
        
        data[f'formal_{etiqueta}_total'] = formal_e
        data[f'informal_{etiqueta}_total'] = informal_e
        data[f'pct_formal_{etiqueta}_total'] = safe_pct(formal_e, den_total)
        
        # Horas mensuales por nivel
        hrs_sem_total = weighted_average(df_rem_nivel, 'hrsocup', PONDERATOR)
        data[f'hrs_mens_rem_{etiqueta}_total'] = hrs_sem_total * WEEKS_PER_MONTH if pd.notna(hrs_sem_total) else np.nan

    return pd.Series(data)

# ----------------------------------------------------------------------
# 4. EJECUCIÓN Y DEFLACTACIÓN FINAL
# ----------------------------------------------------------------------

if __name__ == '__main__':
    periodos = pedir_rango_trimestral()
    resultados = []

    print(f"\nProcesando {len(periodos)} trimestres a nivel nacional...")
    for y, q in periodos:
        print(f"  -> {y} Q{q}")
        res = procesar_trimestre_nacional(y, q)
        if res is not None:
            resultados.append(res)

    if resultados:
        df_final = pd.DataFrame(resultados)
        
        # --- APLICAR DEFLACTOR ---
        print("\nAplicando deflactor base 2018=100...")
        df_final['deflactor_inpc'] = df_final.apply(lambda row: DEFLACTOR_DICT.get((row['year'], row['quarter']), np.nan), axis=1)
        
        # Detectar columnas monetarias automáticamente ('ing_', 'wage_', 'masa_salarial')
        cols_monetarias = [c for c in df_final.columns if any(keyword in c for keyword in ['ing_', 'wage_', 'masa_salarial'])]
        
        for col in cols_monetarias:
            # Fórmula: (Valor Nominal / INPC) * 100
            df_final[f"def_{col}"] = (df_final[col] / df_final['deflactor_inpc']) * 100

        # Guardar resultados
        today = datetime.now().strftime('%Y%m%d')
        output_file = f"ENOE_Nacional_Deflactado_{today}.csv"
        df_final.to_csv(output_file, index=False)
        print(f"\n¡Éxito! Base nacional guardada en: {output_file}")
        print(f"Dimensiones: {df_final.shape[0]} trimestres, {df_final.shape[1]} variables.")
        
        if df_final['deflactor_inpc'].isna().any():
            faltantes = df_final[df_final['deflactor_inpc'].isna()][['year', 'quarter']]
            print("\nNota: Faltan valores del deflactor para los siguientes trimestres:")
            print(faltantes.to_string(index=False))
    else:
        print("\nNo se encontraron datos para procesar.")