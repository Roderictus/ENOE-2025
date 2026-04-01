import pandas as pd
import numpy as np
import os
from datetime import datetime

# ----------------------------------------------------------------------
# DICCIONARIOS
# ----------------------------------------------------------------------
ENTIDADES = {
    1: 'Aguascalientes', 2: 'Baja California', 3: 'Baja California Sur', 4: 'Campeche',
    5: 'Coahuila', 6: 'Colima', 7: 'Chiapas', 8: 'Chihuahua', 9: 'Ciudad de Mexico',
    10: 'Durango', 11: 'Guanajuato', 12: 'Guerrero', 13: 'Hidalgo', 14: 'Jalisco',
    15: 'Mexico', 16: 'Michoacan', 17: 'Morelos', 18: 'Nayarit', 19: 'Nuevo Leon',
    20: 'Oaxaca', 21: 'Puebla', 22: 'Queretaro', 23: 'Quintana Roo', 24: 'San Luis Potosi',
    25: 'Sinaloa', 26: 'Sonora', 27: 'Tabasco', 28: 'Tamaulipas', 29: 'Tlaxcala',
    30: 'Veracruz', 31: 'Yucatan', 32: 'Zacatecas'
}

EDUC_GRUPOS = {
    'Primaria': [2], 'Secundaria': [3], 'Preparatoria': [4],
    'Profesional': [7], 'Posgrado': [8, 9]
}

# INPC (Base 2018=100) simplificado para deflactar
INPC_DICT = {
    (2025, 4): 142.465, # Añade más según necesites históricamente
}

def weighted_average(df, value_col, weight_col):
    df_f = df.dropna(subset=[value_col, weight_col])
    if df_f.empty or df_f[weight_col].sum() == 0: return np.nan
    return np.average(df_f[value_col], weights=df_f[weight_col])

def obtener_nombre_archivo(year, quarter):
    year_short = str(year)[-2:]
    base_name = f"ENOE_SDEMT{quarter}{year_short}".upper() if year >= 2023 else f"ENOEN_SDEMT{quarter}{year_short}".upper()
    return os.path.join("Data/ENOE_dta", f"ENOE_{year}_{quarter}", f"{base_name}.dta")

def procesar_trimestre_estatal(year, quarter):
    file_path = obtener_nombre_archivo(year, quarter)
    PONDERATOR = 'fac_tri' if ((year == 2020 and quarter >= 3) or year >= 2021) else 'fac'
    
    if not os.path.exists(file_path): return pd.DataFrame()

    print(f"Procesando Estados para {year} T{quarter}...")
    df = pd.read_stata(file_path, convert_categoricals=False)
    df.columns = df.columns.str.lower()
    
    # Manejo de la columna entidad (ent o cve_ent)
    col_ent = 'cve_ent' if 'cve_ent' in df.columns else 'ent'
    
    # Filtro base y POBLACIÓN CON INGRESO > 0 (nuestro hallazgo clave)
    df[PONDERATOR] = pd.to_numeric(df[PONDERATOR], errors='coerce').fillna(0)
    df['r_def'] = df['r_def'].astype(str).str.strip()
    df_base = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3])) & (df['eda'] >= 15)]
    
    df_ocupada = df_base[df_base['clase2'] == 1].copy()
    df_ocupada['ingocup'] = pd.to_numeric(df_ocupada['ingocup'], errors='coerce').fillna(0)
    
    # AISLAMOS SOLO INGRESOS REALES POSITIVOS
    df_ing = df_ocupada[df_ocupada['ingocup'] > 0]
    
    resultados_estatales = []
    
    for ent_code, ent_name in ENTIDADES.items():
        # Filtro por estado
        df_ent = df_ing[df_ing[col_ent] == ent_code]
        if df_ent.empty: continue
            
        total_ing = df_ent[PONDERATOR].sum()
        formal = df_ent[df_ent['emp_ppal'] == 2][PONDERATOR].sum()
        informal = df_ent[df_ent['emp_ppal'] == 1][PONDERATOR].sum()
        
        datos_estado = {
            'year': year, 'quarter': quarter, 'entidad_codigo': ent_code, 'entidad_nombre': ent_name,
            'ocupados_con_ingreso': total_ing,
            'pct_formal': (formal / total_ing) * 100 if total_ing else np.nan,
            'pct_informal': (informal / total_ing) * 100 if total_ing else np.nan,
            'ing_mensual_promedio_general': weighted_average(df_ent, 'ingocup', PONDERATOR)
        }
        
        # Ingreso Promedio por Escolaridad (solo en la sub-población del estado)
        for label, codes in EDUC_GRUPOS.items():
            df_edu = df_ent[df_ent['cs_p13_1'].isin(codes)]
            datos_estado[f'ing_mensual_{label}'] = weighted_average(df_edu, 'ingocup', PONDERATOR)
            
        resultados_estatales.append(datos_estado)
        
    return pd.DataFrame(resultados_estatales)

if __name__ == '__main__':
    # Aquí puedes iterar sobre tu lista de periodos
    periodos = [(2025, 4)] # Ejemplo para T4 2025
    
    dfs = []
    for y, q in periodos:
        df_trim = procesar_trimestre_estatal(y, q)
        if not df_trim.empty:
            dfs.append(df_trim)
            
    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
        # Aquí puedes inyectar el DEFLACTOR como lo hicimos a nivel nacional
        # df_final['def_ing_mensual...'] = df_final['ing_mensual...'] / deflactor * 100
        
        output = f"ENOE_Estados_{datetime.now().strftime('%Y%m%d')}.csv"
        df_final.to_csv(output, index=False)
        print(f"¡Base estatal generada en {output}!")