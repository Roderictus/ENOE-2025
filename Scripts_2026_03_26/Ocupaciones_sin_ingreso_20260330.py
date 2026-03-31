import pandas as pd
import numpy as np
import os

# ----------------------------------------------------------------------
# DICCIONARIOS DE MAPEO
# ----------------------------------------------------------------------
EDUC_GRUPOS = {
    0: 'Sin_Escolaridad', 1: 'Preescolar', 2: 'Primaria', 3: 'Secundaria',
    4: 'Preparatoria', 5: 'Normal', 6: 'Carrera_Tecnica', 7: 'Profesional',
    8: 'Posgrado', 9: 'Posgrado', 99: 'No_Especificada'
}

SEXO_MAP = {1: 'Hombres', 2: 'Mujeres'}
CLASE2_MAP = {1: 'Ocupados', 2: 'Desocupados'}
POS_OCU_MAP = {1: 'Subordinados', 2: 'Empleadores', 3: 'Cuenta Propia', 4: 'No Remunerados'}

# Clasificación SINCO (Sistema Nacional de Clasificación de Ocupaciones) a 1 dígito
SINCO_MAP = {
    '1': '1. Funcionarios, directores y jefes',
    '2': '2. Profesionistas y técnicos',
    '3': '3. Auxiliares en actividades administrativas',
    '4': '4. Comerciantes y agentes de ventas',
    '5': '5. Servicios personales y vigilancia',
    '6': '6. Actividades agrícolas, ganaderas y forestales',
    '7': '7. Trabajadores artesanales',
    '8': '8. Operadores de maquinaria y transporte',
    '9': '9. Actividades elementales y de apoyo'
}

def obtener_nombre_archivo(year, quarter):
    year_short = str(year)[-2:]
    if year >= 2023:
        base_name = f"ENOE_SDEMT{quarter}{year_short}".upper()
    elif year in [2021, 2022] or (year == 2020 and quarter >= 3):
        base_name = f"ENOEN_SDEMT{quarter}{year_short}".upper()
    else:
        base_name = f"SDEMT{quarter}{year_short}".upper()
        
    return os.path.join("Data/ENOE_dta", f"ENOE_{year}_{quarter}", f"{base_name}.dta")

def weighted_value_counts(df, col, weight_col, map_dict=None):
    """Agrupa por una columna y suma los factores de expansión."""
    grouped = df.groupby(col)[weight_col].sum().reset_index()
    if map_dict:
        grouped[col] = grouped[col].map(map_dict).fillna('Otro / No Especificado')
    # Ordenar de mayor a menor población
    grouped = grouped.sort_values(by=weight_col, ascending=False).reset_index(drop=True)
    grouped['Porcentaje (%)'] = (grouped[weight_col] / grouped[weight_col].sum()) * 100
    return grouped

def analizar_ingresos_cero(year, quarter):
    file_path = obtener_nombre_archivo(year, quarter)
    PONDERATOR = 'fac_tri' if ((year == 2020 and quarter >= 3) or year >= 2021) else 'fac'
    
    print(f"Cargando {year} T{quarter}...")
    df = pd.read_stata(file_path, convert_categoricals=False)
    df.columns = df.columns.str.lower()
    
    # Rellenar y asegurar tipos
    df[PONDERATOR] = pd.to_numeric(df[PONDERATOR], errors='coerce').fillna(0)
    df['r_def'] = df['r_def'].astype(str).str.strip()
    df['ingocup_zero'] = pd.to_numeric(df['ingocup'], errors='coerce').fillna(0)
    
    # Extraer el primer dígito del SINCO (c_ocu11c) para la ocupación principal
    if 'c_ocu11c' in df.columns:
        df['sinco_1digito'] = df['c_ocu11c'].astype(str).str.strip().str[0]
    else:
        df['sinco_1digito'] = 'N/A'

    # Filtros: Residentes habituales, 15+ años, PEA y con Ingreso $0
    df_base = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3])) & (df['eda'] >= 15)]
    df_pea = df_base[df_base['clase1'] == 1]
    
    # POBLACIÓN OBJETIVO: PEA con ingreso cero
    df_zero = df_pea[df_pea['ingocup_zero'] == 0].copy()
    total_zero = df_zero[PONDERATOR].sum()
    print(f"\nPOBLACIÓN EN PEA CON INGRESO $0: {total_zero:,.0f} personas\n")

    # 1. Distribución por Condición de Ocupación (Desocupados vs Ocupados)
    dist_clase2 = weighted_value_counts(df_zero, 'clase2', PONDERATOR, CLASE2_MAP)
    print("--- 1. DISTRIBUCIÓN POR CONDICIÓN DE OCUPACIÓN ---")
    print(dist_clase2.to_string(index=False, formatters={PONDERATOR: '{:,.0f}'.format, 'Porcentaje (%)': '{:.2f}%'.format}))
    print("\n")

    # 2. Distribución por Sexo
    dist_sexo = weighted_value_counts(df_zero, 'sex', PONDERATOR, SEXO_MAP)
    print("--- 2. DISTRIBUCIÓN POR SEXO ---")
    print(dist_sexo.to_string(index=False, formatters={PONDERATOR: '{:,.0f}'.format, 'Porcentaje (%)': '{:.2f}%'.format}))
    print("\n")

    # 3. Distribución por Escolaridad
    dist_edu = weighted_value_counts(df_zero, 'cs_p13_1', PONDERATOR, EDUC_GRUPOS)
    print("--- 3. DISTRIBUCIÓN POR GRADO DE ESCOLARIDAD ---")
    print(dist_edu.to_string(index=False, formatters={PONDERATOR: '{:,.0f}'.format, 'Porcentaje (%)': '{:.2f}%'.format}))
    print("\n")

    # ---------------------------------------------------------
    # ZOOM A LOS "OCUPADOS" PERO CON INGRESO CERO
    # ---------------------------------------------------------
    df_ocup_zero = df_zero[df_zero['clase2'] == 1].copy()
    total_ocup_zero = df_ocup_zero[PONDERATOR].sum()
    print(f"--- ZOOM: OCUPADOS PERO CON INGRESO $0 ({total_ocup_zero:,.0f} personas) ---\n")

    # 4. Posición en la Ocupación
    dist_pos = weighted_value_counts(df_ocup_zero, 'pos_ocu', PONDERATOR, POS_OCU_MAP)
    print("--- 4A. POR POSICIÓN EN LA OCUPACIÓN ---")
    print(dist_pos.to_string(index=False, formatters={PONDERATOR: '{:,.0f}'.format, 'Porcentaje (%)': '{:.2f}%'.format}))
    print("\n")

    # 5. Tipo de Ocupación (SINCO)
    dist_sinco = weighted_value_counts(df_ocup_zero, 'sinco_1digito', PONDERATOR, SINCO_MAP)
    print("--- 4B. POR TIPO DE OCUPACIÓN (SISTEMA SINCO) ---")
    print(dist_sinco.to_string(index=False, formatters={PONDERATOR: '{:,.0f}'.format, 'Porcentaje (%)': '{:.2f}%'.format}))
    print("\n")

# ---------------------------------------------------------
    # FORMALIDAD VS INFORMALIDAD EN OCUPADOS CON INGRESO CERO
    # ---------------------------------------------------------
    EMP_PPAL_MAP = {1: 'Informal', 2: 'Formal'}
    
    # Crear columna mapeada de formalidad
    df_ocup_zero['condicion_formal'] = df_ocup_zero['emp_ppal'].map(EMP_PPAL_MAP).fillna('No Especificado / Otro')

    # 6. Distribución General de Formalidad en Ocupados con Ingreso $0
    dist_formal = weighted_value_counts(df_ocup_zero, 'condicion_formal', PONDERATOR)
    print("--- 6. DISTRIBUCIÓN POR FORMALIDAD / INFORMALIDAD ---")
    print(dist_formal.to_string(index=False, formatters={PONDERATOR: '{:,.0f}'.format, 'Porcentaje (%)': '{:.2f}%'.format}))
    print("\n")

    # 7. Cruce: Sectores (SINCO) por Condición de Formalidad
    print("--- 7. TIPO DE OCUPACIÓN (SINCO) DIVIDIDO POR FORMALIDAD ---")
    
    # Mapear el SINCO para el cruce
    df_ocup_zero['sinco_desc'] = df_ocup_zero['sinco_1digito'].map(SINCO_MAP).fillna('Otro / No Especificado')
    
    # Crear la tabla cruzada (crosstab)
    cruce_sinco_formal = pd.crosstab(
        index=df_ocup_zero['sinco_desc'],
        columns=df_ocup_zero['condicion_formal'],
        values=df_ocup_zero[PONDERATOR],
        aggfunc='sum'
    ).fillna(0)
    
    # Calcular totales y porcentajes para una lectura más fácil
    cruce_sinco_formal['Total'] = cruce_sinco_formal.sum(axis=1)
    cruce_sinco_formal = cruce_sinco_formal.sort_values(by='Total', ascending=False)
    
    # Formatear la tabla para imprimirla en consola
    format_dict = {col: lambda x: f"{x:,.0f}" for col in cruce_sinco_formal.columns}
    print(cruce_sinco_formal.to_string(formatters=format_dict))
    print("\n")

if __name__ == '__main__':
    analizar_ingresos_cero(2025, 4)