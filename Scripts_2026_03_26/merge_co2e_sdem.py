import pandas as pd
import os

def analizar_coe2t_ingreso_cero(year, quarter):
    # 1. Definir nombres de archivos (ajustado a la nomenclatura post-2023)
    year_short = str(year)[-2:]
    base_sdemt = f"ENOE_SDEMT{quarter}{year_short}.dta"
    base_coe2t = f"ENOE_COE2T{quarter}{year_short}.dta"
    
    path_sdemt = os.path.join("Data/ENOE_dta", f"ENOE_{year}_{quarter}", base_sdemt)
    path_coe2t = os.path.join("Data/ENOE_dta", f"ENOE_{year}_{quarter}", base_coe2t)
    
    print(f"Cargando archivos para {year} T{quarter}...")
    
    # 2. Cargar bases de datos (solo las columnas necesarias para no saturar la RAM)
    # Llaves primarias para unir a la misma persona:
    llaves = ['cd_a', 'ent', 'con', 'v_sel', 'n_hog', 'h_mud', 'n_ren']
    
    cols_sdemt = llaves + ['fac_tri', 'clase1', 'clase2', 'emp_ppal', 'ingocup', 'eda']
    # En COE2T buscamos la serie P6 (ingresos crudos y motivos de no pago)
    cols_coe2t = llaves + ['p6b2', 'p6c', 'p6a3'] 
    
    # Manejo de excepciones en caso de que alguna columna de P6 varíe ligeramente de nombre
    try:
        df_sdemt = pd.read_stata(path_sdemt, columns=cols_sdemt, convert_categoricals=False)
        # Cargamos COE2T. Si tira error por columnas faltantes, lee todo y filtramos después
        df_coe2t = pd.read_stata(path_coe2t, convert_categoricals=False)
        df_coe2t.columns = df_coe2t.columns.str.lower()
    except Exception as e:
        print(f"Error al cargar: {e}")
        return

    df_sdemt.columns = df_sdemt.columns.str.lower()
    
    # 3. Hacer el Merge (Inner Join) para tener a la misma persona con sus respuestas sociodemográficas y de empleo
    print("Fusionando bases de datos SDEMT y COE2T...")
    df_merge = pd.merge(df_sdemt, df_coe2t, on=llaves, how='inner')
    
    # 4. Limpiar y Filtrar (Población ocupada mayor de 15 años con ingreso cero)
    df_merge['fac_tri'] = pd.to_numeric(df_merge['fac_tri'], errors='coerce').fillna(0)
    df_ocupados = df_merge[(df_merge['eda'] >= 15) & (df_merge['clase1'] == 1) & (df_merge['clase2'] == 1)].copy()
    
    df_ocupados['ingocup'] = pd.to_numeric(df_ocupados['ingocup'], errors='coerce').fillna(0)
    df_zero = df_ocupados[df_ocupados['ingocup'] == 0].copy()
    
    total_zero = df_zero['fac_tri'].sum()
    print(f"\nPOBLACIÓN OCUPADA CON INGRESO CERO (Verificada con COE2T): {total_zero:,.0f} personas")
    
    # 5. Explorar la variable p6a3 (Razón de no recibir pago)
    # Nota: Si 'p6a3' es nula para muchos, puede que hayan contestado que "no les pagan" en otra variable de filtro inicial.
    if 'p6a3' in df_zero.columns:
        print("\n--- RAZÓN DECLARADA PARA NO RECIBIR INGRESO ESTA SEMANA (Variable cruda p6a3) ---")
        motivos = df_zero.groupby('p6a3')['fac_tri'].sum().reset_index()
        motivos = motivos.sort_values(by='fac_tri', ascending=False)
        motivos['Porcentaje'] = (motivos['fac_tri'] / total_zero) * 100
        
        # Mapeo tentativo (depende del cuestionario exacto del trimestre, estos son los más comunes)
        MAPEO_P6A3 = {
            1: 'Trabajador no remunerado / Ayuda familiar',
            2: 'Retraso en el pago / Falta de liquidez de la empresa',
            3: 'Negocio propio sin utilidades esta semana',
            4: 'Trabajo a comisión/destajo sin ventas',
            9: 'No sabe / No respondió'
        }
        motivos['Descripción'] = motivos['p6a3'].map(MAPEO_P6A3).fillna('Otro motivo')
        
        format_dict = {'fac_tri': '{:,.0f}'.format, 'Porcentaje': '{:.2f}%'.format}
        print(motivos[['p6a3', 'Descripción', 'fac_tri', 'Porcentaje']].to_string(index=False, formatters=format_dict))
    else:
        print("La variable 'p6a3' no se encontró en esta edición de la COE2T.")

    # 6. Cruzar la razón de no pago por sector formal vs informal
    if 'p6a3' in df_zero.columns and 'emp_ppal' in df_zero.columns:
        print("\n--- MOTIVO DE INGRESO CERO DIVIDIDO POR FORMALIDAD ---")
        df_zero['condicion_formal'] = df_zero['emp_ppal'].map({1: 'Informal', 2: 'Formal'}).fillna('N/E')
        
        cruce = pd.crosstab(
            index=df_zero['p6a3'].map(MAPEO_P6A3).fillna('Otro motivo'),
            columns=df_zero['condicion_formal'],
            values=df_zero['fac_tri'],
            aggfunc='sum'
        ).fillna(0)
        
        # Formato numérico para la impresión
        format_cruce = {col: lambda x: f"{x:,.0f}" for col in cruce.columns}
        print(cruce.to_string(formatters=format_cruce))

if __name__ == '__main__':
    analizar_coe2t_ingreso_cero(2025, 4)