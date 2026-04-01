import pandas as pd
import os

def analizar_coe2t_ingreso_cero(year, quarter):
    # 1. Rutas a los archivos
    year_short = str(year)[-2:]
    path_sdemt = os.path.join("Data/ENOE_dta", f"ENOE_{year}_{quarter}", f"ENOE_SDEMT{quarter}{year_short}.dta")
    path_coe2t = os.path.join("Data/ENOE_dta", f"ENOE_{year}_{quarter}", f"ENOE_COE2T{quarter}{year_short}.dta")
    
    print(f"--- Iniciando análisis COE2T para {year} T{quarter} ---")
    
    try:
        # 2. Carga y estandarización de columnas
        print("Cargando base SDEMT...")
        df_sdemt = pd.read_stata(path_sdemt, convert_categoricals=False)
        df_sdemt.columns = df_sdemt.columns.str.lower()
        
        # Homologar nombre de la entidad si es necesario
        if 'cve_ent' not in df_sdemt.columns and 'ent' in df_sdemt.columns:
            df_sdemt.rename(columns={'ent': 'cve_ent'}, inplace=True)
            
        print("Cargando base COE2T...")
        df_coe2t = pd.read_stata(path_coe2t, convert_categoricals=False)
        df_coe2t.columns = df_coe2t.columns.str.lower()
        
    except Exception as e:
        print(f"Error al cargar archivos: {e}")
        return

    # 3. Definir llaves reales basadas en tus datos
    llaves_posibles = ['cd_a', 'cve_ent', 'con', 'v_sel', 'n_hog', 'h_mud', 'n_ren']
    
    # Filtrar solo las llaves que existen en AMBAS bases
    llaves_merge = [k for k in llaves_posibles if k in df_sdemt.columns and k in df_coe2t.columns]
    print(f"Llaves detectadas para el merge: {llaves_merge}")
    
    # 4. Seleccionar solo las columnas que nos interesan para no saturar memoria
    # De SDEMT: Llaves + Clase1 (PEA), Clase2 (Ocupado), Emp_Ppal (Formalidad), IngOcup, Factor, Edad
# 4. Seleccionar solo las columnas que nos interesan
    cols_sdemt_keep = llaves_merge + ['fac_tri', 'clase1', 'clase2', 'emp_ppal', 'ingocup', 'eda', 'pos_ocu']
    cols_coe2t_keep = llaves_merge + ['p6a3', 'p6c', 'p6_99', 'p6b2']
    
    # Filtrar DataFrames
    df_sdemt = df_sdemt[[c for c in cols_sdemt_keep if c in df_sdemt.columns]]
    df_coe2t = df_coe2t[[c for c in cols_coe2t_keep if c in df_coe2t.columns]]

    # 5. Ejecutar el Merge
    print("Ejecutando fusión (Inner Join)...")
    df_merge = pd.merge(df_sdemt, df_coe2t, on=llaves_merge, how='inner')
    
    # 6. Filtrar Población Ocupada (Clase1=1, Clase2=1) >= 15 años
    df_merge['fac_tri'] = pd.to_numeric(df_merge['fac_tri'], errors='coerce').fillna(0)
    df_ocupados = df_merge[(df_merge['eda'] >= 15) & (df_merge['clase1'] == 1) & (df_merge['clase2'] == 1)].copy()
    
    # Filtrar solo los que ganan $0
    df_ocupados['ingocup'] = pd.to_numeric(df_ocupados['ingocup'], errors='coerce').fillna(0)
    df_zero = df_ocupados[df_ocupados['ingocup'] == 0].copy()
    
    total_zero = df_zero['fac_tri'].sum()
    print(f"\nPOBLACIÓN OCUPADA CON INGRESO $0 (SDEMT + COE2T): {total_zero:,.0f} personas")
    
    # 7. Tabular variable p6a3
    if 'p6a3' in df_zero.columns:
        print("\n--- P6A3: ¿POR QUÉ NO RECIBIÓ INGRESOS ESTA SEMANA? ---")
        df_zero['p6a3'] = pd.to_numeric(df_zero['p6a3'], errors='coerce').fillna(-1)
        
        # Mapeo oficial del Cuestionario ENOE
        MAPEO_P6A3 = {
            1: '1. Es trabajador no remunerado',
            2: '2. Su negocio propio no dio utilidades',
            3: '3. Falta de ventas (Trabaja por comisión/destajo)',
            4: '4. La empresa se retrasó en el pago',
            5: '5. Tuvo ausencias temporales sin goce de sueldo',
            6: '6. Otra razón',
            9: '9. No sabe',
            -1: 'No aplica / No contestó esta pregunta en específico'
        }
        
        res_p6a3 = df_zero.groupby('p6a3')['fac_tri'].sum().reset_index()
        res_p6a3 = res_p6a3.sort_values(by='fac_tri', ascending=False)
        res_p6a3['Porcentaje'] = (res_p6a3['fac_tri'] / total_zero) * 100
        res_p6a3['Descripción'] = res_p6a3['p6a3'].map(MAPEO_P6A3).fillna('Desconocido')
        
        format_dict = {'fac_tri': '{:,.0f}'.format, 'Porcentaje': '{:.2f}%'.format}
        print(res_p6a3[['p6a3', 'Descripción', 'fac_tri', 'Porcentaje']].to_string(index=False, formatters=format_dict))
        
        # Cruzar con formalidad
        print("\n--- CRUCE: MOTIVO (P6A3) VS SECTOR FORMAL/INFORMAL ---")
        df_zero['Condicion_Formal'] = df_zero['emp_ppal'].map({1: 'Informal', 2: 'Formal'}).fillna('N/E')
        
        cruce = pd.crosstab(
            index=df_zero['p6a3'].map(MAPEO_P6A3),
            columns=df_zero['Condicion_Formal'],
            values=df_zero['fac_tri'],
            aggfunc='sum'
        ).fillna(0)
        
        format_cruce = {col: lambda x: f"{x:,.0f}" for col in cruce.columns}
        print(cruce.to_string(formatters=format_cruce))

        print("\n--- EL MISTERIO RESUELTO: ¿SON NO RESPUESTAS AL INGRESO? ---")
    
    # Evaluar rechazos directos a declarar ingresos
    if 'p6_99' in df_zero.columns and 'p6b2' in df_zero.columns:
        # Convertir a numérico para evitar errores
        df_zero['p6_99'] = pd.to_numeric(df_zero['p6_99'], errors='coerce').fillna(-1)
        df_zero['p6b2'] = pd.to_numeric(df_zero['p6b2'], errors='coerce').fillna(-1)
        
        # Clasificar a la población de ingreso cero
        def clasificar_cero(row):
            if row['pos_ocu'] == 4:
                return '1. Cero Real: Trabajador No Remunerado'
            elif row['p6_99'] == 99 or row['p6_99'] == 9:
                return '2. Falso Cero: Se negó a contestar (No sabe/No responde)'
            elif row['p6b2'] == 99 or row['p6b2'] == 9:
                return '2. Falso Cero: Se negó a dar el monto exacto'
            elif pd.isna(row['p6b2']) or row['p6b2'] == -1:
                return '3. Falso Cero: Salto de cuestionario / Datos en blanco'
            else:
                return '4. Otros ceros (Comisiones, negocios sin utilidad, etc.)'

        df_zero['Clasificacion_Cero'] = df_zero.apply(clasificar_cero, axis=1)
        
        # Tabular resultados
        res_clasif = df_zero.groupby('Clasificacion_Cero')['fac_tri'].sum().reset_index()
        res_clasif = res_clasif.sort_values(by='fac_tri', ascending=False)
        res_clasif['Porcentaje'] = (res_clasif['fac_tri'] / total_zero) * 100
        
        format_dict2 = {'fac_tri': '{:,.0f}'.format, 'Porcentaje': '{:.2f}%'.format}
        print(res_clasif.to_string(index=False, formatters=format_dict2))

if __name__ == '__main__':
    analizar_coe2t_ingreso_cero(2025, 4)
