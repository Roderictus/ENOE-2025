import pandas as pd
import os

# 1. Definir rutas (Asegúrate de que los nombres coincidan con tus archivos)
path_sdemt = "Data/ENOE_dta/ENOE_2025_4/ENOE_SDEMT425.dta"
path_coe1t = "Data/ENOE_dta/ENOE_2025_4/ENOE_COE1T425.dta"

print("Cargando bases de datos (SDEMT y COE1T)...")
try:
    # Cargar bases y pasar a minúsculas
    df_sdemt = pd.read_stata(path_sdemt, convert_categoricals=False)
    df_sdemt.columns = df_sdemt.columns.str.lower()
    
    df_coe1t = pd.read_stata(path_coe1t, convert_categoricals=False)
    df_coe1t.columns = df_coe1t.columns.str.lower()
    
    # Estandarizar nombre de entidad si es necesario
    if 'cve_ent' not in df_sdemt.columns and 'ent' in df_sdemt.columns:
        df_sdemt.rename(columns={'ent': 'cve_ent'}, inplace=True)
    if 'cve_ent' not in df_coe1t.columns and 'ent' in df_coe1t.columns:
        df_coe1t.rename(columns={'ent': 'cve_ent'}, inplace=True)

    # 2. Definir llaves para el Merge
    llaves_posibles = ['cd_a', 'cve_ent', 'con', 'v_sel', 'n_hog', 'h_mud', 'n_ren']
    llaves_merge = [k for k in llaves_posibles if k in df_sdemt.columns and k in df_coe1t.columns]
    
    # Buscar la columna SINCO en la COE1T
    col_sinco = 'sinco' if 'sinco' in df_coe1t.columns else ('p3' if 'p3' in df_coe1t.columns else None)
    if not col_sinco:
        print("No se encontró variable de ocupación (sinco o p3) en COE1T.")
        exit()

    # Columnas a conservar para no saturar RAM
    PONDERATOR = 'fac_tri' if 'fac_tri' in df_sdemt.columns else 'fac'
    cols_sdemt = llaves_merge + [PONDERATOR, 'r_def', 'c_res', 'eda', 'clase1', 'clase2', 'emp_ppal', 'ingocup', 'sex']
    cols_coe1t = llaves_merge + [col_sinco]
    
    df_sdemt = df_sdemt[[c for c in cols_sdemt if c in df_sdemt.columns]]
    df_coe1t = df_coe1t[[c for c in cols_coe1t if c in df_coe1t.columns]]
    
    print(f"Haciendo Merge usando llaves: {llaves_merge}")
    df_merge = pd.merge(df_sdemt, df_coe1t, on=llaves_merge, how='inner')
    
    # 3. Filtrar Población Ocupada (PEA) con Ingresos > 0
    df_merge[PONDERATOR] = pd.to_numeric(df_merge[PONDERATOR], errors='coerce').fillna(0)
    df_merge['r_def'] = df_merge['r_def'].astype(str).str.strip()
    
    df_ocupados = df_merge[(df_merge['r_def'] == '0.0') & 
                           (df_merge['c_res'].isin([1, 3])) & 
                           (df_merge['eda'] >= 15) & 
                           (df_merge['clase1'] == 1) & 
                           (df_merge['clase2'] == 1)].copy()
                           
    df_ocupados['ingocup'] = pd.to_numeric(df_ocupados['ingocup'], errors='coerce').fillna(0)
    df_ing = df_ocupados[df_ocupados['ingocup'] > 0].copy()
    
    # 4. Agrupar por Profesión (SINCO a 4 dígitos)
    # Rellenar con ceros a la izquierda por si se leyó como numérico (ej. 211 -> 0211)
    df_ing['codigo_profesion'] = df_ing[col_sinco].astype(str).str.split('.').str[0].str.zfill(4)
    total_ing = df_ing[PONDERATOR].sum()

    resultados = []
    grupos = df_ing.groupby('codigo_profesion')
    
    for cod, grupo in grupos:
        personas = grupo[PONDERATOR].sum()
        if personas == 0: continue
            
        ingreso_prom = (grupo['ingocup'] * grupo[PONDERATOR]).sum() / personas
        formales = grupo[grupo['emp_ppal'] == 2][PONDERATOR].sum()
        hombres = grupo[grupo['sex'] == 1][PONDERATOR].sum()
        
        resultados.append({
            'Código SINCO': cod,
            'Personas': personas,
            '% del Total Ocupado': (personas / total_ing) * 100,
            '% Formalidad': (formales / personas) * 100,
            'Ingreso Mensual Promedio': ingreso_prom,
            '% Hombres': (hombres / personas) * 100,
            '% Mujeres': ((personas - hombres) / personas) * 100
        })
        
    df_res = pd.DataFrame(resultados).sort_values(by='Personas', ascending=False).head(20)
    
    print("\n--- EL VERDADERO TOP 20 DE PROFESIONES EN MÉXICO (T4 2025) ---")
    format_dict = {'Personas': '{:,.0f}'.format, '% del Total Ocupado': '{:.2f}%'.format, 
                   '% Formalidad': '{:.1f}%'.format, 'Ingreso Mensual Promedio': '${:,.0f}'.format,
                   '% Hombres': '{:.1f}%'.format, '% Mujeres': '{:.1f}%'.format}
                   
    print(df_res.to_string(index=False, formatters=format_dict))
    
    # Guardar en CSV para cruzar los nombres después
    df_res.to_csv("Top_20_Profesiones_SINCO.csv", index=False)

except Exception as e:
    print(f"Ocurrió un error: {e}")