import pandas as pd
import numpy as np
import os

def analyze_professions():
    year = 2025
    quarter = 4
    year_short = str(year)[-2:]
    
    path_sdemt = os.path.join("Data/ENOE_dta", f"ENOE_{year}_{quarter}", f"ENOE_SDEMT{quarter}{year_short}.dta")
    path_coe1t = os.path.join("Data/ENOE_dta", f"ENOE_{year}_{quarter}", f"ENOE_COE1T{quarter}{year_short}.dta")
    
    print("Cargando SDEMT...")
    df_sdemt = pd.read_stata(path_sdemt, convert_categoricals=False)
    df_sdemt.columns = df_sdemt.columns.str.lower()
    
    print("Cargando COE1T...")
    df_coe1t = pd.read_stata(path_coe1t, convert_categoricals=False)
    df_coe1t.columns = df_coe1t.columns.str.lower()
    
    # Homologar nombre de entidad si varió
    if 'cve_ent' not in df_sdemt.columns and 'ent' in df_sdemt.columns:
        df_sdemt.rename(columns={'ent': 'cve_ent'}, inplace=True)
    if 'cve_ent' not in df_coe1t.columns and 'ent' in df_coe1t.columns:
        df_coe1t.rename(columns={'ent': 'cve_ent'}, inplace=True)
        
    llaves_posibles = ['cd_a', 'cve_ent', 'con', 'v_sel', 'n_hog', 'h_mud', 'n_ren']
    llaves_merge = [k for k in llaves_posibles if k in df_sdemt.columns and k in df_coe1t.columns]
    
    # Seleccionar columnas para optimizar la Memoria RAM
    PONDERATOR = 'fac_tri' if 'fac_tri' in df_sdemt.columns else 'fac'
    cols_sdemt = llaves_merge + [PONDERATOR, 'clase1', 'clase2', 'eda', 'emp_ppal', 'ingocup', 'sex', 'r_def', 'c_res']
    cols_coe1t = llaves_merge + ['p4a']
    
    df_sdemt = df_sdemt[[c for c in cols_sdemt if c in df_sdemt.columns]]
    df_coe1t = df_coe1t[[c for c in cols_coe1t if c in df_coe1t.columns]]
    
    print("Fusionando bases (Inner Join)...")
    df_merge = pd.merge(df_sdemt, df_coe1t, on=llaves_merge, how='inner')
    
    df_merge[PONDERATOR] = pd.to_numeric(df_merge[PONDERATOR], errors='coerce').fillna(0)
    df_merge['r_def'] = df_merge['r_def'].astype(str).str.strip()
    
    # Filtrar estrictamente PEA Ocupada
    df_ocu = df_merge[(df_merge['r_def'] == '0.0') & 
                      (df_merge['c_res'].isin([1, 3])) & 
                      (df_merge['eda'] >= 15) & 
                      (df_merge['clase1'] == 1) & 
                      (df_merge['clase2'] == 1)].copy()
                      
    df_ocu['ingocup'] = pd.to_numeric(df_ocu['ingocup'], errors='coerce').fillna(0)
    # Limpiar el código SINCO a 4 dígitos
    df_ocu['p4a'] = df_ocu['p4a'].astype(str).str.split('.').str[0].str.zfill(4)
    
    total_pea = df_ocu[PONDERATOR].sum()
    
    print("Calculando métricas por profesión...")
    resultados = []
    
    # Agrupar por profesión
    for code, group in df_ocu.groupby('p4a'):
        # Ignorar códigos nulos o mal formados
        if code in ['9999', '0000', 'nan', '0nan']:
            continue
            
        personas = group[PONDERATOR].sum()
        if personas == 0:
            continue
            
        pct_pea = (personas / total_pea) * 100
        
        # Formalidad
        formal = group[group['emp_ppal'] == 2][PONDERATOR].sum()
        pct_formal = (formal / personas) * 100
        
        # Género
        hombres = group[group['sex'] == 1][PONDERATOR].sum()
        mujeres = group[group['sex'] == 2][PONDERATOR].sum()
        pct_hombres = (hombres / personas) * 100
        pct_mujeres = (mujeres / personas) * 100
        
        # Ingreso Promedio (Solo personas que sí declararon ingresos > 0)
        group_ing = group[group['ingocup'] > 0]
        if group_ing.empty or group_ing[PONDERATOR].sum() == 0:
            ing_promedio = np.nan
        else:
            ing_promedio = np.average(group_ing['ingocup'], weights=group_ing[PONDERATOR])
            
        resultados.append({
            'Código': code,
            'Personas': personas,
            '% PEA': pct_pea,
            '% Formal': pct_formal,
            'Ingreso Mensual': ing_promedio,
            '% Hombres': pct_hombres,
            '% Mujeres': pct_mujeres
        })
        
    # Crear DataFrame y ordenar de mayor a menor población
    df_res = pd.DataFrame(resultados).sort_values(by='Personas', ascending=False)
    
    # Diccionario SINCO para las profesiones más dominantes
    SINCO_MAP = {
        '4111': 'Comerciantes en establecimientos',
        '4211': 'Empleados de ventas y dependientes',
        '6111': 'Trabajadores agrícolas (maíz, frijol)',
        '5411': 'Trabajadores domésticos',
        '7111': 'Albañiles y mamposteros',
        '8342': 'Choferes de taxis y autos con ruta',
        '5116': 'Trabajadores de limpieza',
        '4311': 'Comerciantes ambulantes',
        '4312': 'Vendedores ambulantes de alimentos',
        '2221': 'Profesores de primaria',
        '3111': 'Auxiliares administrativos',
        '8341': 'Choferes de camiones',
        '5312': 'Vigilantes y guardias',
        '7311': 'Mecánicos automotrices',
        '9311': 'Peones agropecuarios',
        '5412': 'Peluqueros y estilistas',
        '7121': 'Carpinteros',
        '2411': 'Contadores y auditores',
        '2421': 'Abogados',
        '2111': 'Médicos generales',
        '5111': 'Cocineros',
        '5114': 'Meseros',
        '9111': 'Peones de construcción',
        '3112': 'Cajeros'
    }
    
    # Insertar el nombre de la profesión
    df_res.insert(1, 'Profesión', df_res['Código'].map(SINCO_MAP).fillna('Ocupación específica (Ver Catálogo SINCO)'))
    
    # Exportar el dataset completo para tu análisis profundo
    df_res.to_csv("ENOE_Tabla_Profesiones.csv", index=False)
    
    print("\n--- TOP 20 PROFESIONES EN MÉXICO ---")
    formatters = {
        'Personas': '{:,.0f}'.format,
        '% PEA': '{:.2f}%'.format,
        '% Formal': '{:.1f}%'.format,
        'Ingreso Mensual': '${:,.0f}'.format,
        '% Hombres': '{:.1f}%'.format,
        '% Mujeres': '{:.1f}%'.format
    }
    # Imprimir la tabla en consola
    print(df_res.head(20).to_string(index=False, formatters=formatters, na_rep="N/D"))
    print("\n¡Éxito! Se ha exportado el archivo 'ENOE_Tabla_Profesiones.csv' en tu carpeta con las métricas de las +400 profesiones del país.")

if __name__ == "__main__":
    analyze_professions()