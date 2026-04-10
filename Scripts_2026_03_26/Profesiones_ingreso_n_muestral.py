import pandas as pd
import numpy as np
import os

def procesar_profesiones_2_digitos_completo_obs():
    # =========================================================================
    # 1. RUTAS DE LOS ARCHIVOS
    # =========================================================================
    path_sdemt = "Data/ENOE_dta/ENOE_2025_4/ENOE_SDEMT425.dta"
    path_coe1t = "Data/ENOE_dta/ENOE_2025_4/ENOE_COE1T425.dta"

    print("Cargando bases de datos (SDEMT y COE1T)...")
    try:
        df_sdemt = pd.read_stata(path_sdemt, convert_categoricals=False)
        df_sdemt.columns = df_sdemt.columns.str.lower()
        
        df_coe1t = pd.read_stata(path_coe1t, convert_categoricals=False)
        df_coe1t.columns = df_coe1t.columns.str.lower()
        
        # Estandarizar nombre de la entidad federativa
        if 'cve_ent' not in df_sdemt.columns and 'ent' in df_sdemt.columns:
            df_sdemt.rename(columns={'ent': 'cve_ent'}, inplace=True)
        if 'cve_ent' not in df_coe1t.columns and 'ent' in df_coe1t.columns:
            df_coe1t.rename(columns={'ent': 'cve_ent'}, inplace=True)

        llaves_merge = ['cd_a', 'cve_ent', 'con', 'v_sel', 'n_hog', 'h_mud', 'n_ren']
        col_sinco = 'sinco' if 'sinco' in df_coe1t.columns else 'p3'
        
        PONDERATOR = 'fac_tri' if 'fac_tri' in df_sdemt.columns else 'fac'
        
        cols_sdemt = llaves_merge + [PONDERATOR, 'r_def', 'c_res', 'eda', 'clase1', 'clase2', 'ingocup', 'sex', 'hrsocup', 'emp_ppal']
        cols_coe1t = llaves_merge + [col_sinco]
        
        df_sdemt = df_sdemt[[c for c in cols_sdemt if c in df_sdemt.columns]]
        df_coe1t = df_coe1t[[c for c in cols_coe1t if c in df_coe1t.columns]]
        
        print("Fusionando bases de datos...")
        df_merge = pd.merge(df_sdemt, df_coe1t, on=llaves_merge, how='inner')
        df_merge[PONDERATOR] = pd.to_numeric(df_merge[PONDERATOR], errors='coerce').fillna(0)
        df_merge['r_def'] = df_merge['r_def'].astype(str).str.strip()
        
        # =========================================================================
        # 2. UNIVERSOS Y AGREGACIÓN A 2 DÍGITOS
        # =========================================================================
        df_ocupados = df_merge[(df_merge['r_def'] == '0.0') & 
                               (df_merge['c_res'].isin([1, 3])) & 
                               (df_merge['eda'] >= 15) & 
                               (df_merge['clase1'] == 1) & 
                               (df_merge['clase2'] == 1)].copy()
        
        def get_2digit_code(x):
            if pd.isna(x): return '00'
            x = str(x).strip()
            if x == '0nan' or x.lower() == 'nan': return '00'
            try:
                return str(int(float(x))).zfill(4)[:2]
            except:
                return '00'

        df_ocupados['codigo_2d'] = df_ocupados[col_sinco].apply(get_2digit_code)
        total_ocupados_nacional = df_ocupados[PONDERATOR].sum()

        df_ocupados['ingocup'] = pd.to_numeric(df_ocupados['ingocup'], errors='coerce').fillna(0)
        df_ocupados['hrsocup'] = pd.to_numeric(df_ocupados['hrsocup'], errors='coerce').fillna(0)
        
        df_ing = df_ocupados[df_ocupados['ingocup'] > 0].copy()
        df_ing['horas_mensuales'] = df_ing['hrsocup'] * 4.345
        df_ing_hrs = df_ing[df_ing['horas_mensuales'] > 0].copy()
        df_ing_hrs['ingreso_hora'] = df_ing_hrs['ingocup'] / df_ing_hrs['horas_mensuales']

        def weighted_avg(df_filtered, val_col):
            if df_filtered.empty: return np.nan
            val = df_filtered[val_col].values
            wt = df_filtered[PONDERATOR].values
            valid = ~np.isnan(val) & ~np.isnan(wt)
            if not valid.any() or wt[valid].sum() == 0: return np.nan
            return np.average(val[valid], weights=wt[valid])

        print("Calculando métricas agregadas (con conteo de observaciones en muestra)...")
        resultados = []
        
        for cod in df_ocupados['codigo_2d'].unique():
            g_total = df_ocupados[df_ocupados['codigo_2d'] == cod]
            g_ing = df_ing[df_ing['codigo_2d'] == cod]
            g_hrs = df_ing_hrs[df_ing_hrs['codigo_2d'] == cod]
            
            personas_total = g_total[PONDERATOR].sum()
            if personas_total == 0: continue
            
            # --- CONTEO DE OBSERVACIONES MUESTRALES SIN EXPANDIR ---
            obs_total = len(g_total)
            obs_ing_pos = len(g_ing)
            
            # Cálculos de poblaciones
            personas_formales = g_total[g_total['emp_ppal'] == 2][PONDERATOR].sum()
            personas_informales = g_total[g_total['emp_ppal'] == 1][PONDERATOR].sum()
            
            personas_hombres = g_total[g_total['sex'] == 1][PONDERATOR].sum()
            personas_mujeres = g_total[g_total['sex'] == 2][PONDERATOR].sum()
            
            personas_ing_pos_p = g_ing[PONDERATOR].sum()
                
            resultados.append({
                'Código SINCO': cod,
                'Personas (Total Ocupados)': personas_total,
                '% de la PEA Ocupada': (personas_total / total_ocupados_nacional) * 100,
                '%Formal': (personas_formales / personas_total) * 100 if personas_total > 0 else 0,
                '%Informal': (personas_informales / personas_total) * 100 if personas_total > 0 else 0,
                '%Hombres': (personas_hombres / personas_total) * 100 if personas_total > 0 else 0,
                '%Mujeres': (personas_mujeres / personas_total) * 100 if personas_total > 0 else 0,
                'Horas Mensuales Trabajadas': weighted_avg(g_hrs, 'horas_mensuales'),
                'Ingreso Mensual Promedio': weighted_avg(g_ing, 'ingocup'),
                'Ingreso x Hora (General)': weighted_avg(g_hrs, 'ingreso_hora'),
                'Ingreso x Hora (Hombres)': weighted_avg(g_hrs[g_hrs['sex'] == 1], 'ingreso_hora'),
                'Ingreso x Hora (Mujeres)': weighted_avg(g_hrs[g_hrs['sex'] == 2], 'ingreso_hora'),
                'Personas con ingresos positivos': personas_ing_pos_p,
                'Personas con ingresos positivos como % de Personas de la PEA Ocupada': (personas_ing_pos_p / personas_total) * 100 if personas_total > 0 else 0,
                'Observaciones (Muestra Total)': obs_total,
                'Observaciones (Muestra con Ingresos)': obs_ing_pos
            })
            
        df_res = pd.DataFrame(resultados).sort_values(by='Personas (Total Ocupados)', ascending=False)
        
        # =========================================================================
        # 3. DICCIONARIO COMPLETO (SINCO a 2 Dígitos)
        # =========================================================================
        cat_map_2d = {
            '11': 'Funcionarios y directores de los sectores público, privado y social',
            '12': 'Coordinadores y jefes de área',
            '13': 'Directores y gerentes en producción, tecnología y transporte',
            '14': 'Directores y gerentes de ventas, restaurantes, hoteles y otros establecimientos',
            '15': 'Coordinadores y jefes de área en servicios financieros, administrativos y sociales',
            '16': 'Coordinadores y jefes de área en producción y tecnología',
            '17': 'Coordinadores y jefes de área de ventas, restaurantes, hoteles',
            '19': 'Otros directores, funcionarios, gerentes y jefes de área',
            '21': 'Especialistas en ciencias económico-administrativas y sociales',
            '22': 'Especialistas en ciencias exactas, ingeniería y tecnología',
            '23': 'Profesores y especialistas en docencia',
            '24': 'Médicos, enfermeras y especialistas en salud',
            '25': 'Especialistas en tecnología de la información',
            '26': 'Otros profesionistas y técnicos',
            '27': 'Auxiliares y técnicos en educación, instructores y capacitadores',
            '28': 'Enfermeras, técnicos en medicina y trabajadores de apoyo en salud',
            '29': 'Otros profesionistas y técnicos no clasificados anteriormente',
            '31': 'Trabajadores de apoyo administrativo y secretarial',
            '32': 'Trabajadores que brindan información al público',
            '33': 'Auxiliares y técnicos en administración y contabilidad',
            '39': 'Otros trabajadores auxiliares en actividades administrativas',
            '41': 'Comerciantes en establecimientos',
            '42': 'Empleados de ventas y despachadores',
            '43': 'Agentes de ventas y comercio exterior',
            '49': 'Otros comerciantes, empleados en ventas y agentes de ventas',
            '51': 'Trabajadores en preparación de alimentos y bebidas',
            '52': 'Trabajadores en cuidados personales y belleza',
            '53': 'Trabajadores de protección y seguridad',
            '54': 'Trabajadores en el cuidado de personas',
            '61': 'Trabajadores en actividades agrícolas',
            '62': 'Trabajadores en actividades ganaderas',
            '63': 'Trabajadores en silvicultura, pesca y caza',
            '69': 'Otros trabajadores en actividades agrícolas, ganaderas, forestales, caza y pesca',
            '71': 'Trabajadores en extracción y construcción',
            '72': 'Trabajadores en metalmecánica y orfebrería',
            '73': 'Trabajadores de la madera, papel y artes gráficas',
            '74': 'Trabajadores en la industria textil y del cuero',
            '75': 'Trabajadores en la elaboración de alimentos y bebidas',
            '76': 'Operadores de plantas e instalaciones',
            '79': 'Otros trabajadores artesanales',
            '81': 'Operadores de maquinaria industrial',
            '82': 'Ensambladores de manufactura',
            '83': 'Conductores de transporte y maquinaria móvil',
            '89': 'Otros operadores de maquinaria, ensambladores y conductores',
            '91': 'Trabajadores de apoyo en actividades agropecuarias',
            '92': 'Trabajadores de apoyo en minería y construcción',
            '93': 'Trabajadores de apoyo en la industria y transporte',
            '94': 'Ayudantes en la preparación de alimentos',
            '95': 'Vendedores ambulantes',
            '96': 'Trabajadores domésticos y de limpieza',
            '97': 'Trabajadores de apoyo diversos',
            '98': 'Ocupaciones no especificadas',
            '99': 'Ocupaciones no clasificadas',
            '00': 'Ocupación no especificada / Nulos',
            '0n': 'Ocupación no especificada / Nulos'
        }
        
        df_res.insert(1, 'Grupo Profesional (Subgrupo Principal)', df_res['Código SINCO'].map(cat_map_2d).fillna("Grupo " + df_res['Código SINCO']))

        # =========================================================================
        # 4. ORDEN FINAL Y FORMATO PARA GOOGLE SHEETS
        # =========================================================================
        columnas_finales = [
            'Código SINCO',
            'Grupo Profesional (Subgrupo Principal)',
            'Personas (Total Ocupados)',
            '% de la PEA Ocupada',
            '%Formal',
            '%Informal',
            '%Hombres',
            '%Mujeres',
            'Horas Mensuales Trabajadas',
            'Ingreso Mensual Promedio',
            'Ingreso x Hora (General)',
            'Ingreso x Hora (Hombres)',
            'Ingreso x Hora (Mujeres)',
            'Personas con ingresos positivos',
            'Personas con ingresos positivos como % de Personas de la PEA Ocupada',
            'Observaciones (Muestra Total)',
            'Observaciones (Muestra con Ingresos)'
        ]
        
        df_res = df_res[columnas_finales]

        # Columnas que deben ser números enteros (sin decimales)
        int_cols = [
            'Personas (Total Ocupados)',
            'Horas Mensuales Trabajadas',
            'Ingreso Mensual Promedio',
            'Ingreso x Hora (General)',
            'Ingreso x Hora (Hombres)',
            'Ingreso x Hora (Mujeres)',
            'Personas con ingresos positivos',
            'Observaciones (Muestra Total)',
            'Observaciones (Muestra con Ingresos)'
        ]

        # Columnas que deben ser porcentajes (conservar un decimal y usar coma)
        pct_cols = [
            '% de la PEA Ocupada',
            '%Formal',
            '%Informal',
            '%Hombres',
            '%Mujeres',
            'Personas con ingresos positivos como % de Personas de la PEA Ocupada'
        ]

        # Aplicar el redondeo de enteros
        for col in int_cols:
            df_res[col] = df_res[col].fillna(0).round(0).astype(int)

        # Aplicar el redondeo de porcentajes y convertir a formato LATAM (con coma)
        for col in pct_cols:
            df_res[col] = df_res[col].fillna(0).round(1)
            # Descomenta la siguiente línea si quieres exportar forzosamente con comas en lugar de puntos. 
            # Si tu Google Sheets está configurado en español, lee el CSV y separa por comas los campos
            # de manera nativa (si guardas con sep=';').
            df_res[col] = df_res[col].astype(str).str.replace('.', ',')

        # Convertir también enteros a texto para prevenir notaciones científicas en Excel/Sheets
        for col in int_cols:
            df_res[col] = df_res[col].astype(str)

        # Exportar CSV usando ';' como delimitador para que LATAM/Google Sheets lo respete sin romper
        output_filename = "Grupos_Profesionales_Analisis_Obs.csv"
        df_res.to_csv(output_filename, sep=';', index=False, encoding='utf-8-sig')
        
        print("\n¡Éxito! El archivo CSV está listo, con las muestras verificadas y listo para tabular.")
        print(f"Búscalo como: '{output_filename}' en tu carpeta local.\n")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == '__main__':
    procesar_profesiones_2_digitos_completo_obs()