import pandas as pd
import numpy as np
import os
import gc

def procesar_profesiones_1_digito_completo():
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
        df_coe1t = df_coe1t.drop_duplicates(subset=llaves_merge)
        df_merge = pd.merge(df_sdemt, df_coe1t, on=llaves_merge, how='inner')
        df_merge[PONDERATOR] = pd.to_numeric(df_merge[PONDERATOR], errors='coerce').fillna(0)
        df_merge['r_def'] = df_merge['r_def'].astype(str).str.strip()
        
        # =========================================================================
        # 2. UNIVERSOS Y AGREGACIÓN A 1 DÍGITO
        # =========================================================================
        df_ocupados = df_merge[(df_merge['r_def'] == '0.0') & 
                               (df_merge['c_res'].isin([1, 3])) & 
                               (df_merge['eda'] >= 15) & 
                               (df_merge['clase1'] == 1) & 
                               (df_merge['clase2'] == 1)].copy()
        
        # Función ajustada para extraer SÓLO EL PRIMER DÍGITO
        def get_1digit_code(x):
            if pd.isna(x): return '0'
            x = str(x).strip()
            if x == '0nan' or x.lower() == 'nan': return '0'
            try:
                # Convertir a 4 dígitos para estabilizar, luego tomar el primero
                return str(int(float(x))).zfill(4)[:1]
            except:
                return '0'

        df_ocupados['codigo_1d'] = df_ocupados[col_sinco].apply(get_1digit_code)
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

        print("Calculando métricas agregadas por Grupo Principal (1 dígito)...")
        resultados = []
        
        for cod in df_ocupados['codigo_1d'].unique():
            g_total = df_ocupados[df_ocupados['codigo_1d'] == cod]
            g_ing = df_ing[df_ing['codigo_1d'] == cod]
            g_hrs = df_ing_hrs[df_ing_hrs['codigo_1d'] == cod]
            
            personas_total = g_total[PONDERATOR].sum()
            if personas_total == 0: continue
            
            # Conteo de observaciones (Muestra)
            obs_total = len(g_total)
            obs_ing_pos = len(g_ing)
            
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
            
        # Ordenar de mayor a menor volumen de ocupados
        df_res = pd.DataFrame(resultados).sort_values(by='Personas (Total Ocupados)', ascending=False)
        
        # =========================================================================
        # 3. DICCIONARIO COMPLETO (SINCO a 1 Dígito)
        # =========================================================================
        cat_map_1d = {
            '1': 'Funcionarios, directores y jefes',
            '2': 'Profesionistas y técnicos',
            '3': 'Trabajadores auxiliares en actividades administrativas',
            '4': 'Comerciantes, empleados en ventas y agentes de ventas',
            '5': 'Trabajadores en servicios personales y vigilancia',
            '6': 'Trabajadores en actividades agrícolas, ganaderas, forestales, caza y pesca',
            '7': 'Trabajadores artesanales, en la construcción y otros oficios',
            '8': 'Operadores de maquinaria industrial, ensambladores, choferes y conductores de transporte',
            '9': 'Trabajadores en actividades elementales y de apoyo',
            '0': 'Ocupación no especificada / Nulos'
        }
        
        df_res.insert(1, 'Grupo Profesional (Grupo Principal)', df_res['Código SINCO'].map(cat_map_1d).fillna("Grupo " + df_res['Código SINCO']))

        # =========================================================================
        # 4. ORDEN FINAL Y FORMATO PARA GOOGLE SHEETS
        # =========================================================================
        columnas_finales = [
            'Código SINCO',
            'Grupo Profesional (Grupo Principal)',
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

        # Formatear enteros
        int_cols = [
            'Personas (Total Ocupados)', 'Horas Mensuales Trabajadas', 'Ingreso Mensual Promedio',
            'Ingreso x Hora (General)', 'Ingreso x Hora (Hombres)', 'Ingreso x Hora (Mujeres)',
            'Personas con ingresos positivos', 'Observaciones (Muestra Total)', 'Observaciones (Muestra con Ingresos)'
        ]
        for col in int_cols:
            df_res[col] = df_res[col].fillna(0).round(0).astype(int).astype(str)

        # Formatear porcentajes (1 decimal y coma)
        pct_cols = [
            '% de la PEA Ocupada', '%Formal', '%Informal', '%Hombres', '%Mujeres',
            'Personas con ingresos positivos como % de Personas de la PEA Ocupada'
        ]
        for col in pct_cols:
            df_res[col] = df_res[col].fillna(0).round(1).astype(str).str.replace('.', ',')

        # Exportar a CSV con punto y coma
        output_filename = "Grupos_Profesionales_Agregados_1D_LATAM.csv"
        df_res.to_csv(output_filename, sep=';', index=False, encoding='utf-8-sig')
        
        print("\n" + "="*80)
        print(" ¡ANÁLISIS A 1 DÍGITO COMPLETADO!")
        print(" Se ha condensado toda la economía mexicana en los 9 Grandes Grupos.")
        print("="*80 + "\n")
        print(f"Archivo generado exitosamente: '{output_filename}'")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == '__main__':
    procesar_profesiones_1_digito_completo()