import pandas as pd
import numpy as np

def procesar_top50_profesiones_completo():
    # =========================================================================
    # 1. RUTAS DE LOS ARCHIVOS (Ajusta si están en otra carpeta)
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
        
        # Añadido: emp_ppal (necesario para calcular % Formal)
        cols_sdemt = llaves_merge + [PONDERATOR, 'r_def', 'c_res', 'eda', 'clase1', 'clase2', 'ingocup', 'sex', 'hrsocup', 'emp_ppal']
        cols_coe1t = llaves_merge + [col_sinco]
        
        df_sdemt = df_sdemt[[c for c in cols_sdemt if c in df_sdemt.columns]]
        df_coe1t = df_coe1t[[c for c in cols_coe1t if c in df_coe1t.columns]]
        
        print("Fusionando bases de datos...")
        df_merge = pd.merge(df_sdemt, df_coe1t, on=llaves_merge, how='inner')
        df_merge[PONDERATOR] = pd.to_numeric(df_merge[PONDERATOR], errors='coerce').fillna(0)
        df_merge['r_def'] = df_merge['r_def'].astype(str).str.strip()
        
        # =========================================================================
        # 2. DEFINIR UNIVERSOS: OCUPADOS TOTALES E INGRESOS POSITIVOS
        # =========================================================================
        df_ocupados = df_merge[(df_merge['r_def'] == '0.0') & 
                               (df_merge['c_res'].isin([1, 3])) & 
                               (df_merge['eda'] >= 15) & 
                               (df_merge['clase1'] == 1) & 
                               (df_merge['clase2'] == 1)].copy()
        
        # Limpiar códigos
        def clean_code(x):
            if pd.isna(x): return x
            x = str(x).strip()
            if x == '0nan' or x.lower() == 'nan': return '0nan'
            try:
                return str(int(float(x))).zfill(4)
            except:
                return x

        df_ocupados['codigo_profesion'] = df_ocupados[col_sinco].apply(clean_code)
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

        print("Calculando todas las métricas (Ingresos, Horas, Género, Formalidad)...")
        resultados = []
        
        for cod in df_ocupados['codigo_profesion'].unique():
            g_total = df_ocupados[df_ocupados['codigo_profesion'] == cod]
            g_ing = df_ing[df_ing['codigo_profesion'] == cod]
            g_hrs = df_ing_hrs[df_ing_hrs['codigo_profesion'] == cod]
            
            personas_total = g_total[PONDERATOR].sum()
            if personas_total == 0: continue
            
            # Cálculos adicionales requeridos
            personas_formales = g_total[g_total['emp_ppal'] == 2][PONDERATOR].sum()
            personas_hombres = g_total[g_total['sex'] == 1][PONDERATOR].sum()
            personas_ing_pos = g_ing[PONDERATOR].sum()
                
            resultados.append({
                'Código SINCO': cod,
                'Personas (Total Ocupados)': personas_total,
                '% de la PEA Ocupada': (personas_total / total_ocupados_nacional) * 100,
                '%Formal': (personas_formales / personas_total) * 100 if personas_total > 0 else 0,
                '%Hombres': (personas_hombres / personas_total) * 100 if personas_total > 0 else 0,
                'Horas Mensuales Trabajadas': weighted_avg(g_hrs, 'horas_mensuales'),
                'Ingreso Mensual Promedio': weighted_avg(g_ing, 'ingocup'),
                'Ingreso x Hora (General)': weighted_avg(g_hrs, 'ingreso_hora'),
                'Ingreso x Hora (Hombres)': weighted_avg(g_hrs[g_hrs['sex'] == 1], 'ingreso_hora'),
                'Ingreso x Hora (Mujeres)': weighted_avg(g_hrs[g_hrs['sex'] == 2], 'ingreso_hora'),
                'Personas con ingresos positivos': personas_ing_pos,
                'Personas con ingresos positivos como % de Personas de la PEA Ocupada': (personas_ing_pos / personas_total) * 100 if personas_total > 0 else 0
            })
            
        # Ordenar por el volumen total de personas en la profesión
        df_res = pd.DataFrame(resultados).sort_values(by='Personas (Total Ocupados)', ascending=False).head(50)
        
        # =========================================================================
        # 3. MAPEO DE PROFESIONES (Catálogo Consolidado)
        # =========================================================================
        cat_map = {
            '4211': 'Empleados de ventas y dependientes en comercios',
            '4111': 'Comerciantes en establecimientos',
            '9611': 'Trabajadores domésticos',
            '9111': 'Peones agrícolas y de apoyo',
            '7121': 'Albañiles, mamposteros y afines',
            '9221': 'Peones de la construcción',
            '8342': 'Conductores de taxis y automóviles',
            '9621': 'Trabajadores de limpieza (excepto hoteles)',
            '8341': 'Conductores de camiones de carga',
            '7513': 'Sastres, costureros y confeccionistas',
            '5114': 'Cocineros',
            '6111': 'Trabajadores en el cultivo de maíz y frijol',
            '5313': 'Vigilantes y guardias de seguridad',
            '9521': 'Vendedores ambulantes de artículos diversos',
            '3115': 'Auxiliares administrativos y capturistas',
            '3121': 'Cajeros, receptores de apuestas y taquilleros',
            '5116': 'Preparadores de comida rápida',
            '9512': 'Vendedores ambulantes de alimentos',
            '3132': 'Almacenistas y despachadores de inventario',
            '5111': 'Meseros',
            '8349': 'Otros conductores de transporte terrestre',
            '2111': 'Contadores y auditores',
            '2411': 'Abogados',
            '2211': 'Médicos generales y especialistas',
            '2311': 'Profesores de educación primaria',
            '5212': 'Peluqueros, barberos, estilistas y peinadores',
            '8112': 'Mecánicos en mantenimiento de vehículos',
            '3111': 'Secretarias',
            '7311': 'Carpinteros, ebanistas y afines',
            '7111': 'Electricistas y linieros',
            '6112': 'Trabajadores en el cultivo de hortalizas',
            '6114': 'Trabajadores en el cultivo de árboles frutales',
            '7211': 'Carniceros',
            '2331': 'Profesores de educación secundaria',
            '1121': 'Directores y gerentes en ventas',
            '2511': 'Desarrolladores y analistas de software',
            '2121': 'Administradores y especialistas en recursos humanos',
            '2221': 'Enfermeras',
            '7131': 'Plomeros, fontaneros e instaladores',
            '7411': 'Pintores de viviendas y edificios',
            '8351': 'Operadores de maquinaria pesada (construcción)',
            '9311': 'Barrenderos y trabajadores de limpieza pública',
            '1111': 'Presidentes y directores generales',
            '2141': 'Analistas financieros',
            '2321': 'Profesores de educación preescolar',
            '2341': 'Profesores universitarios',
            '2151': 'Mercadólogos y publicistas',
            '2521': 'Ingenieros en telecomunicaciones',
            '8131': 'Operadores de máquinas de coser',
            '7122': 'Techadores y colocadores de pisos',
            '5112': 'Cantineros y preparadores de bebidas',
            '2332': 'Profesores de enseñanza primaria',
            '2632': 'Mecánicos en mantenimiento y reparación de vehículos de motor',
            '9411': 'Ayudantes en la preparación de alimentos',
            '9231': 'Trabajadores de apoyo en elaboración y mantenimiento de equipos',
            '5211': 'Peluqueros, barberos, estilistas y peinadores',
            '8212': 'Ensambladores de partes eléctricas y electrónicas',
            '2135': 'Abogados',
            '8344': 'Conductores de motocicleta',
            '2512': 'Auxiliares en contabilidad, finanzas y agentes de bolsa',
            '8211': 'Ensambladores de maquinaria, equipos y productos metálicos',
            '7341': 'Sastres, modistos, costureras y confeccionadores',
            '2271': 'Desarrolladores y analistas de software y multimedia',
            '9236': 'Trabajadores de apoyo en la industria de alimentos y bebidas',
            '2436': 'Enfermeras y paramédicos profesionales',
            '8101': 'Supervisores de operadores de maquinaria industrial',
            '9331': 'Cargadores',
            '3211': 'Recepcionistas y trabajadores que brindan información',
            '4224': 'Vendedores por catálogo',
            '8133': 'Operadores de máquinas para productos de plástico y hule',
            '8153': 'Operadores de máquinas de costura, bordado y de corte',
            '4221': 'Agentes y representantes de ventas y consignatarios',
            '0nan': 'Ocupación no especificada / Nulos'
        }
        
        df_res.insert(1, 'Profesión', df_res['Código SINCO'].map(cat_map).fillna("Código " + df_res['Código SINCO'] + " (SINCO)"))

        # =========================================================================
        # 4. ORDEN FINAL Y REDONDEO
        # =========================================================================
        columnas_finales = [
            'Código SINCO',
            'Profesión',
            'Personas (Total Ocupados)',
            '% de la PEA Ocupada',
            '%Formal',
            '%Hombres',
            'Horas Mensuales Trabajadas',
            'Ingreso Mensual Promedio',
            'Ingreso x Hora (General)',
            'Ingreso x Hora (Hombres)',
            'Ingreso x Hora (Mujeres)',
            'Personas con ingresos positivos',
            'Personas con ingresos positivos como % de Personas de la PEA Ocupada'
        ]
        
        df_res = df_res[columnas_finales]

        # Redondear números a un decimal
        numeric_cols = df_res.select_dtypes(include=['float64']).columns
        df_res[numeric_cols] = df_res[numeric_cols].round(1)

        # Guardar en CSV
        output_filename = "Top_50_Profesiones_Final_Completo.csv"
        df_res.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        print("\n¡Éxito! El archivo CSV se ha generado con todas las columnas ordenadas.")
        print(f"Búscalo como: '{output_filename}' en tu carpeta.\n")
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == '__main__':
    procesar_top50_profesiones_completo()