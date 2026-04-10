import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================
# 1. CONFIGURACIÓN Y DICCIONARIOS
# =========================================================================
sns.set_theme(style="whitegrid")
WATERMARK = "www.mexicoendatos.org"

# INPC Histórico (Base 2018=100) para deflactar la serie
inpc_historico = [
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
    138.743, 140.012, 140.948, 142.465, 144.150
]

DEFLACTOR_DICT = {}
idx = 0
for y in range(2005, 2027):
    for q in range(1, 5):
        if idx < len(inpc_historico):
            DEFLACTOR_DICT[(y, q)] = inpc_historico[idx]
            idx += 1

# Catálogo a 2 dígitos resumido para gráficas limpias
cat_map_2d = {
    '11': 'Funcionarios y directores', '12': 'Coordinadores y jefes', '13': 'Directores (Producción/Transporte)',
    '14': 'Directores (Ventas/Hoteles)', '15': 'Jefes (Serv. Financieros)', '16': 'Jefes (Producción)',
    '17': 'Jefes (Ventas/Hoteles)', '19': 'Otros directores y jefes', '21': 'Especialistas Económico-Admin',
    '22': 'Ingenieros y Tecnólogos', '23': 'Profesores y Docentes', '24': 'Médicos y Especialistas Salud',
    '25': 'Especialistas en TI', '26': 'Otros profesionistas', '27': 'Técnicos en educación', '28': 'Enfermeras y Técnicos en Salud',
    '29': 'Otros profesionistas técnicos', '31': 'Apoyo Administrativo', '32': 'Atención al público',
    '33': 'Auxiliares contables', '39': 'Otros auxiliares admin', '41': 'Comerciantes en establecimientos',
    '42': 'Empleados de ventas', '43': 'Agentes de ventas', '49': 'Otros comerciantes',
    '51': 'Preparación de alimentos', '52': 'Cuidados y belleza', '53': 'Protección y seguridad',
    '54': 'Cuidado de personas', '61': 'Actividades agrícolas', '62': 'Actividades ganaderas', '63': 'Silvicultura y pesca',
    '69': 'Otros trabajadores agrícolas', '71': 'Construcción y Albañilería', '72': 'Metalmecánica',
    '73': 'Madera y papel', '74': 'Textil y cuero', '75': 'Elaboración de alimentos',
    '76': 'Operadores de plantas', '79': 'Otros artesanales', '81': 'Operadores maquinaria industrial',
    '82': 'Ensambladores', '83': 'Conductores de transporte', '89': 'Otros operadores',
    '91': 'Apoyo agropecuario', '92': 'Apoyo en construcción', '93': 'Apoyo en industria',
    '94': 'Ayudantes en alimentos', '95': 'Vendedores ambulantes', '96': 'Trabajadores domésticos',
    '97': 'Apoyo diversos', '98': 'No especificadas', '99': 'No clasificadas', '00': 'Nulos'
}

def get_file_path(year, quarter, prefix):
    y_short = str(year)[-2:]
    base_dir = f"Data/ENOE_dta/ENOE_{year}_{quarter}"
    candidates = [
        f"ENOE_{prefix}{quarter}{y_short}.dta", f"ENOEN_{prefix}{quarter}{y_short}.dta",
        f"{prefix}{quarter}{y_short}.dta", f"enoe_{prefix.lower()}{quarter}{y_short}.dta"
    ]
    for c in candidates:
        p = os.path.join(base_dir, c)
        if os.path.exists(p): return p
    return None

def weighted_avg(df_filtered, val_col, wt_col):
    if df_filtered.empty: return np.nan
    val, wt = df_filtered[val_col].values, df_filtered[wt_col].values
    valid = ~np.isnan(val) & ~np.isnan(wt)
    if not valid.any() or wt[valid].sum() == 0: return np.nan
    return np.average(val[valid], weights=wt[valid])

# =========================================================================
# 2. EXTRACCIÓN HISTÓRICA (A PARTIR DE LA IMPLEMENTACIÓN DEL SINCO)
# =========================================================================
print("Iniciando extracción histórica desde 2012-T3 (Era SINCO)...")
resultados_historicos = []

for year in range(2012, 2026):
    for quarter in range(1, 5):
        # 1. Ignorar datos anteriores a la adopción del SINCO
        if year == 2012 and quarter < 3: continue 
        # 2. Ignorar el vacío del confinamiento COVID-19
        if year == 2020 and quarter == 2: continue 
        
        p_sdemt = get_file_path(year, quarter, 'SDEMT')
        p_coe1t = get_file_path(year, quarter, 'COE1T')
        
        if not p_sdemt or not p_coe1t:
            print(f"  [Aviso] No se encontró la base {year}-T{quarter}")
            continue
            
        print(f"  Procesando {year}-T{quarter}...")
        try:
            df_sdemt = pd.read_stata(p_sdemt, convert_categoricals=False)
            df_coe1t = pd.read_stata(p_coe1t, convert_categoricals=False)
            
            df_sdemt.columns = df_sdemt.columns.str.lower()
            df_coe1t.columns = df_coe1t.columns.str.lower()
            
            if 'ent' in df_sdemt.columns and 'cve_ent' not in df_sdemt.columns: df_sdemt.rename(columns={'ent': 'cve_ent'}, inplace=True)
            if 'ent' in df_coe1t.columns and 'cve_ent' not in df_coe1t.columns: df_coe1t.rename(columns={'ent': 'cve_ent'}, inplace=True)

            PONDERATOR = 'fac_tri' if 'fac_tri' in df_sdemt.columns else 'fac'
            col_sinco = 'sinco' if 'sinco' in df_coe1t.columns else 'p3'
            llaves_merge = [k for k in ['cd_a', 'cve_ent', 'con', 'v_sel', 'n_hog', 'h_mud', 'n_ren'] if k in df_sdemt.columns]

            df_sdemt = df_sdemt[llaves_merge + [PONDERATOR, 'r_def', 'c_res', 'eda', 'clase1', 'clase2', 'ingocup', 'sex', 'hrsocup', 'emp_ppal']]
            df_coe1t = df_coe1t[llaves_merge + [col_sinco]]
            
            df_merge = pd.merge(df_sdemt, df_coe1t, on=llaves_merge, how='inner')
            df_merge[PONDERATOR] = pd.to_numeric(df_merge[PONDERATOR], errors='coerce').fillna(0)
            df_merge['r_def'] = df_merge['r_def'].astype(str).str.strip()
            
            # Filtro PEA Ocupada
            df_oc = df_merge[(df_merge['r_def'] == '0.0') & (df_merge['c_res'].isin([1, 3])) & 
                             (df_merge['eda'] >= 15) & (df_merge['clase1'] == 1) & (df_merge['clase2'] == 1)].copy()
                             
            def clean_2d(x):
                if pd.isna(x): return '00'
                x = str(x).strip()
                if x == '0nan' or x.lower() == 'nan': return '00'
                try: return str(int(float(x))).zfill(4)[:2]
                except: return '00'

            df_oc['cod_2d'] = df_oc[col_sinco].apply(clean_2d)
            
            df_oc['ingocup'] = pd.to_numeric(df_oc['ingocup'], errors='coerce').fillna(0)
            df_oc['hrsocup'] = pd.to_numeric(df_oc['hrsocup'], errors='coerce').fillna(0)
            
            df_ing = df_oc[df_oc['ingocup'] > 0].copy()
            df_ing['horas_mes'] = df_ing['hrsocup'] * 4.345
            df_ing_hrs = df_ing[df_ing['horas_mes'] > 0].copy()
            df_ing_hrs['ing_hora'] = df_ing_hrs['ingocup'] / df_ing_hrs['horas_mes']
            
            deflactor = DEFLACTOR_DICT.get((year, quarter), 100)

            for cod in df_oc['cod_2d'].unique():
                g_tot = df_oc[df_oc['cod_2d'] == cod]
                g_ing = df_ing[df_ing['cod_2d'] == cod]
                g_hrs = df_ing_hrs[df_ing_hrs['cod_2d'] == cod]
                
                personas = g_tot[PONDERATOR].sum()
                if personas == 0: continue
                
                formales = g_tot[g_tot['emp_ppal'] == 2][PONDERATOR].sum()
                informales = g_tot[g_tot['emp_ppal'] == 1][PONDERATOR].sum()
                hombres = g_tot[g_tot['sex'] == 1][PONDERATOR].sum()
                mujeres = g_tot[g_tot['sex'] == 2][PONDERATOR].sum()
                
                # Calcular Ratios Puros (X por cada 1)
                ratio_formal = (formales / informales) if informales > 0 else np.nan
                ratio_genero = (hombres / mujeres) if mujeres > 0 else np.nan
                
                ing_mensual_nom = weighted_avg(g_ing, 'ingocup', PONDERATOR)
                ing_hora_nom = weighted_avg(g_hrs, 'ing_hora', PONDERATOR)
                hrs_mes = weighted_avg(g_hrs, 'horas_mes', PONDERATOR)
                
                resultados_historicos.append({
                    'year': year, 'quarter': quarter, 'Periodo': f"{year}-T{quarter}",
                    'Codigo': cod, 'Personas': personas,
                    'Ratio_Formal': ratio_formal, 'Ratio_HombreMujer': ratio_genero,
                    'Horas_Mensuales': hrs_mes,
                    'Ing_Mensual_Real': (ing_mensual_nom / deflactor) * 100 if pd.notna(ing_mensual_nom) else np.nan,
                    'Ing_Hora_Real': (ing_hora_nom / deflactor) * 100 if pd.notna(ing_hora_nom) else np.nan
                })
                
            del df_sdemt, df_coe1t, df_merge, df_oc, df_ing, df_ing_hrs
            gc.collect() # Liberar memoria ram
            
        except Exception as e:
            print(f"Error procesando {year}-T{quarter}: {e}")

# =========================================================================
# 3. PROCESAMIENTO Y TOP 10 HISTÓRICO
# =========================================================================
df_hist = pd.DataFrame(resultados_historicos)

# Encontrar el Top 10 histórico que agrupa a más mexicanos
top10_codigos = df_hist.groupby('Codigo')['Personas'].sum().sort_values(ascending=False).head(10).index.tolist()
df_top10 = df_hist[df_hist['Codigo'].isin(top10_codigos)].copy()

# Mapear nombres para las leyendas
df_top10['Grupo Ocupacional'] = df_top10['Codigo'].map(cat_map_2d).fillna("Cód: " + df_top10['Codigo'])

# =========================================================================
# 4. GENERACIÓN DE LAS 5 GRÁFICAS SOLICITADAS
# =========================================================================
print("\nBase de datos histórica consolidada. Generando las gráficas...")

graficas_config = [
    ('Ing_Hora_Real', 'Evolución del Ingreso Promedio por Hora (MXN Reales)', 'Ingreso Real por Hora (Base 2018)', '01_ingreso_hora_top10.png'),
    ('Ing_Mensual_Real', 'Evolución del Ingreso Mensual Promedio (MXN Reales)', 'Ingreso Mensual Real (Base 2018)', '02_ingreso_mensual_top10.png'),
    ('Horas_Mensuales', 'Evolución de la Carga Laboral Promedio', 'Horas Trabajadas al Mes', '03_horas_trabajadas_top10.png'),
    ('Ratio_Formal', 'Evolución de la Formalidad (Ratio)\nInterpretación: Por cada 1 trabajador informal, existen "X" formales', 'Ratio (Formales / Informales)', '04_ratio_formalidad_top10.png'),
    ('Ratio_HombreMujer', 'Brecha de Género Ocupacional (Ratio)\nInterpretación: Por cada 1 mujer, existen "X" hombres trabajando', 'Ratio (Hombres / Mujeres)', '05_ratio_genero_top10.png')
]

# Definir una paleta de colores contrastantes para las 10 líneas
colores = sns.color_palette("tab10", n_colors=10)

for metrica, titulo, ylabel, filename in graficas_config:
    plt.figure(figsize=(15, 8))
    
    # Dibujar líneas
    sns.lineplot(data=df_top10, x='Periodo', y=metrica, hue='Grupo Ocupacional', palette=colores, linewidth=2.5)
    
    plt.title(titulo + '\n(Top 10 Grupos Profesionales SINCO, ENOE 2012-2025)', fontsize=15, pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Año y Trimestre', fontsize=12)
    
    # Limpiar el eje X para que no se sature (Mostrar solo los Trimestres 1 y 3)
    periodos_unicos = sorted(df_top10['Periodo'].unique())
    ticks_to_show = [p for p in periodos_unicos if ('-T1' in p or '-T3' in p)]
    plt.xticks(ticks_to_show, rotation=45)
    
    # Colocar leyenda fuera de la gráfica para no tapar los datos
    plt.legend(title='Grupo Profesional (2 Dígitos)', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    
    # Marca de agua
    plt.text(0.99, 0.01, WATERMARK, transform=plt.gca().transAxes, 
             ha='right', va='bottom', fontsize=12, color='black', alpha=0.6)
             
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Guardar la base de datos subyacente para validaciones
df_top10.to_csv("Evolucion_Top10_SINCO_2012_2025.csv", index=False, encoding='utf-8-sig')

print("¡Proceso finalizado! Revisa tu carpeta local, las 5 gráficas y el CSV han sido exportados con éxito.")