import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def procesar_sectores_y_ramas_economicas():
    # =========================================================================
    # 0. CONFIGURACIÓN DE DIRECTORIOS Y PREFIJOS
    # =========================================================================
    dir_graficas = "Resultados_Graficas"
    dir_bases = "Resultados Bases"
    
    os.makedirs(dir_graficas, exist_ok=True)
    os.makedirs(dir_bases, exist_ok=True)
    
    fecha_prefix = datetime.now().strftime("%Y%m%d_")

    # =========================================================================
    # 1. INPC HISTÓRICO (Base 2018) Y DICCIONARIOS
    # =========================================================================
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

    dict_ramas = {
        1: 'Agricultura y ganadería',
        2: 'Industria extractiva y electricidad',
        3: 'Industria manufacturera',
        4: 'Construcción',
        5: 'Comercio',
        6: 'Restaurantes y alojamiento',
        7: 'Transportes y comunicaciones',
        8: 'Servicios profesionales y corporativos',
        9: 'Servicios sociales',
        10: 'Servicios diversos',
        11: 'Gobierno y org. internacionales'
    }

    def get_sdemt_path(year, quarter):
        y_short = str(year)[-2:]
        base_dir = f"Data/ENOE_dta/ENOE_{year}_{quarter}"
        candidates = [
            f"ENOE_SDEMT{quarter}{y_short}.dta", f"ENOEN_SDEMT{quarter}{y_short}.dta",
            f"SDEMT{quarter}{y_short}.dta", f"enoe_sdemt{quarter}{y_short}.dta"
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
    # 2. EXTRACCIÓN Y AGREGACIÓN HISTÓRICA (2005-2025)
    # =========================================================================
    print("Iniciando extracción de Sectores y Ramas Económicas...")
    resultados = []

    for year in range(2005, 2026):
        for quarter in range(1, 5):
            if year == 2020 and quarter == 2: continue # Pandemia
            
            p_sdemt = get_sdemt_path(year, quarter)
            if not p_sdemt: continue
                
            try:
                print(f"  Procesando {year}-T{quarter}...")
                df = pd.read_stata(p_sdemt, convert_categoricals=False)
                df.columns = df.columns.str.lower()
                
                PONDERATOR = 'fac_tri' if 'fac_tri' in df.columns else 'fac'
                df[PONDERATOR] = pd.to_numeric(df[PONDERATOR], errors='coerce').fillna(0)
                df['r_def'] = df['r_def'].astype(str).str.strip()
                
                # Universo PEA Ocupada
                df_oc = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3])) & 
                           (df['eda'] >= 15) & (df['clase1'] == 1) & (df['clase2'] == 1)].copy()
                
                df_oc['ingocup'] = pd.to_numeric(df_oc.get('ingocup', 0), errors='coerce').fillna(0)
                
                # 2.1 Mapeo de Sector (3 niveles)
                s = pd.to_numeric(df_oc.get('rama_est2', 12), errors='coerce').fillna(12)
                conds = [s == 1, s.isin([2, 3, 4]), s.isin([5, 6, 7, 8, 9, 10, 11])]
                choices = ['Primario', 'Secundario', 'Terciario']
                df_oc['Sector'] = np.select(conds, choices, default='No Especificado')
                
                # 2.2 Mapeo de Rama (11 niveles)
                df_oc['Rama'] = s.map(dict_ramas).fillna('No Especificado')
                
                total_ocupados = df_oc[PONDERATOR].sum()
                deflactor = DEFLACTOR_DICT.get((year, quarter), 100)
                periodo_str = f"{year}-T{quarter}"
                
                # Agregación estructurada (Tidy Data)
                def extraer_metricas(df_grupo, nombre_categoria, nivel):
                    pob = df_grupo[PONDERATOR].sum()
                    pct = (pob / total_ocupados * 100) if total_ocupados > 0 else 0
                    
                    df_ing = df_grupo[df_grupo['ingocup'] > 0]
                    ing_nom = weighted_avg(df_ing, 'ingocup', PONDERATOR)
                    ing_real = (ing_nom / deflactor * 100) if pd.notna(ing_nom) else np.nan
                    
                    if pob > 0:
                        resultados.append({
                            'Periodo': periodo_str,
                            'Nivel_Clasificacion': nivel,
                            'Categoria': nombre_categoria,
                            'Poblacion_Ocupada': pob,
                            'Porcentaje_PEA': pct,
                            'Ingreso_Mensual_Real': ing_real
                        })

                # Extraer Sectores
                for sector in ['Primario', 'Secundario', 'Terciario']:
                    extraer_metricas(df_oc[df_oc['Sector'] == sector], sector, 'Sector')
                
                # Extraer Ramas
                for rama in dict_ramas.values():
                    extraer_metricas(df_oc[df_oc['Rama'] == rama], rama, 'Rama')

            except Exception as e:
                print(f"Error procesando {year}-T{quarter}: {e}")

    df_res = pd.DataFrame(resultados)

    # =========================================================================
    # 3. FUNCIÓN GENERADORA DE GRÁFICAS (CON REGLAS ESTRICTAS DE FORMATO)
    # =========================================================================
    print("Generando gráficas de Sectores y Ramas...")
    sns.set_theme(style="whitegrid")
    ticks_to_show = [p for p in df_res['Periodo'].unique() if '-T1' in p or '-T3' in p]

    def graficar_metrica(df_plot, metric_col, titulo, ylabel, is_pct, is_currency, archivo_salida):
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Paleta de colores (usamos husl para que las 11 ramas se distingan bien)
        categorias = df_plot['Categoria'].unique()
        colores = sns.color_palette("husl", len(categorias))
        
        for i, cat in enumerate(categorias):
            df_cat = df_plot[df_plot['Categoria'] == cat].sort_values('Periodo')
            ax.plot(df_cat['Periodo'], df_cat[metric_col], label=cat, linewidth=3, color=colores[i])

            # Rótulos en el primer y último periodo
            if not df_cat.empty:
                val_first = df_cat[metric_col].iloc[0]
                val_last = df_cat[metric_col].iloc[-1]
                
                # Reglas de decimales según el tipo de métrica
                if is_pct:
                    fmt = "{:.1f}%"
                elif is_currency:
                    fmt = "${:,.0f}"
                else: # Población
                    fmt = "{:,.0f}" if val_first < 1000 else "{:,.0f}" 

                ax.text(0, val_first, fmt.format(val_first) + " ", ha='right', va='center', 
                        fontsize=9, fontweight='bold', color=colores[i])
                ax.text(len(df_cat)-1, val_last, " " + fmt.format(val_last), ha='left', va='center', 
                        fontsize=9, fontweight='bold', color=colores[i])

        ax.set_title(titulo, fontsize=17, pad=20, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xticks(ticks_to_show)
        ax.tick_params(axis='x', rotation=45)
        
        # Formato del eje Y para población (Millones)
        if not is_pct and not is_currency:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e6:.0f}M"))
            
        # Leyenda de Pesos Constantes si aplica
        if is_currency:
            ax.text(0.01, 0.02, "Valores expresados en Pesos de 2018", transform=ax.transAxes, 
                    ha='left', fontsize=11, style='italic', color='#555555')
            
        # Marca de agua de la plataforma
        ax.text(0.99, 0.01, "www.mexicoendatos.org", transform=ax.transAxes, 
                ha='right', fontsize=12, color='black', alpha=0.5)

        # Ubicar leyenda fuera del gráfico para que no tape las 11 líneas
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=11)
        
        # Ajustar márgenes para que quepan los rótulos y la leyenda
        ax.set_xlim(-1, len(df_cat) + 1)
        plt.tight_layout()
        
        # Guardar en el subdirectorio con el prefijo de fecha
        ruta_completa = os.path.join(dir_graficas, f"{fecha_prefix}{archivo_salida}")
        plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Separar DataFrames por Nivel
    df_sectores = df_res[df_res['Nivel_Clasificacion'] == 'Sector'].copy()
    df_ramas = df_res[df_res['Nivel_Clasificacion'] == 'Rama'].copy()

    # --- 3 GRÁFICAS DE SECTORES ---
    graficar_metrica(df_sectores, 'Poblacion_Ocupada', 'Población Ocupada por Sector Económico (2005-2025)', 'Millones de Personas', False, False, "Sectores_1_Poblacion.png")
    graficar_metrica(df_sectores, 'Porcentaje_PEA', 'Composición de la Fuerza Laboral por Sector', 'Porcentaje de la PEA Ocupada (%)', True, False, "Sectores_2_Porcentajes.png")
    graficar_metrica(df_sectores, 'Ingreso_Mensual_Real', 'Ingreso Promedio por Sector', 'Ingreso Mensual ($ MXN)', False, True, "Sectores_3_Ingresos.png")

    # --- 3 GRÁFICAS DE RAMAS ---
    graficar_metrica(df_ramas, 'Poblacion_Ocupada', 'Población Ocupada por Rama de Actividad (2005-2025)', 'Millones de Personas', False, False, "Ramas_1_Poblacion.png")
    graficar_metrica(df_ramas, 'Porcentaje_PEA', 'Composición de la Fuerza Laboral por Rama de Actividad', 'Porcentaje de la PEA Ocupada (%)', True, False, "Ramas_2_Porcentajes.png")
    graficar_metrica(df_ramas, 'Ingreso_Mensual_Real', 'Ingreso Promedio por Rama de Actividad', 'Ingreso Mensual ($ MXN)', False, True, "Ramas_3_Ingresos.png")

    # =========================================================================
    # 4. EXPORTACIÓN DE BASES A CSV (Formato LATAM)
    # =========================================================================
    # Pivoteamos la tabla para que sea más fácil de leer en Excel/Sheets
    df_export = df_res.copy()
    
    # Formateo de números según tus reglas
    df_export['Poblacion_Ocupada'] = df_export['Poblacion_Ocupada'].fillna(0).round(0).astype(int).astype(str)
    df_export['Ingreso_Mensual_Real'] = df_export['Ingreso_Mensual_Real'].fillna(0).round(0).astype(int).astype(str)
    df_export['Porcentaje_PEA'] = df_export['Porcentaje_PEA'].fillna(0).round(1).astype(str).str.replace('.', ',')

    # Guardar en su respectivo subdirectorio con prefijo
    ruta_csv = os.path.join(dir_bases, f"{fecha_prefix}Evolucion_Sectores_y_Ramas_LATAM.csv")
    df_export.to_csv(ruta_csv, sep=';', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print(" ¡PROCESO DE SECTORES Y RAMAS FINALIZADO!")
    print(f" -> Las 6 gráficas se guardaron en: ./{dir_graficas}/")
    print(f" -> La base de datos se guardó en: ./{dir_bases}/")
    print("="*70)

if __name__ == '__main__':
    procesar_sectores_y_ramas_economicas()