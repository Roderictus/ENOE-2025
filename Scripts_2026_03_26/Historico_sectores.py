import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def procesar_sectores_economicos():
    # =========================================================================
    # 1. INPC HISTÓRICO (Base 2018=100) PARA DEFLACTAR INGRESOS
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
    print("Iniciando extracción de Sectores Económicos (2005-2025)...")
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
                
                # Universo: PEA Ocupada (Residentes, >=15 años, Ocupados)
                df_oc = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3])) & 
                           (df['eda'] >= 15) & (df['clase1'] == 1) & (df['clase2'] == 1)].copy()
                
                df_oc['ingocup'] = pd.to_numeric(df_oc.get('ingocup', 0), errors='coerce').fillna(0)
                
                # Clasificación de Sectores (Usando rama_est1 o rama_est2)
                if 'rama_est1' in df_oc.columns:
                    s = pd.to_numeric(df_oc['rama_est1'], errors='coerce').fillna(4)
                    conds = [s == 1, s == 2, s == 3]
                    choices = ['Primario', 'Secundario', 'Terciario']
                    df_oc['Sector'] = np.select(conds, choices, default='No Especificado')
                elif 'rama_est2' in df_oc.columns:
                    s = pd.to_numeric(df_oc['rama_est2'], errors='coerce').fillna(12)
                    conds = [s == 1, s.isin([2, 3, 4]), s.isin([5, 6, 7, 8, 9, 10, 11])]
                    choices = ['Primario', 'Secundario', 'Terciario']
                    df_oc['Sector'] = np.select(conds, choices, default='No Especificado')
                else:
                    df_oc['Sector'] = 'No Especificado'

                # Agregación Trimestral
                total_ocupados = df_oc[PONDERATOR].sum()
                deflactor = DEFLACTOR_DICT.get((year, quarter), 100)
                
                res_dict = {'Periodo': f"{year}-T{quarter}"}
                
                for sector in ['Primario', 'Secundario', 'Terciario', 'No Especificado']:
                    g_sec = df_oc[df_oc['Sector'] == sector]
                    
                    # 1. Población
                    pob = g_sec[PONDERATOR].sum()
                    
                    # 2. Porcentaje
                    pct = (pob / total_ocupados * 100) if total_ocupados > 0 else 0
                    
                    # 3. Ingreso Mensual Promedio Real (solo los que reportan ingresos > 0)
                    g_ing = g_sec[g_sec['ingocup'] > 0]
                    ing_nom = weighted_avg(g_ing, 'ingocup', PONDERATOR)
                    ing_real = (ing_nom / deflactor * 100) if pd.notna(ing_nom) else np.nan
                    
                    res_dict[f'Pob_{sector}'] = pob
                    res_dict[f'Pct_{sector}'] = pct
                    res_dict[f'Ingreso_{sector}'] = ing_real
                    
                resultados.append(res_dict)
                
            except Exception as e:
                print(f"Error procesando {year}-T{quarter}: {e}")

    df_res = pd.DataFrame(resultados)

    # =========================================================================
    # 3. GENERACIÓN DE GRÁFICAS
    # =========================================================================
    print("Generando gráficas de los sectores económicos...")
    sns.set_theme(style="whitegrid")
    
    # Paleta de colores temáticos
    c_prim = '#27ae60' # Verde (Agricultura/Naturaleza)
    c_sec = '#e67e22'  # Naranja (Industria/Construcción)
    c_ter = '#2980b9'  # Azul (Servicios/Comercio)
    
    ticks_to_show = [p for p in df_res['Periodo'] if '-T1' in p or '-T3' in p]

    # --- Gráfica 1: Población Ocupada Absoluta ---
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    ax1.plot(df_res['Periodo'], df_res['Pob_Terciario'], color=c_ter, linewidth=3.5, label='Sector Terciario (Servicios y Comercio)')
    ax1.plot(df_res['Periodo'], df_res['Pob_Secundario'], color=c_sec, linewidth=3.5, label='Sector Secundario (Industria y Construcción)')
    ax1.plot(df_res['Periodo'], df_res['Pob_Primario'], color=c_prim, linewidth=3.5, label='Sector Primario (Agricultura y Ganadería)')
    
    ax1.set_title('Evolución de la Población Ocupada por Sector Económico (2005 - 2025)', fontsize=16, pad=15)
    ax1.set_ylabel('Millones de Personas Ocupadas', fontsize=13)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e6:.0f}M"))
    ax1.set_xticks(ticks_to_show)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax1.text(0.99, 0.01, "www.mexicoendatos.org", transform=ax1.transAxes, ha='right', fontsize=12, color='black', alpha=0.6)
    plt.tight_layout()
    plt.savefig('10_Sector_Poblacion.png', dpi=300)
    plt.close(fig1)

    # --- Gráfica 2: Composición Porcentual ---
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    ax2.plot(df_res['Periodo'], df_res['Pct_Terciario'], color=c_ter, linewidth=3.5, label='Sector Terciario')
    ax2.plot(df_res['Periodo'], df_res['Pct_Secundario'], color=c_sec, linewidth=3.5, label='Sector Secundario')
    ax2.plot(df_res['Periodo'], df_res['Pct_Primario'], color=c_prim, linewidth=3.5, label='Sector Primario')
    
    ax2.set_title('Composición de la Fuerza Laboral (% de la PEA Ocupada)', fontsize=16, pad=15)
    ax2.set_ylabel('Porcentaje de Participación (%)', fontsize=13)
    ax2.set_ylim(0, 70)
    ax2.set_xticks(ticks_to_show)
    ax2.tick_params(axis='x', rotation=45)
    
    # Etiquetas de final de línea
    ax2.text(len(df_res)-1, df_res['Pct_Terciario'].iloc[-1], f" {df_res['Pct_Terciario'].iloc[-1]:.1f}%", color=c_ter, fontweight='bold', fontsize=12)
    ax2.text(len(df_res)-1, df_res['Pct_Secundario'].iloc[-1], f" {df_res['Pct_Secundario'].iloc[-1]:.1f}%", color=c_sec, fontweight='bold', fontsize=12)
    ax2.text(len(df_res)-1, df_res['Pct_Primario'].iloc[-1], f" {df_res['Pct_Primario'].iloc[-1]:.1f}%", color=c_prim, fontweight='bold', fontsize=12)
    
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax2.text(0.99, 0.01, "www.mexicoendatos.org", transform=ax2.transAxes, ha='right', fontsize=12, color='black', alpha=0.6)
    plt.tight_layout()
    plt.savefig('11_Sector_Porcentajes.png', dpi=300)
    plt.close(fig2)

    # --- Gráfica 3: Ingreso Mensual Promedio ---
    fig3, ax3 = plt.subplots(figsize=(15, 8))
    ax3.plot(df_res['Periodo'], df_res['Ingreso_Terciario'], color=c_ter, linewidth=3.5, label='Sector Terciario')
    ax3.plot(df_res['Periodo'], df_res['Ingreso_Secundario'], color=c_sec, linewidth=3.5, label='Sector Secundario')
    ax3.plot(df_res['Periodo'], df_res['Ingreso_Primario'], color=c_prim, linewidth=3.5, label='Sector Primario')
    
    ax3.set_title('Brecha Salarial por Sector (Pesos Reales Base 2018)\nPromedio mensual de quienes reportan ingresos > $0', fontsize=16, pad=15)
    ax3.set_ylabel('Ingreso Mensual Real ($ MXN)', fontsize=13)
    ax3.set_xticks(ticks_to_show)
    ax3.tick_params(axis='x', rotation=45)
    
    ax3.text(len(df_res)-1, df_res['Ingreso_Terciario'].iloc[-1], f" ${df_res['Ingreso_Terciario'].iloc[-1]:,.0f}", color=c_ter, va='bottom', fontweight='bold', fontsize=12)
    ax3.text(len(df_res)-1, df_res['Ingreso_Secundario'].iloc[-1], f" ${df_res['Ingreso_Secundario'].iloc[-1]:,.0f}", color=c_sec, va='top', fontweight='bold', fontsize=12)
    ax3.text(len(df_res)-1, df_res['Ingreso_Primario'].iloc[-1], f" ${df_res['Ingreso_Primario'].iloc[-1]:,.0f}", color=c_prim, va='center', fontweight='bold', fontsize=12)
    
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax3.text(0.99, 0.01, "www.mexicoendatos.org", transform=ax3.transAxes, ha='right', fontsize=12, color='black', alpha=0.6)
    plt.tight_layout()
    plt.savefig('12_Sector_Ingresos.png', dpi=300)
    plt.close(fig3)

    # =========================================================================
    # 4. EXPORTACIÓN A CSV (Formato LATAM)
    # =========================================================================
    df_out = df_res.copy()
    
    # Enteros
    int_cols = [c for c in df_out.columns if 'Pob_' in c or 'Ingreso_' in c]
    for col in int_cols:
        df_out[col] = df_out[col].fillna(0).round(0).astype(int).astype(str)
        
    # Decimales (%)
    pct_cols = [c for c in df_out.columns if 'Pct_' in c]
    for col in pct_cols:
        df_out[col] = df_out[col].fillna(0).round(1).astype(str).str.replace('.', ',')

    csv_filename = 'Evolucion_Sectores_2005_2025_LATAM.csv'
    df_out.to_csv(csv_filename, sep=';', index=False, encoding='utf-8-sig')
    
    print(f"\n¡Análisis de Sectores Finalizado!")
    print(f"Base de datos exportada a: {csv_filename}")
    print("Gráficas generadas: 10_Sector_Poblacion.png, 11_Sector_Porcentajes.png, 12_Sector_Ingresos.png")

if __name__ == '__main__':
    procesar_sectores_economicos()