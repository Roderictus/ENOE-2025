import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def procesar_brechas_ingresos_horas():
    # =========================================================================
    # 1. INPC HISTÓRICO (Base 2018=100) PARA DEFLACTAR
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
    # 2. EXTRACCIÓN HISTÓRICA DE INGRESOS Y HORAS
    # =========================================================================
    print("Iniciando extracción de horas y salarios por género (2005-2025)...")
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
                
                # Filtro Ocupados y Residentes definitivos
                df_oc = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3])) & 
                           (df['eda'] >= 15) & (df['clase1'] == 1) & (df['clase2'] == 1)].copy()
                
                df_oc['hrsocup'] = pd.to_numeric(df_oc.get('hrsocup', 0), errors='coerce').fillna(0)
                df_oc['ingocup'] = pd.to_numeric(df_oc.get('ingocup', 0), errors='coerce').fillna(0)
                
                # Universo: Tienen ingresos y horas trabajadas registradas
                df_valido = df_oc[(df_oc['hrsocup'] > 0) & (df_oc['ingocup'] > 0)].copy()
                
                # Convertir ingreso mensual a ingreso por hora
                df_valido['ingreso_hora'] = df_valido['ingocup'] / (df_valido['hrsocup'] * 4.345)
                
                deflactor = DEFLACTOR_DICT.get((year, quarter), 100)
                res_dict = {'Periodo': f"{year}-T{quarter}"}
                
                for sex_code, sex_label in [(1, 'Hombres'), (2, 'Mujeres')]:
                    df_sex = df_valido[df_valido['sex'] == sex_code]
                    
                    # Promedios ponderados
                    hrs_sem_nom = weighted_avg(df_sex, 'hrsocup', PONDERATOR)
                    ing_men_nom = weighted_avg(df_sex, 'ingocup', PONDERATOR)
                    ing_hor_nom = weighted_avg(df_sex, 'ingreso_hora', PONDERATOR)
                    
                    # Deflactar ingresos
                    res_dict[f'Horas_Semanales_{sex_label}'] = hrs_sem_nom
                    res_dict[f'Ingreso_Mensual_Real_{sex_label}'] = (ing_men_nom / deflactor) * 100 if pd.notna(ing_men_nom) else np.nan
                    res_dict[f'Ingreso_Hora_Real_{sex_label}'] = (ing_hor_nom / deflactor) * 100 if pd.notna(ing_hor_nom) else np.nan
                    
                resultados.append(res_dict)
                
            except Exception as e:
                print(f"Error procesando {year}-T{quarter}: {e}")

    df_res = pd.DataFrame(resultados)

    # =========================================================================
    # 3. GENERACIÓN DE GRÁFICAS
    # =========================================================================
    print("Generando gráficas...")
    sns.set_theme(style="whitegrid")
    color_h = '#1f77b4'
    color_m = '#2ca02c'
    ticks_to_show = [p for p in df_res['Periodo'] if '-T1' in p or '-T3' in p]

    # Gráfica 1: Horas Semanales
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    ax1.plot(df_res['Periodo'], df_res['Horas_Semanales_Hombres'], color=color_h, linewidth=3.5, label='Hombres')
    ax1.plot(df_res['Periodo'], df_res['Horas_Semanales_Mujeres'], color=color_m, linewidth=3.5, label='Mujeres')
    ax1.set_title('Brecha de Tiempo: Promedio de Horas Semanales Trabajadas (2005 - 2025)\n(Solo en el trabajo remunerado)', fontsize=16, pad=15)
    ax1.set_ylabel('Horas a la Semana', fontsize=13)
    ax1.set_ylim(35, 50) # Escala para notar la diferencia
    ax1.set_xticks(ticks_to_show)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='lower right', fontsize=12)
    ax1.text(0.99, 0.01, "www.mexicoendatos.org", transform=ax1.transAxes, ha='right', fontsize=12, color='black', alpha=0.6)
    
    u_h = df_res['Horas_Semanales_Hombres'].iloc[-1]
    u_m = df_res['Horas_Semanales_Mujeres'].iloc[-1]
    ax1.text(len(df_res)-1, u_h, f' {u_h:.1f} hrs', color=color_h, va='bottom', fontweight='bold', fontsize=12)
    ax1.text(len(df_res)-1, u_m, f' {u_m:.1f} hrs', color=color_m, va='top', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('07_brecha_horas_semanales.png', dpi=300)
    plt.close(fig1)

    # Gráfica 2: Ingreso Mensual Real
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    ax2.plot(df_res['Periodo'], df_res['Ingreso_Mensual_Real_Hombres'], color=color_h, linewidth=3.5, label='Hombres')
    ax2.plot(df_res['Periodo'], df_res['Ingreso_Mensual_Real_Mujeres'], color=color_m, linewidth=3.5, label='Mujeres')
    ax2.set_title('Brecha Salarial Mensual (Pesos Reales Base 2018)\n¿Cuánto llevan a casa a fin de mes?', fontsize=16, pad=15)
    ax2.set_ylabel('Ingreso Promedio Mensual ($)', fontsize=13)
    ax2.set_xticks(ticks_to_show)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(loc='lower right', fontsize=12)
    ax2.text(0.99, 0.01, "www.mexicoendatos.org", transform=ax2.transAxes, ha='right', fontsize=12, color='black', alpha=0.6)
    
    u_hm = df_res['Ingreso_Mensual_Real_Hombres'].iloc[-1]
    u_mm = df_res['Ingreso_Mensual_Real_Mujeres'].iloc[-1]
    ax2.text(len(df_res)-1, u_hm, f' ${u_hm:,.0f}', color=color_h, va='bottom', fontweight='bold', fontsize=12)
    ax2.text(len(df_res)-1, u_mm, f' ${u_mm:,.0f}', color=color_m, va='top', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('08_brecha_ingreso_mensual.png', dpi=300)
    plt.close(fig2)

    # Gráfica 3: Ingreso por Hora Real
    fig3, ax3 = plt.subplots(figsize=(15, 8))
    ax3.plot(df_res['Periodo'], df_res['Ingreso_Hora_Real_Hombres'], color=color_h, linewidth=3.5, label='Hombres')
    ax3.plot(df_res['Periodo'], df_res['Ingreso_Hora_Real_Mujeres'], color=color_m, linewidth=3.5, label='Mujeres')
    ax3.set_title('Valoración Real del Trabajo: Ingreso por Hora (Base 2018)\nEliminando el efecto de la jornada laboral prolongada', fontsize=16, pad=15)
    ax3.set_ylabel('Ingreso Promedio por Hora ($)', fontsize=13)
    ax3.set_xticks(ticks_to_show)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(loc='lower right', fontsize=12)
    ax3.text(0.99, 0.01, "www.mexicoendatos.org", transform=ax3.transAxes, ha='right', fontsize=12, color='black', alpha=0.6)
    
    u_hh = df_res['Ingreso_Hora_Real_Hombres'].iloc[-1]
    u_mh = df_res['Ingreso_Hora_Real_Mujeres'].iloc[-1]
    ax3.text(len(df_res)-1, u_hh, f' ${u_hh:,.1f}', color=color_h, va='bottom', fontweight='bold', fontsize=12)
    ax3.text(len(df_res)-1, u_mh, f' ${u_mh:,.1f}', color=color_m, va='top', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('09_brecha_ingreso_hora.png', dpi=300)
    plt.close(fig3)

    # =========================================================================
    # 4. EXPORTACIÓN DEL CSV (Formato LATAM)
    # =========================================================================
    for col in df_res.columns:
        if 'Ingreso_Mensual' in col:
            df_res[col] = df_res[col].round(0).astype(int).astype(str)
        elif 'Horas' in col or 'Ingreso_Hora' in col:
            df_res[col] = df_res[col].round(1).astype(str).str.replace('.', ',')

    csv_out = 'Brechas_Horas_Ingresos_2005_2025_LATAM.csv'
    df_res.to_csv(csv_out, sep=';', index=False, encoding='utf-8-sig')
    
    print("\n¡Proceso Finalizado!")
    print("Se generaron las siguientes gráficas en alta resolución:")
    print(" 1. 07_brecha_horas_semanales.png")
    print(" 2. 08_brecha_ingreso_mensual.png")
    print(" 3. 09_brecha_ingreso_hora.png")
    print(f"-> Archivo de datos guardado: '{csv_out}'")

if __name__ == '__main__':
    procesar_brechas_ingresos_horas()