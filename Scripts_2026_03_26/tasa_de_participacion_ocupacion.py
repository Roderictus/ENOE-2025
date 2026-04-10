import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def generar_graficas_genero_separadas():
    # =========================================================================
    # 1. EXTRACCIÓN DE DATOS MACROECONÓMICOS (SDEMT)
    # =========================================================================
    print("Iniciando extracción de datos (2012-T3 a 2025)...")
    resultados_macro = []

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

    for year in range(2012, 2026):
        for quarter in range(1, 5):
            if year == 2012 and quarter < 3: continue 
            if year == 2020 and quarter == 2: continue 
            
            p_sdemt = get_sdemt_path(year, quarter)
            if not p_sdemt: continue
                
            try:
                df = pd.read_stata(p_sdemt, convert_categoricals=False)
                df.columns = df.columns.str.lower()
                
                PONDERATOR = 'fac_tri' if 'fac_tri' in df.columns else 'fac'
                df[PONDERATOR] = pd.to_numeric(df[PONDERATOR], errors='coerce').fillna(0)
                df['r_def'] = df['r_def'].astype(str).str.strip()
                
                # Filtro Base: Residentes definitivos
                df_base = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3]))].copy()
                
                # 1. Población Mayor de 15 años
                df_15 = df_base[df_base['eda'] >= 15]
                pob15_h = df_15[df_15['sex'] == 1][PONDERATOR].sum()
                pob15_m = df_15[df_15['sex'] == 2][PONDERATOR].sum()
                
                # 2. PEA (clase1 == 1)
                df_pea = df_15[df_15['clase1'] == 1]
                pea_h = df_pea[df_pea['sex'] == 1][PONDERATOR].sum()
                pea_m = df_pea[df_pea['sex'] == 2][PONDERATOR].sum()
                
                # 3. Ocupados (clase2 == 1)
                df_oc = df_pea[df_pea['clase2'] == 1]
                oc_h = df_oc[df_oc['sex'] == 1][PONDERATOR].sum()
                oc_m = df_oc[df_oc['sex'] == 2][PONDERATOR].sum()

                resultados_macro.append({
                    'Periodo': f"{year}-T{quarter}",
                    'Pob_Mayor15_Hombres': pob15_h,
                    'Pob_Mayor15_Mujeres': pob15_m,
                    'PEA_Hombres': pea_h,
                    'PEA_Mujeres': pea_m,
                    'Ocupados_Hombres': oc_h,
                    'Ocupados_Mujeres': oc_m,
                    
                    # CÁLCULO DE LAS TASAS
                    'Tasa_Participacion_Hombres': (pea_h / pob15_h * 100) if pob15_h > 0 else 0,
                    'Tasa_Participacion_Mujeres': (pea_m / pob15_m * 100) if pob15_m > 0 else 0,
                    
                    'Tasa_Ocupacion_Hombres': (oc_h / pea_h * 100) if pea_h > 0 else 0,
                    'Tasa_Ocupacion_Mujeres': (oc_m / pea_m * 100) if pea_m > 0 else 0
                })
                
            except Exception as e:
                print(f"Error procesando {year}-T{quarter}: {e}")

    df_res = pd.DataFrame(resultados_macro)

    # =========================================================================
    # 2. CONFIGURACIÓN VISUAL GENERAL
    # =========================================================================
    print("Datos extraídos. Generando gráficas individuales...")
    sns.set_theme(style="whitegrid")
    
    periodos = df_res['Periodo'].tolist()
    ticks_to_show = [p for p in periodos if '-T1' in p or '-T3' in p] # Mostrar etiquetas intercaladas
    
    color_hombres = '#1f77b4' # Azul oscuro/medio
    color_mujeres = '#2ca02c' # Verde fuerte

    # =========================================================================
    # GRÁFICA 1: TASA DE PARTICIPACIÓN
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.plot(df_res['Periodo'], df_res['Tasa_Participacion_Hombres'], color=color_hombres, linewidth=3.5, label='Hombres')
    ax1.plot(df_res['Periodo'], df_res['Tasa_Participacion_Mujeres'], color=color_mujeres, linewidth=3.5, label='Mujeres')
    
    # Textos de inicio y fin (Participación)
    prim_part_h, ult_part_h = df_res['Tasa_Participacion_Hombres'].iloc[0], df_res['Tasa_Participacion_Hombres'].iloc[-1]
    prim_part_m, ult_part_m = df_res['Tasa_Participacion_Mujeres'].iloc[0], df_res['Tasa_Participacion_Mujeres'].iloc[-1]
    
    ax1.text(0, prim_part_h, f'{prim_part_h:.1f}% ', color=color_hombres, ha='right', va='center', fontweight='bold', fontsize=12)
    ax1.text(len(df_res)-1, ult_part_h, f' {ult_part_h:.1f}%', color=color_hombres, ha='left', va='center', fontweight='bold', fontsize=12)
    ax1.text(0, prim_part_m, f'{prim_part_m:.1f}% ', color=color_mujeres, ha='right', va='center', fontweight='bold', fontsize=12)
    ax1.text(len(df_res)-1, ult_part_m, f' {ult_part_m:.1f}%', color=color_mujeres, ha='left', va='center', fontweight='bold', fontsize=12)

    ax1.set_title('Evolución de la Tasa de Participación por Género (2012 - 2025)\n(Población en la PEA / Población Mayor de 15 años)', fontsize=16, pad=15)
    ax1.set_ylabel('Porcentaje de Participación (%)', fontsize=12)
    ax1.set_xlabel('Año y Trimestre', fontsize=12)
    ax1.set_ylim(35, 85) # Escala fija para evidenciar la brecha
    ax1.set_xticks(ticks_to_show)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='lower right', fontsize=12)
    ax1.text(0.99, 0.01, "www.mexicoendatos.org", transform=ax1.transAxes, ha='right', va='bottom', fontsize=12, color='black', alpha=0.6)
    
    # Expandir márgenes para que quepan los textos laterales
    ax1.set_xlim(-2, len(df_res) + 2)
    
    plt.tight_layout()
    plt.savefig('grafica_01_tasa_participacion.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # =========================================================================
    # GRÁFICA 2: TASA DE OCUPACIÓN
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    
    ax2.plot(df_res['Periodo'], df_res['Tasa_Ocupacion_Hombres'], color=color_hombres, linewidth=3.5, label='Hombres')
    ax2.plot(df_res['Periodo'], df_res['Tasa_Ocupacion_Mujeres'], color=color_mujeres, linewidth=3.5, label='Mujeres')
    
    # Textos de inicio y fin (Ocupación)
    prim_ocup_h, ult_ocup_h = df_res['Tasa_Ocupacion_Hombres'].iloc[0], df_res['Tasa_Ocupacion_Hombres'].iloc[-1]
    prim_ocup_m, ult_ocup_m = df_res['Tasa_Ocupacion_Mujeres'].iloc[0], df_res['Tasa_Ocupacion_Mujeres'].iloc[-1]
    
    ax2.text(0, prim_ocup_h, f'{prim_ocup_h:.1f}% ', color=color_hombres, ha='right', va='bottom', fontweight='bold', fontsize=12)
    ax2.text(len(df_res)-1, ult_ocup_h, f' {ult_ocup_h:.1f}%', color=color_hombres, ha='left', va='bottom', fontweight='bold', fontsize=12)
    ax2.text(0, prim_ocup_m, f'{prim_ocup_m:.1f}% ', color=color_mujeres, ha='right', va='top', fontweight='bold', fontsize=12)
    ax2.text(len(df_res)-1, ult_ocup_m, f' {ult_ocup_m:.1f}%', color=color_mujeres, ha='left', va='top', fontweight='bold', fontsize=12)

    ax2.set_title('Evolución de la Tasa de Ocupación por Género (2012 - 2025)\n(Población Ocupada / Población en la PEA)', fontsize=16, pad=15)
    ax2.set_ylabel('Porcentaje de Ocupación (%)', fontsize=12)
    ax2.set_xlabel('Año y Trimestre', fontsize=12)
    ax2.set_ylim(92, 100) # Escala enfocada en la parte alta (el desempleo suele ser bajo del 3-5%)
    ax2.set_xticks(ticks_to_show)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(loc='lower right', fontsize=12)
    ax2.text(0.99, 0.01, "www.mexicoendatos.org", transform=ax2.transAxes, ha='right', va='bottom', fontsize=12, color='black', alpha=0.6)
    
    # Expandir márgenes para que quepan los textos laterales
    ax2.set_xlim(-2, len(df_res) + 2)
    
    plt.tight_layout()
    plt.savefig('grafica_02_tasa_ocupacion.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # =========================================================================
    # 3. EXPORTACIÓN DEL CSV (Formato Google Sheets LATAM)
    # =========================================================================
    # Formatear números enteros
    int_cols = ['Pob_Mayor15_Hombres', 'Pob_Mayor15_Mujeres', 'PEA_Hombres', 'PEA_Mujeres', 'Ocupados_Hombres', 'Ocupados_Mujeres']
    for col in int_cols:
        df_res[col] = df_res[col].fillna(0).round(0).astype(int).astype(str)
        
    # Formatear porcentajes (1 decimal y comas)
    pct_cols = ['Tasa_Participacion_Hombres', 'Tasa_Participacion_Mujeres', 'Tasa_Ocupacion_Hombres', 'Tasa_Ocupacion_Mujeres']
    for col in pct_cols:
        df_res[col] = df_res[col].fillna(0).round(1).astype(str).str.replace('.', ',')

    csv_filename = 'Evolucion_Tasas_Macro_Genero_Final.csv'
    df_res.to_csv(csv_filename, sep=';', index=False, encoding='utf-8-sig')

    print("\n¡Proceso Finalizado!")
    print(f"-> Se generó la imagen: 'grafica_01_tasa_participacion.png'")
    print(f"-> Se generó la imagen: 'grafica_02_tasa_ocupacion.png'")
    print(f"-> Se generó la base de datos: '{csv_filename}'")

if __name__ == '__main__':
    generar_graficas_genero_separadas()