import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------
# 1. DICCIONARIO DEL DEFLACTOR (Base 2018=100)
# ----------------------------------------------------------------------
inpc_historico_trimestral = [
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
        if idx < len(inpc_historico_trimestral):
            DEFLACTOR_DICT[(y, q)] = inpc_historico_trimestral[idx]
            idx += 1

# ----------------------------------------------------------------------
# 2. FUNCIONES DE EXTRACCIÓN
# ----------------------------------------------------------------------
def weighted_average(df, value_col, weight_col):
    df_filtered = df.dropna(subset=[value_col, weight_col])
    if df_filtered.empty or df_filtered[weight_col].sum() == 0:
        return np.nan
    return np.average(df_filtered[value_col], weights=df_filtered[weight_col])

def obtener_nombre_archivo(year, quarter):
    year_short = str(year)[-2:]
    if year >= 2023:
        base_name = f"ENOE_SDEMT{quarter}{year_short}".upper()
    elif year in [2021, 2022] or (year == 2020 and quarter >= 3):
        base_name = f"ENOEN_SDEMT{quarter}{year_short}".upper()
    else:
        base_name = f"SDEMT{quarter}{year_short}".upper()
    return os.path.join("Data/ENOE_dta", f"ENOE_{year}_{quarter}", f"{base_name}.dta")

def procesar_formalidad(year, quarter):
    file_path = obtener_nombre_archivo(year, quarter)
    PONDERATOR = 'fac_tri' if ((year == 2020 and quarter >= 3) or year >= 2021) else 'fac'
    
    if not os.path.exists(file_path): 
        return None

    try:
        df = pd.read_stata(file_path, convert_categoricals=False)
        df.columns = df.columns.str.lower()
    except Exception as e:
        print(f"Error {year} T{quarter}: {e}")
        return None

    df[PONDERATOR] = pd.to_numeric(df.get(PONDERATOR, df.get('fac')), errors='coerce').fillna(0)
    df['r_def'] = df['r_def'].astype(str).str.strip()
    
    # Filtro: Residentes, >15 años, Ocupados
    df_base = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3])) & (df['eda'] >= 15)]
    df_ocupada = df_base[df_base['clase2'] == 1].copy()
    
    # Filtro Clave: Solo Ingresos > 0 (Para evitar los 'Falsos Ceros')
    df_ocupada['ingocup'] = pd.to_numeric(df_ocupada['ingocup'], errors='coerce').fillna(0)
    df_ing = df_ocupada[df_ocupada['ingocup'] > 0]
    
    if df_ing.empty: return None

    ing_general = weighted_average(df_ing, 'ingocup', PONDERATOR)
    ing_formal = weighted_average(df_ing[df_ing['emp_ppal'] == 2], 'ingocup', PONDERATOR)
    ing_informal = weighted_average(df_ing[df_ing['emp_ppal'] == 1], 'ingocup', PONDERATOR)
    
    return {
        'year': year, 'quarter': quarter,
        'ing_nominal_general': ing_general,
        'ing_nominal_formal': ing_formal,
        'ing_nominal_informal': ing_informal
    }

# ----------------------------------------------------------------------
# 3. EJECUCIÓN Y GRÁFICA
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print("Extrayendo salarios formales e informales desde los microdatos (Esto tomará unos minutos)...")
    
    periodos = []
    y, q = 2005, 1
    while (y, q) <= (2025, 4):
        if not (y == 2020 and q in [1, 2]):
            periodos.append((y, q))
        q += 1
        if q > 4: q = 1; y += 1

    resultados = []
    for y, q in periodos:
        res = procesar_formalidad(y, q)
        if res:
            resultados.append(res)
            print(f"  -> Procesado {y} T{q}")

    df_final = pd.DataFrame(resultados)
    
    # Aplicar deflactor a valores reales
    df_final['deflactor'] = df_final.apply(lambda row: DEFLACTOR_DICT.get((row['year'], row['quarter']), np.nan), axis=1)
    df_final['def_ing_general'] = (df_final['ing_nominal_general'] / df_final['deflactor']) * 100
    df_final['def_ing_formal'] = (df_final['ing_nominal_formal'] / df_final['deflactor']) * 100
    df_final['def_ing_informal'] = (df_final['ing_nominal_informal'] / df_final['deflactor']) * 100
    
    # Calcular Porcentaje ("Premium de Formalidad": Cuánto % MÁS gana el formal)
    df_final['brecha_pct'] = ((df_final['def_ing_formal'] / df_final['def_ing_informal']) - 1) * 100
    
    df_final['Periodo'] = df_final['year'].astype(int).astype(str) + '-T' + df_final['quarter'].astype(int).astype(str)
    
    df_final.to_csv("ENOE_Ingresos_Formalidad.csv", index=False)
    print("\n¡Bases calculadas! Generando gráfica...")
    
    # ==========================
    # CREACIÓN DE LA GRÁFICA
    # ==========================
    fig, ax1 = plt.subplots(figsize=(16, 9))
    
    # 1. Graficar Líneas de Ingreso en el Eje Principal (Izquierdo)
    ax1.plot(df_final['Periodo'], df_final['def_ing_formal'], label='Ingreso Promedio Formal', color='#27ae60', linewidth=3.5)
    ax1.plot(df_final['Periodo'], df_final['def_ing_general'], label='Ingreso Promedio General', color='#2c3e50', linewidth=2.5, linestyle='-.')
    ax1.plot(df_final['Periodo'], df_final['def_ing_informal'], label='Ingreso Promedio Informal', color='#e74c3c', linewidth=3.5)
    
    # Etiquetas de texto al inicio y final (sin decimales)
    primer_formal, ultimo_formal = df_final['def_ing_formal'].iloc[0], df_final['def_ing_formal'].iloc[-1]
    primer_informal, ultimo_informal = df_final['def_ing_informal'].iloc[0], df_final['def_ing_informal'].iloc[-1]
    primer_gral, ultimo_gral = df_final['def_ing_general'].iloc[0], df_final['def_ing_general'].iloc[-1]
    
    ax1.text(0, primer_formal, f'${primer_formal:,.0f} ', color='#27ae60', ha='right', va='center', fontweight='bold', fontsize=10)
    ax1.text(len(df_final)-1, ultimo_formal, f' ${ultimo_formal:,.0f}', color='#27ae60', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax1.text(0, primer_informal, f'${primer_informal:,.0f} ', color='#e74c3c', ha='right', va='center', fontweight='bold', fontsize=10)
    ax1.text(len(df_final)-1, ultimo_informal, f' ${ultimo_informal:,.0f}', color='#e74c3c', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax1.text(0, primer_gral, f'${primer_gral:,.0f} ', color='#2c3e50', ha='right', va='center', fontweight='bold', fontsize=10)
    ax1.text(len(df_final)-1, ultimo_gral, f' ${ultimo_gral:,.0f}', color='#2c3e50', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('Año y Trimestre', fontsize=12)
    ax1.set_ylabel('Ingreso Mensual Real (MXN Base 2018)', fontsize=12)
    ticks_to_show = df_final['Periodo'][df_final['quarter'] == 1]
    ax1.set_xticks(df_final[df_final['quarter'] == 1].index)
    ax1.set_xticklabels(ticks_to_show, rotation=45)
    
    # 2. Graficar Porcentaje (Premium) en el Eje Secundario (Derecho)
    ax2 = ax1.twinx()
    ax2.plot(df_final['Periodo'], df_final['brecha_pct'], label='Premium de Formalidad (%)', color='#8e44ad', linewidth=2.5, linestyle=':')
    ax2.set_ylabel('Premium de Formalidad: ¿Cuánto % más paga ser formal?', fontsize=12, color='#8e44ad')
    ax2.tick_params(axis='y', labelcolor='#8e44ad')
    
    primer_pct, ultimo_pct = df_final['brecha_pct'].iloc[0], df_final['brecha_pct'].iloc[-1]
    ax2.text(0, primer_pct, f'{primer_pct:.1f}% ', color='#8e44ad', ha='right', va='center', fontweight='bold', fontsize=10)
    ax2.text(len(df_final)-1, ultimo_pct, f' {ultimo_pct:.1f}%', color='#8e44ad', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # 3. Título y Leyenda Unificada
    plt.title('Evolución del Ingreso Real: Sector Formal vs Informal (2005-2025)\n(Pesos constantes base 2018, población ocupada con ingresos > 0)', fontsize=15, pad=15)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=12)
    
    # 4. Marca de Agua
    ax1.text(0.99, 0.01, "www.mexicoendatos.org", transform=ax1.transAxes, 
             ha='right', va='bottom', fontsize=12, color='black', alpha=0.8)
    
    ax1.set_xlim(-3, len(df_final) + 3)
    plt.tight_layout()
    
    output_filename = 'ingreso_formal_vs_informal.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gráfica '{output_filename}' exportada exitosamente.")