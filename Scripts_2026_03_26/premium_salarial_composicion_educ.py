import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Cargar los datos nacionales generados previamente
df = pd.read_csv('ENOE_Series_Tiempo_20260330.csv')

# Crear columna de periodo para el eje X
df['Periodo'] = df['year'].astype(int).astype(str) + '-T' + df['quarter'].astype(int).astype(str)
df = df.sort_values(by=['year', 'quarter']).reset_index(drop=True)

# ==========================================
# 1. Calcular Premium Salarial Anualizado
# ==========================================
# Se resta 1 para obtener la tasa de crecimiento, se divide por los años, y se multiplica por 100 para %
df['premio_secundaria'] = ((df['def_ing_mensual_ing_Secundaria'] / df['def_ing_mensual_ing_Primaria']) - 1) / 3 * 100
df['premio_prepa'] = ((df['def_ing_mensual_ing_Preparatoria'] / df['def_ing_mensual_ing_Secundaria']) - 1) / 3 * 100
df['premio_profesional'] = ((df['def_ing_mensual_ing_Profesional'] / df['def_ing_mensual_ing_Preparatoria']) - 1) / 4 * 100

# Graficar Premium Anualizado
plt.figure(figsize=(14, 7))
plt.plot(df['Periodo'], df['premio_secundaria'], label='Secundaria vs Primaria (3 años)', linewidth=2.5, color='#e67e22')
plt.plot(df['Periodo'], df['premio_prepa'], label='Preparatoria vs Secundaria (3 años)', linewidth=2.5, color='#27ae60')
plt.plot(df['Periodo'], df['premio_profesional'], label='Universidad vs Preparatoria (4 años)', linewidth=2.5, color='#8e44ad')

plt.title('Premium Salarial Anualizado por Grado Educativo (2005 - 2025)\n(% de incremento real en el ingreso por cada AÑO extra de estudio)', fontsize=15, pad=15)
plt.ylabel('Rendimiento Salarial Anualizado (%)', fontsize=12)
plt.xlabel('Año y Trimestre', fontsize=12)

# Filtrar etiquetas del eje X (1 por año)
ticks_to_show = df['Periodo'][df['quarter'] == 1]
plt.xticks(ticks_to_show, rotation=45)

plt.legend(title='Transición Educativa', fontsize=11, title_fontsize=12)
plt.tight_layout()
plt.savefig('premium_salarial_anualizado.png', dpi=300)

plt.show()

# ==========================================
# 2. Evolución de la Composición Educativa
# ==========================================
plt.figure(figsize=(14, 7))
plt.plot(df['Periodo'], df['pct_ing_edu_Primaria'], label='Primaria', linewidth=2.5, color='#5ba0d0')
plt.plot(df['Periodo'], df['pct_ing_edu_Secundaria'], label='Secundaria', linewidth=2.5, color='#2878b5')
plt.plot(df['Periodo'], df['pct_ing_edu_Preparatoria'], label='Preparatoria', linewidth=2.5, color='#0f4c81')
plt.plot(df['Periodo'], df['pct_ing_edu_Profesional'], label='Profesional (Universidad)', linewidth=2.5, color='#17202a')
plt.plot(df['Periodo'], df['pct_ing_edu_Posgrado'], label='Posgrado', linewidth=2.5, color='#c0392b')

plt.title('Evolución de la Escolaridad en la Población Ocupada (2005 - 2025)\n(% respecto al total de trabajadores con ingresos > 0)', fontsize=15, pad=15)
plt.ylabel('Porcentaje de la Población Ocupada (%)', fontsize=12)
plt.xlabel('Año y Trimestre', fontsize=12)

plt.xticks(ticks_to_show, rotation=45)
plt.legend(title='Nivel Educativo', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)
plt.tight_layout()
plt.savefig('composicion_educativa_evolucion.png', dpi=300)



plt.show()