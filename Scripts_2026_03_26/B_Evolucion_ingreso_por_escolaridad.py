import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
sns.set_theme(style="whitegrid")

# Cargar los datos
df = pd.read_csv('ENOE_Series_Tiempo_20260330.csv')

# Crear columna de periodo para el eje X
df['Periodo'] = df['year'].astype(int).astype(str) + '-T' + df['quarter'].astype(int).astype(str)
df = df.sort_values(by=['year', 'quarter']).reset_index(drop=True)

# Inicializar figura y ejes
fig, ax = plt.subplots(figsize=(15, 8))

# Diccionario de variables a graficar
categorias = {
    'Primaria': ('def_ing_mensual_ing_Primaria', '#5ba0d0'),
    'Secundaria': ('def_ing_mensual_ing_Secundaria', '#2878b5'),
    'Preparatoria': ('def_ing_mensual_ing_Preparatoria', '#0f4c81'),
    'Profesional (Universitaria)': ('def_ing_mensual_ing_Profesional', '#17202a')
}

# 1. Graficar Niveles Educativos y sus etiquetas
for label, (col, color) in categorias.items():
    ax.plot(df['Periodo'], df[col], label=label, linewidth=2.5, color=color)
    
    primer_valor = df[col].iloc[0]
    ultimo_valor = df[col].iloc[-1]
    
    # Etiqueta al inicio
    ax.text(0, primer_valor, f'${primer_valor:,.0f} ', color=color, 
             ha='right', va='center', fontweight='bold', fontsize=10)
    
    # Etiqueta al final
    ax.text(len(df)-1, ultimo_valor, f' ${ultimo_valor:,.0f}', color=color, 
             ha='left', va='center', fontweight='bold', fontsize=10)

# 2. Graficar Salario Mínimo Mensual Real y sus etiquetas
col_sm = 'def_sm_mensual_estimado'
color_sm = '#c0392b'
ax.plot(df['Periodo'], df[col_sm], label='Salario Mínimo Mensual (Real)', 
         linewidth=3, color=color_sm, linestyle='--')

primer_sm = df[col_sm].iloc[0]
ultimo_sm = df[col_sm].iloc[-1]

ax.text(0, primer_sm, f'${primer_sm:,.0f} ', color=color_sm, 
         ha='right', va='center', fontweight='bold', fontsize=10)
ax.text(len(df)-1, ultimo_sm, f' ${ultimo_sm:,.0f}', color=color_sm, 
         ha='left', va='center', fontweight='bold', fontsize=10)

# Títulos y ejes
ax.set_title('Evolución del Ingreso Mensual Real por Escolaridad vs Salario Mínimo (2005-2025)\n(Pesos constantes base 2018, población ocupada con ingresos > 0)', fontsize=15, pad=15)
ax.set_xlabel('Año y Trimestre', fontsize=12)
ax.set_ylabel('Ingreso Mensual Real (MXN Base 2018)', fontsize=12)

# Filtrar etiquetas del eje X
ticks_to_show = df['Periodo'][df['quarter'] == 1]
ax.set_xticks(df[df['quarter'] == 1].index)
ax.set_xticklabels(ticks_to_show, rotation=45)

# --- AGREGAR MARCA DE AGUA NEGRA ---
WATERMARK_URL = "www.mexicoendatos.org"
ax.text(0.99, 0.01, WATERMARK_URL, transform=ax.transAxes, 
         ha='right', va='bottom', fontsize=12, color='black', alpha=0.8)

# Leyenda y márgenes (Expandir el eje X para que los números quepan)
ax.set_xlim(-2.5, len(df) + 2.5) 
ax.legend(title='Nivel Educativo / Referencia', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)

plt.tight_layout()
plt.savefig('B_Evolucion_ingreso_por_escolaridad.png', dpi=300, bbox_inches='tight')
plt.show()