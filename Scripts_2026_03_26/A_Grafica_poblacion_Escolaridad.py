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

# Diccionario con las categorías, columnas correspondientes y colores
categorias = {
    'Sin Escolaridad': ('pct_ing_edu_Sin_Escolaridad', '#7f8c8d'), # Gris
    'Primaria': ('pct_ing_edu_Primaria', '#5ba0d0'),             # Azul claro
    'Secundaria': ('pct_ing_edu_Secundaria', '#2878b5'),         # Azul medio
    'Preparatoria': ('pct_ing_edu_Preparatoria', '#0f4c81'),     # Azul oscuro
    'Profesional (Universidad)': ('pct_ing_edu_Profesional', '#17202a'), # Casi negro
    'Posgrado': ('pct_ing_edu_Posgrado', '#c0392b')              # Rojo
}

# Inicializar figura
plt.figure(figsize=(15, 8))

# Graficar cada línea y agregar las etiquetas al inicio y al final
for label, (col, color) in categorias.items():
    plt.plot(df['Periodo'], df[col], label=label, linewidth=2.5, color=color)
    
    # Extraer valores de la primera y última observación
    primer_valor = df[col].iloc[0]
    ultimo_valor = df[col].iloc[-1]
    
    # Etiqueta al inicio (2005 T1)
    plt.text(0, primer_valor, f'{primer_valor:.1f}% ', color=color, 
             ha='right', va='center', fontweight='bold', fontsize=10)
    
    # Etiqueta al final (2025 T4)
    plt.text(len(df)-1, ultimo_valor, f' {ultimo_valor:.1f}%', color=color, 
             ha='left', va='center', fontweight='bold', fontsize=10)

# Títulos y ejes
plt.title('Evolución de la Escolaridad en la Población Ocupada (2005 - 2025)\n(% respecto al total de trabajadores con ingresos > 0)', fontsize=15, pad=15)
plt.ylabel('Porcentaje de la Población Ocupada (%)', fontsize=12)
plt.xlabel('Año y Trimestre', fontsize=12)

# Filtrar etiquetas del eje X para no saturar visualmente
ticks_to_show = df['Periodo'][df['quarter'] == 1]
plt.xticks(df[df['quarter'] == 1].index, ticks_to_show, rotation=45)

# Agregar la Marca de Agua (Watermark)
WATERMARK_URL = "www.mexicoendatos.org"
plt.text(0.99, 0.01, WATERMARK_URL, transform=plt.gca().transAxes, 
         ha='right', va='bottom', fontsize=12, color='black', alpha=0.7)

# Leyenda y márgenes
# Expandir un poco el límite X para que las etiquetas iniciales/finales no se corten
plt.xlim(-2, len(df) + 2) 
plt.legend(title='Nivel Educativo', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)
plt.tight_layout()

# Guardar la gráfica en alta calidad
output_filename = 'composicion_educativa_actualizada.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()