import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
sns.set_theme(style="whitegrid")

# Cargar los datos nacionales
df = pd.read_csv('ENOE_Series_Tiempo_20260330.csv')

# Crear columna de periodo para el eje X
df['Periodo'] = df['year'].astype(int).astype(str) + '-T' + df['quarter'].astype(int).astype(str)
df = df.sort_values(by=['year', 'quarter']).reset_index(drop=True)

# ==========================================
# 1. Calcular Premium Salarial Anualizado
# ==========================================
# Supuestos de años de estudio: Primaria(6), Secundaria(3), Prepa(3), Profesional(4)
# Ya no calculamos Posgrado
df['premio_primaria'] = ((df['def_ing_mensual_ing_Primaria'] / df['def_ing_mensual_ing_Sin_Escolaridad']) - 1) / 6 * 100
df['premio_secundaria'] = ((df['def_ing_mensual_ing_Secundaria'] / df['def_ing_mensual_ing_Primaria']) - 1) / 3 * 100
df['premio_prepa'] = ((df['def_ing_mensual_ing_Preparatoria'] / df['def_ing_mensual_ing_Secundaria']) - 1) / 3 * 100
df['premio_profesional'] = ((df['def_ing_mensual_ing_Profesional'] / df['def_ing_mensual_ing_Preparatoria']) - 1) / 4 * 100

plt.figure(figsize=(16, 9))

# Diccionario para iterar y graficar (Excluyendo Posgrado)
categorias_premium = {
    'Primaria vs Sin Escolaridad (6 años)': ('premio_primaria', '#5ba0d0'),
    'Secundaria vs Primaria (3 años)': ('premio_secundaria', '#2878b5'),
    'Preparatoria vs Secundaria (3 años)': ('premio_prepa', '#0f4c81'),
    'Universidad vs Preparatoria (4 años)': ('premio_profesional', '#17202a')
}

for label, (col, color) in categorias_premium.items():
    if col in df.columns:
        plt.plot(df['Periodo'], df[col], label=label, linewidth=2.5, color=color, alpha=0.9)
        
        # Filtrar NAs para obtener el primer y último valor válido
        valid_data = df[col].dropna()
        if not valid_data.empty:
            primer_valor = valid_data.iloc[0]
            ultimo_valor = valid_data.iloc[-1]
            
            # Etiqueta al inicio
            plt.text(0, primer_valor, f'{primer_valor:.1f}% ', color=color, 
                     ha='right', va='center', fontweight='bold', fontsize=10)
            
            # Etiqueta al final
            plt.text(len(df)-1, ultimo_valor, f' {ultimo_valor:.1f}%', color=color, 
                     ha='left', va='center', fontweight='bold', fontsize=10)

# Títulos y ejes
plt.title('Premium Salarial Anualizado por Grado Educativo (2005 - 2025)\n(% de incremento real en el ingreso por cada AÑO extra de estudio)', fontsize=15, pad=15)
plt.ylabel('Rendimiento Salarial Anualizado (%)', fontsize=12)
plt.xlabel('Año y Trimestre', fontsize=12)

# Filtrar etiquetas del eje X para limpieza
ticks_to_show = df['Periodo'][df['quarter'] == 1]
plt.xticks(df[df['quarter'] == 1].index, ticks_to_show, rotation=45)

# --- Marca de agua ---
WATERMARK_URL = "www.mexicoendatos.org"
plt.text(0.99, 0.01, WATERMARK_URL, transform=plt.gca().transAxes, 
         ha='right', va='bottom', fontsize=12, color='black', alpha=0.8)

# Leyenda y márgenes
plt.xlim(-3, len(df) + 3)
plt.legend(title='Transición Educativa', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)
plt.tight_layout()

# Guardar y mostrar
output_filename = 'premium_salarial_anualizado_sin_posgrado.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()