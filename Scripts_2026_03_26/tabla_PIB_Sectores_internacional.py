import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Datos provenientes de los indicadores del Banco Mundial (~2022)
data = {
    'País': ['Estados Unidos', 'Alemania', 'Japón', 'Corea del Sur', 'México', 'Sierra Leona'],
    'Sector Primario': [1, 1, 1, 2, 4, 58],
    'Sector Secundario': [21, 36, 30, 35, 36, 10],
    'Sector Terciario': [78, 63, 69, 63, 60, 32]
}
df = pd.DataFrame(data)

# Configurar la gráfica
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

# Colores característicos para cada sector
colors = ['#2ca02c', '#1f77b4', '#ff7f0e'] # Verde, Azul, Naranja
labels = ['Sector Primario', 'Sector Secundario', 'Sector Terciario']

# Variable 'lefts' para controlar dónde empieza horizontalmente cada nueva barra apilada
lefts = np.zeros(len(df))

# Generar las barras apiladas mediante un ciclo
for i, col in enumerate(labels):
    widths = df[col]
    bars = ax.barh(df['País'], widths, left=lefts, label=col, color=colors[i], edgecolor='white', height=0.7)
    
    # Lógica matemática para centrar los porcentajes en cada bloque
    for bar, width in zip(bars, widths):
        if width > 0:
            # Obtener el centro en X sumando la coordenada inicial + la mitad del ancho del bloque
            x_center = bar.get_x() + (bar.get_width() / 2)
            # Obtener el centro en Y sumando la coordenada inicial + la mitad del alto
            y_center = bar.get_y() + (bar.get_height() / 2)
            
            # Cambiar color de la fuente si el bloque es muy angosto (para los 1% y 2%)
            text_color = 'white' if width > 5 else 'black'
            
            ax.text(x_center, y_center, f'{width}%', ha='center', va='center', 
                    color=text_color, fontweight='bold', fontsize=10)
            
    # Sumar el ancho del bloque actual al array 'lefts' para que el siguiente sector empiece justo donde este termina
    lefts += widths

# Diseño general y limpieza visual
ax.set_xlabel('Porcentaje del PIB (%)', fontsize=12)
ax.set_title('Distribución del PIB por Sectores Económicos (Aprox. 2022)', fontsize=14, pad=20, fontweight='bold')
ax.set_xlim(0, 100) # Fijar el límite del eje X al 100%
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Extraer la leyenda fuera del cuadro principal para no tapar los datos
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Sectores")

# --- AGREGADO 1: Marca de agua ---
fig.text(0.5, 0.5, 'www.mexicoendatos.org', fontsize=40, color='gray', 
         ha='center', va='center', alpha=0.1, rotation=20)

# --- AGREGADO 2: Pie de página con la fuente ---
fig.text(0.01, -0.05, 'Fuente: Indicadores de Desarrollo del Banco Mundial (World Bank Open Data).\nCódigos: NV.AGR.TOTL.ZS, NV.IND.TOTL.ZS, NV.SRV.TOTL.ZS (~2022).', 
         fontsize=10, color='#555555', ha='left')

plt.tight_layout()
plt.savefig('distribucion_sectores_pib.png', dpi=300, bbox_inches='tight')