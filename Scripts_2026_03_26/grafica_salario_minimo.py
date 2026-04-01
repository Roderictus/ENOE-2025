import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
sns.set_theme(style="whitegrid")

try:
    df = pd.read_csv('ENOE_Series_Tiempo_20260330.csv')
    df['Periodo'] = df['year'].astype(int).astype(str) + '-T' + df['quarter'].astype(int).astype(str)
    df = df.sort_values(by=['year', 'quarter'])

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Eje Izquierdo: % de Informales ganando menos de 1 SM
    color1 = '#c0392b'
    ax1.set_xlabel('Año y Trimestre', fontsize=12)
    ax1.set_ylabel('% de Informales que ganan < 1 Salario Mínimo', color=color1, fontsize=12)
    line1 = ax1.plot(df['Periodo'], df['pct_informal_menos_1sm'], color=color1, linewidth=3, 
                     label='% Informales < 1 SM')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Reducir etiquetas del eje X
    ticks_to_show = df['Periodo'][df['quarter'] == 1]
    ax1.set_xticks(ticks_to_show)
    ax1.set_xticklabels(ticks_to_show, rotation=45)

    # Eje Derecho: Valor del Salario Mínimo Mensual Deflactado
    ax2 = ax1.twinx()  
    color2 = '#2980b9'
    ax2.set_ylabel('Salario Mínimo Mensual Real (MXN Base 2018)', color=color2, fontsize=12)
    line2 = ax2.plot(df['Periodo'], df['def_sm_mensual_estimado'], color=color2, linewidth=3, 
                     linestyle='--', label='Salario Mínimo Real')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Impacto de los aumentos al Salario Mínimo sobre el Sector Informal (2005-2025)', fontsize=15, pad=15)
    
    # Unir leyendas
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig('impacto_sm_informalidad.png', dpi=300, bbox_inches='tight')
    print("Gráfica 'impacto_sm_informalidad.png' generada con éxito.")

except Exception as e:
    print(f"Error: {e}")