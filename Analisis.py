import plotly.express as px
import pandas as pd


df_final = pd.read_csv('Nacional_deflactado.csv')

# --- GRÁFICA 1: Ingresos Reales vs Nominales ---
# Esto te permite ver el efecto de la inflación
fig_ingresos = px.line(df_final, x='fecha', 
                       y=['ing_prom_mes_total', 'ing_prom_mes_real'],
                       title='Ingreso Promedio Mensual (Nominal vs. Real Base 2005)',
                       labels={'value': 'Ingreso (MXN)', 'fecha': 'Fecha'},
                       hover_data={'variable': True, 'value': ':.2f'})
fig_ingresos.update_layout(hovermode="x unified")
fig_ingresos.show()


# --- GRÁFICA 2: Población Ocupada vs Desocupada ---
# Nota: La gráfica mostrará un HUECO para T1 y T2 de 2020 si los datos faltan
fig_pea = px.line(df_final, x='fecha', 
                  y=['ocupada_total', 'desocupada_total'],
                  title='Población Ocupada vs. Desocupada',
                  labels={'value': 'Personas', 'fecha': 'Fecha'})
fig_pea.update_layout(hovermode="x unified")
fig_pea.show()


# --- GRÁFICA 3: Tasa de Ocupación Informal ---
# Creamos la tasa (Informales / Total Ocupados)
df_final['tasa_informalidad'] = (df_final['ocupacion_informal'] / df_final['ocupada_total']) * 100

fig_informal = px.line(df_final, x='fecha', 
                       y='tasa_informalidad',
                       title='Tasa de Ocupación Informal (%)',
                       labels={'tasa_informalidad': 'Porcentaje (%)', 'fecha': 'Fecha'})
fig_informal.update_traces(fill='tozeroy') # Rellena el área bajo la línea
fig_informal.show()


# --- GRÁFICA 4: Subocupación ---
fig_subocupacion = px.line(df_final, x='fecha', 
                           y='subocupacion',
                           title='Población Subocupada',
                           labels={'subocupacion': 'Personas', 'fecha': 'Fecha'})
fig_subocupacion.update_traces(fill='tozeroy', line_color='orange')
fig_subocupacion.show()