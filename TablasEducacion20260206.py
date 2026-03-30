import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('ENOE_Education_Wages_National.csv')

# Create a clean 'Period' label
df['Periodo'] = df['year'].astype(str) + "-T" + df['quarter'].astype(str)

# Select the most recent quarter for the snapshot graph
latest = df.iloc[-1] 
print(f"Analyzing data for: {latest['Periodo']}")

# ----------------------------------------------------------------------
# A. PREPARE DATA FOR PLOTTING
# ----------------------------------------------------------------------
# We want to pivot the columns back into rows for plotting
# Columns look like: wage_Primaria_total, wage_Doctorado_total, etc.

educ_levels = ['Primaria', 'Secundaria', 'Preparatoria', 'CarreraTec', 'Profesional', 'Maestria', 'Doctorado']
plot_data = []

for educ in educ_levels:
    plot_data.append({
        'Nivel Educativo': educ,
        'Salario Promedio': latest[f'wage_{educ}_total'],
        'Población (Millones)': latest[f'pop_{educ}_total'] / 1_000_000,
        'Grupo': 'Total'
    })
    # Add Sex breakdown if needed
    plot_data.append({
        'Nivel Educativo': educ,
        'Salario Promedio': latest[f'wage_{educ}_hombres'],
        'Grupo': 'Hombres'
    })
    plot_data.append({
        'Nivel Educativo': educ,
        'Salario Promedio': latest[f'wage_{educ}_mujeres'],
        'Grupo': 'Mujeres'
    })

df_plot = pd.DataFrame(plot_data)

# ----------------------------------------------------------------------
# B. VISUALIZATION 1: The Wage Premium (Bar Chart)
# ----------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# Plot Men vs Women vs Total wages by Education
chart = sns.barplot(
    data=df_plot, 
    x='Nivel Educativo', 
    y='Salario Promedio', 
    hue='Grupo',
    palette={'Total': 'gray', 'Hombres': '#1f77b4', 'Mujeres': '#e377c2'}
)

plt.title(f'Salario Mensual Promedio por Nivel Educativo ({latest["Periodo"]})', fontsize=16)
plt.ylabel('Ingreso Mensual (MXN)')
plt.xticks(rotation=45)
plt.legend(title='Sexo')

# Add values on top of bars
for container in chart.containers:
    chart.bar_label(container, fmt='${:,.0f}', padding=3, fontsize=9)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# C. VISUALIZATION 2: Population Composition (Pie/Bar)
# ----------------------------------------------------------------------
# How many people with income actually have these degrees?
df_pop = df_plot[df_plot['Grupo'] == 'Total'].copy()

plt.figure(figsize=(10, 6))
sns.barplot(data=df_pop, x='Nivel Educativo', y='Población (Millones)', color='teal')
plt.title(f'Población Ocupada con Ingreso por Nivel Educativo ({latest["Periodo"]})', fontsize=16)
plt.ylabel('Personas (Millones)')
plt.show()

# ----------------------------------------------------------------------
# D. SUMMARY TABLE
# ----------------------------------------------------------------------
summary_cols = ['year', 'quarter', 
                'pop_ing_positivo_total', 
                'ingreso_prom_mensual_total', 
                'wage_Profesional_total', 
                'wage_Doctorado_total']

print("\n--- Resumen Rápido (Últimos periodos) ---")
print(df[summary_cols].tail())

# Calculate "Education Premium" (Doctorate vs Primary)
premium = latest['wage_Doctorado_total'] / latest['wage_Primaria_total']
print(f"\nUn Doctorado gana {premium:.1f} veces más que alguien con Primaria en promedio.")