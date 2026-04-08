import pandas as pd

df = pd.read_stata("Data/ENOE_dta/ENOE_2025_4/ENOE_SDEMT425.dta", convert_categoricals=False)
df.columns = df.columns.str.lower()

# Buscar columnas candidatas a tener la ocupación o el sector
columnas_clave = [col for col in df.columns if col in ['sinco', 'scian', 'cbo', 'rama', 'rama_est1', 'rama_est2']]
print("Columnas de clasificación encontradas:", columnas_clave)

# Imprimir una muestra de los valores
print(df[columnas_clave].dropna().head())