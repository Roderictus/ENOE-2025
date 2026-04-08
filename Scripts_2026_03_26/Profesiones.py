import pandas as pd
import os

# 1. Definir la ruta de la base de datos (último trimestre)
year = 2025
quarter = 4
year_short = str(year)[-2:]
path = os.path.join("Data/ENOE_dta", f"ENOE_{year}_{quarter}", f"ENOE_SDEMT{quarter}{year_short}.dta")

print(f"Cargando microdatos del T{quarter} {year}...")

try:
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = df.columns.str.lower()
    
    # Estandarizar el ponderador
    PONDERATOR = 'fac_tri' if 'fac_tri' in df.columns else 'fac'
    df[PONDERATOR] = pd.to_numeric(df[PONDERATOR], errors='coerce').fillna(0)
    df['r_def'] = df['r_def'].astype(str).str.strip()
    
    # 2. Filtrar a la PEA Ocupada (Residentes, >15 años, clase1=1, clase2=1)
    df_ocupados = df[(df['r_def'] == '0.0') & 
                     (df['c_res'].isin([1, 3])) & 
                     (df['eda'] >= 15) & 
                     (df['clase1'] == 1) & 
                     (df['clase2'] == 1)].copy()

    # Mapeo oficial de los Grandes Grupos SINCO (1 Dígito)
    sinco_1d_map = {
        '1': '1. Funcionarios, directores y jefes',
        '2': '2. Profesionistas y técnicos',
        '3': '3. Auxiliares en actividades administrativas',
        '4': '4. Comerciantes, empleados y agentes de ventas',
        '5': '5. Servicios personales y vigilancia',
        '6': '6. Act. agrícolas, ganaderas y forestales',
        '7': '7. Trabajadores artesanales y fabriles',
        '8': '8. Operadores de maquinaria y transporte',
        '9': '9. Actividades elementales y de apoyo'
    }

    # Diccionario de profesiones frecuentes en México (4 Dígitos)
    sinco_4d_map = {
        '4111': 'Comerciantes en establecimientos',
        '4211': 'Empleados de ventas y dependientes en comercios',
        '6111': 'Trabajadores agrícolas (maíz, frijol, etc.)',
        '5411': 'Trabajadores domésticos',
        '7111': 'Albañiles, mamposteros y afines',
        '8342': 'Choferes de taxis y automóviles con ruta',
        '5116': 'Trabajadores de limpieza (oficinas y hoteles)',
        '4311': 'Comerciantes ambulantes (bienes no alimenticios)',
        '4312': 'Vendedores ambulantes de alimentos',
        '2221': 'Profesores de enseñanza primaria',
        '3111': 'Auxiliares administrativos y oficinistas',
        '8341': 'Choferes de camiones y tráileres',
        '5312': 'Vigilantes y guardias de seguridad',
        '7311': 'Mecánicos en mantenimiento y reparación automotriz',
        '9311': 'Peones agropecuarios',
        '5412': 'Peluqueros, barberos y estilistas',
        '7121': 'Carpinteros',
        '2411': 'Contadores y auditores',
        '2421': 'Abogados',
        '2111': 'Médicos generales'
    }

    if 'c_ocu11c' in df_ocupados.columns:
        # Limpiar la columna (Quitar decimales y rellenar con 0 a la izquierda para tener 4 dígitos)
        df_ocupados['c_ocu11c_str'] = df_ocupados['c_ocu11c'].astype(str).str.split('.').str[0].str.zfill(4)
        
        # Ignorar los "No Especificados" o valores en blanco que el INEGI clasifica como 9999 o 0000
        df_ocupados = df_ocupados[~df_ocupados['c_ocu11c_str'].isin(['9999', '0000', 'nan'])]
        
        # Extraer el primer dígito para los grandes grupos
        df_ocupados['sinco_1'] = df_ocupados['c_ocu11c_str'].str[0]
        
        total_ocupados = df_ocupados[PONDERATOR].sum()

        # --- AGRUPACIÓN A 1 DÍGITO ---
        dist_1d = df_ocupados.groupby('sinco_1')[PONDERATOR].sum().reset_index()
        dist_1d['Descripción'] = dist_1d['sinco_1'].map(sinco_1d_map).fillna('Ocupación no especificada')
        dist_1d = dist_1d.sort_values(by=PONDERATOR, ascending=False)
        dist_1d['Porcentaje'] = (dist_1d[PONDERATOR] / total_ocupados) * 100
        
        print("\n--- DISTRIBUCIÓN DE LA PEA OCUPADA (GRANDES SECTORES) ---")
        format_dict = {PONDERATOR: '{:,.0f}'.format, 'Porcentaje': '{:.2f}%'.format}
        print(dist_1d[['Descripción', PONDERATOR, 'Porcentaje']].to_string(index=False, formatters=format_dict))
        
        # --- AGRUPACIÓN A 4 DÍGITOS (TOP 15) ---
        print("\n--- TOP 15 PROFESIONES ESPECÍFICAS MÁS COMUNES EN MÉXICO ---")
        dist_4d = df_ocupados.groupby('c_ocu11c_str')[PONDERATOR].sum().reset_index()
        dist_4d = dist_4d.sort_values(by=PONDERATOR, ascending=False).head(15)
        dist_4d['Porcentaje'] = (dist_4d[PONDERATOR] / total_ocupados) * 100
        dist_4d['Profesión'] = dist_4d['c_ocu11c_str'].map(sinco_4d_map).fillna('Código SINCO específico (no en dict)')
        
        # Reordenar columnas para imprimir
        cols_print = ['c_ocu11c_str', 'Profesión', PONDERATOR, 'Porcentaje']
        print(dist_4d[cols_print].to_string(index=False, formatters=format_dict))
        
    else:
        print("Error: La columna 'c_ocu11c' no se encuentra en el DataFrame procesado.")

except Exception as e:
    print(f"Ocurrió un error: {e}")
