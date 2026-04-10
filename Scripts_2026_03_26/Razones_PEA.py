import pandas as pd
import numpy as np
import os

path_sdemt = "Data/ENOE_dta/ENOE_2025_4/ENOE_SDEMT425.dta"
path_coe1t = "Data/ENOE_dta/ENOE_2025_4/ENOE_COE1T425.dta"

print("Cargando bases y cruzando con el Cuestionario Oficial (COE1T)...")
try:
    df_sdemt = pd.read_stata(path_sdemt, convert_categoricals=False)
    df_coe1t = pd.read_stata(path_coe1t, convert_categoricals=False)

    df_sdemt.columns = df_sdemt.columns.str.lower()
    df_coe1t.columns = df_coe1t.columns.str.lower()

    # Estandarizar identificadores geográficos
    if 'ent' in df_sdemt.columns and 'cve_ent' not in df_sdemt.columns: df_sdemt.rename(columns={'ent': 'cve_ent'}, inplace=True)
    if 'ent' in df_coe1t.columns and 'cve_ent' not in df_coe1t.columns: df_coe1t.rename(columns={'ent': 'cve_ent'}, inplace=True)

    llaves_merge = [k for k in ['cd_a', 'cve_ent', 'con', 'v_sel', 'n_hog', 'h_mud', 'n_ren'] if k in df_sdemt.columns]

    PONDERATOR = 'fac_tri' if 'fac_tri' in df_sdemt.columns else 'fac'
    df_sdemt[PONDERATOR] = pd.to_numeric(df_sdemt[PONDERATOR], errors='coerce').fillna(0)
    df_sdemt['r_def'] = df_sdemt['r_def'].astype(str).str.strip()

    # Filtrar universo base en SDEMT
    df_base = df_sdemt[(df_sdemt['r_def'] == '0.0') & (df_sdemt['c_res'].isin([1, 3])) & (df_sdemt['eda'] >= 15)].copy()

    # Detectar cuál es la pregunta de inactividad (p1b o p1c dependiendo del trimestre)
    col_razon = None
    if 'p1b' in df_coe1t.columns: col_razon = 'p1b'
    elif 'p1c' in df_coe1t.columns: col_razon = 'p1c'
    elif 'p1d' in df_coe1t.columns: col_razon = 'p1d'

    if not col_razon:
        raise ValueError("No se encontró la pregunta p1b o p1c en el cuestionario COE1T.")

    # Cruzar bases
    df_merge = pd.merge(df_base, df_coe1t[llaves_merge + [col_razon]], on=llaves_merge, how='left')
    df_merge[col_razon] = pd.to_numeric(df_merge[col_razon], errors='coerce')

    resultados = []

    for sex_code, sex_label in [(1, 'Hombres'), (2, 'Mujeres')]:
        df_sex = df_merge[df_merge['sex'] == sex_code]

        pob_total = df_sex[PONDERATOR].sum()
        pea = df_sex[df_sex['clase1'] == 1][PONDERATOR].sum()
        pnea = df_sex[df_sex['clase1'] == 2][PONDERATOR].sum()

        # Aislar a los inactivos
        df_pnea = df_sex[df_sex['clase1'] == 2]

        # Mapeo según el cuestionario oficial del INEGI
        # 1 a 12 = Disponibles (Desempleo Oculto)
        # 13 = Quehaceres del hogar
        # 14 = Estudiante
        # 15 = Jubilado/pensionado
        # 16 = Incapacitado
        # 17 o más, o nulos = Otros motivos

        pnea_disp = df_pnea[df_pnea[col_razon] <= 12][PONDERATOR].sum()
        pnea_quehac = df_pnea[df_pnea[col_razon] == 13][PONDERATOR].sum()
        pnea_estud = df_pnea[df_pnea[col_razon] == 14][PONDERATOR].sum()
        pnea_jubil = df_pnea[df_pnea[col_razon] == 15][PONDERATOR].sum()
        pnea_incap = df_pnea[df_pnea[col_razon] == 16][PONDERATOR].sum()
        pnea_otros = df_pnea[(df_pnea[col_razon] >= 17) | (df_pnea[col_razon].isna())][PONDERATOR].sum()

        res_dict = {
            'Género': sex_label,
            'Población > 15': pob_total,
            'PEA': pea,
            'PNEA Total': pnea,
            'Disponibles (Personas)': pnea_disp,
            'Quehaceres del hogar (Personas)': pnea_quehac,
            'Estudiantes (Personas)': pnea_estud,
            'Jubilados / Pensionados (Personas)': pnea_jubil,
            'Incapacitados (Personas)': pnea_incap,
            'Otros motivos (Personas)': pnea_otros
        }

        # Calcular porcentajes relativos a la PNEA
        for k in list(res_dict.keys()):
            if '(Personas)' in k:
                label_pct = k.replace('(Personas)', '(%)')
                res_dict[label_pct] = (res_dict[k] / pnea * 100) if pnea > 0 else 0

        resultados.append(res_dict)

    df_res = pd.DataFrame(resultados)
    pd.options.display.float_format = '{:,.1f}'.format

    print(f"\n=== DESGLOSE DE PNEA POR CUESTIONARIO DIRECTO (Pregunta {col_razon.upper()}) ===")
    for index, row in df_res.iterrows():
        print(f"\n--- {row['Género']} ---")
        print(f"Población > 15 años: {row['Población > 15']:,.0f}")
        print(f"PEA:                 {row['PEA']:,.0f}")
        print(f"PNEA:                {row['PNEA Total']:,.0f}")
        print("Razones para no buscar empleo:")
        print(f"  - Quehaceres del hogar: {row['Quehaceres del hogar (Personas)']:,.0f} ({row['Quehaceres del hogar (%)']:.1f}%)")
        print(f"  - Estudiantes:          {row['Estudiantes (Personas)']:,.0f} ({row['Estudiantes (%)']:.1f}%)")
        print(f"  - Jubilados/Pensiones:  {row['Jubilados / Pensionados (Personas)']:,.0f} ({row['Jubilados / Pensionados (%)']:.1f}%)")
        print(f"  - Disponibles (Oculto): {row['Disponibles (Personas)']:,.0f} ({row['Disponibles (%)']:.1f}%)")
        print(f"  - Incapacitados:        {row['Incapacitados (Personas)']:,.0f} ({row['Incapacitados (%)']:.1f}%)")
        print(f"  - Otros:                {row['Otros motivos (Personas)']:,.0f} ({row['Otros motivos (%)']:.1f}%)")
        
    df_res.to_csv("PNEA_Motivos_Cuestionario_Q4_2025.csv", index=False, sep=';', decimal=',')
    print("\nArchivo de respaldo generado exitosamente: 'PNEA_Motivos_Cuestionario_Q4_2025.csv'")

except Exception as e:
    print(f"Ocurrió un error al procesar los microdatos: {e}")