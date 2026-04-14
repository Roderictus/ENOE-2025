import pandas as pd
import os

def calcular_pnea_femenina_real():
    path_sdemt = "Data/ENOE_dta/ENOE_2025_4/ENOE_SDEMT425.dta"
    path_coe1t = "Data/ENOE_dta/ENOE_2025_4/ENOE_COE1T425.dta"

    print("Cruzando bases para extraer motivos directos de la Pregunta P1...")
    try:
        # 1. Cargar bases
        df_sdemt = pd.read_stata(path_sdemt, convert_categoricals=False)
        df_coe1t = pd.read_stata(path_coe1t, convert_categoricals=False)

        df_sdemt.columns = df_sdemt.columns.str.lower()
        df_coe1t.columns = df_coe1t.columns.str.lower()

        # Estandarizar identificador geográfico
        if 'ent' in df_sdemt.columns and 'cve_ent' not in df_sdemt.columns: df_sdemt.rename(columns={'ent': 'cve_ent'}, inplace=True)
        if 'ent' in df_coe1t.columns and 'cve_ent' not in df_coe1t.columns: df_coe1t.rename(columns={'ent': 'cve_ent'}, inplace=True)

        llaves_merge = [k for k in ['cd_a', 'cve_ent', 'con', 'v_sel', 'n_hog', 'h_mud', 'n_ren'] if k in df_sdemt.columns]

        PONDERATOR = 'fac_tri' if 'fac_tri' in df_sdemt.columns else 'fac'
        df_sdemt[PONDERATOR] = pd.to_numeric(df_sdemt[PONDERATOR], errors='coerce').fillna(0)
        df_sdemt['r_def'] = df_sdemt['r_def'].astype(str).str.strip()

        # 2. Filtrar Universo: Mujeres, mayores de 15 años, residentes definitivas y en la PNEA
        df_mujeres_pnea = df_sdemt[(df_sdemt['sex'] == 2) & 
                                   (df_sdemt['r_def'] == '0.0') & 
                                   (df_sdemt['c_res'].isin([1, 3])) & 
                                   (df_sdemt['eda'] >= 15) & 
                                   (df_sdemt['clase1'] == 2)].copy()

        # 3. Cruzar con el cuestionario COE1T para traer la variable 'p1'
        df_merge = pd.merge(df_mujeres_pnea, df_coe1t[llaves_merge + ['p1']], on=llaves_merge, how='left')
        df_merge['p1'] = pd.to_numeric(df_merge['p1'], errors='coerce')

        # 4. Calcular los volúmenes expandidos (Total nacional de Mujeres en PNEA)
        total_mujeres_pnea = df_merge[PONDERATOR].sum()

        # Extraer por motivo (Respuestas de la Pregunta 1)
        estudiantes = df_merge[df_merge['p1'] == 4][PONDERATOR].sum()
        quehaceres = df_merge[df_merge['p1'] == 5][PONDERATOR].sum()
        jubiladas = df_merge[df_merge['p1'] == 6][PONDERATOR].sum()
        incapacitadas = df_merge[df_merge['p1'] == 7][PONDERATOR].sum()
        otras_razones = df_merge[(df_merge['p1'] == 8) | (df_merge['p1'].isna())][PONDERATOR].sum()

        # 5. Imprimir el dato auditable
        print("\n" + "="*70)
        print(f" MUJERES EN INACTIVIDAD LABORAL (PNEA) - Q4 2025")
        print("="*70)
        print(f"Total de Mujeres Inactivas (>15 años): {total_mujeres_pnea:,.0f} personas (100.0%)")
        print("-" * 70)
        print(f"  -> Dedicadas a Quehaceres del hogar:  {quehaceres:,.0f} ({(quehaceres/total_mujeres_pnea)*100:.1f}%)")
        print(f"  -> Estudiantes:                       {estudiantes:,.0f} ({(estudiantes/total_mujeres_pnea)*100:.1f}%)")
        print(f"  -> Jubiladas o Pensionadas:           {jubiladas:,.0f} ({(jubiladas/total_mujeres_pnea)*100:.1f}%)")
        print(f"  -> Incapacitadas permanentemente:     {incapacitadas:,.0f} ({(incapacitadas/total_mujeres_pnea)*100:.1f}%)")
        print(f"  -> Otras razones (Desalentadas):      {otras_razones:,.0f} ({(otras_razones/total_mujeres_pnea)*100:.1f}%)")
        print("="*70)

    except Exception as e:
        print(f"Error al procesar: {e}")

if __name__ == '__main__':
    calcular_pnea_femenina_real()