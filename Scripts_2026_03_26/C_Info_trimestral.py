import pandas as pd
import numpy as np
import os
import json
import warnings

warnings.filterwarnings("ignore")

# =========================================================================
# INPC HISTÓRICO (Base segunda quincena jul 2018 = 100)
# Cobertura: 2005-T1 a 2026-T1 (84 trimestres)
# =========================================================================
INPC_HISTORICO = [
    58.821, 59.137, 59.404, 60.118, 60.913, 60.928, 61.517, 62.575,
    63.460, 63.464, 63.745, 65.185, 66.108, 66.683, 67.630, 69.130,
    70.026, 70.544, 70.996, 71.812, 73.328, 73.249, 73.805, 74.751,
    75.576, 75.422, 75.745, 77.266, 78.807, 78.579, 79.451, 80.662,
    81.866, 82.345, 82.450, 83.813, 85.365, 85.296, 85.841, 87.338,
    88.050, 87.874, 88.195, 89.383, 90.445, 90.201, 90.753, 92.382,
    95.047, 95.735, 96.637, 98.477, 100.087, 100.025, 101.133, 102.957,
    103.929, 104.115, 104.556, 106.067, 107.538, 107.029, 108.307, 109.473,
    111.497, 113.048, 114.577, 117.114, 119.580, 121.832, 124.326, 126.495,
    128.538, 128.844, 130.126, 132.100, 133.767, 134.339, 136.032, 137.400,
    138.743, 140.012, 140.948, 142.465, 144.150
]

DEFLACTOR_DICT = {}
_idx = 0
for _y in range(2005, 2027):
    for _q in range(1, 5):
        if _idx < len(INPC_HISTORICO):
            DEFLACTOR_DICT[(_y, _q)] = INPC_HISTORICO[_idx]
            _idx += 1


class ENOEDataEngine:
    def __init__(self, target_year, target_quarter, base_dir="Data/ENOE_dta", verbose=True):
        """
        Inicializa el motor de datos.
        target_year: Año base para el análisis y la deflactación (ej. 2026).
        target_quarter: Trimestre base (ej. 1).
        base_dir: Directorio raíz donde se alojan las carpetas de la ENOE.
        verbose: Si True, imprime el progreso y los resultados intermedios.
        """
        self.t_year = target_year
        self.t_quarter = target_quarter
        self.base_dir = base_dir
        self.verbose = verbose
        self.periodos = self._calcular_periodos()
        self.inpc = DEFLACTOR_DICT

        if self.verbose:
            print("=" * 70)
            print("ENOEDataEngine — Inicialización")
            print("=" * 70)
            print(f"  Año/trimestre base (deflactación): {self.t_year}-T{self.t_quarter}")
            print(f"  INPC base: {self.inpc.get((self.t_year, self.t_quarter), 'N/A')}")
            print(f"  Directorio de datos: {self.base_dir}")
            print(f"  Deflactor INPC: {len(self.inpc)} trimestres (2005-T1 a 2026-T1)")
            print("\n  Periodos a procesar:")
            for key, (y, q) in self.periodos.items():
                inpc_val = self.inpc.get((y, q), "N/A")
                print(f"    {key:8s} -> {y}-T{q}  (INPC={inpc_val})")
            print()

    def _calcular_periodos(self):
        """Calcula dinámicamente las tuplas (Año, Trimestre) para los 5 periodos."""
        y, q = self.t_year, self.t_quarter

        # Trimestre anterior (T-1)
        y_t1, q_t1 = (y, q - 1) if q > 1 else (y - 1, 4)

        return {
            "Actual": (y, q),
            "T-1": (y_t1, q_t1),
            "Y-1": (y - 1, q),
            "Y-5": (y - 5, q),
            "Y-10": (y - 10, q)
        }

    def _get_stata_path(self, year, quarter, prefix):
        """Busca el archivo .dta entre varios nombres posibles."""
        y_short = str(year)[-2:]
        folder = os.path.join(self.base_dir, f"ENOE_{year}_{quarter}")
        candidates = [
            os.path.join(folder, f"ENOE_{prefix}{quarter}{y_short}.dta"),
            os.path.join(folder, f"ENOEN_{prefix}{quarter}{y_short}.dta"),
            os.path.join(folder, f"{prefix}{quarter}{y_short}.dta"),
            os.path.join(folder, f"enoe_{prefix.lower()}{quarter}{y_short}.dta"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _cargar_y_limpiar_trimestre(self, year, quarter):
        """Carga SDEMT y COE1T, maneja excepciones históricas y evita producto cartesiano."""
        if self.verbose:
            print(f"  [Carga] Buscando archivos para {year}-T{quarter}...")

        sdemt_path = self._get_stata_path(year, quarter, "SDEMT")
        coe1t_path = self._get_stata_path(year, quarter, "COE1T")

        if not sdemt_path or not coe1t_path:
            if self.verbose:
                print(f"  [Carga] ADVERTENCIA: Archivos no encontrados para {year}-T{quarter}")
                if not sdemt_path:
                    print(f"           SDEMT: no encontrado")
                if not coe1t_path:
                    print(f"           COE1T: no encontrado")
            return pd.DataFrame()

        try:
            if self.verbose:
                print(f"  [Carga] SDEMT -> {sdemt_path}")
                print(f"  [Carga] COE1T -> {coe1t_path}")

            df_sdemt = pd.read_stata(sdemt_path, convert_categoricals=False)
            df_coe1t = pd.read_stata(coe1t_path, convert_categoricals=False)

            df_sdemt.columns = df_sdemt.columns.str.lower()
            df_coe1t.columns = df_coe1t.columns.str.lower()

            # Excepciones Históricas: Estandarización de columnas
            for df in [df_sdemt, df_coe1t]:
                if 'ent' in df.columns and 'cve_ent' not in df.columns:
                    df.rename(columns={'ent': 'cve_ent'}, inplace=True)
                if 'mun' in df.columns and 'cve_mun' not in df.columns:
                    df.rename(columns={'mun': 'cve_mun'}, inplace=True)

            # Identificadores únicos de la ENOE
            llaves = ['cd_a', 'cve_ent', 'con', 'v_sel', 'n_hog', 'h_mud', 'n_ren']
            llaves = [k for k in llaves if k in df_sdemt.columns and k in df_coe1t.columns]

            dupes = df_coe1t.duplicated(subset=llaves).sum()
            if self.verbose and dupes > 0:
                print(f"  [Carga] Eliminando {dupes:,} duplicados en COE1T (evitar producto cartesiano)")

            df_coe1t = df_coe1t.drop_duplicates(subset=llaves)

            # Seleccionar columnas antes del merge para evitar sufijos _x/_y
            ponderador = 'fac_tri' if 'fac_tri' in df_sdemt.columns else 'fac'
            cols_sdemt = llaves + [ponderador, 'r_def', 'c_res', 'eda', 'clase1', 'clase2',
                                   'ingocup', 'sex', 'hrsocup']
            for col_rama in ['rama_est2', 'rama_est1']:
                if col_rama in df_sdemt.columns:
                    cols_sdemt.append(col_rama)
                    break

            cols_coe1t = llaves.copy()
            for col_prof in ['sinco', 'p3']:
                if col_prof in df_coe1t.columns:
                    cols_coe1t.append(col_prof)

            df_sdemt = df_sdemt[[c for c in cols_sdemt if c in df_sdemt.columns]]
            df_coe1t = df_coe1t[[c for c in cols_coe1t if c in df_coe1t.columns]]

            df_merge = pd.merge(df_sdemt, df_coe1t, on=llaves, how='inner')
            df_merge['r_def'] = df_merge['r_def'].astype(str).str.strip()

            if self.verbose:
                print(f"  [Carga] Registros fusionados: {len(df_merge):,}")

            return df_merge

        except Exception as e:
            print(f"  [Carga] ERROR en {year}-T{quarter}: {e}")
            return pd.DataFrame()

    def _deflactar_ingresos(self, df, year, quarter):
        """Convierte los ingresos nominales a pesos constantes del target_year/target_quarter."""
        inpc_actual = self.inpc.get((year, quarter))
        inpc_target = self.inpc.get((self.t_year, self.t_quarter))

        if inpc_actual is None:
            if self.verbose:
                print(f"  [Deflactación] ADVERTENCIA: INPC no disponible para {year}-T{quarter}, usando 100")
            inpc_actual = 100
        if inpc_target is None:
            if self.verbose:
                print(f"  [Deflactación] ADVERTENCIA: INPC no disponible para base {self.t_year}-T{self.t_quarter}, usando 100")
            inpc_target = 100

        factor_deflactor = inpc_target / inpc_actual

        if self.verbose:
            print(f"  [Deflactación] INPC origen={inpc_actual:.3f}, INPC base={inpc_target:.3f}, "
                  f"factor={factor_deflactor:.4f}")

        df['ingocup'] = pd.to_numeric(df.get('ingocup', 0), errors='coerce').fillna(0)
        df['ingocup_real'] = df['ingocup'] * factor_deflactor

        # Calcular ingreso por hora real (Asumiendo 4.345 semanas por mes)
        df['hrsocup'] = pd.to_numeric(df.get('hrsocup', 0), errors='coerce').fillna(0)
        df['horas_mensuales'] = df['hrsocup'] * 4.345

        df['ingreso_hora_real'] = np.where(
            df['horas_mensuales'] > 0,
            df['ingocup_real'] / df['horas_mensuales'],
            0
        )
        return df

    def _resolver_ponderador(self, df):
        """Identifica la columna de factor de expansión."""
        for col in ['fac_tri', 'fac']:
            if col in df.columns:
                return col
        return None

    def _weighted_avg(self, df, val_col, wt_col):
        """Calcula el promedio ponderado usando el factor de expansión."""
        if df.empty:
            return np.nan
        val = df[val_col].values
        wt = df[wt_col].values
        valid = ~np.isnan(val) & ~np.isnan(wt)
        if not valid.any() or wt[valid].sum() == 0:
            return np.nan
        return np.average(val[valid], weights=wt[valid])

    def procesar_metricas(self):
        """Procesa todos los periodos y retorna el JSON/Dict maestro estructurado."""
        if self.verbose:
            print("=" * 70)
            print("Procesando métricas trimestrales")
            print("=" * 70)

        resultados = {
            "metadata": {
                "periodos_labels": [],
                "base_inflacion": self.t_year,
                "base_trimestre": self.t_quarter,
                "inpc_base": self.inpc.get((self.t_year, self.t_quarter))
            },
            "seccion_A_macro": {"pob_total": [], "pea_total": [], "tasa_ocupacion_pct": []},
            "seccion_B_genero": {
                "hombres": {"ingreso_hora_real": [], "ingreso_mensual_real": [], "horas_mensuales": []},
                "mujeres": {"ingreso_hora_real": [], "ingreso_mensual_real": [], "horas_mensuales": []}
            },
            "seccion_C_sectores": {
                "primario": {"pct_pea": [], "horas_promedio": []},
                "secundario": {"pct_pea": [], "horas_promedio": []},
                "terciario": {"pct_pea": [], "horas_promedio": []}
            },
            "seccion_D_top_ocupaciones": {}
        }

        orden_keys = ["Y-10", "Y-5", "Y-1", "T-1", "Actual"]

        for key in orden_keys:
            y, q = self.periodos[key]
            label = f"Q{q} {y}"

            if self.verbose:
                print(f"\n--- Periodo: {key} ({label}) ---")

            resultados["metadata"]["periodos_labels"].append(label)

            df = self._cargar_y_limpiar_trimestre(y, q)
            if df.empty:
                if self.verbose:
                    print(f"  [Métricas] Sin datos — se omite {label}")
                continue

            fac = self._resolver_ponderador(df)
            if fac is None:
                if self.verbose:
                    print(f"  [Métricas] ERROR: No se encontró ponderador (fac/fac_tri) en {label}")
                continue

            df[fac] = pd.to_numeric(df[fac], errors='coerce').fillna(0)

            if self.verbose:
                print(f"  [Métricas] Ponderador utilizado: '{fac}'")

            df = self._deflactar_ingresos(df, y, q)

            # Filtros Universales
            df['eda'] = pd.to_numeric(df.get('eda', 0), errors='coerce')
            pob_15_mas = df[(df['r_def'] == '0.0') &
                            (df['c_res'].isin([1, 3])) & (df['eda'] >= 15)]

            pea = pob_15_mas[pob_15_mas['clase1'] == 1]
            ocupados = pea[pea['clase2'] == 1]
            ocup_con_ingreso = ocupados[ocupados['ingocup_real'] > 0]

            if self.verbose:
                print(f"  [Métricas] Población 15+: {len(pob_15_mas):,} registros "
                      f"(expansión: {pob_15_mas[fac].sum():,.0f})")
                print(f"  [Métricas] PEA: {len(pea):,} registros (expansión: {pea[fac].sum():,.0f})")
                print(f"  [Métricas] Ocupados: {len(ocupados):,} registros (expansión: {ocupados[fac].sum():,.0f})")

            # --- SECCIÓN A: MACRO ---
            tot_pob = pob_15_mas[fac].sum()
            tot_pea = pea[fac].sum()
            tot_ocup = ocupados[fac].sum()
            tasa_ocup = (tot_ocup / tot_pea * 100) if tot_pea else 0

            resultados["seccion_A_macro"]["pob_total"].append(tot_pob)
            resultados["seccion_A_macro"]["pea_total"].append(tot_pea)
            resultados["seccion_A_macro"]["tasa_ocupacion_pct"].append(tasa_ocup)

            if self.verbose:
                print(f"  [Sección A] Población 15+: {tot_pob:,.0f}")
                print(f"  [Sección A] PEA:            {tot_pea:,.0f}")
                print(f"  [Sección A] Tasa ocupación: {tasa_ocup:.2f}%")

            # --- SECCIÓN B: BRECHAS DE GÉNERO ---
            for sexo, label_sexo in [(1, 'hombres'), (2, 'mujeres')]:
                df_sex = ocup_con_ingreso[ocup_con_ingreso['sex'] == sexo]
                ing_mensual = self._weighted_avg(df_sex, 'ingocup_real', fac)
                ing_hora = self._weighted_avg(df_sex, 'ingreso_hora_real', fac)
                horas = self._weighted_avg(ocupados[ocupados['sex'] == sexo], 'horas_mensuales', fac)

                resultados["seccion_B_genero"][label_sexo]["ingreso_mensual_real"].append(ing_mensual)
                resultados["seccion_B_genero"][label_sexo]["ingreso_hora_real"].append(ing_hora)
                resultados["seccion_B_genero"][label_sexo]["horas_mensuales"].append(horas)

                if self.verbose:
                    print(f"  [Sección B] {label_sexo.capitalize()}:")
                    print(f"              Ingreso mensual real: ${ing_mensual:,.0f}" if pd.notna(ing_mensual)
                          else f"              Ingreso mensual real: N/A")
                    print(f"              Ingreso hora real:    ${ing_hora:,.2f}" if pd.notna(ing_hora)
                          else f"              Ingreso hora real:    N/A")
                    print(f"              Horas mensuales:      {horas:,.1f}" if pd.notna(horas)
                          else f"              Horas mensuales:      N/A")

            # --- SECCIÓN C: SECTORES ---
            col_rama = 'rama_est2' if 'rama_est2' in ocupados.columns else 'rama_est1'
            if self.verbose:
                print(f"  [Sección C] Variable sectorial: '{col_rama}'")

            rama = pd.to_numeric(ocupados[col_rama], errors='coerce').fillna(12)
            ocupados = ocupados.copy()
            ocupados['Sector'] = np.select(
                [rama == 1, rama.isin([2, 3, 4]), rama.isin([5, 6, 7, 8, 9, 10, 11])],
                ['primario', 'secundario', 'terciario'],
                default='otro'
            )

            for sec in ['primario', 'secundario', 'terciario']:
                df_sec = ocupados[ocupados['Sector'] == sec]
                pob_sec = df_sec[fac].sum()
                pct_pea = (pob_sec / tot_pea * 100) if tot_pea else 0
                horas_prom = self._weighted_avg(df_sec, 'horas_mensuales', fac)

                resultados["seccion_C_sectores"][sec]["pct_pea"].append(pct_pea)
                resultados["seccion_C_sectores"][sec]["horas_promedio"].append(horas_prom)

                if self.verbose:
                    print(f"  [Sección C] {sec.capitalize()}: {pct_pea:.2f}% de la PEA, "
                          f"horas promedio={horas_prom:,.1f}" if pd.notna(horas_prom)
                          else f"  [Sección C] {sec.capitalize()}: {pct_pea:.2f}% de la PEA, horas promedio=N/A")

            # --- SECCIÓN D: TOP OCUPACIONES (Solo periodo Actual) ---
            if key == "Actual":
                col_sinco = 'sinco' if 'sinco' in ocupados.columns else 'p3'
                if self.verbose:
                    print(f"  [Sección D] Ranking top ocupaciones (variable '{col_sinco}')")

                df_prof = ocupados.groupby(col_sinco).agg(
                    personas=(fac, 'sum'),
                    ingreso_promedio=('ingreso_hora_real', lambda x: np.average(
                        x, weights=ocupados.loc[x.index, fac]
                    ))
                ).reset_index().sort_values('personas', ascending=False)

                top5 = df_prof.head(5).to_dict('records')
                resultados["seccion_D_top_ocupaciones"] = top5

                if self.verbose:
                    print(f"  [Sección D] Top 5 ocupaciones:")
                    for i, row in enumerate(top5, 1):
                        print(f"              {i}. Código {row[col_sinco]}: "
                              f"{row['personas']:,.0f} personas, "
                              f"${row['ingreso_promedio']:,.2f}/hr real")

        if self.verbose:
            print("\n" + "=" * 70)
            print("Procesamiento finalizado")
            print("=" * 70)

        return resultados


def mostrar_resultados(resultados):
    """Imprime en pantalla el diccionario completo de resultados de forma legible."""
    print("\n" + "#" * 70)
    print("RESULTADOS COMPLETOS")
    print("#" * 70)

    labels = resultados["metadata"]["periodos_labels"]
    print(f"\nMetadata:")
    print(f"  Periodos: {labels}")
    print(f"  Base inflación: {resultados['metadata']['base_inflacion']}-T{resultados['metadata']['base_trimestre']}")
    print(f"  INPC base: {resultados['metadata']['inpc_base']}")

    print(f"\n--- Sección A: Indicadores Macroeconómicos ---")
    for i, lbl in enumerate(labels):
        pob = resultados["seccion_A_macro"]["pob_total"][i] if i < len(resultados["seccion_A_macro"]["pob_total"]) else "N/A"
        pea = resultados["seccion_A_macro"]["pea_total"][i] if i < len(resultados["seccion_A_macro"]["pea_total"]) else "N/A"
        toc = resultados["seccion_A_macro"]["tasa_ocupacion_pct"][i] if i < len(resultados["seccion_A_macro"]["tasa_ocupacion_pct"]) else "N/A"
        if isinstance(pob, (int, float)):
            print(f"  {lbl}: Población={pob:,.0f}, PEA={pea:,.0f}, Tasa ocupación={toc:.2f}%")
        else:
            print(f"  {lbl}: Sin datos")

    print(f"\n--- Sección B: Brechas de Género ---")
    for sexo in ['hombres', 'mujeres']:
        print(f"  {sexo.capitalize()}:")
        for metrica in ['ingreso_mensual_real', 'ingreso_hora_real', 'horas_mensuales']:
            vals = resultados["seccion_B_genero"][sexo][metrica]
            print(f"    {metrica}:")
            for j, lbl in enumerate(labels):
                if j < len(vals) and pd.notna(vals[j]):
                    if 'mensual' in metrica and 'horas' not in metrica:
                        fmt = f"${vals[j]:,.0f}"
                    elif 'hora' in metrica:
                        fmt = f"${vals[j]:,.2f}"
                    else:
                        fmt = f"{vals[j]:,.1f}"
                    print(f"      {lbl}: {fmt}")
                elif j < len(vals):
                    print(f"      {lbl}: N/A")

    print(f"\n--- Sección C: Sectores Económicos ---")
    for sec in ['primario', 'secundario', 'terciario']:
        print(f"  {sec.capitalize()}:")
        for metrica in ['pct_pea', 'horas_promedio']:
            vals = resultados["seccion_C_sectores"][sec][metrica]
            print(f"    {metrica}:")
            for j, lbl in enumerate(labels):
                if j < len(vals) and pd.notna(vals[j]):
                    fmt = f"{vals[j]:.2f}%" if 'pct' in metrica else f"{vals[j]:,.1f} hrs"
                    print(f"      {lbl}: {fmt}")
                elif j < len(vals):
                    print(f"      {lbl}: N/A")

    print(f"\n--- Sección D: Top 5 Ocupaciones (periodo actual) ---")
    top = resultados["seccion_D_top_ocupaciones"]
    if top:
        for i, row in enumerate(top, 1):
            codigo = list(row.keys())[0]
            print(f"  {i}. {codigo}={row[codigo]} | Personas={row['personas']:,.0f} | "
                  f"Ingreso/hr=${row['ingreso_promedio']:,.2f}")
    else:
        print("  (Sin datos)")

    print(f"\n--- JSON estructurado ---")
    print(json.dumps(resultados, indent=2, default=str))


if __name__ == '__main__':
    motor = ENOEDataEngine(target_year=2026, target_quarter=1, verbose=True)
    datos_infografia = motor.procesar_metricas()
    mostrar_resultados(datos_infografia)
