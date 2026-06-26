import pandas as pd
import numpy as np
import os
import json
import warnings

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_BASE_DIR = os.path.join(_PROJECT_ROOT, "Data", "ENOE_dta")

# INPC histórico (base jul-2018 = 100), cobertura 2005-T1 a 2026-T1
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
    def __init__(self, target_year, target_quarter, base_dir=None, verbose=True):
        self.t_year = target_year
        self.t_quarter = target_quarter
        self.base_dir = base_dir or DEFAULT_BASE_DIR
        self.verbose = verbose
        self.periodos = self._calcular_periodos()
        self.inpc = DEFLACTOR_DICT

        if self.verbose:
            print("=" * 70)
            print("ENOEDataEngine — Inicialización")
            print("=" * 70)
            print(f"  Año/trimestre base: {self.t_year}-T{self.t_quarter}")
            print(f"  INPC base: {self.inpc.get((self.t_year, self.t_quarter), 'N/A')}")
            print(f"  Directorio de datos: {self.base_dir}")
            for key, (y, q) in self.periodos.items():
                print(f"    {key:8s} -> {y}-T{q}  (INPC={self.inpc.get((y, q), 'N/A')})")
            print()

    def _calcular_periodos(self):
        y, q = self.t_year, self.t_quarter
        y_t1, q_t1 = (y, q - 1) if q > 1 else (y - 1, 4)
        return {
            "Actual": (y, q),
            "T-1": (y_t1, q_t1),
            "Y-1": (y - 1, q),
            "Y-5": (y - 5, q),
            "Y-10": (y - 10, q),
        }

    def _get_stata_path(self, year, quarter, prefix):
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
        if self.verbose:
            print(f"  [Carga] Buscando archivos para {year}-T{quarter}...")

        sdemt_path = self._get_stata_path(year, quarter, "SDEMT")
        coe1t_path = self._get_stata_path(year, quarter, "COE1T")

        if not sdemt_path or not coe1t_path:
            if self.verbose:
                print(f"  [Carga] ADVERTENCIA: Archivos no encontrados para {year}-T{quarter}")
            return pd.DataFrame()

        try:
            if self.verbose:
                print(f"  [Carga] SDEMT -> {sdemt_path}")
                print(f"  [Carga] COE1T -> {coe1t_path}")

            df_sdemt = pd.read_stata(sdemt_path, convert_categoricals=False)
            df_coe1t = pd.read_stata(coe1t_path, convert_categoricals=False)

            df_sdemt.columns = df_sdemt.columns.str.lower()
            df_coe1t.columns = df_coe1t.columns.str.lower()

            for df in [df_sdemt, df_coe1t]:
                if 'ent' in df.columns and 'cve_ent' not in df.columns:
                    df.rename(columns={'ent': 'cve_ent'}, inplace=True)
                if 'mun' in df.columns and 'cve_mun' not in df.columns:
                    df.rename(columns={'mun': 'cve_mun'}, inplace=True)

            llaves = ['cd_a', 'cve_ent', 'con', 'v_sel', 'n_hog', 'h_mud', 'n_ren']
            llaves = [k for k in llaves if k in df_sdemt.columns and k in df_coe1t.columns]

            dupes = df_coe1t.duplicated(subset=llaves).sum()
            if self.verbose and dupes > 0:
                print(f"  [Carga] Eliminando {dupes:,} duplicados en COE1T")

            df_coe1t = df_coe1t.drop_duplicates(subset=llaves)

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
        inpc_actual = self.inpc.get((year, quarter), 100)
        inpc_target = self.inpc.get((self.t_year, self.t_quarter), 100)
        factor_deflactor = inpc_target / inpc_actual

        if self.verbose:
            print(f"  [Deflactación] INPC {year}-T{quarter}={inpc_actual:.3f}, "
                  f"base={inpc_target:.3f}, factor={factor_deflactor:.4f}")

        df['ingocup'] = pd.to_numeric(df.get('ingocup', 0), errors='coerce').fillna(0)
        df['ingocup_real'] = df['ingocup'] * factor_deflactor
        df['hrsocup'] = pd.to_numeric(df.get('hrsocup', 0), errors='coerce').fillna(0)
        df['horas_mensuales'] = df['hrsocup'] * 4.345
        df['ingreso_hora_real'] = np.where(
            df['horas_mensuales'] > 0,
            df['ingocup_real'] / df['horas_mensuales'],
            0,
        )
        return df

    def _resolver_ponderador(self, df):
        for col in ['fac_tri', 'fac']:
            if col in df.columns:
                return col
        return None

    def _weighted_avg(self, df, val_col, wt_col):
        if df.empty:
            return np.nan
        val = df[val_col].values
        wt = df[wt_col].values
        valid = ~np.isnan(val) & ~np.isnan(wt)
        if not valid.any() or wt[valid].sum() == 0:
            return np.nan
        return np.average(val[valid], weights=wt[valid])

    def procesar_metricas(self):
        if self.verbose:
            print("=" * 70)
            print("Procesando métricas trimestrales")
            print("=" * 70)

        resultados = {
            "metadata": {
                "periodos_labels": [],
                "base_inflacion": self.t_year,
                "base_trimestre": self.t_quarter,
                "inpc_base": self.inpc.get((self.t_year, self.t_quarter)),
            },
            "seccion_A_macro": {"pob_total": [], "pea_total": [], "tasa_ocupacion_pct": []},
            "seccion_B_genero": {
                "hombres": {"ingreso_hora_real": [], "ingreso_mensual_real": [], "horas_mensuales": []},
                "mujeres": {"ingreso_hora_real": [], "ingreso_mensual_real": [], "horas_mensuales": []},
            },
            "seccion_C_sectores": {
                "primario": {"pct_pea": [], "horas_promedio": []},
                "secundario": {"pct_pea": [], "horas_promedio": []},
                "terciario": {"pct_pea": [], "horas_promedio": []},
            },
            "seccion_D_top_ocupaciones": {},
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
                continue

            fac = self._resolver_ponderador(df)
            if fac is None:
                print(f"  ERROR: No se encontró ponderador en {label}")
                continue

            df[fac] = pd.to_numeric(df[fac], errors='coerce').fillna(0)
            df = self._deflactar_ingresos(df, y, q)

            df['eda'] = pd.to_numeric(df.get('eda', 0), errors='coerce')
            pob_15_mas = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3])) & (df['eda'] >= 15)]
            pea = pob_15_mas[pob_15_mas['clase1'] == 1]
            ocupados = pea[pea['clase2'] == 1].copy()
            ocup_con_ingreso = ocupados[ocupados['ingocup_real'] > 0]

            tot_pob = pob_15_mas[fac].sum()
            tot_pea = pea[fac].sum()
            tot_ocup = ocupados[fac].sum()
            tasa_ocup = (tot_ocup / tot_pea * 100) if tot_pea else 0

            resultados["seccion_A_macro"]["pob_total"].append(tot_pob)
            resultados["seccion_A_macro"]["pea_total"].append(tot_pea)
            resultados["seccion_A_macro"]["tasa_ocupacion_pct"].append(tasa_ocup)

            if self.verbose:
                print(f"  [A] Población={tot_pob:,.0f}, PEA={tot_pea:,.0f}, Ocupación={tasa_ocup:.2f}%")

            for sexo, label_sexo in [(1, 'hombres'), (2, 'mujeres')]:
                df_sex = ocup_con_ingreso[ocup_con_ingreso['sex'] == sexo]
                ing_m = self._weighted_avg(df_sex, 'ingocup_real', fac)
                ing_h = self._weighted_avg(df_sex, 'ingreso_hora_real', fac)
                horas = self._weighted_avg(ocupados[ocupados['sex'] == sexo], 'horas_mensuales', fac)
                resultados["seccion_B_genero"][label_sexo]["ingreso_mensual_real"].append(ing_m)
                resultados["seccion_B_genero"][label_sexo]["ingreso_hora_real"].append(ing_h)
                resultados["seccion_B_genero"][label_sexo]["horas_mensuales"].append(horas)

            col_rama = 'rama_est2' if 'rama_est2' in ocupados.columns else 'rama_est1'
            rama = pd.to_numeric(ocupados[col_rama], errors='coerce').fillna(12)
            ocupados['Sector'] = np.select(
                [rama == 1, rama.isin([2, 3, 4]), rama.isin([5, 6, 7, 8, 9, 10, 11])],
                ['primario', 'secundario', 'terciario'],
                default='otro',
            )

            for sec in ['primario', 'secundario', 'terciario']:
                df_sec = ocupados[ocupados['Sector'] == sec]
                pob_sec = df_sec[fac].sum()
                resultados["seccion_C_sectores"][sec]["pct_pea"].append(
                    (pob_sec / tot_pea * 100) if tot_pea else 0
                )
                resultados["seccion_C_sectores"][sec]["horas_promedio"].append(
                    self._weighted_avg(df_sec, 'horas_mensuales', fac)
                )

            if key == "Actual":
                col_sinco = 'sinco' if 'sinco' in ocupados.columns else 'p3'
                df_prof = ocupados.groupby(col_sinco).agg(
                    personas=(fac, 'sum'),
                    ingreso_promedio=('ingreso_hora_real', lambda x: np.average(
                        x, weights=ocupados.loc[x.index, fac]
                    )),
                ).reset_index().sort_values('personas', ascending=False)
                resultados["seccion_D_top_ocupaciones"] = df_prof.head(5).to_dict('records')

        if self.verbose:
            print("\n" + "=" * 70)
            print("Procesamiento finalizado")
            print("=" * 70)

        return resultados


def mostrar_resultados(resultados):
    print("\n" + "#" * 70)
    print("RESULTADOS COMPLETOS")
    print("#" * 70)
    print(json.dumps(resultados, indent=2, default=str))


if __name__ == '__main__':
    motor = ENOEDataEngine(target_year=2026, target_quarter=1)
    datos_infografia = motor.procesar_metricas()
    mostrar_resultados(datos_infografia)
