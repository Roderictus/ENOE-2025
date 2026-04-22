import pandas as pd
import numpy as np
import os

def procesar_cero_ingresos_regional_completo():
    path_sdemt = "Data/ENOE_dta/ENOE_2025_4/ENOE_SDEMT425.dta"
    print("Cargando la base sociodemográfica (SDEMT)...")
    
    try:
        df = pd.read_stata(path_sdemt, convert_categoricals=False)
        df.columns = df.columns.str.lower()
        
        # 1. ESTANDARIZAR NOMBRES DE COLUMNAS
        if 'ent' in df.columns and 'cve_ent' not in df.columns:
            df.rename(columns={'ent': 'cve_ent'}, inplace=True)
        if 'cve_mun' in df.columns and 'mun' not in df.columns:
            df.rename(columns={'cve_mun': 'mun'}, inplace=True)
            
        PONDERATOR = 'fac_tri' if 'fac_tri' in df.columns else 'fac'
        df[PONDERATOR] = pd.to_numeric(df[PONDERATOR], errors='coerce').fillna(0)
        df['r_def'] = df['r_def'].astype(str).str.strip()
        df['ingocup'] = pd.to_numeric(df.get('ingocup', 0), errors='coerce').fillna(0)
        df['cd_a'] = pd.to_numeric(df.get('cd_a', np.nan), errors='coerce')
        
        # 2. FILTROS DEL UNIVERSO
        # PEA Ocupada (Residentes, >15 años, clase1=1, clase2=1)
        df_oc = df[(df['r_def'] == '0.0') & (df['c_res'].isin([1, 3])) & 
                   (df['eda'] >= 15) & (df['clase1'] == 1) & (df['clase2'] == 1)].copy()
                   
        # Indicador de Cero Ingresos (Trabaja, pero reporta $0 o Nulo)
        df_oc['cero_ingresos'] = (df_oc['ingocup'] <= 0).astype(int)
        
        # Diccionarios de Catálogos INEGI
        estados = {
            1:'Aguascalientes', 2:'Baja California', 3:'Baja California Sur', 4:'Campeche',
            5:'Coahuila', 6:'Colima', 7:'Chiapas', 8:'Chihuahua', 9:'Ciudad de México',
            10:'Durango', 11:'Guanajuato', 12:'Guerrero', 13:'Hidalgo', 14:'Jalisco',
            15:'Estado de México', 16:'Michoacán', 17:'Morelos', 18:'Nayarit', 19:'Nuevo León',
            20:'Oaxaca', 21:'Puebla', 22:'Querétaro', 23:'Quintana Roo', 24:'San Luis Potosí',
            25:'Sinaloa', 26:'Sonora', 27:'Tabasco', 28:'Tamaulipas', 29:'Tlaxcala',
            30:'Veracruz', 31:'Yucatán', 32:'Zacatecas'
        }

        ciudades = {
            1:'México', 2:'Guadalajara', 3:'Monterrey', 4:'Puebla', 5:'León',
            6:'Torreón', 7:'San Luis Potosí', 8:'Mérida', 9:'Chihuahua', 10:'Tampico',
            11:'Ciudad Juárez', 12:'Tijuana', 13:'Veracruz', 14:'Acapulco', 15:'Aguascalientes',
            16:'Morelia', 17:'Toluca', 18:'Saltillo', 19:'Villahermosa', 20:'Tuxtla Gutiérrez',
            21:'Hermosillo', 22:'Culiacán', 23:'Xalapa', 24:'Cancún', 25:'Oaxaca',
            26:'Querétaro', 27:'Tlaxcala', 28:'Tepic', 29:'Campeche', 30:'Cuernavaca',
            31:'La Paz', 32:'Colima', 33:'Zacatecas', 34:'Durango', 35:'Ciudad Victoria',
            36:'Pachuca', 37:'Chilpancingo', 38:'Chetumal', 39:'Tapachula'
        }

        # ==========================================
        # FUNCIÓN MAESTRA DE AGREGACIÓN
        # ==========================================
        def procesar_nivel(df_base, df_ocupados, group_cols, is_mun=False):
            # A. Muestra total
            muestra = df_base.groupby(group_cols).size().reset_index(name='Total_Entrevistados_General')
            
            # B. Estadísticas de la población Ocupada
            oc_stats = df_ocupados.groupby(group_cols).agg(
                Ocupados_Muestra=('cero_ingresos', 'count'),
                CeroIng_Muestra=('cero_ingresos', 'sum'),
                Ocupados_Pob=(PONDERATOR, 'sum')
            ).reset_index()
            
            # C. Población expandida SIN ingresos
            ing_pob = df_ocupados[df_ocupados['cero_ingresos'] == 1].groupby(group_cols)[PONDERATOR].sum().reset_index(name='CeroIng_Pob')
            
            # Unir todo
            res = muestra.merge(oc_stats, on=group_cols, how='inner')
            res = res.merge(ing_pob, on=group_cols, how='left').fillna({'CeroIng_Pob': 0})
            
            # Filtro de 500 entrevistas para municipios
            if is_mun:
                res = res[res['Total_Entrevistados_General'] >= 500].copy()
                
            # Porcentajes
            res['% Cero Ingresos (Muestral)'] = (res['CeroIng_Muestra'] / res['Ocupados_Muestra']) * 100
            res['% Cero Ingresos (Poblacional)'] = (res['CeroIng_Pob'] / res['Ocupados_Pob']) * 100
            
            return res.sort_values('% Cero Ingresos (Poblacional)', ascending=False)

        # ==========================================
        # 1. NIVEL ESTADOS
        # ==========================================
        print("Calculando Nivel: Estados...")
        df_est = procesar_nivel(df, df_oc, ['cve_ent'])
        df_est.insert(1, 'Estado', df_est['cve_ent'].map(estados))
        
        # ==========================================
        # 2. NIVEL CIUDADES
        # ==========================================
        print("Calculando Nivel: Ciudades Autorrepresentadas...")
        df_base_city = df[df['cd_a'].isin(ciudades.keys())].copy()
        df_oc_city = df_oc[df_oc['cd_a'].isin(ciudades.keys())].copy()
        
        df_ciu = procesar_nivel(df_base_city, df_oc_city, ['cve_ent', 'cd_a'])
        df_ciu.insert(2, 'Estado', df_ciu['cve_ent'].map(estados))
        df_ciu.insert(3, 'Ciudad', df_ciu['cd_a'].map(ciudades))
        
        # ==========================================
        # 3. NIVEL MUNICIPIOS
        # ==========================================
        print("Calculando Nivel: Municipios (Filtro >= 500 entrevistas)...")
        df_mun = procesar_nivel(df, df_oc, ['cve_ent', 'mun'], is_mun=True)
        df_mun.insert(2, 'Estado', df_mun['cve_ent'].map(estados))
        df_mun.rename(columns={'mun': 'Clave_Municipio'}, inplace=True)
        
        # ==========================================
        # FORMATO LATAM Y EXPORTACIÓN
        # ==========================================
        def formatear_latam(dataframe):
            df_out = dataframe.copy()
            
            # 1. Separar estrictamente las columnas de %
            pct_cols = [c for c in df_out.columns if '%' in c]
            
            # 2. Convertir a enteros solo las que NO tienen %
            int_cols = [c for c in df_out.columns if ('Muestra' in c or 'Pob' in c or 'General' in c) and c not in pct_cols]
            
            for col in int_cols:
                df_out[col] = df_out[col].fillna(0).round(0).astype(int).astype(str)
                
            for col in pct_cols:
                df_out[col] = df_out[col].fillna(0).round(1).astype(str).str.replace('.', ',')
                
            return df_out

        formatear_latam(df_est).to_csv('01_Cero_Ingresos_Estados_Q42025.csv', sep=';', index=False, encoding='utf-8-sig')
        formatear_latam(df_ciu).to_csv('02_Cero_Ingresos_Ciudades_Q42025.csv', sep=';', index=False, encoding='utf-8-sig')
        formatear_latam(df_mun).to_csv('03_Cero_Ingresos_Municipios_Q42025.csv', sep=';', index=False, encoding='utf-8-sig')
        
        print("\n" + "="*60)
        print("¡Proceso Finalizado con Éxito!")
        print("Archivos generados en tu carpeta local:")
        print("  -> '01_Cero_Ingresos_Estados_Q42025.csv'")
        print("  -> '02_Cero_Ingresos_Ciudades_Q42025.csv'")
        print("  -> '03_Cero_Ingresos_Municipios_Q42025.csv'")
        print("="*60)

    except Exception as e:
        print(f"Ocurrió un error general: {e}")

if __name__ == '__main__':
    procesar_cero_ingresos_regional_completo()