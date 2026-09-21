"""
Módulo de preprocesamiento de datos e ingeniería de características 
para el dataset CMAPSS FD001.
"""
import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    # 1. Definir columnas
    nombres_columnas = ['id_motor', 'ciclo', 'ajuste_1', 'ajuste_2', 'ajuste_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # 2. Leer archivo
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=nombres_columnas)
    
    # 3. Calcular Remaining Useful Life (RUL)
    max_ciclos = df.groupby('id_motor')['ciclo'].max().reset_index()
    max_ciclos.columns = ['id_motor', 'max_ciclo']
    df = df.merge(max_ciclos, on='id_motor', how='left')
    df['RUL'] = df['max_ciclo'] - df['ciclo']
    df = df.drop(columns=['max_ciclo'])
    
    # 4. Eliminar sensores constantes (ruido)
    desviaciones = df.std()
    columnas_constantes = desviaciones[desviaciones == 0].index
    df = df.drop(columns=columnas_constantes)
    
    return df

def feature_engineering(df):
    """Calcula medias y desviaciones móviles para capturar tendencias de degradación."""
    df_avanzado = df.copy()
    columnas_sensores = df_avanzado.columns[2:-1] # Omitimos id_motor, ciclo y RUL
    
    for motor in df_avanzado['id_motor'].unique():
        filas_motor = df_avanzado['id_motor'] == motor
        for col in columnas_sensores:
            # Tendencia suave (Media 10 vuelos)
            df_avanzado.loc[filas_motor, f'{col}_media_10'] = df_avanzado.loc[filas_motor, col].rolling(window=10, min_periods=1).mean()
            # Inestabilidad (Desviación 10 vuelos)
            df_avanzado.loc[filas_motor, f'{col}_desv_10'] = df_avanzado.loc[filas_motor, col].rolling(window=10, min_periods=2).std().fillna(0)
            
    return df_avanzado

if __name__ == "__main__":
    print("Preprocesando datos...")
    df_raw = load_and_clean_data("../data/train_FD001.txt")
    df_features = feature_engineering(df_raw)
    print(f"Dimensiones finales del dataset: {df_features.shape}")
