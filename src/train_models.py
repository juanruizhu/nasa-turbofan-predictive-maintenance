"""
Módulo de entrenamiento de modelos predictivos usando XGBoost.
Incluye Regresión (estimación de RUL) y Clasificación (Estado de Salud).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from xgboost import XGBRegressor, XGBClassifier

def train_rul_regressor(df_avanzado):
    """Entrena el modelo de regresión XGBoost optimizado para predecir RUL."""
    motores = df_avanzado['id_motor'].unique()
    motores_train, motores_test = train_test_split(motores, test_size=0.2, random_state=42)
    
    train_df = df_avanzado[df_avanzado['id_motor'].isin(motores_train)]
    test_df = df_avanzado[df_avanzado['id_motor'].isin(motores_test)]
    
    X_train = train_df.drop(columns=['id_motor', 'ciclo', 'RUL'])
    y_train = train_df['RUL']
    X_test = test_df.drop(columns=['id_motor', 'ciclo', 'RUL'])
    y_test = test_df['RUL']
    
    # XGBoost con hiperparámetros optimizados mediante RandomizedSearchCV
    pipe_xgb = Pipeline([
        ('escalador', StandardScaler()),
        ('regresor', XGBRegressor(
            n_estimators=200, 
            max_depth=3, 
            learning_rate=0.05, 
            subsample=0.8,
            colsample_bytree=1.0,
            random_state=42, 
            n_jobs=-1
        ))
    ])
    
    pipe_xgb.fit(X_train, y_train)
    predicciones = pipe_xgb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    
    print(f"Modelo de Regresión XGBoost Entrenado. RMSE Legítimo: +/- {rmse:.2f} vuelos.")
    return pipe_xgb, X_test, y_test

def train_health_classifier(df_avanzado):
    """Entrena un clasificador para categorizar el estado de salud del motor."""
    def etiquetar_estado(rul):
        if rul > 50: return 0  # Sano
        elif rul > 15: return 1  # Alerta
        else: return 2  # Crítico
        
    df_class = df_avanzado.copy()
    df_class['estado'] = df_class['RUL'].apply(etiquetar_estado)
    
    motores = df_class['id_motor'].unique()
    motores_train, motores_test = train_test_split(motores, test_size=0.2, random_state=42)
    
    train_df = df_class[df_class['id_motor'].isin(motores_train)]
    test_df = df_class[df_class['id_motor'].isin(motores_test)]
    
    X_train_c = train_df.drop(columns=['id_motor', 'ciclo', 'RUL', 'estado'])
    y_train_c = train_df['estado']
    X_test_c = test_df.drop(columns=['id_motor', 'ciclo', 'RUL', 'estado'])
    y_test_c = test_df['estado']
    
    pipe_clasificador = Pipeline([
        ('escalador', StandardScaler()),
        ('clasificador', XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1))
    ])
    
    pipe_clasificador.fit(X_train_c, y_train_c)
    predicciones = pipe_clasificador.predict(X_test_c)
    
    print("\nReporte de Clasificación (Estado Operacional):")
    print(classification_report(y_test_c, predicciones, target_names=['Sano (0)', 'Alerta (1)', 'Crítico (2)']))
    
    return pipe_clasificador
