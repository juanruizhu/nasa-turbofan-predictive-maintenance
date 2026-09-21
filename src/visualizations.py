"""
Módulo de visualización para el análisis de resultados, 
residuos, interpretabilidad (SHAP) y matrices de confusión.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_real_vs_predicted(y_test, predicciones, filename="real_vs_pred.png"):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predicciones, alpha=0.3, color='teal')
    plt.plot([0, 300], [0, 300], color='red', linestyle='--', linewidth=2)
    plt.title('Vida Útil Real vs. Predicciones (XGBoost Optimizado)')
    plt.xlabel('RUL Real (Vuelos restantes verdaderos)')
    plt.ylabel('RUL Predicho')
    plt.xlim(0, 300)
    plt.ylim(0, 300)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_shap_summary(modelo_xgb, X_test_escalado, X_test_df, filename="shap_summary.png"):
    explainer = shap.TreeExplainer(modelo_xgb)
    shap_values = explainer.shap_values(X_test_escalado)
    
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test_df, plot_type="dot", show=False)
    plt.title('Impacto de cada Sensor en la Predicción (Valores SHAP)', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_confusion_matrix_health(y_test_clases, predicciones_clases, filename="confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    matriz = confusion_matrix(y_test_clases, predicciones_clases)
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sano', 'Alerta', 'Crítico'],
                yticklabels=['Sano', 'Alerta', 'Crítico'])
    plt.title('Matriz de Confusión Operacional')
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Realidad Física')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
def plot_sensor_degradation(motor_df, top_sensores, filename="sensor_degradation.png"):
    plt.figure(figsize=(12, 6))
    for sensor in top_sensores:
        sensor_norm = (motor_df[sensor] - motor_df[sensor].min()) / (motor_df[sensor].max() - motor_df[sensor].min())
        plt.plot(motor_df['ciclo'], sensor_norm, label=sensor, linewidth=2.5)

    plt.title('Evolución Física de los Sensores Críticos', fontsize=14)
    plt.xlabel('Ciclos de Vuelo (Edad del motor)')
    plt.ylabel('Valor de la Media Móvil (Normalizado 0-1)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    ciclo_final = motor_df['ciclo'].max()
    plt.axvspan(ciclo_final - 50, ciclo_final, color='red', alpha=0.1, label='Zona Crítica (<50 vuelos)')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
