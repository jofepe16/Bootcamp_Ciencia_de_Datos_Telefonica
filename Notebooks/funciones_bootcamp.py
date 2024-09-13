import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



def compute_metrics(y_true, y_pred):
     """ Calcula y devuelve las métricas de rendimiento de un modelo de clasificación para los conjuntos de entrenamiento y prueba.

    Parámetros:
    y_entreno_true (array-like): Valores verdaderos para el conjunto .
    y_entreno_pred (array-like): Valores predichos para el conjunto .
    
    Retorna:
    pd.DataFrame: Un DataFrame con las métricas de rendimiento, que incluyen:
        - 'accuracy': Precisión del modelo para los conjuntos deseado.
        - 'precision': Precisión del modelo para la clase positiva (1) en los conjuntos deseado.
        - 'recall': Recall del modelo para la clase positiva (1) en el conjunto deseado."""
        
    # Accuracy para conjuntos de entrenamiento y prueba
     accuracy = round(100.0 * accuracy_score(y_true, y_pred), 2)
     
    # contruimos en reporte de la matrzi de clasificación
     report = classification_report(y_true, y_pred, output_dict=True)
     
    # extraemos las medidas a utilizar para la clase 1 (sobrevivientes), precision recall y accuracy
     precision = round(report['1']['precision'], 3)
     recall = round(report['1']['recall'], 3)
     
    # CCreamos dataframe con las metricas
     metrics_df = pd.DataFrame({
         'accuracy': [accuracy],
         'precision': [precision],
         'recall': [recall]
     }, index=['metricas'])
     return metrics_df



def plot_confusion_matrix_and_reports(y_true, y_pred, title='Matriz de Confusión', cmap=plt.cm.Blues):
    """
    Plotea las matrices de confusión y los reportes de métricas para los conjuntos de entrenamiento y prueba.

    Esta función utiliza la función `compute_metrics` para obtener las métricas de precisión, recall y accuracy,
    y luego genera gráficos de las matrices de confusión y reportes de métricas para los conjuntos de entrenamiento y prueba.

    Parámetros:
    y_entreno_true (array-like): Valores reales para el conjunto de entrenamiento.
    y_entreno_pred (array-like): Valores predichos para el conjunto de entrenamiento.
    
    Retorna:
    None: La función muestra dos gráficos: uno para la matriz de confusión del conjunto deseado"""

    # utilizamos funcion definida anteriormente
    metrics_df = compute_metrics(y_true, y_pred)
    # matrices de confusion (Conjunto Entrenamiento)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(cm, metrics):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # etiquetas matrices de consusion
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                    xticklabels=['No Sobr.', 'Sobr.'], 
                    yticklabels=['No Sobr.', 'Sobr.'], ax=axes[0])
        axes[0].set_title(title)
        axes[0].set_xlabel('Predicción')
        axes[0].set_ylabel('Real')

        # Formatea las métricas en una cadena de texto
        metrics_text = f"Accuracy: {metrics['accuracy']/100:.3f} - ({metrics['accuracy']:.1f}%)\n"
        metrics_text += f"Precision: {metrics['precision']:.3f} - ({metrics['precision']*100:.1f}%)\n"
        metrics_text += f"Recall: {metrics['recall']:.3f} - ({metrics['recall']*100:.1f}%)\n"
        
        # Añade el texto formateado
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, metrics_text, horizontalalignment='center', 
                    verticalalignment='center', fontsize=12)

        plt.tight_layout()
        plt.show()

    plot_confusion_matrix(conf_matrix, metrics_df.loc['metricas'])