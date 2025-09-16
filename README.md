# Predicción del Precio de Cierre del Oro con Machine Learning

Este proyecto implementa dos modelos, uno de regresión lineal múltiple desde cero,por otro lado, el segundo modelo utiliza Baggin Regressor. Utilizando Python, para predecir el precio de cierre del oro a partir de datos históricos de precios y volumen. El objetivo es mostrar el flujo completo de un proyecto de machine learning: desde la preparación de los datos hasta la visualización y análisis de resultados.

## Descripción del proyecto

El modelo utiliza como variables de entrada los precios de apertura, máximo, mínimo y el volumen de transacciones del oro. La variable objetivo es el precio de cierre del oro. El proceso incluye la normalización de los datos, el entrenamiento del modelo mediante gradiente descendente y la evaluación de los resultados a través de diferentes visualizaciones.

## Archivos incluidos

- `main.py`: Script principal con todo el código del modelo y visualización.
- `financial_regression.csv`: Archivo de datos históricos de precios y volúmenes de metales preciosos.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/tu_usuario/tu_repositorio.git
   cd tu_repositorio
   ```
2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

Asegúrate de que el archivo `financial_regression.csv` esté en la misma carpeta que el script principal. Luego ejecuta:

```
python main.py
python main_frame.py
```

Los scripts entrenarán un modelo y mostrarán varias gráficas:

- Predicciones vs valores reales
- Curva de error (MSE) durante el entrenamiento
- Histograma de errores absolutos

## Requisitos

- Python 3.x
- numpy
- pandas
- matplotlib

## Notas sobre los datos

El archivo `financial_regression.csv` contiene más de 3900 registros históricos de precios y volúmenes de oro, plata y otros metales preciosos. Para este proyecto, solo se utilizan las columnas relacionadas con el oro. Los datos se normalizan antes de entrenar el modelo para mejorar la eficiencia y precisión del aprendizaje.

## Autor

Rodrigo Antonio Benítez De La Portilla

---

Si tienes dudas o sugerencias, no dudes en abrir un issue o un pull request.
