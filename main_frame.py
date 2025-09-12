from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('financial_regression.csv')
df = df.dropna() # Esta parte también es de ETL 
features = ['gold open','gold high','gold low','gold volume']
x_raw = df[features].values
y_raw = df['gold close'].values

# División del dataset con slicing simple. Investigue y lo mejor era 70 % train, 15 % val, 15 % test
total_size = len(x_raw)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)


x_train_raw = x_raw[:train_size]
y_train_raw = y_raw[:train_size]

x_val_raw = x_raw[train_size:train_size + val_size]
y_val_raw = y_raw[train_size:train_size + val_size]

x_test_raw = x_raw[train_size + val_size:]
y_test_raw = y_raw[train_size + val_size:]

x_mean = np.mean(x_train_raw, axis=0)
x_std = np.std(x_train_raw, axis=0)
y_mean = np.mean(y_train_raw)
y_std = np.std(y_train_raw)

# Aplicar normalización a todos los conjuntos
x_train = ((x_train_raw - x_mean) / x_std).tolist()
y_train = ((y_train_raw - y_mean) / y_std).tolist()

x_val = ((x_val_raw - x_mean) / x_std).tolist()
y_val = ((y_val_raw - y_mean) / y_std).tolist()

x_test = ((x_test_raw - x_mean) / x_std).tolist()
y_test = ((y_test_raw - y_mean) / y_std).tolist()
# El estimador de decisiontree regressor no es el índicado para este pr


# Entrenar 
bagging = BaggingRegressor(estimator=LinearRegression(), n_estimators=50, random_state=42)
bagging.fit(x_train, y_train)


pred_train = bagging.predict(x_train)
pred_val = bagging.predict(x_val)
pred_test = bagging.predict(x_test)


mse_train = mean_squared_error(y_train, pred_train)
mse_val = mean_squared_error(y_val, pred_val)
mse_test = mean_squared_error(y_test, pred_test)

r2_train = r2_score(y_train, pred_train)
r2_val = r2_score(y_val, pred_val)
r2_test = r2_score(y_test, pred_test)

print(f"\nResultados del modelo Bagging Regressor:")
print(f'MSE Train: {mse_train:.6f}, Val: {mse_val:.6f}, Test: {mse_test:.4f}')
print(f'R² Train: {r2_train:.6f}, Val: {r2_val:.6f}, Test: {r2_test:.4f}')


pred_test_real = pred_test * y_std + y_mean
y_test_real = np.array(y_test) * y_std + y_mean


errores_finales = [mse_train, mse_val, mse_test]

# Visualización
plt.figure(figsize=(15, 5))

# Predicciones vs Valores reales (test set )
plt.subplot(131)
plt.scatter(y_test_real, pred_test_real, alpha=0.6, label="Predicciones")
plt.plot([min(y_test_real), max(y_test_real)], [min(y_test_real), max(y_test_real)], 
         color='red', linestyle='--', label='Línea ideal')
plt.xlabel("Precio real de cierre del oro")
plt.ylabel("Precio predicho de cierre del oro")
plt.title(f"Test Set: Predicciones vs Reales\nMSE: {mse_test:.4f} (normalizado)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Comparación de errores 
plt.subplot(132)
conjuntos = ['Train', 'Validation', 'Test']
plt.bar(conjuntos, errores_finales, color=['blue', 'orange', 'green'])
plt.ylabel("Error MSE (normalizado)")
plt.title("Comparación de errores - Bagging")
plt.grid(True, linestyle='--', alpha=0.5)

# Comparación de R² por conjunto
plt.subplot(133)
r2_scores = [r2_train, r2_val, r2_test]
plt.bar(conjuntos, r2_scores, color=['blue', 'orange', 'green'])
plt.ylabel("R² Score")
plt.title("Comparación de R² - Bagging")
plt.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Evaluación del modelo Bagging con Train | Validation | Test')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
