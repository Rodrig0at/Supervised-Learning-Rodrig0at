
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('financial_regression.csv')
df = df.dropna() # Esta parte también es de ETL 
features = ['gold open','gold high','gold low','gold volume']
x_raw = df[features].values
y_raw = df['gold close'].values

#Aquí se inicia el ETL
# Normalización de x,y
x_mean = np.mean(x_raw, axis=0)
x_std = np.std(x_raw, axis=0)
x = ((x_raw - x_mean) / x_std).tolist()

y_mean = np.mean(y_raw)
y_std = np.std(y_raw)
y = ((y_raw - y_mean) / y_std).tolist()


theta = [0.0] * len(x[0])
b = 0.0
alfa = 0.001
#Función de hipótesis
def hyp(x, theta, b):
    y_hat = b
    for i in range(len(x)):
        y_hat += x[i] * theta[i]
    return y_hat

#Función de costo
def mse(x, theta, b, y):
    cost = 0
    m = len(x)
    for i in range(m):
        cost += (hyp(x[i], theta, b) - y[i])**2
    return cost / m

#Función de actualización
def update(x, theta, b, y, alfa):
    new_theta = theta.copy()
    m = len(x)
    n = len(theta)

    #Aquí empieza el loop para hacer todo el despapaye

    # Actualizar theta
    for j in range(n):
        grad_theta = 0
        for i in range(m):
            grad_theta += (hyp(x[i], theta, b) - y[i]) * x[i][j]
        new_theta[j] = theta[j] - (alfa/m) * grad_theta

    # Actualizar b
    grad_b = 0
    for i in range(m):
        grad_b += (hyp(x[i], theta, b) - y[i])
    new_b = b - (alfa/m) * grad_b

    return new_theta, new_b

#Aquí aplicamos todo lo del despapaye en un while

# Entrenamiento
epoca = 10000
errores = []
i = 0
while i < epoca:
    error_actual = mse(x, theta, b, y)
    errores.append(error_actual)

    if i % 100 == 0:  # Mostrar cada 100 épocas
            print(f"Época {i+1}: Error = {error_actual:.6f}")

    if error_actual == 0:
        break


    theta, b = update(x, theta, b, y, alfa)
    i += 1

print(f"Parámetros finales:")
print(f"Theta: {theta}")
print(f"Bias: {b}")
print(f"Errores finales: {errores[-3:]}")  # Últimos 3 errores

# Desnormalizar predicciones y valores reales para graficar
y_pred_norm = [hyp(xi, theta, b) for xi in x]
y_pred = [yp * y_std + y_mean for yp in y_pred_norm]
y_real = [yi * y_std + y_mean for yi in y]

# Visualización
plt.figure(figsize=(18, 5))

#Predicciones vs Valores reales
plt.subplot(131)
plt.scatter(y_real, y_pred, alpha=0.6, label="Predicciones")
plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], color='red', linestyle='--', label='Línea ideal')
plt.xlabel("Precio real de cierre del oro")
plt.ylabel("Precio predicho de cierre del oro")
mse_final = errores[-1] if errores else 0
plt.title(f"Predicciones vs Valores Reales\nMSE final: {mse_final:.4f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Curva de error (MSE)
plt.subplot(132)
plt.plot(range(len(errores)), errores, color='blue')
plt.xlabel("Época")
plt.ylabel("Error MSE")
plt.title("Curva de error durante el entrenamiento")
plt.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Evaluación del modelo de regresión')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
