import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

#Aquí se inicia el ETL

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


theta = [0.0] * len(x_train[0])
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

#Función R²
def r2(x, theta, b, y):
    
    y_pred = []
    for i in range(len(x)):
        y_pred.append(hyp(x[i], theta, b))
    
    y_mean = 0
    for i in range(len(y)):
        y_mean += y[i]
    y_mean = y_mean / len(y)
    
    ss_res = 0
    for i in range(len(y)):
        ss_res += (y[i] - y_pred[i])**2
    
    ss_tot = 0
    for i in range(len(y)):
        ss_tot += (y[i] - y_mean)**2
    
 
    if ss_tot == 0:
        return 1.0  
    
    return 1 - (ss_res / ss_tot)

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

# Entrenamiento con validación
epoca = 10000
errores_train = []
errores_val = []
i = 0

while i < epoca:
    
    error_train = mse(x_train, theta, b, y_train)
    errores_train.append(error_train)
    
    
    error_val = mse(x_val, theta, b, y_val)
    errores_val.append(error_val)

    #if i % 101 == 0:  
    #    print(f"Época {i+1}: Train Error = {error_train:.6f}, Val Error = {error_val:.6f}")


    if error_train == 0:
        print(f"Se detuvo por error cero en época {i+1}")
        break

    theta, b = update(x_train, theta, b, y_train, alfa)
    i += 1



error_test = mse(x_test, theta, b, y_test)


r2_train = r2(x_train, theta, b, y_train)
r2_val = r2(x_val, theta, b, y_val)
r2_test = r2(x_test, theta, b, y_test)

print(f"\nResultados del modelo Gradient Descent:")
print (f"Theta: {theta} ")
print (f"Bias : {b}")
print(f"MSE - Train: {errores_train[-1]:.6f}, Val: {errores_val[-1]:.6f}, Test: {error_test:.6f}")
print(f"R²  - Train: {r2_train:.6f}, Val: {r2_val:.6f}, Test: {r2_test:.6f}")

# Desnormalizar predicciones y valores reales para graficar (test set)
y_pred_test_norm = [hyp(xi, theta, b) for xi in x_test]
y_pred_test = [yp * y_std + y_mean for yp in y_pred_test_norm]
y_real_test = [yi * y_std + y_mean for yi in y_test]

# Visualización
plt.figure(figsize=(15, 10))  

#Predicciones vs Valores reales (test set)
plt.subplot(221) 
plt.scatter(y_real_test, y_pred_test, alpha=0.6, label="Predicciones")
plt.plot([min(y_real_test), max(y_real_test)], [min(y_real_test), max(y_real_test)], color='red', linestyle='--', label='Línea ideal')
plt.xlabel("Precio real de cierre del oro")
plt.ylabel("Precio predicho de cierre del oro")
plt.title(f"Test Set: Predicciones vs Reales\nMSE: {error_test:.4f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Curva de error (train vs validation)
plt.subplot(222) 
plt.plot(range(len(errores_train)), errores_train, color='blue', label='Train')
plt.plot(range(len(errores_val)), errores_val, color='orange', label='Validation')
plt.xlabel("Época")
plt.ylabel("Error MSE")
plt.title("Curvas de error durante el entrenamiento")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Comparación de errores finales
plt.subplot(223) 
conjuntos = ['Train', 'Validation', 'Test']
errores_finales = [errores_train[-1], errores_val[-1], error_test]
plt.bar(conjuntos, errores_finales, color=['blue', 'orange', 'green'])
plt.ylabel("Error MSE")
plt.title("Comparación de errores finales")
plt.grid(True, linestyle='--', alpha=0.5)

# Comparación de R² por conjunto
plt.subplot(224) 
r2_scores = [r2_train, r2_val, r2_test]
plt.bar(conjuntos, r2_scores, color=['blue', 'orange', 'green'])
plt.ylabel("R² Score")
plt.title("Comparación de R² - Gradient Descent")
plt.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Evaluación del modelo Gradient Descent con Train | Validation | Test')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

