import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def main():
    # Task 1.1: Leer el archivo CSV y almacenarlo en un array
    headers = ["id", "date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]
    data = np.genfromtxt('./data/kc_house_data.csv', delimiter=',', skip_header=1)
    # data = pd.read_csv('./data/kc_house_data.csv')

    # Extraer la columna sqft_living y price para realizar la regresión lineal
    X = data[:, 5].reshape(-1, 1)
    y = data[:, 2].reshape(-1, 1)

    # Task 1.2: Ajustar un modelo polinomial en base al juego de datos
    # Crear una instancia de PolynomialFeatures con grado máximo de 10
    poly = PolynomialFeatures(degree=10)

    # Transformar la matriz X en una matriz de características polinomiales
    X_poly = poly.fit_transform(X)

    # Crear una instancia de LinearRegression y ajustar el modelo
    model = LinearRegression()
    model.fit(X_poly, y)

    # Task 1.3: Implementar el algoritmo de regresión lineal con descenso del gradiente
    # Definir la tasa de aprendizaje y el número máximo de iteraciones
    alpha = 0.0000001
    max_iters = 10000

    # Inicializar los parámetros con valores aleatorios
    theta = np.random.rand(X_poly.shape[1], 1)

    # Implementar el descenso del gradiente
    for i in range(max_iters):
        h = X_poly.dot(theta)
        error = h - y
        gradient = X_poly.T.dot(error)
        theta = theta - alpha * gradient

    # Task 1.4: Usar cross-validation para encontrar el grado del polinomio que mejor describe los datos
    degrees = range(1, 20)
    mse_scores = []

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        mse = -1 * cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error').mean()
        mse_scores.append(mse)

    best_degree = degrees[np.argmin(mse_scores)]
    print("El mejor grado del polinomio es:", best_degree)

    # Task 1.5: Realizar un análisis de los hallazgos
    plt.plot(degrees, mse_scores)
    plt.xlabel('Grado del polinomio')
    plt.ylabel('MSE')
    plt.title('Curva de validación cruzada')
    plt.show()

if __name__ == "__main__":
    main()