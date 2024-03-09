import numpy as np

##DUDA -> Puedo/debo usar la sigmoide como funcion de activacion?
def sigmoid(z):
    """Función de activación sigmoidal."""
    return 1 / (1 + np.exp(-z))



# Función de predicción
def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.
    
    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)
    theta2: array_like
        Weights for the second layer in the neural network.
        It has shape (output layer size x 2nd hidden layer size)
    X : array_like
        The image inputs having shape (number of examples x image dimensions).
    
    Return
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """
    #Añadir el sesgo usando la funcion hstack
    m = X.shape[0]
    X1s = np.hstack([np.ones((m, 1)), X])
    
    # Calcular la salida de la capa oculta
    z2 = np.dot(X1s, theta1.T)
    #Aplicar su funcion de activacion
    a2 = sigmoid(z2)

     ##DUDA -> quitar o añadir sesgo a la capa oculta no cambia nada del resultado
	 ##Pero hay que añadirlo?
    a2 = np.hstack([np.ones((m, 1)), a2])
	 
    # Calcular la salida de la capa de salida
    z3 = np.dot(a2, theta2.T)
    #Aplicar su funcion de activacion
    a3 = sigmoid(z3)

	##La etiqueta predicha para cada ejemplo se determina seleccionando la clase con la mayor activación en la capa de salida
    p = np.argmax(a3, axis=1)  

    return p
