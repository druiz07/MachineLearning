import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from utils import load_data

#########################################################################
# Cost function
#
def compute_cost(x, y, w, b):
	"""
	Computes the cost function for linear regression.

	Args:
		x (ndarray): Shape (m,) Input to the model (Population of cities)
		y (ndarray): Shape (m,) Label (Actual profits for the cities)
		w, b (scalar): Parameters of the model

	Returns
		total_cost (float): The cost of using w,b as the parameters for linear regression
			   to fit the data points in x and y
	"""
	#Primero cargamos los datos de utilsn load data que nos devuelven las componentes X e Y, importando las librerias
	# Cargar los datos del archivo


	##Para calcular el coste sabemos que es la siguiente formula 1/2m *  Sumatorio i=1 hasta m (F(W,B)-y(i)^2
	##F(w,b) w *x +b
	m = len(y)  # número de ejemplos de entrenamiento
	##Aprovechamos las operaciones vectoriales vistas en clase para la multiplicacion de estos arrays
	predictions = w * x + b  # calcular las F(w,b) del modelo, teniendo en cuenta que son multiplicaciones de arrays
	squared_errors = (predictions - y) ** 2 
	total_cost = (1 / (2 * m)) * np.sum(squared_errors)  # calcular el costo total [Usamos sum de numpy para que se haga vectorialmente]
	
	##Como se puede observar tanto la w como la b son parametros externos que decidimos nosotros
	##Podemos probarlo dando valores a la b y a la w  
	
	

	return total_cost


#########################################################################
# Gradient function
#
def compute_gradient(x, y, w, b):
	"""
	Computes the gradient for linear regression 
	Args:
	  x (ndarray): Shape (m,) Input to the model (Population of cities) 
	  y (ndarray): Shape (m,) Label (Actual profits for the cities)
	w,b (scalar): Parameters of the model  
	Returns
	dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
	dj_db (scalar): The gradient of the cost w.r.t. the parameter b	 
	 """

	##Necesitamos las derivadas parciales de w y de b , siguiendo la formula :
	##Derivada para w -> 1/M*(Sumatorio desde i=1 hasta M ((w*x+b[Prediccion])-y [dato real])*Xi
	##Derivada para b -> 1/M*(Sumatorio desde i=1 hasta M ((w*x+b[Prediccion])-y [dato real])
	##Vamos a empezar calculando la M y f(w,b)
	predictions = w * x + b # calcular las f(w,b) del modelo, teniendo en cuenta que son multiplicaciones de arrays
	m=len(y)  ##Numerode ejemplos para el entrenamiento 
	##Ahora calculamos el error que es restar a la prediccion lo real (w*x+b[Prediccion])-y [dato real]
	errors = predictions - y
	##Ahora aplicamos la derivada parcial en diferencia para w o para b 
	# Calcular el gradiente para w usando sum y multiplicación elemento a elemento
	dj_dw = (1 / m) * np.sum(errors * x)

	# Calcular el gradiente para b usando np.sum
	dj_db = (1 / m) * np.sum(errors)
	
	return dj_dw, dj_db


#########################################################################
# gradient descent
#
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
	"""
	Performs batch gradient descent to learn theta. Updates theta by taking 
	num_iters gradient steps with learning rate alpha

	Args:
	  x :	(ndarray): Shape (m,)
	  y :	(ndarray): Shape (m,)
	  w_in, b_in : (scalar) Initial values of parameters of the model
	  cost_function: function to compute cost
	  gradient_function: function to compute the gradient
	  alpha : (float) Learning rate
	  num_iters : (int) number of iterations to run gradient descent
	Returns
	  w : (ndarray): Shape (1,) Updated values of parameters of the model after
		  running gradient descent
	  b : (scalar) Updated value of parameter of the model after
		  running gradient descent
	  J_history : (ndarray): Shape (num_iters,) J at each iteration,
		  primarily for graphing later
	"""
	
	##Vamos a asumir que ya tenemos todo bien , con lo cual tenemos que ir aplicando el gradiente calculado para ir dreduciendo al minimo el	 error en cada iteracion
	##Tenemos claro que hay que crear el array que registra el error , actualizar w y b con formulas , y llamar a los metodos
	#Primer paso , w y b nuevas y creacion del vector de registro por si luego hay grafica
	w=w_in
	b=b_in
	J_history= np.zeros(num_iters) ## Creamis un array vacio con contenido 0 para asegurar que no hay errores 
	##LLamada a los metodos que obtienen lo siguiente calculado :
	##Derivada para w -> 1/M*(Sumatorio desde i=1 hasta M ((w*x+b[Prediccion])-y [dato real])*Xi -> dj_dw
	##Derivada para b -> 1/M*(Sumatorio desde i=1 hasta M ((w*x+b[Prediccion])-y [dato real])-> dj_db
	for i in range (num_iters):
		cost = cost_function(x, y, w, b)
		dj_dw, dj_db = gradient_function(x, y, w, b)
		w = w - (alpha * dj_dw)
		b = b - (alpha * dj_db)
		J_history[i] = cost


	return w, b, J_history

# Ejemplo de uso:
# w_inicial, b_inicial son los valores iniciales para w y b
# alpha es la tasa de aprendizaje, num_iters es el número de iteraciones
# las funciones son las creadas previamente que devuelven el coste para el array J[Numiters] y para las derivadas que nos da el descenso de gradientes





