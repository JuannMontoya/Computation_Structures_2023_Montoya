import nltk
from nltk.corpus import brown

# Recopilación de datos
def get_data():
  """Obtiene un conjunto de datos de texto del corpus Brown."""
  return brown.sents()

# Preprocesamiento de datos
def preprocess_data(data):
  """Preprocesa un conjunto de datos de texto."""
  return [nltk.word_tokenize(sent) for sent in data]

# Entrenamiento de la IA
def train_model(data):
  """Entrena un modelo de IA para analizar texto."""
  # Creamos un modelo de bag-of-words
  model = nltk.CountVectorizer()

  # Entrenamos el modelo con el conjunto de datos
  model.fit(data)

  return model

# Evaluación de la IA
def evaluate_model(model, data):
  """Evalúa un modelo de IA para analizar texto."""
  # Predecimos las etiquetas para el conjunto de datos de prueba
  predictions = model.predict(data)

  # Calculamos la precisión del modelo
  accuracy = nltk.accuracy(predictions, data)

  return accuracy

# Implementación de la IA
def main():
  # Obtenemos el conjunto de datos
  data = get_data()

  # Preprocesamos el conjunto de datos
  data = preprocess_data(data)

  # Entrenamos el modelo
  model = train_model(data)

  # Evaluamos el modelo
  accuracy = evaluate_model(model, data)

  # Imprimimos la precisión del modelo
  print("Precisión:", accuracy)

if __name__ == "__main__":
  main()