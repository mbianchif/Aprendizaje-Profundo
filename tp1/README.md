# Hopfield

A lo largo de las actividades se implementan, entrenan y analizan estas redes con distintos conjuntos de patrones, estudiando su comportamiento frente a ruido, correlación y eliminación de conexiones.

## Contenidos principales
- **Entrenamiento de redes de Hopfield con imágenes binarias.**  
  - Verificación de aprendizaje de patrones.  
  - Recuperación de patrones a partir de versiones alteradas (con ruido, incompletas o modificadas).  
  - Análisis de estados espurios (patrones inversos y combinaciones).  
  - Estudio de la capacidad de almacenamiento con distintos números de patrones.

- **Capacidad estadística de la red.**  
  - Estimación experimental de la cantidad máxima de patrones pseudo-aleatorios que puede almacenar la red en función de su tamaño.  
  - Estudio del impacto de la correlación entre patrones en la capacidad.

- **Robustez de la red frente a fallas sinápticas.**  
  - Análisis de cómo varía el error y la capacidad al eliminar aleatoriamente un porcentaje de las conexiones.

## Objetivo
El objetivo de estas guías es **comprender el funcionamiento, limitaciones y propiedades de generalización de las redes de Hopfield**, combinando simulaciones experimentales con el análisis teórico de la literatura.

## Extracción de Imágenes

Para extraer las imágenes ejecutar:
```sh
unzip images.zip -d images
```

## Ejecución del Notebook

Para correr el notebook de forma local ejcutar:
```sh
uv run jupyter notebook
```

## Dependencias
- **unzip**: Para descomprimir las imágenes
```sh
sudo apt install unzip  # ubuntu
brew install unzip      # macOS
```

- **uv**: Para ejecutar el notebook
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh # linux + macOS
```
