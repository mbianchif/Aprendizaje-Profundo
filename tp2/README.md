# Perceptrones Multicapa, Máquinas de Boltzmann y Redes Convolucionales

A lo largo de estas actividades se implementan, entrenan y analizan distintos modelos neuronales —**Perceptrones Multicapa (MLP)**, **Máquinas de Boltzmann** y **Redes Neuronales Convolucionales (CNN)**— aplicados a problemas supervisados y no supervisados. Se estudia su proceso de aprendizaje, capacidades de representación y comportamiento frente a distintos niveles de ruido y complejidad en los datos.

## Contenidos principales

### **Perceptrones Multicapa (MLP)**

* **Entrenamiento supervisado para clasificación.**

  * Construcción de arquitecturas fully-connected de distintas profundidades.
  * Exploración del impacto del número de capas, neuronas y funciones de activación.
  * Análisis de curvas de aprendizaje, overfitting y regularización.
* **Generalización y ruido.**

  * Evaluación del desempeño con datos perturbados.
  * Estudio del efecto del tamaño del dataset y la complejidad del modelo.

### **Máquinas de Boltzmann**

* **Entrenamiento no supervisado.**

  * Implementación del algoritmo de aprendizaje estocástico con energía y probabilidad de estados.
  * Uso de Máquinas de Boltzmann restringidas (RBM) cuando corresponde.
* **Modelado de distribuciones de probabilidad.**

  * Reconstrucción de patrones y análisis de convergencia.
  * Evaluación del comportamiento frente a ruido y variación de la temperatura.
* **Estudio de capacidades y limitaciones.**

  * Análisis de estados metaestables, modos y energía de la red.

### **Redes Neuronales Convolucionales (CNN)**

* **Clasificación de imágenes.**

  * Definición de arquitecturas convolucionales simples y profundas.
  * Extracción jerárquica de características mediante filtros y pooling.
  * Evaluación de accuracy y matrices de confusión.
* **Visualización e interpretación.**

  * Mapas de activación.
  * Efecto de diferentes arquitecturas y parámetros sobre el desempeño.

## Objetivo

El objetivo de estas guías es **comprender el funcionamiento, las capacidades de representación y las limitaciones de modelos neuronales modernos**, abarcando tanto arquitecturas supervisadas (MLPs, CNNs) como modelos generativos no supervisados (Máquinas de Boltzmann).
Se combinan experimentos prácticos con análisis conceptual para profundizar en su comportamiento.

## Extracción de Imágenes

Si el proyecto incluye imágenes (datasets o resultados exportados), extraerlas con:

```bash
unzip images.zip -d images
```

## Ejecución del Notebook

Para correr el notebook de forma local ejecutar:

```bash
uv run jupyter notebook
```

## Dependencias

* **unzip**: Para descomprimir las imágenes

```bash
sudo apt install unzip  # ubuntu
brew install unzip      # macOS
```

* **uv**: Para ejecutar el notebook

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # linux + macOS
```

---

If you want, I can also generate a short English version, add images, include badges, or format the README more like an academic report.
