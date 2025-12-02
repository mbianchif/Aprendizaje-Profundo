# Redes de Kohonen (Self-Organizing Maps)

A lo largo de estas actividades se implementan y analizan **Mapas Autoorganizados de Kohonen (SOM)** aplicados tanto a datos sintéticos. Se estudia su proceso de aprendizaje competitivo, su capacidad de preservación topológica y su desempeño al organizar y visualizar datos de alta dimensión.

## Contenidos principales

* **Implementación del algoritmo SOM.**
  * Inicialización del mapa y de los vectores de pesos.
  * Actualización competitiva y vecinal del BMU (*Best Matching Unit*).
  * Disminución temporal de la tasa de aprendizaje y del radio vecinal.

* **Entrenamiento con distintos conjuntos de datos.**
  * Distribuciones sintéticas en 2D (circunferencias ruidosas, cuadrículas, clusters).
  * Conjuntos de datos de mayor dimensionalidad.
  * Comparación entre diferentes topologías de mapa (rectangular, hexagonal).

* **Visualización y análisis del mapa.**
  * Representación de los pesos como una cuadrícula de neuronas.
  * Visualización del **U-Matrix** para estudiar separaciones y densidades.
  * Mapas de hit frequency y activación de neuronas.

* **Propiedades emergentes y comportamiento del algoritmo.**
  * Preservación de la topología de los datos.
  * Formación de clusters.
  * Efectos del ruido, del tamaño del mapa y de parámetros de entrenamiento.

## Objetivo

El objetivo de estas guías es **comprender el funcionamiento, las propiedades emergentes y las limitaciones de los Mapas Autoorganizados de Kohonen**, profundizando en cómo estas redes no supervisadas pueden descubrir estructuras internas en los datos y representarlas de manera visual e interpretable.

## Extracción de Imágenes

Para extraer los datos ejecutar:

```bash
unzip data.zip -d data
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
