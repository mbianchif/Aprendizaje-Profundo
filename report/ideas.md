# Ideas

El informe se centra en las diferencias entre distintas cantidades de epochs offline usando los distintos algoritmos distribuidos del sistema ONO que implementamos como trabajo práctico profesional.

## ¿Qué es una offline epoch?

El sistema distribuido necesita algún método de sincronización entre los distintos nodos del sistema para mantaner un estado coherente de los parámetros del modelo siendo entrenado. Este punto de sincronización se hace (por default) cada un epoch pero esta cantidad es configurable y constante durante el entrenamiento.

Se supone que incrementar la cantidad de epochs offline logra mayor velocidad porque hay menos overhead de comunicación, puesto que se ahorra un ida y vuelta por la red.

Esto a su vez implica una mayor divergencia entre los distintos workers. Cada worker tiene de forma local un gradiente propio y por ende una dirección a la cual llevar los parámetros del modelo. Una cantidad elevada de epochs offline puede generar mas inestabilidad en el entrenamiento. Las etapas de sincronización generan correcciones más generosas para mayor cantidad de epochs offline.

## Algoritmos Distribuidos

El sistema soporta 3 algoritmos, estos son:

### Parameter Server

Una implementación centralizada, donde cada worker calcula un gradiente parcial, lo publica a un/os servidor/es para agregarse junto con los demás gradientes de otros workers. Una etapa siguiente los agrega y aplica una optimización sobre los parámetros del modelo para luego redistribuirlos a los workers.

Tiene 2 implementaciones, una bloqueante donde los workers están siempre sincronizados, hasta que uno no termine el resto no sigue calculando la siguiente epoch. Y otra no bloqueante donde cada worker puede publicar y leer los parámetros del modelo a demanda.

### All Reduce

Una implementación descentralizada, donde los workers se organizan en una topología de anillo para compartir gradientes parciales y agregarlos de a partes para luego en parelelo optimizar el modelo.

### Strategy Switch (capaz no incluído en el informe)

Arranca con All Reduce y cambia de algoritmo durante el entrenamiento si se cumple una condición dependiente del error de entrenamiento. Si se cumple esa condición cambia a Parameter Server y termina el entrenamiento de esa forma.

# Secciones

1. Introducción y Objectivos:
  - Presentación del sistema ONO, hipótesis sobre epochs offline y qué se busca evaluar.
2. Arquitectura del sistema y Mecanismos de Sicronización.
  - Explicar los detalles técnicos, algoritmo de entrenamiento, algoritmo distribuido.
3. Metodología Experimental
  - Hardware utilizado
  - Dataset
  - Modelo y arquitectura
  - Configuración de los ensayos a medir
4. Análisis de Resultados
  - Rendimiento del sistema (tiempo, cantidad de epochs offline)
  - Convergencia del modelo (epochs, efectividad de las epochs offline)
5. Conclusiones y lecciones aprendidas
  - Balance final entre velocidad de comunicación y estabilidad del entrenamiento
