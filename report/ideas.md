# Ideas

El informe se centra en las diferencias entre distintas cantidades de epochs offline usando los distintos algoritmos distribuidos del sistema ONO que implementamos como trabajo práctico profesional.

## ¿Qué es una offline epoch?

El sistema distribuido necesita algún método de sincronización entre los distintos nodos del sistema para mantaner un estado coherente de los parámetros del modelo siendo entrenado. Este punto de sincronización se hace (por default) cada un epoch pero esta cantidad es configurable y constante durante el entrenamiento.

Se supone que incrementar la cantidad de epochs offline logra mayor velocidad porque hay menos overhead de comunicación, puesto que se ahorra un ida y vuelta de mensajes en la red.

Esto a su vez implica una mayor divergencia entre los distintos workers. Cada worker tiene de forma local un gradiente propio y por ende una dirección a la cual llevar los parámetros del modelo. Una cantidad elevada de epochs offline puede generar mas inestabilidad en el entrenamiento. Las etapas de sincronización generan correcciones más generosas para mayor cantidad de epochs offline.

## Algoritmos Distribuidos

El sistema soporta 3 algoritmos, Parameter Server, All Reduce y Strategy Switch. En este informe vamos a quedarnos con los 2 primeros.

### Parameter Server

Una implementación centralizada, donde cada worker calcula un gradiente parcial, lo publica a un/os servidor/es para agregarse junto con los demás gradientes de otros workers. Una etapa siguiente los agrega y aplica una optimización sobre los parámetros del modelo para luego redistribuirlos a los workers.

Tiene 2 implementaciones, una bloqueante donde los workers están siempre sincronizados, hasta que uno no termine el resto no sigue calculando la siguiente epoch. Y otra no bloqueante donde cada worker puede publicar y leer los parámetros del modelo a demanda.

### All Reduce

Una implementación descentralizada, donde los workers se organizan en una topología de anillo para compartir gradientes parciales y agregarlos de a partes para luego en parelelo optimizar el modelo.

# Secciones

0. Resúmen
1. Introducción y Objectivos:
  - Presentación del sistema ONO.
  - Hipótesis sobre epochs offline y qué se busca evaluar.
2. Arquitectura del sistema y Mecanismos de Sicronización.
  - Explicar los detalles técnicos, algoritmo de entrenamiento, algoritmo distribuido.
  - Diagramas de topologias.
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
6. Bibliografía

# 0 - Resémen

Como altera la cantidad de epochs offline al entrenamiento y su velocidad de ejecuión en algoritmos distribuidos de aprendizaje profundo como Parameter Server y All Reduce

kw:
- Epochs Offline
- Entrenamiento Distribuido
- Aprendizaje Profundo
- Parameter Server
- All Reduce

# 1 - Introducción y Objetivos

## 1.1 - Presentación del sistema ONO, hipótesis sobre epochs offline y qué se busca evaluar.

En el mundo del entrenamiento distribuido existen varios algoritmos, entre ellos fueron 2 los que implementamos en nuestro sistema ONO para nuestro trabajo práctico profesional junto a mis compañeros. Estos son Parameter Server y All Reduce. ONO le permite al usuario de forma declarativa definir una arquitectura, dataset, hiperparámetros y algoritmo distribuido para entrenar su modelo.

## 1.2 - Hipótesis sobre epochs offline y qué se busca evaluar.

El objectivo de este informe es medir y analizar los resultados de ejecuciones de entrenamientos utilizando distintas configuraciones, en particular la cantidad de épocas offline. Se quiere saber como altera la velocidad de entrenamiento y eficacia del modelo entrenado tomando como referencia una arquitectura como la del modelo LeNet5. Se espera que a mayor cantidad de épocas offline, el entrenamiento sea más rápido pero a su vez deteriore un poco la calidad del entrenamiento.

# 2 - Arquitectura del sistema y Mecanísmos de sincronización.

## 2.1 Algoritmos Distribuidos.

### 2.1.1 Orchestrator, Workers, Servers

El `Orchestrator` es la entidad que orquesta al sistema, le permite al usuario declarar una configuración para su entrenamiento y es responsable de traducir dicha configuración en especificaciones que envían los hiperparámetros necesarios a los nodos entrenadores. Funciona también como agregador de los resultados parciales de los otros nodos una vez que el entrenamiento finaliza. Solo hay uno por entrenamiento y para fines del análisis se utilizará su versión de ffi de python para una mejor automatización.

Los `Worker`s son nodos que dados una arquitectura y dataset, computa gradientes y (dependiendo del algoritmo escogido) también aplica pasos de optimización sobre los parámetros del modelo siendo entrenado. Cuando se entrena utilizando Parameter Server la optimización de parámetros del modelo global la aplica el servidor, en cambio en All Reduce, al ser un algoritmo descentralizado, de esto se encarga cada uno de los workers.

Los `Server`s son nodos que dados un algoritmo de inicialización de pesos, crean e inicializan lo pesos iniciales del modelo y esperan por gradientes calculados por los workers. Estos luego son agreagdos entre así y aplicados en los parámetros previamente dichos. Una vez se aplican los gradientes, los pesos actualizados son compartidos con los workers para repetir el proceso hasta que estos determinen que el entrenamiento finalizó.

### 2.1.2 Parameter Server

El algoritmo de `Parameter Server` es un algoritmo centralizado que se basa en tener servidores que guarden los parámetros del modelo y workers que publican gradientes que son luego agregados, aplicados en los pesos y redistribuidos hacia los workers, esto se hace en un loop hasta que el entrenamiento finalice.

> Agregar diagrama...

### 2.1.3 All Reduce

El algoritmo de `All Reduce` es un algoritmo descentralizado que se basa en tener varios workers en una topología de anillo donde cada uno computa un gradiente y es compartido con su próximo en la cadena de a partes, esto agiliza el proceso de agregación minimizando los mensajes totales. Luego cada uno aplica el gradiente total y repite el proceso hasta finalizar el entrenamiento.

> Agregar diagrama...

## 2.2 Sincronización.

Cada uno de estos algoritmos tiene un punto de sincronización, en el caso de `Parameter Server`, este es al momento de aplicar los pesos, en su modo bloqueante un servidor espera por los gradientes de todos los nodos antes de aplicar la optimización sobre los parámetros. En el caso de `All Reduce`, la sincronización ocurre al momento de compartir los gradientes calculados para esa epoch.

Si nosotros configuramos `E = epochs offline > 0` entonces este punto de sincronización sucede menos frecuentemente, lo que agiliza el entrenamiento porque tiene menos overhead de comunicación, la diferencia entre $$E = 0$$ y $$E = 1$$ es dividir la cantidad de mensajes y puntos de sincronización a la mitad. A su vez por sincronizar menos veces, se le permite a cada worker navegar la función de pérdida con más libertad, por lo que van a tender a discrepar más fuerte sobre cual es la dirección correcta del gradiente total, haciendo el entrenamiento más inestable.

# 3. Metodología Experimental

## 3.1.1 Hardware utilizado.

Para hacer las mediciones se utilizaron varias computadoras

> TODO: Agregar un listado de las computadoras que se utilizaron.

## 3.1.2 Dataset, modelo y arquitectura.

> TODO: Explicar el dataset LeNet5, el modelo y su arquitectura. Capaz mostrar el código que lo describe en python.

## 3.1.3 Configuración de los ensayos a medir.

> TODO: Explicar la configuración particular de ONO para esta parte, estilo max epochs, offline epochs, etc...

# 4 - Resultados y análisis.

> TODO: Estaría bueno que se me ocurra alguna cosa interesante que juegue con las epochs offline.

## 4.1 Ejecuciones de Parameter Server con distintas cantidades de epochs offline.

## 4.2 Ejecuciones de All Reduce con distintas cantidades de epochs offline.

## 4.3 Capaz una comparación entre los algoritmos (seria del informe de Loren pero tirando más hacía las epochs offline).

# 5 - Conclusiones y lecciones aprendidas.

## 5.1 Balance final entre velocidad de comunicación y estabilidad del entrenamiento.

# 6 - Bibliografía.
