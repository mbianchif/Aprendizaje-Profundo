#!/usr/bin/env -S uv run

from utils.measurement import measure_training_results, available_results
from pprint import pprint


def main() -> None:
    if not available_results():
        print("There are no past measurements, execute the `train.py` program to generate training session results")
        return

    mesaurements = measure_training_results()
    pprint(mesaurements)

    """
        Acá estaría bueno agrupar las mediciones por algoritmo, cantidad de nodos, cantidad de offline epochs.
        Me interesa mucho más la de offline epochs pero sí puedo mostrar los otros datos capaz es interesante.

        Me interesan gráficos que muestren la estabilidad del entrenamiento, estilo, que para cada configuración de
        offline epochs tenga un gráfico de pérdida superpuesto. Podria ser un gráfico de tiempo (muy parecido al de la
        tui) que muestre como el error disminuye y a medida que suben las epochs offline estos parezcan más "toscos".

        Me interesa también mostrar el tiempo que toma el entrenamiento dependiendo de las epochs offline. Podria ser un
        gráfico de barras que muestre como a medida que suben las epochs offline este también baje.

        Me interesa también mostrar la accuracy promedio agrupando por epochs offline, para ver si termina dando mejores
        resultados (acá no creo que sea mejor), se podría haber dado el caso que hay algún tipo de tendencia.

        Capaz estaria bueno ver la relación de cantidad de nodos y epochs offline, si son más nodos debería haber mayor
        ganancia por epochs offline. Teniendo den cuenta que parameter server es más bloqueante, mayores epochs offline
        también deberían dejar mayor ganancia y no deberian generar peor convergencia por ser ps.

        No creo que necesite algo más para sacar conclusiones.
    """


main()
