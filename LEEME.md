# El arnés de punto en el tiempo de Brota

Código de las pruebas que sostienen la disciplina de punto en el tiempo del dataset
[`Alexallonson/brota-publicable`](https://huggingface.co/datasets/Alexallonson/brota-publicable), más el
código que construye y evalúa esa tabla.

**Se publica por una razón concreta:** la ficha del dataset afirma que hay dos pruebas de punto en el
tiempo y un banco de canarios, y esas afirmaciones no valían nada mientras el código fuera privado. Ahora
se pueden correr.

## Correrlo

```bash
pip install numpy pandas pyarrow hypothesis scikit-learn
python3 -m unittest discover -s pruebas
```

**No hace falta ningún dato.** Todas las pruebas construyen su propio almacén sintético
(`pruebas/test_dataset.py::almacen_sintetico`): unas pocas emisoras con precios generados, un espejo del
SIC, índices, divisas y macro. Tarda un par de minutos.

## Qué hay aquí

| | |
|---|---|
| `laboratorio/dataset.py` | Construye la tabla: una fila por emisora y día, cada variable con la fecha en que **se conocía** |
| `laboratorio/fundamentales.py` | Estados financieros de la SEC con su fecha de presentación |
| `laboratorio/evaluar.py` | Mide cualquier puntaje contra líneas base, con costos y tope por liquidez |
| `laboratorio/evaluar_caidas.py` | Lo mismo para caídas fuertes, con calibración y descomposición de Murphy |
| `laboratorio/entrenar.py` | Entrenamiento con validación por tiempo (*walk-forward*) |
| `pruebas/test_propiedades.py` | **Las dos pruebas de punto en el tiempo** (Hypothesis) |
| `pruebas/canarios.py` + `test_canarios.py` | **El banco de canarios** |
| el resto de `pruebas/` | Propiedades de métricas, cartera, fundamentales y evaluación |

## Las dos pruebas que importan

**1. Futuro alterado.** Se construye la tabla, se cambia el futuro —precios, volúmenes, reportes,
calendario— y se exige que **ninguna fila anterior al corte se mueva**. Cubre todas las columnas de los
grupos de variables; los objetivos quedan fuera a propósito, porque miran al futuro por diseño.

**2. Split futuro.** Se aplica un split y se reexpresa el historial como lo hace Yahoo. El pasado
tampoco puede moverse.

Las dos han encontrado fugas reales, y varias **no habrían cambiado ningún resultado**: al corregirlas
los números se quedaban igual. Ésas son las peligrosas, porque ninguna revisión de resultados las
delata.

## El banco de canarios, y por qué existe

Una prueba en verde dice que el código cumple lo que la prueba mira. **No dice que la prueba mire.**

El 2026-09-18 metí una fuga deliberada en una variable para comprobar que la prueba de punto en el
tiempo servía, y **pasaba**. Tres causas encadenadas, todas de la misma forma: *el caso que dispara la
fuga nunca se genera.* La serie afectada no estaba entre las que la prueba perturbaba; la perturbación
multiplicaba el volumen pero nunca lo apagaba, así que la condición «operó» jamás cambiaba; y con tres
ejemplos el muestreador podía no llegar nunca al caso.

De ahí salió esto: **siete fugas escritas a propósito que el arnés está obligado a rechazar**. Si un
canario pasa, la suite falla con el nombre del agujero que se abrió, **aunque todo lo demás esté en
verde**.

Cubren siete de doce tipos de fuga identificados. **Faltan cinco**, y están nombrados en el código:
sobre todo las tres pruebas de **sensibilidad**, que son las que la invariancia no puede sustituir —
la invariancia dice «lo prohibido no influye», no dice «lo permitido influye».

Y un dato que preferimos publicar antes que callar: **seis de nueve canarios escritos pasaron en su
primer intento por defectos del propio canario.** Escribir la fuga es fácil; construir el caso que la
dispara es lo difícil, que es exactamente por lo que el arnés tenía el agujero original. Por eso el
banco tiene su propio control.

## Lo que NO está, y por qué

- **Ningún dato.** Ni precios, ni el almacén, ni las tablas de prueba con datos reales. Los datos de
  mercado del proyecto vienen de Yahoo Finance, con restricción contractual, y no se redistribuyen.
- **La app.** Este paquete es el laboratorio, no el producto.
- **Ningún modelo entrenado.** No hay pesos ni puntajes.
- **Una prueba de `test_evaluar.py`** (`test_costos_leidos_de_la_app`) lee las constantes de comisión
  desde el código de la app, que no está aquí, así que da error. Es la única, y se deja a la vista en
  lugar de borrarla para que el conteo cuadre.

## Qué se puede verificar con esto

Que la disciplina existe y que las pruebas muerden. Se puede meter una fuga a propósito en
`laboratorio/dataset.py` y comprobar que el arnés la ve — que es exactamente lo que hicimos y lo que
descubrió que a veces **no** la veía.

## Advertencia

Brota es una app **didáctica** con **dinero ficticio**, sin recomendaciones individualizadas. Esto no es
consejo de inversión, y el laboratorio concluyó que **no hay nada publicable** del lado de las señales:
existe orden en los datos y no sobrevive a los costos reales de operar.
