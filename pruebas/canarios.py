"""Canarios: fugas escritas A PROPÓSITO que el arnés está obligado a rechazar.

Idea de Fable (`RESPUESTA_FABLE_K_2026-09-18.md` §4.5) y nace de un hecho: el 2026-09-18 metí una fuga
deliberada en `dias_sic_12m` para comprobar que la prueba de punto en el tiempo servía, y **pasaba**. Tres
causas encadenadas, todas de la misma forma: **el caso que dispara la fuga nunca se genera.**

Una prueba en verde dice que el código cumple lo que la prueba mira. Un canario dice otra cosa, que es la
que faltaba: **que la prueba mira.** Si un canario pasa, el arnés se rompió aunque todo esté en verde.

Cada canario declara:
  - `nombre`: qué fuga introduce
  - `agujero`: cuál de los agujeros de Fable cubre
  - `parche`: función que monkeypatchea `dataset.py` para introducirla
  - `perturbacion`: qué perturbación del futuro debería delatarla
  - `columnas`: dónde hay que mirar

Regla de mantenimiento (de Fable): **toda fuga real atrapada se convierte en canario el mismo día.** Así
la tabla de defectos deja de ser un registro y pasa a ser una suite.
"""
import numpy as np
import pandas as pd

CORTE = pd.Timestamp('2020-09-30')


# ================================================================ perturbaciones del futuro
def perturbar_precio(p, tickers=('AAA',)):
    """La de siempre: el futuro de unos tickers se multiplica."""
    m = (pd.to_datetime(p['fecha']) > CORTE) & (p['ticker'].isin(tickers))
    p.loc[m, 'close'] = p.loc[m, 'close'] * 3.0
    return p


def perturbar_volumen(p, tickers=('AAA', 'AAA.MX')):
    """Apagar el volumen futuro. Sin esto, `volumen > 0` nunca cambia y nada que dependa de «si operó»
    está protegido. Es el agujero que se encontró el 2026-09-18."""
    m = (pd.to_datetime(p['fecha']) > CORTE) & (p['ticker'].isin(tickers))
    p.loc[m, 'volumen'] = 0
    return p


def perturbar_corte_entero(p, tickers=None):
    """Agujero 1 de Fable: las variables transversales (medianas de sector, rangos dentro del libro)
    dependen de OTROS tickers. Perturbar uno solo nunca las toca."""
    m = pd.to_datetime(p['fecha']) > CORTE
    p.loc[m, 'close'] = p.loc[m, 'close'] * 2.5
    return p


def perturbar_nulos(p, tickers=('AAA',)):
    """Agujero 7: NaN en bloques del futuro. Delata los rellenos hacia atrás (`bfill`)."""
    m = (pd.to_datetime(p['fecha']) > CORTE) & (p['ticker'].isin(tickers))
    p.loc[m, ['close', 'high', 'low']] = np.nan
    return p


def perturbar_deslistamiento(p, tickers=('AAA',)):
    """Agujero 6: borrar el futuro entero de un ticker. Si algo en D cambia, hay supervivencia en el código."""
    m = (pd.to_datetime(p['fecha']) > CORTE) & (p['ticker'].isin(tickers))
    return p[~m].copy()




def perturbar_calendario(p, tickers=('AAA',)):
    """Agujero 5 de Fable: quitar un día hábil FUTURO. Delata `shift(-1)`, `iloc` y ventanas por posición
    que se corren un día cuando el calendario cambia. Una perturbación de valores nunca produce este caso."""
    f = pd.to_datetime(p['fecha'])
    dias = sorted(f[f > CORTE].unique())
    if len(dias) < 10: return p
    # tienen que desaparecer para TODOS: el calendario del dataset es la unión de los tickers de tipo
    # empresa, así que quitarle días a uno solo no lo mueve. Primera versión de este canario escapó por eso.
    # y tienen que ser los PRIMEROS días después del corte: si se quitan más lejos, un desplazamiento
    # corto por posición nunca los alcanza desde una fila pasada y el canario escapa (comprobado).
    fuera = set(pd.to_datetime(dias[:3]))
    return p[~f.isin(fuera)].copy()


def perturbar_permutacion(p, tickers=('AAA',)):
    """Agujero 4: multiplicar el futuro por una constante CONSERVA el orden entre fechas. Una fuga que use
    rangos futuros pasa. Aquí el futuro se baraja, que rompe el orden sin cambiar el conjunto de valores."""
    m = (pd.to_datetime(p['fecha']) > CORTE) & (p['ticker'].isin(tickers))
    if m.sum() < 5: return p
    azar = np.random.default_rng(20260918)
    v = p.loc[m, 'close'].to_numpy()
    p.loc[m, 'close'] = azar.permutation(v)
    return p

PERTURBACIONES = {'precio': perturbar_precio, 'volumen': perturbar_volumen,
                  'corte_entero': perturbar_corte_entero, 'nulos': perturbar_nulos,
                  'deslistamiento': perturbar_deslistamiento, 'calendario': perturbar_calendario,
                  'permutacion': perturbar_permutacion}


# ================================================================ los canarios
def _fuga_shift(DS):
    """Clásica: una variable de liquidez que mira 5 días adelante."""
    orig = DS.grupo_liquidez
    def fugado(m, close_mxn):
        f = orig(m, close_mxn)
        f['importe_20d'] = f['importe_20d'].shift(-5)
        return f
    DS.grupo_liquidez = fugado
    return lambda: setattr(DS, 'grupo_liquidez', orig)


def _fuga_bfill(DS):
    """Relleno HACIA ATRÁS: trae un precio futuro al pasado.

    El canario abre el hueco él mismo, en LAS DOS construcciones, justo antes del corte. Sin hueco no hay
    nada que rellenar y el `bfill` es inocuo: la primera versión de este canario escapó por eso, y el
    defecto era del canario y no del arnés. Un hueco antes del corte es además realista —una suspensión—
    y es exactamente el caso en que un relleno hacia atrás trae el futuro al pasado."""
    orig = DS.matrices
    def fugado(precios, desde_lookback):
        m = orig(precios, desde_lookback)
        c = m['close'].copy()
        hueco = (c.index > CORTE - pd.Timedelta(days=12)) & (c.index <= CORTE)
        c.loc[hueco, :] = np.nan          # el hueco existe en las dos corridas: no es la perturbación
        m['close'] = c.bfill()            # y aquí el futuro entra al pasado
        return m
    DS.matrices = fugado
    return lambda: setattr(DS, 'matrices', orig)


def _fuga_volumen_futuro(DS):
    """La fuga real del 2026-09-18, conservada como canario: `dias_sic_12m` mirando adelante."""
    orig = DS.liquidez_sic
    def fugado(m, columnas, **kw):
        f = orig(m, columnas, **kw)
        f['dias_sic_12m'] = f['dias_sic_12m'].shift(-5)
        return f
    DS.liquidez_sic = fugado
    return lambda: setattr(DS, 'liquidez_sic', orig)


def _fuga_transversal(DS):
    """Agujero 1 de Fable: las variables TRANSVERSALES dependen de otros tickers en la misma fecha.

    La fuga resta a cada emisora la media DEL CORTE de tres días adelante. Es transversal y es futura a la
    vez, que es justo la combinación que la perturbación de un solo ticker no puede delatar: hace falta
    perturbar el corte entero. Por eso su perturbación es `corte_entero` y no `precio`."""
    orig = DS.grupo_liquidez
    def fugado(m, close_mxn):
        f = orig(m, close_mxn)
        x = f['log_importe_20d']
        f['log_importe_20d'] = x.sub(x.mean(axis=1).shift(-3), axis=0)
        return f
    DS.grupo_liquidez = fugado
    return lambda: setattr(DS, 'grupo_liquidez', orig)


def _fuga_supervivencia(DS):
    """Agujero 6: el universo de D definido como «tickers que existen en el archivo», que incluye saber
    quién sobrevivió. Si borrar el futuro de una emisora cambia algo en D, hay supervivencia en el código."""
    orig = DS.matrices
    def fugado(precios, desde_lookback):
        m = orig(precios, desde_lookback)
        c = m['close'].copy()
        # «vivo» = tiene precio en el ÚLTIMO día del archivo. Saber eso en D es saber quién sobrevivió.
        # La fuga inclina el precio de los vivos en una ventana del pasado; no anula nada, porque anular
        # rompe módulos de aguas abajo y un canario tiene que fallar por la fuga, no por una excepción.
        vivos = c.iloc[-1].notna()
        ventana = c.index[(c.index > CORTE - pd.Timedelta(days=45)) & (c.index <= CORTE)]
        for t in c.columns[vivos]:
            c.loc[ventana, t] = c.loc[ventana, t] * 1.05
        m['close'] = c
        return m
    DS.matrices = fugado
    return lambda: setattr(DS, 'matrices', orig)



def _fuga_por_posicion(DS):
    """Agujero 5: una ventana por POSICIÓN en vez de por fecha. Mientras el calendario no cambie parece
    correcta; al quitar un día futuro, la ventana de una fila pasada se corre y trae el futuro."""
    orig = DS.grupo_liquidez
    def fugado(m, close_mxn):
        f = orig(m, close_mxn)
        x = f['amihud_60d']
        f['amihud_60d'] = pd.DataFrame(np.roll(x.to_numpy(), -2, axis=0), index=x.index, columns=x.columns)
        return f
    DS.grupo_liquidez = fugado
    return lambda: setattr(DS, 'grupo_liquidez', orig)


def _fuga_rango_futuro(DS):
    """Agujero 4: la variable usa el rango del precio DE MAÑANA.

    Multiplicar el futuro por una constante no la delata, porque multiplicar conserva el orden y el rango
    no cambia. Barajar el futuro sí. Un rango sobre TODA la serie tampoco sirve como canario —lo probé y
    escapó— porque permutar el futuro no altera el multiconjunto y el rango de un valor pasado queda
    igual: hace falta que la fuga dependa del ORDEN, no del conjunto."""
    orig = DS.grupo_liquidez
    def fugado(m, close_mxn):
        f = orig(m, close_mxn)
        f['prop_ceros_60d'] = m['close'].shift(-1).rank(axis=1, pct=True)
        return f
    DS.grupo_liquidez = fugado
    return lambda: setattr(DS, 'grupo_liquidez', orig)


CANARIOS = [
    {'nombre': 'liquidez_con_shift_negativo', 'agujero': 'clásico: la variable mira adelante',
     'parche': _fuga_shift, 'perturbacion': 'precio', 'columnas': ['importe_20d']},
    {'nombre': 'relleno_hacia_atras', 'agujero': '7 · nulos futuros y `bfill`',
     'parche': _fuga_bfill, 'perturbacion': 'nulos', 'columnas': None},
    {'nombre': 'dias_sic_con_shift_negativo', 'agujero': '2 · series auxiliares (fuga real del 2026-09-18)',
     'parche': _fuga_volumen_futuro, 'perturbacion': 'volumen', 'columnas': ['dias_sic_12m']},
    {'nombre': 'rango_de_sector_con_futuro', 'agujero': '1 · variables transversales',
     'parche': _fuga_transversal, 'perturbacion': 'corte_entero', 'columnas': ['log_importe_20d']},
    {'nombre': 'universo_por_quien_sobrevivio', 'agujero': '6 · deslistamiento futuro y supervivencia',
     'parche': _fuga_supervivencia, 'perturbacion': 'deslistamiento', 'columnas': None},
    {'nombre': 'ventana_por_posicion', 'agujero': '5 · el calendario futuro cambia',
     'parche': _fuga_por_posicion, 'perturbacion': 'calendario', 'columnas': ['amihud_60d']},
    {'nombre': 'rango_sobre_toda_la_serie', 'agujero': '4 · perturbaciones monótonas no lo delatan',
     'parche': _fuga_rango_futuro, 'perturbacion': 'permutacion', 'columnas': ['prop_ceros_60d']},
]
