#!/usr/bin/env python3
"""
laboratorio/evaluar.py — evaluación fuera de muestra de puntajes: líneas base y modelos (LABORATORIO_IA.md §2.5, §2.6 y §3.8).

Uso:
    python3 laboratorio/evaluar.py --lineas-base                               # dataset en laboratorio/datos (lo escribe dataset.py)
    python3 laboratorio/evaluar.py --lineas-base --dataset RUTA --solo-app
    python3 laboratorio/evaluar.py --puntajes pred.parquet --nombre lgbm-v1 --variables ret_6m,vol_60d [--ablacion]   (lo usará entrenar.py)

Protocolo:
  - Rebalanceo mensual: primer día hábil de cada mes. El puntaje de ese día solo usa lo que el dataset permite (precios ≤ D, datos con conocido < D).
  - IC = correlación de Spearman entre puntaje y rendimiento total en pesos a 21 y 63 días, por fecha, con al menos MIN_EMPRESAS.
    t con Newey-West: a 63 días las ventanas de meses seguidos se traslapan (rezagos = 2); a 21 días, sin rezagos.
  - Quintiles iguales por fecha (Q1 = mejor puntaje). Costo = rotación × comisión × (1 + IVA), leídos de js/metas.js para no duplicarlos.
    Q1−Q5 neto (ambas patas pagan) y Q1 neto contra el universo completo neto (lo que puede hacer quien no vende en corto).
  - Estabilidad: correlación de rangos del puntaje entre un mes y el siguiente (baja = el modelo es ruido caro).
  - Criterio para entrar a la app (§2.6, versión aprobada por el dueño el 2026-09-15, pedido 89): IC > 0.03 con t ≥ 3 en ≥ 60 meses; Q1−Q5 neto con
    IC 90 % por bootstrap de bloques de 12 meses que excluya cero; Sharpe desinflado (Bailey y López de Prado, 2014) con el registro de intentos;
    peor año de Q1 contra el universo, neto, no peor que −20 % (pedido 151); y vencer al compuesto lineal por ejes (pedido 94). Compuerta de publicación sobre variables Y objetivo (publicable.verificar, pedido 70),
    cobertura de emisoras que salieron de bolsa (pedido 72) y analistas (grupo C) solo en ablación (pedido 64).
Resultados que solo crecen: laboratorio/resultados/AAAA-MM-DD_<nombre>.json y un reporte .md por corrida; nunca se sobrescriben.
"""
import argparse, datetime as dt, json, os, re, subprocess, sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
import publicable  # noqa: E402

VERSION = 'evaluar-v2'
HORIZONTES = (21, 63)
MIN_EMPRESAS = 25
SEMILLA = 20260914
# Regla de §2.6 vigente. Dueño, 2026-09-15: aprobó t ≥ 3, 60 meses, intervalo por bloques, ajuste por intentos, peor año −8 % (pedido 89) y, tras la
# tanda 3 de Science Mercado, decidió: (1) se juzga Q1 topado por importe contra el universo, neto (Q1−Q5 queda como diagnóstico de monotonía);
# (2) DSR ≥ 0.95 como probabilidad, con N = 1 en el periodo que no se usó para elegir (el DSR con todos los intentos se reporta al lado);
# (3) la referencia a vencer es el impulso 12-1 (pedidos 119, 120, 121, 123). Tras la tanda 4 (pedido 151): peor año −20 % (con el riesgo de factor de
# Q1 ninguna cartera de quintil llega a −8 %) y pliegue externo desde 2021-01 (68 meses con 2022; ahí va el DSR con N = 1).
CRITERIO = {'horizonte': 21, 'ic_medio_min': 0.03, 't_min': 3.0, 'meses_min': 60,
            'cartera': 'Q1 topado por importe contra el universo, neto',
            'ic90_excluye_cero': True, 'bloque_bootstrap_meses': 12, 'remuestreos': 2000,
            'dsr_min': 0.95, 'dsr_intentos': 1,
            'peor_anio_relativo_min': -0.20, 'meses_min_por_anio': 6, 'pliegue_externo_desde': '2021-01-01',
            'referencia': 'impulso_12m_ex1m', 'desempate': 'sharpe_anual_margen'}
# Dueño, 2026-09-16 (tanda 6, M31): la condición 6 se decide por el SHARPE del margen de Q1 topado contra el universo, no por IC y media.
# Con 108 meses la diferencia de media entre candidatos es ruido (t −0.26) y donde sí difieren es en riesgo.
#
# COMPUERTA DE DOS CONDICIONES — decisión del dueño del 2026-09-17 (pedido 191), sobre la propuesta convergente de Mercado (su D3) y Fable
# (su G3). No son dos formas de medir lo mismo: son dos preguntas distintas, y **cada afirmación que aparezca en la app tiene que declarar
# de cuál de las dos viene**. Una condición no sustituye a la otra ni la implica.
#
#   SEÑAL          Q1 topado − canasta al azar con la MISMA rotación, el mismo tope y los mismos costos (`control_igual_rotacion`).
#                  Libre de referencia —se cancela— y robusta al costo —el control lo paga igual—.
#                  Habilita: «ordenar así separó ganadoras de perdedoras».
#                  Al 2026-09-17 el impulso 12-1 LA CUMPLE: +0.52 pp/mes en 2007-2020.
#
#   COBRABILIDAD   Q1 topado − índice invertible del libro, con el costo COMPLETO y la corrida al DOBLE como obligatoria.
#                  Habilita: «rindió más que dejarlo en un fondo».
#                  Al 2026-09-17 NADIE LA CUMPLE: el margen no excluye el cero con el costo completo.
#
# Por qué dos y no una: con una sola referencia una canasta al azar igual ponderada pasaría, porque el índice arrastra la prima de igual
# ponderación (+0.425 pp/mes medido). Y exigir las dos cuesta 0.003 de potencia (RESULTADO_POTENCIA_COMPUERTA_2026-09-17.md), o sea nada.
CONDICIONES = {
    'senal': {'contra': 'control al azar con rotación, tope y costos igualados', 'columna': 'q1_topado_vs_control_neto',
              'habilita': 'ordenar así separó ganadoras de perdedoras', 'costo': 'se cancela'},
    'cobrabilidad': {'contra': 'índice invertible del libro', 'columna': 'q1_topado_vs_indice_neto',
                     'habilita': 'rindió más que dejarlo en un fondo', 'costo': 'completo, y al doble como corrida obligatoria'},
}
FAMILIA_RANKING = 'ranking_21d'
REGISTRO_INTENTOS = 'registro_intentos.jsonl'   # pedido 115: solo crece; un renglón por candidato evaluado (no diagnósticos ni controles)
SUBCONJUNTOS = {'todas': lambda d: d, 'mexico': lambda d: d[d['pais'] == 'MX'], 'extranjero': lambda d: d[d['pais'] != 'MX']}

# nombre: (variables, puntaje(tabla) → Serie, descripción). Mayor puntaje = mejor.
LINEAS_BASE = {
    'impulso_6m': (['ret_6m'], lambda t: t['ret_6m'], 'Impulso: rendimiento total de 6 meses (línea base a de §2.5)'),
    'impulso_12m_ex1m': (['ret_12m_ex1m'], lambda t: t['ret_12m_ex1m'], 'Impulso clásico: 12 meses sin el último'),
    'valor_pe_sector': (['pe_rel_sector'], lambda t: -t['pe_rel_sector'], 'Valor: P/E reconstruido relativo a la mediana del sector, menor = mejor (línea base b)'),
    'baja_volatilidad': (['vol_250d'], lambda t: -t['vol_250d'], 'Baja volatilidad: menor volatilidad de un año = mejor'),
    'placebo': ([], None, 'Control: puntaje al azar con semilla fija; su IC debe rondar 0'),
}
CONTROLES = {'placebo'}
# Pedido 94: compuesto lineal por ejes, cada variable como rango del día centrado (−0.5 a 0.5) con su signo, promedio por eje y promedio de ejes
EJES_COMPUESTO = {
    'impulso': [('ret_12m_ex1m', 1), ('ret_6m', 1)],
    'valor': [('pe_rel_sector', -1), ('libros_mercado', 1)],
    'calidad': [('roa_4t', 1), ('rentabilidad_bruta', 1), ('devengos_4t', -1), ('apalancamiento_libros', -1)],
    'baja_volatilidad': [('vol_250d', -1), ('beta_mercado_1a', -1)],
    # sin eje de liquidez (Science Mercado, M12, argumento previo al resultado): no es eje de rendimiento a 21 días y en universo mezclado
    # es indicadora de país; la liquidez queda como restricción (cartera topada) y control
    'riesgo_caida': [('ivol_60d', -1), ('max_ret_1m', -1)],
}
VARIABLES_COMPUESTO = [c for vs in EJES_COMPUESTO.values() for c, _ in vs]


def puntaje_compuesto(tabla, ejes=EJES_COMPUESTO, min_ejes=3):
    """Suma de ejes entre el número TOTAL de ejes: un eje sin datos vale 0 (neutral en el rango del día). Promediar solo los ejes disponibles
    comparaba emisoras puntuadas con 3 ejes (mexicanas sin grupo D) contra otras con 5, y la cobertura es indicadora de país (tanda 4, §5)."""
    por_eje = []
    for vs in ejes.values():
        cols = [(tabla.groupby('fecha')[c].rank(pct=True) - 0.5) * sg for c, sg in vs if c in tabla]
        if cols: por_eje.append(pd.concat(cols, axis=1).mean(axis=1))
    ejes_df = pd.concat(por_eje, axis=1)
    return (ejes_df.fillna(0).sum(axis=1) / len(ejes)).where(ejes_df.notna().sum(axis=1) >= min_ejes)


# nombre nuevo: el registro de intentos conserva el compuesto anterior (media de ejes disponibles) como otro intento que cuenta para el DSR
# Tanda 6, M32.4: compuesto de rangos de las cinco variables con mayor importancia por permutación en el modelo, con el signo
# de su relación conocida (las dos de beta en negativo, que es la dirección que midió el diagnóstico del canal de beta).
# No estima nada, pero la SELECCIÓN de las cinco salió de mirar el modelo: se declara y cuenta como intento.
CINCO_IMPORTANTES = [('caja_activos', 1), ('sorpresa_prom_4', 1), ('beta_ipc_1a', -1), ('beta_mercado_1a', -1), ('devengos_4t', -1)]


def puntaje_cinco(tabla):
    cols = [(tabla.groupby('fecha')[c].rank(pct=True) - 0.5) * sg for c, sg in CINCO_IMPORTANTES if c in tabla]
    d = pd.concat(cols, axis=1)
    return d.mean(axis=1).where(d.notna().sum(axis=1) >= 3)


LINEAS_BASE['compuesto_cinco'] = ([c for c, _ in CINCO_IMPORTANTES], puntaje_cinco,
    'Compuesto de rangos de las cinco variables más importantes del modelo (caja/activos, sorpresa media, −beta IPC, −beta del mercado propio, −devengos): sin ajuste, explicable en una tabla (tanda 6, M32.4)')
LINEAS_BASE['compuesto_lineal_neutral'] = (VARIABLES_COMPUESTO, puntaje_compuesto, 'Compuesto lineal de cinco ejes (impulso, valor, calidad, baja volatilidad, riesgo de caída) con eje faltante neutral; la liquidez salió por M12 (pedido 94)')
PENDIENTES_BASE = {'tres_ejes_de_la_app': 'Promedio de los tres ejes (línea base c): sin historia evaluable. Los ejes usan márgenes, deuda y valuación de Yahoo '
                                'que solo se archivan desde 2026-09 (fotos_valuacion); se podrá evaluar cuando haya ≥ 24 meses o cuando los '
                                'financieros de EDGAR/CNBV entren al dataset (grupo D, pedido 65).'}
LIMITACIONES = [
    'sesgo de supervivencia: la mayoría de las emisoras que salieron de bolsa no tiene historia en el dataset (ver cobertura de salidas)',
    'una emisora que deja de cotizar antes de 21/63 días queda sin desenlace y fuera de ese mes; sus pérdidas no cuentan (ver sin_desenlace)',
    'costos: comisión + IVA, medio diferencial de compra-venta estimado de máximos y mínimos (Abdi-Ranaldo, 2017) y conversión de divisa '
    'en el SIC; no incluye impacto de mercado ni ISR. El estimador del diferencial sesga a la BAJA cuando hay días sin operar, así que el '
    'costo cobrado es un piso: por eso la corrida obligatoria al doble',
    'rendimientos en pesos: el corte transversal mezcla emisoras de varios países, así que el tipo de cambio pesa; ver subconjuntos mexico/extranjero',
    'las líneas base no se ajustan con datos, así que todo su periodo es fuera de muestra; un modelo entrenado solo cuenta los meses posteriores a su entrenamiento',
]


# ================================================================ entradas
def leer_dataset(ruta, columnas):
    import pyarrow.dataset as pads
    base = os.path.join(ruta, 'dataset')
    d = pads.dataset(base, format='parquet', partitioning='hive')
    cols = [c for c in dict.fromkeys(columnas) if c in d.schema.names]
    t = d.to_table(columns=cols)
    tabla = t.to_pandas(split_blocks=True, self_destruct=True); del t   # memoria: sin copia doble de la tabla completa
    tabla['ticker'] = tabla['ticker'].astype(str)
    meta_ruta = os.path.join(ruta, 'dataset_meta.json')
    meta = {}
    if os.path.exists(meta_ruta):
        with open(meta_ruta, encoding='utf-8') as fh: meta = json.load(fh)
    return tabla, meta


def costos_app(raiz=RAIZ):
    """Comisión por operación (METAS_DEF.comision) e IVA (LEY_MX.iva) tal como los usa la app (js/metas.js, js/rebalanceo.js)."""
    with open(os.path.join(raiz, 'js', 'metas.js'), encoding='utf-8') as fh: txt = fh.read()
    com = re.search(r'METAS_DEF\s*=\s*\{[^}]*?\bcomision:\s*([0-9.]+)', txt)
    iva = re.search(r'\biva:\s*([0-9.]+)', txt)
    if not com or not iva: raise SystemExit('No encontré METAS_DEF.comision o LEY_MX.iva en js/metas.js')
    c, i = float(com.group(1)), float(iva.group(1))
    return {'comision': c, 'iva': i, 'costo_por_peso_operado': c * (1 + i), 'fuente': 'js/metas.js (METAS_DEF.comision, LEY_MX.iva)'}


# ================================================================ costos por operación (pedidos 99 y 184)
# Tres sumandos por cada peso operado, y los tres se cobran en la compra y en la venta:
#   1. comisión + IVA, que es lo que cobra la app (`costos_app`);
#   2. el MEDIO diferencial de compra-venta de la emisora ese mes (`COL_DIFERENCIAL`, Abdi-Ranaldo en dataset.py): quien compra paga la
#      mitad superior del diferencial y quien vende la inferior, así que el costo de una operación es medio diferencial, no uno entero;
#   3. y en las extranjeras, el diferencial del SIC EN LUGAR del de su bolsa de origen. Los precios del dataset son los de Nueva York, pero
#      un inversionista mexicano compra ese papel en el SIC, cotizado en pesos y con un diferencial más ancho porque el volumen del SIC en
#      un papel estadounidense es una fracción del de su mercado natural. Se cobra POR OPERACIÓN —no es una conversión que se evite
#      quedándose en dólares, es un precio peor en cada compra y en cada venta— y **sustituye** al de origen, no se le suma: quien opera en
#      el SIC paga el diferencial del SIC, no el de Nueva York más un recargo (corrección de Science Mercado, D1, 2026-09-17).
#      El 0.628 % es un SUPUESTO —la suma que se cobraba antes de la corrección— y está por medir: las emisoras del SIC tienen su propia
#      clave con máximo, mínimo y cierre, así que el mismo estimador de Abdi-Ranaldo se puede correr sobre el papel que de hecho se compra.
COL_DIFERENCIAL = 'medio_dif_ar_60d'
# MEDIDO el 2026-09-17 (pedido 198, autorizado por el dueño), ya no supuesto. Se bajaron las claves del SIC de las 90 extranjeras y se les
# corrió el mismo estimador de Abdi-Ranaldo: 64 tienen serie y 26 no. En la muestra FIABLE —22 claves con menos del 10 % de días sin rango
# intradía— la mediana del medio diferencial del SIC es 0.254 % y el percentil 75 es 0.376 %; en la bolsa de origen, 0.073 %. O sea que el
# SIC sí es ~3.5 veces más caro que Nueva York, y el supuesto anterior de 0.628 % estaba por encima incluso del percentil 90 (0.512 %).
#
# SE COBRA EL PERCENTIL 75 (0.376 %) Y NO LA MEDIANA, y la elección se declara antes de volver a correr nada porque el cambio abarata los
# costos y por lo tanto me favorece. Tres razones: (1) las 42 claves con muchos días sin rango dan estimaciones poco fiables y su verdadero
# diferencial es casi seguro mayor, no menor; (2) las 26 sin serie en el SIC son presumiblemente las menos líquidas de todas; (3) el
# estimador sesga a la baja cuando hay días sin operar, que es exactamente el caso del SIC. Detalle por clave en `laboratorio/sic_medido.csv`.
DIFERENCIAL_SIC = 0.00376
DIFERENCIAL_SIC_MEDIDO = {'mediana_fiables': 0.00254, 'p75_fiables': 0.00376, 'p90_fiables': 0.00512,
                          'mediana_todas_con_serie': 0.00348, 'origen_fiables': 0.00073,
                          'claves_con_serie': 64, 'claves_sin_serie': 26, 'claves_fiables': 22,
                          'supuesto_anterior': 0.00628}


def costo_unitario(filas, k, sic=DIFERENCIAL_SIC):
    """Costo por peso operado de cada emisora: comisión + IVA + su medio diferencial, y en las extranjeras el diferencial del SIC en lugar
    del de su bolsa de origen. Sin la columna del diferencial devuelve k para todas, que es el costo de antes de los pedidos 99 y 184."""
    base = pd.Series(float(k), index=filas.index)
    fuera = (filas['pais'].to_numpy() != 'MX') if 'pais' in filas else np.zeros(len(filas), bool)
    if COL_DIFERENCIAL in filas:
        # a la emisora sin estimación ese mes se le cobra la mediana del día, no cero: faltar el dato no puede salir gratis
        dif = filas[COL_DIFERENCIAL].astype(float)
        dif = dif.fillna(dif.median()).fillna(0.0)
        base = base + (pd.Series(np.where(fuera, 0.0, dif.to_numpy()), index=filas.index) if sic else dif)
    if sic:
        base = base + np.where(fuera, sic, 0.0)
    return base


def commit_codigo(raiz=RAIZ):
    try:
        c = subprocess.run(['git', '-C', raiz, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
        sucio = bool(subprocess.run(['git', '-C', raiz, 'status', '--porcelain', '--', 'laboratorio'], capture_output=True, text=True).stdout.strip())
        return {'commit': c, 'laboratorio_con_cambios_sin_commit': sucio}
    except Exception:
        return {'commit': None, 'laboratorio_con_cambios_sin_commit': None}


# ================================================================ estadística
def fechas_rebalanceo(fechas, cobertura=0.9):
    """Primer día de cada mes con al menos `cobertura` × el máximo de emisoras de ese mes. Con el calendario unión, el primer día hábil puede
    ser feriado en la BMV (p. ej. 2021-02-01) y dejar el corte sin emisoras mexicanas. Si `fechas` no trae repeticiones, es el primer día."""
    cuenta = pd.Series(pd.to_datetime(pd.Series(fechas))).value_counts().sort_index()
    mes = cuenta.index.to_period('M')
    maximo = cuenta.groupby(mes).transform('max')
    ok = cuenta[cuenta >= cobertura * maximo]
    return pd.DatetimeIndex(ok.groupby(ok.index.to_period('M')).apply(lambda s: s.index.min()).to_numpy())


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if len(ra) < 3 or ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def t_newey_west(x, rezagos=0):
    """t de la media con varianza de Newey-West (pesos de Bartlett). Con rezagos=0 es la t usual (varianza con n en el denominador)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < 3: return None
    e = x - x.mean(); s = float(e @ e) / n
    for L in range(1, min(rezagos, n - 1) + 1):
        s += 2 * (1 - L / (rezagos + 1)) * float(e[L:] @ e[:-L]) / n
    # serie (casi) constante: la varianza es ruido de redondeo y la t no significa nada (umbral relativo a la escala, prueba de propiedades)
    return float(x.mean() / np.sqrt(s / n)) if s > 1e-12 * max(float(np.mean(x * x)), 1e-300) else None


def rotacion(anterior, actual):
    """Suma de |cambio de peso| entre dos carteras de pesos iguales (1.0 = se compra todo desde cero)."""
    if not actual: return 0.0
    w1 = {t: 1 / len(actual) for t in actual}
    w0 = {t: 1 / len(anterior) for t in anterior} if anterior else {}
    return float(sum(abs(w1.get(t, 0) - w0.get(t, 0)) for t in set(w1) | set(w0)))


CAPITAL_REFERENCIA = 1_000_000          # pesos: cartera de referencia para topar posiciones por importe operado (pedido 105)
PARTICIPACION_MAX_IMPORTE = 0.01         # una posición no pasa de 1 % del importe diario medio de 20 días (Science Mercado, M6)



# Tope por liquidez del SIC (pedido AT de Fable, `REGLA_TOPE_SIC.md`). Un mexicano compra las extranjeras en el SIC, no en su bolsa
# de origen, y medido el 2026-09-18 el importe de origen sobreestima la capacidad real por miles de veces. Con `tope_sic=True` el
# libro extranjero se topa con `importe_sic_20d`; el mexicano NO cambia, porque ahí la BMV ya es el mercado correcto.
DIAS_SIC_MIN = 0.60                      # corte de elegibilidad propuesto por Fable; no se afina


def importe_para_topar(filas, tope_sic):
    """Importe con el que se topa cada posición: el del SIC para las extranjeras cuando `tope_sic`, el de su bolsa si no."""
    imp = filas['importe_20d'].to_numpy(float) if 'importe_20d' in filas else np.full(len(filas), np.nan)
    if not tope_sic: return imp
    if 'importe_sic_20d' not in filas:   # falla RUIDOSA: si faltara en silencio, la corrida diría «no cambia nada» cuando en verdad no corrió
        raise ValueError('con tope_sic hace falta la columna importe_sic_20d en el dataset (dataset-v7)')
    es_ext = (filas['pais'] != 'MX').to_numpy() if 'pais' in filas else np.zeros(len(filas), bool)
    return np.where(es_ext, filas['importe_sic_20d'].to_numpy(float), imp)


def elegibles_sic(filas, tope_sic):
    """Máscara de lo comprable desde México: las extranjeras que no operan en el SIC al menos DIAS_SIC_MIN del año quedan fuera."""
    if not tope_sic: return np.ones(len(filas), bool)
    if 'dias_sic_12m' not in filas:
        raise ValueError('con tope_sic hace falta la columna dias_sic_12m en el dataset (dataset-v7)')
    es_ext = (filas['pais'] != 'MX').to_numpy() if 'pais' in filas else np.zeros(len(filas), bool)
    d = filas['dias_sic_12m'].to_numpy(float)
    return ~es_ext | (np.isfinite(d) & (d >= DIAS_SIC_MIN))

def pesos_topados(importe, capital=CAPITAL_REFERENCIA, participacion=PARTICIPACION_MAX_IMPORTE, objetivo=None):
    """Pesos con tope por emisora = participacion × importe_20d / capital; lo que sobra se reparte entre las no topadas en proporción al
    objetivo y lo que no cabe queda en efectivo (rinde 0). Sin importe conocido: sin tope. `objetivo` (por ejemplo 1/beta) son pesos
    relativos antes del tope; por omisión, iguales."""
    imp = np.asarray(importe, float); n = len(imp)
    if n == 0: return np.array([])
    obj = np.ones(n) if objetivo is None else np.asarray(objetivo, float).copy()
    obj = np.where(np.isfinite(obj) & (obj > 0), obj, np.nan)
    if np.isnan(obj).all(): obj = np.ones(n)
    obj = np.where(np.isnan(obj), np.nanmedian(obj), obj)
    tope = np.where(np.isfinite(imp), participacion * imp / capital, np.inf)
    w = np.zeros(n); libres = np.ones(n, bool); resto = 1.0
    while libres.any() and resto > 1e-12:
        peso_libre = obj[libres].sum()
        if peso_libre <= 0: break
        cuota = resto * obj / peso_libre
        topa = libres & (tope <= cuota + w)
        if not topa.any():
            w[libres] += cuota[libres]; resto = 0.0; break
        resto -= (tope[topa] - w[topa]).sum(); w[topa] = tope[topa]; libres &= ~topa
    return w


def rotacion_pesos(anterior, actual):
    """Suma de |cambio de peso| entre dos carteras dadas como {ticker: peso}."""
    anterior = anterior or {}
    return float(sum(abs(actual.get(t, 0) - anterior.get(t, 0)) for t in set(actual) | set(anterior)))


def control_igual_rotacion(anterior, disponibles, tamano, cambios, azar):
    """Pedido 193 (Mercado P4.2): la canasta al azar contra la que se compara Q1 tiene que rotar IGUAL que Q1, o el control paga costos
    que Q1 no paga y la señal sale inflada (el placebo por puntaje al azar rota 147 % mensual contra 47 % de Q1, y eso son 0.29 pp/mes).

    Regla: se conservan los nombres del mes pasado que sigan disponibles; los que ya no cotizan se reemplazan (rotación forzada, la misma
    que sufre Q1); y después se cambian al azar tantos nombres como cambió Q1, hasta completar `tamano`. Así la ÚNICA diferencia entre las
    dos carteras es de dónde sale el nombre nuevo, que es justo lo que se quiere medir."""
    disponibles = list(dict.fromkeys(disponibles))
    tamano = min(int(tamano), len(disponibles))
    if tamano <= 0: return []
    vivos = [t for t in (anterior or []) if t in set(disponibles)]
    fuera = [t for t in disponibles if t not in set(vivos)]
    azar.shuffle(fuera)
    # los que Q1 cambió: se sueltan al azar de los que sobrevivieron, y el resto del hueco se llena igual
    sueltos = min(int(cambios), len(vivos))
    if sueltos:
        quita = set(azar.choice(np.array(vivos, dtype=object), size=sueltos, replace=False).tolist())
        vivos = [t for t in vivos if t not in quita]
    return (vivos[:tamano] + fuera)[:tamano] if len(vivos) >= tamano else vivos + fuera[:tamano - len(vivos)]


def nombres_cartera(mecanismo, histeresis, fecha, v, q, previos):
    """Qué nombres tiene la cartera este mes (REGLA_ROTACION.md, escrita antes de correr las variantes). El mecanismo gobierna SOLO la
    cartera topada, que es la que juzga la compuerta; los cuantiles crudos siguen siendo los del corte de cada mes, como diagnóstico.

    - `mensual`: el corte del cuantil, cada mes. Es lo que se hacía y la línea de comparación.
    - `trimestral`: el corte solo en enero, abril, julio y octubre; en medio se conserva lo que había (los que dejan de cotizar salen).
    - `histeresis`: ENTRA quien está en el cuantil superior y SALE solo al caer bajo el percentil `histeresis` (0.5). Entre las dos
      fronteras se conserva lo que ya se tenía: distingue al que se deterioró de verdad del que ronda el corte y lo cruza por ruido."""
    arriba = list(v['ticker'][q == 1])
    if mecanismo == 'mensual' or not previos: return arriba
    vivos = set(v['ticker'])
    if mecanismo == 'trimestral':
        if fecha.month in (1, 4, 7, 10): return arriba
        quedan = [t for t in previos if t in vivos]
        return quedan or arriba
    if mecanismo == 'histeresis':
        pct = pd.Series(pd.Series(v['puntaje'].to_numpy()).rank(ascending=False, method='first', pct=True).to_numpy(),
                        index=v['ticker'].to_numpy())
        quedan = [t for t in previos if t in pct.index and pct[t] <= histeresis]
        return list(dict.fromkeys(quedan + arriba))
    raise ValueError(f'mecanismo desconocido: {mecanismo}')


def quintiles(puntaje, cuantiles=5):
    """Q1 = mejor puntaje. Empates se rompen por orden de aparición (el dataset va ordenado por ticker). Con cuantiles=3 son terciles."""
    pct = pd.Series(puntaje).rank(ascending=False, method='first', pct=True).to_numpy()
    return np.clip(np.ceil(pct * cuantiles), 1, cuantiles).astype(int)


# ================================================================ evaluación
def evaluar_meses(d, k, cuantiles=5, inclinacion_beta=False, beta='beta_ipc_1a', libro=None, factor_costos=1.0, sic=DIFERENCIAL_SIC,
                  mecanismo='mensual', histeresis=0.5, tope_sic=False, elegibles=None):
    # inclinacion_beta: False; 'pesos' (dentro de Q1, ∝ 1/beta antes del tope); 'seleccion' (Q1 solo entre las de beta bajo la mediana del día)
    """d: filas de fechas de rebalanceo con fecha, ticker, puntaje, y_21, y_63. Devuelve una fila por mes. Con cuantiles=3, Q1 es el tercil
    superior y las columnas q1q5_* comparan el mejor grupo contra el peor (con terciles, Q1 − Q3)."""
    filas, prev = [], {'q1': None, 'q5': None, 'uni': None, 'punt': None, 'q1t': None, 'ctrl': None, 'ctrlt': None, 'nombres': None}
    azar = np.random.default_rng(SEMILLA)
    for fecha, g in d.groupby('fecha', sort=True):
        # `elegibles` se separa de `tope_sic` a propósito (pedido de Fable, 2026-09-18): con las dos juntas no se puede saber cuánto del
        # cambio viene de recortar el universo y cuánto de cobrar la liquidez real. Por omisión sigue a `tope_sic`.
        if (tope_sic if elegibles is None else elegibles): g = g[elegibles_sic(g, True)]
        con = g.dropna(subset=['puntaje'])
        fila = {'fecha': str(fecha.date()), 'con_puntaje': int(len(con)), 'en_fecha': int(len(g))}
        for h in HORIZONTES:
            v = con.dropna(subset=[f'y_{h}'])
            fila[f'n_{h}'] = int(len(v))
            fila[f'ic_{h}'] = spearman(v['puntaje'], v[f'y_{h}']) if len(v) >= MIN_EMPRESAS else np.nan
            fila[f'sin_desenlace_{h}'] = int(len(con) - len(v))
        v = con.dropna(subset=['y_21'])
        if len(v) >= MIN_EMPRESAS:
            universo = v
            if inclinacion_beta == 'seleccion' and beta in v and v[beta].notna().sum() >= MIN_EMPRESAS:
                v = v[v[beta] <= v[beta].median()]          # los cuantiles se forman solo entre las de beta baja; el universo de comparación no cambia
            q = quintiles(v['puntaje'], cuantiles)
            for i in range(1, cuantiles + 1): fila[f'q{i}_21'] = float(v['y_21'].to_numpy()[q == i].mean())
            fila['cuantiles'] = cuantiles
            s1, s5 = set(v['ticker'][q == 1]), set(v['ticker'][q == cuantiles])
            su = set(universo['ticker'])
            fila['rot_q1'], fila['rot_q5'], fila['rot_universo'] = rotacion(prev['q1'], s1), rotacion(prev['q5'], s5), rotacion(prev['uni'], su)
            fila['universo_21'] = float(universo['y_21'].mean())
            fila['q1q5_bruto'] = fila['q1_21'] - fila[f'q{cuantiles}_21']
            # costo por peso operado de CADA cartera: el de sus propias emisoras, no uno igual para todas (pedidos 99 y 184)
            cu, cuu = costo_unitario(v, k, sic), costo_unitario(universo, k, sic)
            fila['costo_q1'] = factor_costos * float(cu.to_numpy()[q == 1].mean())
            fila['costo_q5'] = factor_costos * float(cu.to_numpy()[q == cuantiles].mean())
            fila['costo_universo'] = factor_costos * float(cuu.mean())
            fila['q1q5_neto'] = fila['q1q5_bruto'] - (fila['costo_q1'] * fila['rot_q1'] + fila['costo_q5'] * fila['rot_q5'])
            fila['q1_vs_universo_neto'] = ((fila['q1_21'] - fila['costo_q1'] * fila['rot_q1'])
                                           - (fila['universo_21'] - fila['costo_universo'] * fila['rot_universo']))
            if True:   # Q1 como la propondría la app: topado por importe, lo que no cabe queda en efectivo (sin importe conocido: pesos iguales)
                nombres = nombres_cartera(mecanismo, histeresis, fecha, v, q, prev['nombres'])
                sel = v[v['ticker'].isin(nombres)]
                en_sel = v['ticker'].isin(nombres).to_numpy()
                fila['nombres_cartera'] = int(len(sel)); fila['nombres_en_cuantil'] = int((q == 1).sum())
                # inclinación a beta baja (tanda 6, M31): pesos ∝ 1/beta dentro de Q1 antes del tope. Es una regla de construcción, no un modelo
                obj = 1.0 / np.clip(sel[beta].to_numpy(float), 0.3, 3.0) if (inclinacion_beta == 'pesos' and beta in sel) else None
                w = pesos_topados(importe_para_topar(sel, tope_sic), objetivo=obj)
                pesos = {t: x for t, x in zip(sel['ticker'], w) if x > 0}
                fila['q1_topado_21'] = float(np.dot(w, sel['y_21'].to_numpy(float)))
                fila['invertido_q1_topado'] = float(w.sum())
                fila['rot_q1_topado'] = rotacion_pesos(prev['q1t'], pesos); prev['q1t'] = pesos
                invertido = w.sum()
                fila['costo_q1_topado'] = factor_costos * (float(np.dot(w, cu.to_numpy()[en_sel]) / invertido) if invertido > 0 else 0.0)
                fila['q1_topado_vs_universo_neto'] = ((fila['q1_topado_21'] - fila['costo_q1_topado'] * fila['rot_q1_topado'])
                                                      - (fila['universo_21'] - fila['costo_universo'] * fila['rot_universo']))
                # control al azar con la MISMA rotación que Q1 (pedido 193): misma construcción, mismo tope, mismos costos
                # la rotación que se iguala es la de la cartera de verdad, la que produce el mecanismo, no la del corte crudo del cuantil
                nuevos_q1 = len(set(nombres) - set(prev['nombres'] or [])) if prev['nombres'] else len(nombres)
                ctrl = control_igual_rotacion(prev['ctrl'], list(universo['ticker']), len(sel), nuevos_q1, azar)
                if ctrl:
                    cs = universo[universo['ticker'].isin(ctrl)]
                    wc = pesos_topados(importe_para_topar(cs, tope_sic))
                    pesos_c = {t: x for t, x in zip(cs['ticker'], wc) if x > 0}
                    fila['control_topado_21'] = float(np.dot(wc, cs['y_21'].to_numpy(float)))
                    fila['rot_control'] = rotacion_pesos(prev['ctrlt'], pesos_c); prev['ctrlt'] = pesos_c
                    inv_c = wc.sum()
                    cuc = costo_unitario(cs, k, sic).to_numpy()
                    fila['costo_control'] = factor_costos * (float(np.dot(wc, cuc) / inv_c) if inv_c > 0 else 0.0)
                    fila['q1_topado_vs_control_neto'] = ((fila['q1_topado_21'] - fila['costo_q1_topado'] * fila['rot_q1_topado'])
                                                         - (fila['control_topado_21'] - fila['costo_control'] * fila['rot_control']))
                    fila['q1_topado_vs_control_bruto'] = fila['q1_topado_21'] - fila['control_topado_21']
                prev['ctrl'] = ctrl; prev['nombres'] = nombres
            # segunda referencia, la invertible: índice del mercado propio + dividendo estimado, sin costo de rotación (un fondo no rebalancea cada mes)
            if 'ret_mercado_21' in universo:
                ind = universo['ret_mercado_21'].dropna()
                if len(ind):
                    fila['indice_21'] = float(ind.median()) + dividendo_mensual(libro)
                    fila['q1_topado_vs_indice_neto'] = (fila['q1_topado_21'] - fila['costo_q1_topado'] * fila['rot_q1_topado']) - fila['indice_21']
            prev.update(q1=s1, q5=s5, uni=su)
        if prev['punt'] is not None:
            comun = con.merge(prev['punt'], on='ticker', suffixes=('', '_ant'))
            fila['estabilidad'] = spearman(comun['puntaje'], comun['puntaje_ant']) if len(comun) >= MIN_EMPRESAS else np.nan
        prev['punt'] = con[['ticker', 'puntaje']]
        filas.append(fila)
    return pd.DataFrame(filas)


# Referencia INVERTIBLE (Fable §F2.1, tanda 8 M40): lo que el usuario sí podría haber tenido, un fondo que replica el índice.
# `ret_mercado_21` del dataset es el índice de PRECIOS del mercado propio (IPC para México, S&P 500 en pesos fuera), así que se
# le suma un rendimiento por dividendo estimado: sin ese ajuste el margen quedaría inflado ~0.17 pp/mes (42 % del margen).
# La cifra se declara en el reporte, al lado del resultado. El rendimiento total real llegará con el proveedor con licencia.
DIVIDENDO_INDICE = {'MX': 0.025, 'extranjero': 0.014}   # anual, estimado; IPC y S&P 500


def dividendo_mensual(libro):
    return (1 + DIVIDENDO_INDICE.get(libro, 0.0)) ** (1 / 12) - 1


LIBROS = {'MX': (lambda d: d[d['pais'] == 'MX'], 3), 'extranjero': (lambda d: d[d['pais'] != 'MX'], 5)}   # pedido 161: terciles en México (45 emisoras), quintiles fuera
# Pesos CONGELADOS del 2026-09-16 (tanda 6, B1): riesgo activo igual medido en la ventana de selección 2012-01 → 2020-11 (error de seguimiento
# mensual MX 0.017866, extranjero 0.025258). NO se recalculan: si se midieran en el pliegue externo, la construcción usaría los datos que sirven
# para verificarla y el pliegue dejaría de serlo.
PESOS_LIBRO = {'MX': 0.585705, 'extranjero': 0.414295}


def meses_por_libro(d, k, libros=None, pesos=PESOS_LIBRO, inclinacion_beta=False, factor_costos=1.0, sic=DIFERENCIAL_SIC, tope_sic=False, elegibles=None,
                    mecanismo='mensual', histeresis=0.5):
    """Pedido 161 (tanda 5, M27): Q1 dentro de cada libro —terciles en México, quintiles en extranjero— y las dos carteras combinadas con
    RIESGO ACTIVO IGUAL: w ∝ 1/error de seguimiento del libro (Merton, 1980: la varianza se estima bien y la media no). `pesos` llega
    congelado (PESOS_LIBRO); con pesos=None se recalculan con los datos recibidos, que solo vale para medirlos una vez en la ventana de
    selección: recalcularlos en el pliegue externo lo invalidaría. Devuelve (combinado, por_libro, info)."""
    libros = libros or LIBROS
    por_libro, te = {}, {}
    for nombre, (filtro, cuantiles) in libros.items():
        m = evaluar_meses(filtro(d), k, cuantiles, inclinacion_beta, libro=nombre, factor_costos=factor_costos, sic=sic,
                          mecanismo=mecanismo, histeresis=histeresis, tope_sic=tope_sic, elegibles=elegibles)
        if not len(m) or 'q1_topado_vs_universo_neto' not in m: continue
        por_libro[nombre] = m.set_index('fecha')
        s = m['q1_topado_vs_universo_neto'].dropna()
        te[nombre] = float(s.std(ddof=1)) if len(s) > 2 and s.std(ddof=1) > 0 else None
    if not por_libro: return pd.DataFrame(), {}, {}
    if pesos:
        falta = [b for b in por_libro if b not in pesos]
        if falta: raise ValueError(f'faltan pesos congelados para {falta}')
        usar = {b: float(pesos[b]) for b in por_libro}
        total = sum(usar.values()); usar = {b: v / total for b, v in usar.items()}
        congelados = True
    else:
        inv = {b: (1 / te[b] if te.get(b) else 0.0) for b in por_libro}
        total = sum(inv.values())
        usar = {b: (v / total if total else 1 / len(por_libro)) for b, v in inv.items()}
        congelados = False
    pesos = usar
    fechas = sorted(set().union(*[set(m.index) for m in por_libro.values()]))
    suma = ['con_puntaje', 'en_fecha', 'n_21', 'n_63', 'sin_desenlace_21', 'sin_desenlace_63']
    ponderadas = ['ic_21', 'ic_63', 'q1_21', 'universo_21', 'q1q5_bruto', 'q1q5_neto', 'q1_vs_universo_neto', 'q1_topado_21', 'invertido_q1_topado',
                  'rot_q1', 'rot_q5', 'rot_universo', 'rot_q1_topado', 'nombres_cartera', 'nombres_en_cuantil', 'q1_topado_vs_universo_neto', 'estabilidad', 'indice_21', 'q1_topado_vs_indice_neto',
                  'costo_q1', 'costo_q5', 'costo_universo', 'costo_q1_topado',
                  'control_topado_21', 'rot_control', 'costo_control', 'q1_topado_vs_control_neto', 'q1_topado_vs_control_bruto']
    filas = []
    for f in fechas:
        fila = {'fecha': f}
        for c in suma:
            fila[c] = int(sum(int(m.at[f, c]) for m in por_libro.values() if f in m.index and c in m and pd.notna(m.at[f, c])))
        for c in ponderadas:
            partes = {b: m.at[f, c] for b, m in por_libro.items() if f in m.index and c in m and pd.notna(m.at[f, c])}
            w = sum(pesos[b] for b in partes)
            fila[c] = float(sum(pesos[b] * partes[b] for b in partes) / w) if w > 0 else np.nan
        filas.append(fila)
    return pd.DataFrame(filas), {b: m.reset_index() for b, m in por_libro.items()}, {'pesos': pesos, 'pesos_congelados': congelados, 'error_seguimiento_mensual_medido': te}


def bootstrap_bloques(x, bloque, remuestreos, semilla=SEMILLA):
    """IC 90 % de la media con bootstrap de bloques móviles (circular) de `bloque` meses."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < bloque * 2: return None
    rng = np.random.default_rng(semilla); k = int(np.ceil(n / bloque)); medias = np.empty(remuestreos)
    for b in range(remuestreos):
        inicios = rng.integers(0, n, size=k)
        idx = (inicios[:, None] + np.arange(bloque)[None, :]).ravel()[:n] % n
        medias[b] = x[idx].mean()
    return [float(np.percentile(medias, 5)), float(np.percentile(medias, 95))]


def sharpe_mensual(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std(ddof=1) > 0 else None


def sharpe_desinflado(x, sharpes_intentos, n_intentos=None, sd_intentos=None):
    """Bailey y López de Prado (2014, JPM). SR0 = máximo esperado de N Sharpe nulos con la varianza observada entre intentos;
    DSR = P(SR verdadero > SR0) con corrección por asimetría y curtosis. Todo en unidades mensuales; también en anual (× √12).
    Para el pliegue externo, `n_intentos` y `sd_intentos` permiten fijar N y la dispersión de antemano (regla pre-comprometida: dos
    candidatos y la dispersión medida en la ventana de selección, porque estimarla con dos observaciones es inestable)."""
    from statistics import NormalDist
    x = np.asarray(x, float); x = x[~np.isnan(x)]; T = len(x); sr = sharpe_mensual(x)
    if sr is None or T < 12: return None
    intentos = [v for v in sharpes_intentos if v is not None and np.isfinite(v)]
    N = int(n_intentos) if n_intentos else len(intentos)
    sd = float(sd_intentos) if sd_intentos is not None else (float(np.std(intentos, ddof=1)) if len(intentos) >= 2 else None)
    if N >= 2 and sd:
        g = 0.5772156649; z = NormalDist().inv_cdf
        sr0 = float(sd * ((1 - g) * z(1 - 1 / N) + g * z(1 - 1 / (N * np.e))))
    else:
        sr0 = 0.0
    e = x - x.mean(); m2 = (e ** 2).mean()
    asim = float((e ** 3).mean() / m2 ** 1.5); curt = float((e ** 4).mean() / m2 ** 2)
    den = 1 - asim * sr + (curt - 1) / 4 * sr ** 2
    dsr = float(NormalDist().cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(den))) if den > 0 else None
    return {'sharpe_mensual': sr, 'sharpe_anual': sr * np.sqrt(12), 'intentos': N, 'sd_intentos': sd, 'sharpe0_anual': sr0 * np.sqrt(12),
            'sharpe_desinflado_anual': (sr - sr0) * np.sqrt(12), 'probabilidad_dsr': dsr, 'asimetria': asim, 'curtosis': curt}


def peor_anio_relativo(m, k, meses_min, cartera='q1'):
    """Q1 (igual ponderado o topado) neto contra el universo neto, compuesto por año calendario con al menos `meses_min` meses. (peor, año, todos)."""
    col, rot = ('q1_21', 'rot_q1') if cartera == 'q1' else ('q1_topado_21', 'rot_q1_topado')
    if col not in m: return None, None, {}
    d = m.dropna(subset=[col, 'universo_21']).assign(anio=lambda x: x['fecha'].str[:4])
    # el costo por peso operado de cada mes viene en la tabla (incluye diferencial y conversión); k es el respaldo si la tabla es vieja
    cq = d[f'costo_{cartera if cartera == "q1" else "q1_topado"}'] if f'costo_{cartera if cartera == "q1" else "q1_topado"}' in d else k
    cun = d['costo_universo'] if 'costo_universo' in d else k
    d = d.assign(q1n=d[col] - cq * d[rot], un=d['universo_21'] - cun * d['rot_universo'])
    anual = {a: float(np.prod(1 + g['q1n']) / np.prod(1 + g['un']) - 1) for a, g in d.groupby('anio') if len(g) >= meses_min}
    if not anual: return None, None, {}
    peor = min(anual, key=anual.get)
    return anual[peor], peor, anual


def leer_intentos(carpeta, familia=FAMILIA_RANKING):
    """Último renglón por candidato de la familia (una recorrida por un error no suma intentos, pedido 115). Un candidato marcado como
    corrección de procedimiento de otro (campo `corrige`, justificado por un diagnóstico y no por el desempeño) lo sustituye: no suma intento."""
    ruta = os.path.join(carpeta, REGISTRO_INTENTOS); ultimos = {}
    if os.path.exists(ruta):
        with open(ruta, encoding='utf-8') as fh:
            for linea in fh:
                if linea.strip():
                    e = json.loads(linea)
                    if e.get('familia') == familia: ultimos[e['nombre']] = e
    for e in list(ultimos.values()):
        if e.get('corrige'): ultimos.pop(e['corrige'], None)
    return ultimos


def registrar_intento(carpeta, entrada):
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, REGISTRO_INTENTOS), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(limpiar_nan(entrada), ensure_ascii=False) + '\n')


def resumir(m):
    def media(c): return float(m[c].mean()) if c in m and m[c].notna().any() else None
    def pct_pos(c): return float((m[c].dropna() > 0).mean()) if c in m and m[c].notna().any() else None
    r = {}
    for h in HORIZONTES:
        ic = m[f'ic_{h}'].dropna() if f'ic_{h}' in m else pd.Series(dtype=float)
        r[f'ic_{h}'] = {'meses': int(len(ic)), 'medio': float(ic.mean()) if len(ic) else None, 't_newey_west': t_newey_west(ic, h // 21 - 1),
                        'rezagos_nw': h // 21 - 1, 'pct_meses_positivo': float((ic > 0).mean()) if len(ic) else None}
    r['quintiles_21'] = {f'q{i}_medio': media(f'q{i}_21') for i in range(1, 6)}
    r['quintiles_21'].update({'universo_medio': media('universo_21'), 'q1q5_bruto_medio': media('q1q5_bruto'), 'q1q5_neto_medio': media('q1q5_neto'),
                              'pct_meses_q1q5_neto_positivo': pct_pos('q1q5_neto'), 'q1_vs_universo_neto_medio': media('q1_vs_universo_neto'),
                              'pct_meses_q1_supera_universo_neto': pct_pos('q1_vs_universo_neto'), 'rotacion_q1_media': media('rot_q1'),
                              'rotacion_q5_media': media('rot_q5'), 'q1_topado_vs_universo_neto_medio': media('q1_topado_vs_universo_neto'),
                              'invertido_q1_topado_medio': media('invertido_q1_topado'), 'rotacion_q1_topado_media': media('rot_q1_topado'),
                              'nombres_cartera_medio': media('nombres_cartera'), 'nombres_en_cuantil_medio': media('nombres_en_cuantil')})
    r['quintiles_21'].update({'indice_medio': media('indice_21'), 'q1_topado_vs_indice_neto_medio': media('q1_topado_vs_indice_neto'),
                              'pct_meses_q1_supera_indice': pct_pos('q1_topado_vs_indice_neto')})
    # control al azar con la misma rotación que Q1 (pedido 193): la señal libre de referencia y de costo de construcción
    r['quintiles_21'].update({'control_medio': media('control_topado_21'), 'rotacion_control_media': media('rot_control'),
                              'q1_topado_vs_control_neto_medio': media('q1_topado_vs_control_neto'),
                              'q1_topado_vs_control_bruto_medio': media('q1_topado_vs_control_bruto'),
                              'pct_meses_q1_supera_control': pct_pos('q1_topado_vs_control_neto')})
    for col, nombre in (('q1_topado_vs_universo_neto', 'sharpe_anual_margen'), ('q1q5_neto', 'sharpe_anual_q1q5'),
                        ('q1_topado_vs_indice_neto', 'sharpe_anual_margen_indice'), ('q1_topado_vs_control_neto', 'sharpe_anual_senal_vs_control')):
        sr = sharpe_mensual(m[col]) if col in m else None
        r[nombre] = float(sr * np.sqrt(12)) if sr is not None else None
    r['estabilidad_media'] = media('estabilidad')
    r['cobertura_media'] = float((m['con_puntaje'] / m['en_fecha']).mean()) if len(m) else None
    r['sin_desenlace_21_total'] = int(m['sin_desenlace_21'].iloc[:-2].sum()) if len(m) > 2 else 0   # los 2 últimos meses aún no tienen desenlace
    return r


def por_anio(m):
    if not len(m): return []
    a = m.assign(anio=m['fecha'].str[:4]).groupby('anio')
    salida = []
    for anio, g in a:
        salida.append({'anio': anio, 'meses': int(g['ic_21'].notna().sum()), 'ic_21': float(g['ic_21'].mean()) if g['ic_21'].notna().any() else None,
                       'ic_63': float(g['ic_63'].mean()) if g['ic_63'].notna().any() else None,
                       'q1q5_neto': float(g['q1q5_neto'].mean()) if 'q1q5_neto' in g and g['q1q5_neto'].notna().any() else None})
    return salida


# Compuerta de TRES valores (pedido AO de Fable, 2026-09-17). Con la potencia de 0.40 que se midió en la condición que decide, un intervalo
# que incluye el cero **no es evidencia de que la señal no se cobra: es evidencia de que la muestra no alcanza para saberlo**. Llamar
# «falla» a eso afirma más de lo que los datos permiten.
#
# El criterio NO se relaja. Bajar el umbral para subir la potencia sería ajustar la compuerta al candidato, que es exactamente lo que la
# compuerta existe para impedir; la potencia es propiedad del tamaño de muestra y del efecto, y el camino para subirla son más datos, no
# barras más bajas. Lo único que cambia es **qué significa un «no»**, y todo veredicto —«sí» o «no»— lleva su potencia al lado.
POTENCIA_MIN = 0.80


def veredicto(iv, potencia=None, cumple_lo_demas=True):
    """'pasa', 'falla' o 'indeterminado'. Indeterminado = el intervalo incluye el cero y la potencia no alcanza para concluir nada."""
    if iv is None or iv[0] is None or iv[1] is None: return 'indeterminado'
    if iv[0] > 0: return 'pasa' if cumple_lo_demas else 'falla'
    if iv[1] < 0: return 'falla'
    if potencia is not None and potencia >= POTENCIA_MIN: return 'falla'
    return 'indeterminado'


def criterio(r, extra, referencia=None, es_referencia=False):
    ic = r['ic_21']; c = CRITERIO
    iv = extra.get('ic90_q1_topado_vs_universo_neto')
    # si la corrida trae el DSR pre-comprometido (pliegue externo, N fijado antes de abrir), ése decide
    dsr = ((extra.get('desinflado_q1_topado_precomprometido') or extra.get('desinflado_q1_topado_n1')) or {}).get('probabilidad_dsr')
    pa = extra.get('peor_anio_topado')
    # condición 6: vencer el Sharpe anual del margen de la referencia (decisión del dueño, 2026-09-16)
    vence = None if es_referencia else (False if referencia is None else
            (r.get('sharpe_anual_margen') if r.get('sharpe_anual_margen') is not None else -99) > (referencia.get('sharpe_anual_margen') or 0))
    checks = {'ic_medio': ic['medio'] is not None and ic['medio'] > c['ic_medio_min'], 't': ic['t_newey_west'] is not None and ic['t_newey_west'] >= c['t_min'],
              'meses': ic['meses'] >= c['meses_min'], 'ic90_excluye_cero': iv is not None and iv[0] > 0, 'dsr': dsr is not None and dsr >= c['dsr_min'],
              'peor_anio': pa is not None and pa >= c['peor_anio_relativo_min'], 'vence_referencia': vence}
    cumple = all(v for v in checks.values() if v is not None)
    return {'cumple': cumple, 'veredicto': veredicto(iv, extra.get('potencia'), cumple), 'potencia': extra.get('potencia'),
            'detalle': checks, 'regla': c}


def cobertura_salidas(tabla, universo_csv, fechas):
    """Pedido 72: emisoras que salieron de bolsa con historia UTILIZABLE (puntaje evaluable: y_21 no nulo en una fecha de rebalanceo), por año."""
    if not os.path.exists(universo_csv): return {'nota': 'sin universo.csv'}
    u = pd.read_csv(universo_csv, dtype=str)
    salidas = sorted(u.loc[u['salida'].fillna('').str.strip() != '', 'ticker'])
    d = tabla[tabla['fecha'].isin(fechas) & tabla['ticker'].isin(salidas) & tabla['y_21'].notna()]
    anios = sorted({f.year for f in fechas})
    por = [{'anio': str(a), 'con_historia': sorted(d.loc[d['fecha'].dt.year == a, 'ticker'].unique().tolist())} for a in anios]
    for p in por: p['n'] = len(p['con_historia'])
    con = sorted(d['ticker'].unique().tolist())
    return {'salidas_en_universo': len(salidas), 'con_historia_utilizable': con, 'sin_historia': [t for t in salidas if t not in con], 'por_anio': por,
            'conclusion': f'{len(con)} de {len(salidas)} emisoras que salieron de bolsa entran a la evaluación: el resultado conserva sesgo de supervivencia'}


def revisar_grupos(meta_cols, variables, ablacion):
    """Pedido 64: los analistas (grupo C) solo se usan en ablación; un modelo candidato a la app no puede tenerlos."""
    c = [v for v in variables if (meta_cols.get(v) or {}).get('grupo') == 'C']
    if c and not ablacion:
        raise ValueError(f'Variables de analistas {c}: solo se permiten con --ablacion (pedido 64)')
    return c


def evaluar(tabla, puntaje, nombre, variables, meta, descripcion='', ablacion=False, solo_app=False, universo_csv=None, costos=None,
            referencia=None, carpeta_intentos=None, tipo='candidato', corrige=None, por_libro=False, inclinacion_beta=False, dsr_n=None, dsr_sd=None,
            factor_costos=1.0, sic=DIFERENCIAL_SIC, mecanismo='mensual', histeresis=0.5, tope_sic=False, elegibles=None):
    meta_cols = meta.get('columnas') or {}
    analistas = revisar_grupos(meta_cols, variables, ablacion)
    costos = dict(costos or costos_app())
    costos.update({'factor': float(factor_costos), 'diferencial_sic': float(sic), 'mecanismo': mecanismo,
                   'histeresis': float(histeresis) if mecanismo == 'histeresis' else None,
                   'diferencial': COL_DIFERENCIAL if COL_DIFERENCIAL in tabla else 'no está en el dataset: solo comisión + IVA'})
    t = tabla.assign(puntaje=np.asarray(puntaje, dtype=float))
    if solo_app and 'en_app' in t: t = t[t['en_app'].astype(bool)]
    fechas = fechas_rebalanceo(t['fecha'])
    d = t[t['fecha'].isin(fechas)]
    subs, meses_todas, libros_info = {}, None, None
    if por_libro:
        combinado, por_libro_m, libros_info = meses_por_libro(d, costos['costo_por_peso_operado'], inclinacion_beta=inclinacion_beta,
                                                              factor_costos=factor_costos, sic=sic, mecanismo=mecanismo, histeresis=histeresis,
                                                              tope_sic=tope_sic, elegibles=elegibles)
        if len(combinado):
            meses_todas = combinado
            subs['todas'] = {'resumen': resumir(combinado), 'por_anio': por_anio(combinado)}
            for b, m in por_libro_m.items():
                subs[b] = {'resumen': resumir(m), 'por_anio': por_anio(m)}
    else:
        for s, filtro in SUBCONJUNTOS.items():
            m = evaluar_meses(filtro(d), costos['costo_por_peso_operado'], inclinacion_beta=inclinacion_beta,
                              factor_costos=factor_costos, sic=sic, mecanismo=mecanismo, histeresis=histeresis, tope_sic=tope_sic,
                              elegibles=elegibles)
            if not len(m): continue
            subs[s] = {'resumen': resumir(m), 'por_anio': por_anio(m)}
            if s == 'todas': meses_todas = m
    r = subs.get('todas', {}).get('resumen')
    extra, crit = {}, None
    if r is not None:
        k = costos['costo_por_peso_operado']
        extra['ic90_q1q5_neto'] = bootstrap_bloques(meses_todas['q1q5_neto'] if 'q1q5_neto' in meses_todas else [], CRITERIO['bloque_bootstrap_meses'], CRITERIO['remuestreos'])
        pa, pa_anio, anuales = peor_anio_relativo(meses_todas, k, CRITERIO['meses_min_por_anio'])
        extra.update({'peor_anio': pa, 'peor_anio_cual': pa_anio, 'relativo_por_anio': anuales})
        sr = sharpe_mensual(meses_todas['q1q5_neto']) if 'q1q5_neto' in meses_todas else None
        intentos = leer_intentos(carpeta_intentos) if carpeta_intentos else {}
        if tipo != 'control' and carpeta_intentos:
            st = sharpe_desinflado(meses_todas['q1_topado_vs_universo_neto'], [None]) if 'q1_topado_vs_universo_neto' in meses_todas else None
            sq = sharpe_desinflado(meses_todas['q1q5_neto'], [None]) if 'q1q5_neto' in meses_todas else None
            registrar_intento(carpeta_intentos, {'familia': FAMILIA_RANKING, 'nombre': nombre, 'tipo': tipo, 'corrige': corrige, 'fecha': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                                                 'sharpe_mensual_q1q5_neto': sr, 'sharpe_mensual_q1_topado': (st or {}).get('sharpe_mensual'),
                                                 'asimetria_q1q5': (sq or {}).get('asimetria'), 'curtosis_q1q5': (sq or {}).get('curtosis'),
                                                 'asimetria_q1_topado': (st or {}).get('asimetria'), 'curtosis_q1_topado': (st or {}).get('curtosis'),
                                                 'ic_21': r['ic_21']['medio'], 'dataset': meta.get('version'),
                                                 'codigo': commit_codigo()['commit'], 'universo': 'solo_app' if solo_app else 'todas'})
            intentos[nombre] = {'sharpe_mensual_q1q5_neto': sr, 'sharpe_mensual_q1_topado': (st or {}).get('sharpe_mensual')}
        extra['desinflado'] = sharpe_desinflado(meses_todas['q1q5_neto'] if 'q1q5_neto' in meses_todas else [],
                                                [e.get('sharpe_mensual_q1q5_neto') for e in intentos.values()] if intentos else [sr])
        if 'q1_topado_vs_universo_neto' in meses_todas:
            serie_t = meses_todas['q1_topado_vs_universo_neto']
            extra['ic90_q1_topado_vs_universo_neto'] = bootstrap_bloques(serie_t, CRITERIO['bloque_bootstrap_meses'], CRITERIO['remuestreos'])
            extra['desinflado_q1_topado_n1'] = sharpe_desinflado(serie_t, [None])
            if dsr_n:   # regla pre-comprometida del pliegue externo: N y dispersión fijados antes de abrir
                extra['desinflado_q1_topado_precomprometido'] = sharpe_desinflado(serie_t, [None], n_intentos=dsr_n, sd_intentos=dsr_sd)
            extra['desinflado_q1_topado_registro'] = sharpe_desinflado(serie_t, [e.get('sharpe_mensual_q1_topado') for e in intentos.values()] or [None])
            extra['peor_anio_topado'], extra['peor_anio_topado_cual'], extra['relativo_topado_por_anio'] = peor_anio_relativo(meses_todas, k, CRITERIO['meses_min_por_anio'], 'topado')
        if 'q1_topado_vs_indice_neto' in meses_todas:   # la referencia invertible, reportada al lado (tanda 8, M40)
            si = meses_todas['q1_topado_vs_indice_neto']
            extra['ic90_q1_topado_vs_indice_neto'] = bootstrap_bloques(si, CRITERIO['bloque_bootstrap_meses'], CRITERIO['remuestreos'])
            extra['desinflado_q1_topado_indice_n1'] = sharpe_desinflado(si, [None])
            extra['dividendo_indice_anual'] = DIVIDENDO_INDICE
        if 'q1_vs_universo_neto' in meses_todas:
            extra['ic90_q1_vs_universo_neto'] = bootstrap_bloques(meses_todas['q1_vs_universo_neto'], CRITERIO['bloque_bootstrap_meses'], CRITERIO['remuestreos'])
        crit = criterio(r, extra, referencia, es_referencia=(nombre == CRITERIO['referencia']))
    return {
        'version': VERSION, 'nombre': nombre, 'descripcion': descripcion, 'evaluado': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'variables': list(variables), 'ablacion': bool(ablacion), 'variables_analistas': analistas, 'universo': 'solo_app' if solo_app else 'todas',
        'dataset': {k: meta.get(k) for k in ('version', 'generado', 'almacen_commit', 'filas', 'empresas', 'fecha_min', 'fecha_max')},
        'codigo': commit_codigo(), 'semilla': SEMILLA, 'costos': costos, 'min_empresas_por_fecha': MIN_EMPRESAS,
        'periodo': {'desde': str(fechas.min().date()) if len(fechas) else None, 'hasta': str(fechas.max().date()) if len(fechas) else None, 'rebalanceos': int(len(fechas))},
        'compuerta_publicacion': {f'y_{h}': publicable.verificar(meta_cols, variables, f'y_{h}') for h in HORIZONTES},
        'candidato_a_app': bool(crit and crit['cumple'] and not ablacion and tipo != 'control'),
        'criterio': crit, 'criterio_extra': extra, 'subconjuntos': subs, 'por_libro': libros_info, 'inclinacion_beta': inclinacion_beta or None,
        'meses': [] if meses_todas is None else json.loads(meses_todas.round(6).to_json(orient='records')),
        'cobertura_salidas': cobertura_salidas(t, universo_csv or os.path.join(AQUI, 'universo.csv'), fechas),
        'limitaciones': LIMITACIONES,
    }


def recalcular_desinflado(resultados, carpeta_intentos, referencia):
    """Al final de una corrida: todos los candidatos se desinflan con el MISMO conjunto de intentos (el registro completo), no con el que
    existía cuando le tocó a cada uno."""
    intentos = leer_intentos(carpeta_intentos) if carpeta_intentos else {}
    sharpes = [e.get('sharpe_mensual_q1q5_neto') for e in intentos.values()]
    ref = next((x['subconjuntos']['todas']['resumen'] for x in resultados if x['nombre'] == CRITERIO['referencia'] and x.get('criterio')), referencia)
    for r in resultados:
        if not r.get('criterio'): continue
        serie = [mm.get('q1q5_neto') for mm in r['meses']]
        r['criterio_extra']['desinflado'] = sharpe_desinflado(np.array([np.nan if v is None else v for v in serie], float), sharpes or [None])
        resumen = r['subconjuntos']['todas']['resumen']
        serie_t = [mm.get('q1_topado_vs_universo_neto') for mm in r['meses']]
        if any(v is not None for v in serie_t):
            r['criterio_extra']['desinflado_q1_topado_registro'] = sharpe_desinflado(np.array([np.nan if v is None else v for v in serie_t], float),
                                                                                     [e.get('sharpe_mensual_q1_topado') for e in intentos.values()] or [None])
        r['criterio'] = criterio(resumen, r['criterio_extra'], ref, es_referencia=(r['nombre'] == CRITERIO['referencia']))
        r['candidato_a_app'] = bool(r['criterio']['cumple'] and not r['ablacion'] and r['nombre'] not in CONTROLES)
    return resultados


def puntaje_linea_base(tabla, nombre):
    variables, f, _ = LINEAS_BASE[nombre]
    if f is None:
        return np.random.default_rng(SEMILLA).standard_normal(len(tabla))
    return f(tabla).to_numpy(dtype=float)


# ================================================================ salida
def ruta_libre(carpeta, base, ext):
    ruta, i = os.path.join(carpeta, f'{base}{ext}'), 2
    while os.path.exists(ruta):
        ruta = os.path.join(carpeta, f'{base}_{i}{ext}'); i += 1
    return ruta


def _iv(iv, pct=False):
    return f'[{_f(iv[0], pct, 4)}, {_f(iv[1], pct, 4)}]' if iv else '—'


def _f(x, pct=False, dec=3):
    if x is None or (isinstance(x, float) and np.isnan(x)): return '—'
    return f'{x * 100:.{max(dec - 2, 1)}f} %' if pct else f'{x:.{dec}f}'


def reporte_md(resultados, titulo):
    r0 = resultados[0]
    ds, cod = r0['dataset'], r0['codigo']
    L = [f'# {titulo}', '',
         f"Evaluado {r0['evaluado']} · {VERSION} · código {cod['commit']}{' (con cambios sin commit)' if cod['laboratorio_con_cambios_sin_commit'] else ''} · "
         f"{ds.get('version')} (almacén {ds.get('almacen_commit')}, generado {ds.get('generado')}) · universo {r0['universo']} · "
         f"{r0['periodo']['rebalanceos']} rebalanceos mensuales {r0['periodo']['desde']} → {r0['periodo']['hasta']}", '',
         f"Costos por peso operado: {_f(r0['costos']['comision'], True, 4)} de comisión + IVA {_f(r0['costos']['iva'], True, 2)} ({r0['costos']['fuente']}) "
         f"+ el medio diferencial de compra-venta de cada emisora ({r0['costos'].get('diferencial')}, Abdi-Ranaldo 2017) "
         f"+ {_f(r0['costos'].get('diferencial_sic', 0), True, 3)} de diferencial en las extranjeras, que SUSTITUYE al de su bolsa de origen (supuesto por medir: el papel se compra en el SIC, no en Nueva York)"
         f"{', TODO multiplicado por ' + str(r0['costos'].get('factor')) if r0['costos'].get('factor', 1) != 1 else ''}. "
         f"Criterio para la app (§2.6, decidido por el dueño el 2026-09-15): se juzga {CRITERIO['cartera']}. IC 21 d > {CRITERIO['ic_medio_min']} con t ≥ {CRITERIO['t_min']} "
         f"en ≥ {CRITERIO['meses_min']} meses; IC 90 % (bloques de {CRITERIO['bloque_bootstrap_meses']} meses) de Q1 topado − universo sobre cero; DSR ≥ {CRITERIO['dsr_min']} "
         f"con N = {CRITERIO['dsr_intentos']} en el periodo no usado para elegir; peor año ≥ {CRITERIO['peor_anio_relativo_min']:.0%}; vencer el Sharpe del margen de {CRITERIO['referencia']}. "
         "Tope por importe: 1 % del importe diario medio de 20 días sobre una cartera de un millón de pesos.", '',
         '## Criterio punto por punto', '',
         '| Modelo | IC > 0.03 y t ≥ 3 | ≥ 60 meses | Q1 topado − universo neto/mes, IC 90 % | Invertido | DSR N=1 (y con los intentos del registro) | Peor año | Sharpe del margen | Vence el Sharpe | Cumple |',
         '|---|---|---|---|---|---|---|---|---|---|'] + [
         (lambda r, e, c: f"| {r['nombre']} | {'✔' if c['detalle']['ic_medio'] and c['detalle']['t'] else '✘'} | {'✔' if c['detalle']['meses'] else '✘'} | "
          f"{'✔' if c['detalle']['ic90_excluye_cero'] else '✘'} {_f(r['subconjuntos']['todas']['resumen']['quintiles_21'].get('q1_topado_vs_universo_neto_medio'), True, 4)} "
          f"[{_f((e.get('ic90_q1_topado_vs_universo_neto') or [None, None])[0], True, 4)}, {_f((e.get('ic90_q1_topado_vs_universo_neto') or [None, None])[1], True, 4)}] | "
          f"{_f(r['subconjuntos']['todas']['resumen']['quintiles_21'].get('invertido_q1_topado_medio'), True)} | "
          f"{'✔' if c['detalle']['dsr'] else '✘'} {_f((e.get('desinflado_q1_topado_n1') or {}).get('probabilidad_dsr'), dec=3)} ({_f((e.get('desinflado_q1_topado_registro') or {}).get('probabilidad_dsr'), dec=3)}) | "
          f"{'✔' if c['detalle']['peor_anio'] else '✘'} {_f(e.get('peor_anio_topado'), True)} ({e.get('peor_anio_topado_cual')}) | "
          f"{_f(r['subconjuntos']['todas']['resumen'].get('sharpe_anual_margen'), dec=2)} | "
          f"{'—' if c['detalle']['vence_referencia'] is None else ('✔' if c['detalle']['vence_referencia'] else '✘')} | {'**sí**' if c['cumple'] else 'no'} |")(
             r, r.get('criterio_extra') or {}, r['criterio']) for r in resultados if r.get('criterio')] + ['',
         '## Diagnóstico de monotonía: Q1 − Q5 (no decide)', '',
         '| Modelo | Q1−Q5 neto/mes, IC 90 % | Sharpe anual | DSR con los intentos del registro | Asimetría | Curtosis | Q1 igual ponderado: peor año |', '|---|---|---|---|---|---|---|'] + [
         (lambda r, e: f"| {r['nombre']} | {_f(r['subconjuntos']['todas']['resumen']['quintiles_21'].get('q1q5_neto_medio'), True, 4)} "
          f"[{_f((e.get('ic90_q1q5_neto') or [None, None])[0], True, 4)}, {_f((e.get('ic90_q1q5_neto') or [None, None])[1], True, 4)}] | {_f((e.get('desinflado') or {}).get('sharpe_anual'), dec=2)} | "
          f"{_f((e.get('desinflado') or {}).get('probabilidad_dsr'), dec=3)} | {_f((e.get('desinflado') or {}).get('asimetria'), dec=2)} | {_f((e.get('desinflado') or {}).get('curtosis'), dec=2)} | "
          f"{_f(e.get('peor_anio'), True)} ({e.get('peor_anio_cual')}) |")(r, r.get('criterio_extra') or {}) for r in resultados if r.get('criterio')] + ['',
         '## Resumen (todas las emisoras)', '',
         '| Modelo | IC 21 d | t (NW) | Meses IC>0 | IC 63 d | t (NW, 2 rez.) | Q1−Q5 neto/mes | Meses Q1−Q5>0 | Q1 − universo neto/mes | Rotación Q1 | Estabilidad | Candidato a la app | Publicable |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in resultados:
        s = r['subconjuntos'].get('todas', {}).get('resumen')
        if not s: L.append(f"| {r['nombre']} | sin datos |||||||||||| "); continue
        q = s['quintiles_21']; pub = r['compuerta_publicacion']['y_21']
        L.append(f"| {r['nombre']} | {_f(s['ic_21']['medio'])} | {_f(s['ic_21']['t_newey_west'], dec=2)} | {_f(s['ic_21']['pct_meses_positivo'], True)} | "
                 f"{_f(s['ic_63']['medio'])} | {_f(s['ic_63']['t_newey_west'], dec=2)} | {_f(q['q1q5_neto_medio'], True, 4)} | {_f(q['pct_meses_q1q5_neto_positivo'], True)} | "
                 f"{_f(q['q1_vs_universo_neto_medio'], True, 4)} | {_f(q['rotacion_q1_media'], True)} | {_f(s['estabilidad_media'], dec=2)} | "
                 f"{'sí' if r['candidato_a_app'] else 'no'} | {'sí' if pub['publicable'] else 'no'} |")
    # las claves de subconjunto dependen de la corrida (SUBCONJUNTOS o los libros): se recorren las presentes, no una lista fija (tanda 6, V4)
    claves = list(dict.fromkeys([c for r in resultados for c in r['subconjuntos']]))
    L += ['', '## Margen contra la referencia invertible (índice del mercado propio + dividendo estimado)', '',
          f"Segunda referencia, la que el usuario sí podría haber tenido (tanda 8, M40). El índice del dataset es de PRECIOS, así que se le suma un "
          f"dividendo estimado de {DIVIDENDO_INDICE['MX']:.1%} anual en México y {DIVIDENDO_INDICE['extranjero']:.1%} fuera; **sin ese ajuste el margen "
          f"aparecería ~0.17 pp/mes más alto**. La referencia anterior (universo igual ponderado) se conserva arriba para comparar con lo medido antes.", '',
          '| Modelo | Q1 topado − índice neto/mes | IC 90 % | Sharpe | Meses que supera al índice |', '|---|---|---|---|---|']
    for r in resultados:
        q = r['subconjuntos'].get('todas', {}).get('resumen', {}).get('quintiles_21', {}) or {}
        e = r.get('criterio_extra') or {}
        iv = e.get('ic90_q1_topado_vs_indice_neto')
        L.append(f"| {r['nombre']} | {_f(q.get('q1_topado_vs_indice_neto_medio'), True, 4)} | {_iv(iv, True)} | "
                 f"{_f(r['subconjuntos'].get('todas', {}).get('resumen', {}).get('sharpe_anual_margen_indice'), dec=2)} | "
                 f"{_f(q.get('pct_meses_q1_supera_indice'), True)} |")
    L += ['', '## Señal: Q1 topado contra una canasta al azar que rota igual (pedido 193)', '',
          'Control construido mes a mes con el **mismo número de nombres, el mismo tope, la misma rotación y los mismos costos** que Q1, '
          'cambiando al azar tantos nombres como cambió Q1. La única diferencia entre las dos carteras es de dónde sale el nombre nuevo, '
          'así que esta columna **no depende de qué referencia se use** —la referencia se cancela— y tampoco del costo de construcción, '
          'que las dos pagan igual. Es la cantidad que Science Mercado propone para la compuerta (su V1).', '',
          '| Modelo | Q1 topado − control neto/mes | Bruto | Sharpe | Rotación Q1 | Rotación del control | Meses que supera al control |',
          '|---|---|---|---|---|---|---|']
    for r in resultados:
        s = r['subconjuntos'].get('todas', {}).get('resumen', {}) or {}
        q = s.get('quintiles_21', {}) or {}
        L.append(f"| {r['nombre']} | {_f(q.get('q1_topado_vs_control_neto_medio'), True, 4)} | {_f(q.get('q1_topado_vs_control_bruto_medio'), True, 4)} | "
                 f"{_f(s.get('sharpe_anual_senal_vs_control'), dec=2)} | {_f(q.get('rotacion_q1_topado_media'), True)} | "
                 f"{_f(q.get('rotacion_control_media'), True)} | {_f(q.get('pct_meses_q1_supera_control'), True)} |")
    ctrl = next((r for r in resultados if r['nombre'] in CONTROLES), None)
    if ctrl:
        mc = (ctrl['subconjuntos'].get('todas', {}).get('resumen', {}).get('quintiles_21') or {})
        L += ['', f"**Cómo leer esta tabla:** el control al azar («{ctrl['nombre']}») da {_f(mc.get('q1_topado_vs_indice_neto_medio'), True, 4)} contra el índice "
              f"y {_f(mc.get('q1_topado_vs_universo_neto_medio'), True, 4)} contra el universo. Esa diferencia **no es señal**: es la prima de igual ponderación "
              '—una canasta igual ponderada rindió más que el índice ponderado por capitalización en este periodo—. Por eso la referencia del universo es la '
              'más exigente para una cartera igual ponderada, y es la que decide la compuerta; el margen contra el índice es lo que el usuario compararía '
              'contra un fondo, y hay que presentarlo restando lo que el control ya obtiene sin información.']
    L += ['', '## IC a 21 días por subconjunto', '', '| Modelo | ' + ' | '.join(claves) + ' |', '|---|' + '---|' * len(claves)]
    for r in resultados:
        c = [r['subconjuntos'].get(s, {}).get('resumen', {}).get('ic_21', {}) for s in claves]
        L.append(f"| {r['nombre']} | " + ' | '.join(f"{_f(x.get('medio'))} (t {_f(x.get('t_newey_west'), dec=2)}, {x.get('meses', 0)} m)" for x in c) + ' |')
    anios = sorted({a['anio'] for r in resultados for a in r['subconjuntos'].get('todas', {}).get('por_anio', [])})
    L += ['', '## IC a 21 días por año (todas)', '', '| Año | ' + ' | '.join(r['nombre'] for r in resultados) + ' |', '|---|' + '---|' * len(resultados)]
    for a in anios:
        fila = []
        for r in resultados:
            x = next((p for p in r['subconjuntos'].get('todas', {}).get('por_anio', []) if p['anio'] == a), None)
            fila.append(_f(x['ic_21']) if x else '—')
        L.append(f'| {a} | ' + ' | '.join(fila) + ' |')
    cs = r0['cobertura_salidas']
    L += ['', '## Compuerta de publicación (pedido 70)', '']
    for r in resultados:
        b = r['compuerta_publicacion']['y_21']
        L.append(f"- **{r['nombre']}**: {'publicable' if b['publicable'] else 'NO publicable'} — objetivo publicable: {'sí' if b['objetivo_publicable'] else 'no'}; "
                 f"variables no publicables: {', '.join(b['variables_no_publicables']) or 'ninguna'}.")
    L += ['', '## Emisoras que salieron de bolsa (pedido 72)', '', f"- {cs.get('conclusion', cs.get('nota'))}.",
          f"- Con historia utilizable: {', '.join(cs.get('con_historia_utilizable', [])) or 'ninguna'}.",
          f"- Sin historia: {', '.join(cs.get('sin_historia', [])) or 'ninguna'}."]
    sd = [f"{r['nombre']} {r['subconjuntos']['todas']['resumen']['sin_desenlace_21_total']}" for r in resultados if 'todas' in r['subconjuntos']]
    L += [f"- Filas con puntaje y sin desenlace a 21 días (dejaron de cotizar): {'; '.join(sd)}."]
    if len(resultados) and any(n not in [r['nombre'] for r in resultados] for n in PENDIENTES_BASE):
        L += ['', '## Líneas base pendientes', ''] + [f'- **{n}**: {t}' for n, t in PENDIENTES_BASE.items()]
    L += ['', '## Limitaciones', ''] + [f'- {x}' for x in r0['limitaciones']] + ['']
    return '\n'.join(L)


def guardar(resultados, carpeta, titulo, base_md):
    os.makedirs(carpeta, exist_ok=True)
    hoy = dt.date.today().isoformat()
    rutas = []
    for r in resultados:
        ruta = ruta_libre(carpeta, f"{hoy}_{r['nombre']}" + ('_app' if r.get('universo') == 'solo_app' else ''), '.json')
        with open(ruta, 'w', encoding='utf-8') as fh: json.dump(r, fh, ensure_ascii=False, indent=1, allow_nan=False, default=lambda x: None)
        r['archivo'] = os.path.basename(ruta); rutas.append(ruta)
    ruta = ruta_libre(carpeta, f'{hoy}_{base_md}', '.md')
    with open(ruta, 'w', encoding='utf-8') as fh: fh.write(reporte_md(resultados, titulo))
    return rutas + [ruta]


def limpiar_nan(o):
    if isinstance(o, dict): return {k: limpiar_nan(v) for k, v in o.items()}
    if isinstance(o, list): return [limpiar_nan(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o): return None
    return o


def main(argv=None):
    p = argparse.ArgumentParser(description='Evaluación fuera de muestra de Brota')
    p.add_argument('--dataset', default=os.path.join(AQUI, 'datos')); p.add_argument('--resultados', default=os.path.join(AQUI, 'resultados'))
    p.add_argument('--lineas-base', action='store_true'); p.add_argument('--solo-app', action='store_true')
    p.add_argument('--puntajes', help='parquet con fecha, ticker, puntaje'); p.add_argument('--nombre'); p.add_argument('--variables', default='')
    p.add_argument('--descripcion', default=''); p.add_argument('--ablacion', action='store_true')
    p.add_argument('--desde', help='primera fecha evaluada (incluida)'); p.add_argument('--hasta', help='última fecha evaluada (excluida)')
    p.add_argument('--corrige', help='nombre del candidato que este corrige por un defecto de procedimiento documentado (no suma intento)')
    p.add_argument('--por-libro', action='store_true', help='Q1 dentro de México (terciles) y extranjero (quintiles), combinados con riesgo activo igual (pedido 161)')
    p.add_argument('--dsr-n', type=int, help='N de intentos pre-comprometido para el DSR (pliegue externo)')
    p.add_argument('--dsr-sd', type=float, help='dispersión mensual de Sharpe entre intentos, medida en la ventana de selección')
    p.add_argument('--mecanismo', choices=['mensual', 'trimestral', 'histeresis'], default='mensual', help='mecanismo de rotación de la cartera topada (laboratorio/REGLA_ROTACION.md)')
    p.add_argument('--histeresis', type=float, default=0.5, help='percentil de SALIDA con --mecanismo histeresis; fijado en 0.5 por la regla, no es una malla que se recorra')
    p.add_argument('--factor-costos', type=float, default=1.0, help='multiplica todo el costo por operación; 2 es la corrida obligatoria al doble (pedidos 99 y 184)')
    p.add_argument('--elegibles-sic', dest='elegibles', action='store_true', default=None,
                   help='recortar el libro extranjero a lo comprable desde México SIN cambiar el tope: aísla el efecto del universo')
    p.add_argument('--tope-sic', dest='tope_sic', action='store_true',
                   help='topar el libro extranjero con la liquidez del SIC y no con la de su bolsa de origen, y dejar fuera lo que opera '
                        f'menos de {DIAS_SIC_MIN:.2f} de los días en el SIC (pedido AT de Fable; ver REGLA_TOPE_SIC.md)')
    p.add_argument('--sic', type=float, default=DIFERENCIAL_SIC, help=f'medio diferencial de las extranjeras en el SIC, que SUSTITUYE al de su bolsa de origen (por omisión {DIFERENCIAL_SIC:.5f}; 0 lo apaga y se usa el de origen)')
    p.add_argument('--inclinacion-beta', choices=['pesos', 'seleccion'], help='inclinación a beta baja (tanda 6, M31): pesos ∝ 1/beta dentro de Q1, o Q1 solo entre las de beta bajo la mediana del día')
    a = p.parse_args(argv)
    if not a.lineas_base and not a.puntajes: p.error('usa --lineas-base y/o --puntajes')
    if a.puntajes and not a.nombre: p.error('--puntajes requiere --nombre')
    base_cols = ['fecha', 'ticker', 'pais', 'sector', 'en_app', 'y_21', 'y_63', 'importe_20d', 'beta_ipc_1a', 'ret_mercado_21', COL_DIFERENCIAL,
                  'dias_sic_12m', 'importe_sic_20d']   # beta: inclinación; ret_mercado_21: referencia invertible; COL_DIFERENCIAL: costo
    vars_modelo = [v for v in a.variables.split(',') if v]
    tabla, meta = leer_dataset(a.dataset, base_cols + [c for v, _, _ in LINEAS_BASE.values() for c in v] + vars_modelo)
    if a.desde: tabla = tabla[pd.to_datetime(tabla['fecha']) >= pd.Timestamp(a.desde)]
    if a.hasta: tabla = tabla[pd.to_datetime(tabla['fecha']) < pd.Timestamp(a.hasta)]   # --hasta: el pliegue externo no entra a la selección
    tabla = tabla.sort_values(['fecha', 'ticker']).reset_index(drop=True)
    kw = dict(solo_app=a.solo_app, carpeta_intentos=a.resultados, por_libro=a.por_libro, inclinacion_beta=a.inclinacion_beta, dsr_n=a.dsr_n, dsr_sd=a.dsr_sd,
              factor_costos=a.factor_costos, sic=a.sic, mecanismo=a.mecanismo, histeresis=a.histeresis, tope_sic=a.tope_sic, elegibles=a.elegibles)
    nref = CRITERIO['referencia']
    comp = evaluar(tabla, puntaje_linea_base(tabla, nref), nref, LINEAS_BASE[nref][0], meta, LINEAS_BASE[nref][2], tipo='linea_base', **kw)
    ref = comp['subconjuntos'].get('todas', {}).get('resumen')
    resultados = []
    if a.puntajes:
        pz = pd.read_parquet(a.puntajes, columns=['fecha', 'ticker', 'puntaje']); pz['ticker'] = pz['ticker'].astype(str); pz['fecha'] = pd.to_datetime(pz['fecha'])
        j = tabla.merge(pz, on=['fecha', 'ticker'], how='left')
        resultados.append(evaluar(tabla, j['puntaje'], a.nombre, vars_modelo, meta, a.descripcion, a.ablacion, referencia=ref, tipo='modelo', corrige=a.corrige, **kw))
    resultados.append(comp)
    for nombre, (variables, _, desc) in LINEAS_BASE.items():
        if nombre == nref: continue
        resultados.append(evaluar(tabla, puntaje_linea_base(tabla, nombre), nombre, variables, meta, desc, False, referencia=ref,
                                  tipo='control' if nombre in CONTROLES else 'linea_base', **kw))
    resultados = recalcular_desinflado(resultados, a.resultados, ref)
    resultados.sort(key=lambda r: list(LINEAS_BASE).index(r['nombre']) if r['nombre'] in LINEAS_BASE else -1)
    resultados = [limpiar_nan(r) for r in resultados]
    titulo = (f"Evaluación: {a.nombre} contra líneas base" if a.puntajes else 'Evaluación de líneas base') + (' · por libro (México en terciles, extranjero en quintiles)' if a.por_libro else '') + (' · con inclinación a beta baja' if a.inclinacion_beta else '')
    rutas = guardar(resultados, a.resultados, titulo, (a.nombre or 'lineas-base') + ('_app' if a.solo_app else '') + ('_por-libro' if a.por_libro else '') + ('_beta-baja' if a.inclinacion_beta else ''))
    for r in resultados:
        s = r['subconjuntos'].get('todas', {}).get('resumen') or {}
        print(f"{r['nombre']:18s} IC21 {_f((s.get('ic_21') or {}).get('medio'))} t {_f((s.get('ic_21') or {}).get('t_newey_west'), dec=2)} "
              f"IC63 {_f((s.get('ic_63') or {}).get('medio'))}  Q1−Q5 neto {_f((s.get('quintiles_21') or {}).get('q1q5_neto_medio'), True, 4)}  "
              f"criterio {'cumple' if r['candidato_a_app'] else 'no'}")
    print('Escrito:', *[os.path.relpath(x, RAIZ) for x in rutas], sep='\n  ')


if __name__ == '__main__':
    main()
