#!/usr/bin/env python3
"""
laboratorio/evaluar_caidas.py — evaluación de señales de CAÍDA FUERTE como eventos raros (CAIDAS_FABLE §2–§3, pedidos 81–84).

Uso:
    python3 laboratorio/evaluar_caidas.py --reglas-base                       # dataset-v2 en laboratorio/datos
    python3 laboratorio/evaluar_caidas.py --reglas-base --dataset RUTA
    python3 laboratorio/evaluar_caidas.py --puntajes pred.parquet --nombre lgbm-caidas-v1 --variables a,b [--probabilidades]   (entrenar.py)

Protocolo:
  - Muestra mensual: primer día hábil de cada mes; puntaje mayor = más riesgo. Solo variables del dataset (precios ≤ D, datos con conocido < D).
  - Objetivos (binarios derivados de columnas continuas): principal = trayectoria −20 % en 63 días (min_ret_63 ≤ −0.20); sensibilidades
    −25 %/63, −20 %/21, punto a punto, ajustada por mercado e idiosincrática (ajustada ≤ −u con el mercado arriba de −10 %).
  - Métricas: tasa base; PR-AUC (precisión promedio); ROC-AUC solo como referencia; precisión, cobertura y lift en el 5 % y 10 % de mayor
    riesgo de cada mes (empates en el corte cuentan en fracción, sin azar); Brier y Brier skill contra la mejor referencia ingenua (tasa
    histórica general o del régimen), con
    probabilidades calibradas mes a mes SOLO con meses cuyo desenlace ya se conocía (embargo = horizonte); tabla de calibración.
  - Intervalos al 90 % por bootstrap de bloques circulares de 12 meses, con el efecto de diseño contra meses independientes; episodios independientes; todo por régimen y por año.
  - Mismas compuertas que evaluar.py: publicable.verificar sobre variables y objetivo, analistas solo en ablación, cobertura de deslistadas.
Resultados que solo crecen: laboratorio/resultados/AAAA-MM-DD_caidas_<nombre>.json y un reporte .md por corrida.
"""
import argparse, datetime as dt, glob, json, math, os, sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import evaluar as EV  # noqa: E402
import publicable  # noqa: E402

VERSION = 'evaluar-caidas-v1'
SEMILLA = 20260914
BOOTSTRAP = 300
BLOQUE_MESES = 12                          # mismo bloque que evaluar.py (tanda 4, M14)
K_RIESGO = (0.05, 0.10)
MIN_EMPRESAS = 25
MESES_MIN_CALIBRAR = 24
SUAVIZADO = 20                       # pseudo-observaciones de la tasa base al calibrar un decil con pocos casos
REGIMENES = ('tranquilo', 'estres', 'crisis')

# nombre: (columna del objetivo, horizonte, descripción, función(tabla, u) → bool)
OBJETIVOS = {
    'trayectoria_20_63': ('min_ret_63', 63, 'Tocó −20 % en algún momento de los 63 días (principal)', lambda t: t['min_ret_63'] <= -0.20),
    'trayectoria_25_63': ('min_ret_63', 63, 'Tocó −25 % en 63 días', lambda t: t['min_ret_63'] <= -0.25),
    'trayectoria_20_21': ('min_ret_21', 21, 'Tocó −20 % en 21 días', lambda t: t['min_ret_21'] <= -0.20),
    'punto_20_63': ('y_63', 63, 'Terminó −20 % o peor a los 63 días', lambda t: t['y_63'] <= -0.20),
    'ajustada_20_63': ('y_63_aj_mercado', 63, 'Cayó 20 % más que su mercado (beta) en 63 días', lambda t: t['y_63_aj_mercado'] <= -0.20),
    'idiosincratica_20_63': ('y_63_aj_mercado', 63, 'Cayó 20 % más que su mercado mientras el mercado no cayó más de 10 %',
                             lambda t: (t['y_63_aj_mercado'] <= -0.20) & (t['ret_mercado_63'] > -0.10)),
}
PRINCIPAL = 'trayectoria_20_63'


def _senal(cond):
    return cond.fillna(False).astype(float)


# nombre: (variables, puntaje(tabla), descripción). CAIDAS_FABLE §3.4
REGLAS_BASE = {
    'bajo_ma200': (['dist_ma200'], lambda t: _senal(t['dist_ma200'] < 0), 'Precio debajo de su media de 200 días'),
    'caida_20_desde_max': (['dist_max_52s'], lambda t: _senal(t['dist_max_52s'] <= -0.20), 'Ya cayó 20 % desde su máximo de 52 semanas (regla tardía)'),
    'decil_volatilidad': (['vol_60d'], lambda t: _senal(t.groupby('fecha')['vol_60d'].rank(pct=True) >= 0.90), 'Volatilidad de 60 días en el 10 % más alto del día'),
    'compuesta': (['dist_ma200', 'dist_max_52s', 'vol_60d', 'runup_2a', 'sorpresa_ult'],
                  lambda t: _senal(t['dist_ma200'] < 0) + _senal(t['dist_max_52s'] <= -0.20) + _senal(t.groupby('fecha')['vol_60d'].rank(pct=True) >= 0.90)
                  + _senal(t['runup_2a'] > 1.0) + _senal(t['sorpresa_ult'] < 0),
                  'Suma de las tres anteriores + run-up de 2 años > 100 % + última sorpresa negativa'),
    'placebo': ([], None, 'Control: puntaje al azar con semilla fija; su PR-AUC debe rondar la tasa base'),
}
COLUMNAS_OBJETIVO = ['min_ret_21', 'min_ret_63', 'y_21', 'y_63', 'y_63_aj_mercado', 'ret_mercado_63']


# ================================================================ métricas
def precision_promedio(y, s):
    """PR-AUC como precisión promedio con umbrales en los valores distintos del puntaje (los empates entran juntos)."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    p = y.sum()
    if p == 0 or len(y) == 0: return None
    orden = np.argsort(-s, kind='mergesort'); y, s = y[orden], s[orden]
    fin_grupo = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[fin_grupo]; n = fin_grupo + 1
    precision, recall = tp / n, tp / p
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    p, n = y.sum(), (~y).sum()
    if p == 0 or n == 0: return None
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - p * (p + 1) / 2) / (p * n))


def aciertos_en_k(y, s, k):
    """(marcadas esperadas, aciertos esperados) al marcar la fracción k de mayor puntaje. Empates en el corte: cada una cuenta en fracción."""
    y = np.asarray(y, float); s = np.asarray(s, float); n = len(y)
    c = int(math.ceil(k * n))
    if n == 0 or c == 0: return 0.0, 0.0
    umbral = np.sort(s)[::-1][c - 1]
    arriba, empate = s > umbral, s == umbral
    fraccion = (c - arriba.sum()) / empate.sum()
    return float(c), float(y[arriba].sum() + fraccion * y[empate].sum())


def brier(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2)) if len(y) else None


def episodios(filas):
    """Caídas independientes: meses positivos de la misma emisora separados por ≤ 2 meses son un solo episodio (ventanas traslapadas)."""
    pos = filas[filas['evento'] == 1][['ticker', 'mes_idx']].sort_values(['ticker', 'mes_idx'])
    if not len(pos): return 0
    nuevo = (pos['ticker'] != pos['ticker'].shift()) | (pos['mes_idx'] - pos['mes_idx'].shift() > 3)
    return int(nuevo.sum())


# ================================================================ M39: Platt condicionado y descomposición de Murphy
def _logistica(x, y, iteraciones=25):
    """Regresión logística de dos parámetros por mínimos cuadrados reponderados (IRLS). Sin regularización: se quiere el ajuste que los
    datos pidan, no uno encogido hacia cero por un valor por omisión que nadie eligió."""
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    y = np.asarray(y, float)
    b = np.zeros(2)
    for _ in range(iteraciones):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        w = np.clip(p * (1 - p), 1e-9, None)
        z = X @ b + (y - p) / w
        try:
            nuevo = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * z))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(nuevo)): break
        paso = np.max(np.abs(nuevo - b)); b = nuevo
        if paso < 1e-8: break
    return b


def murphy(y, p, bins=10, por_cuantiles=True):
    """Descomposición de Murphy (1973) del Brier: confiabilidad − resolución + incertidumbre.

    - **confiabilidad**: qué tan lejos está la probabilidad predicha de la frecuencia observada en su tramo. Baja es buena, y se arregla
      recalibrando sin tocar el modelo.
    - **resolución**: cuánto se aparta de la tasa base al separar los casos. Alta es buena, y NO se arregla calibrando: es lo que el
      modelo sabe distinguir.
    - **incertidumbre**: la varianza de la tasa base. No depende del modelo; es el piso del problema.

    Sirve para contestar si un Brier malo viene de estar mal calibrado —que tiene arreglo— o de no distinguir —que no.

    La identidad es exacta sobre el pronóstico AGRUPADO (cada caso con la media de su tramo), no sobre el continuo: al agrupar se pierde la
    dispersión dentro del tramo. Por eso se devuelven `brier` (el de verdad), `brier_agrupado` (el que cumple la identidad) y el
    `residuo_agrupacion` entre los dos, en vez de presentar como exacta una suma que no cierra."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < bins: return None
    base = float(y.mean())
    # Los grupos son CUANTILES del propio pronóstico, no tramos fijos de probabilidad. Corrección de Science Mercado (M1, 2026-09-17): con
    # tramos fijos una transformación monótona mueve observaciones de tramo y cambia la resolución, que entonces deja de medir lo que dice
    # medir. Con cuantiles del pronóstico la resolución es invariante a cualquier transformación monótona —los grupos contienen las mismas
    # observaciones— y eso es lo que la vuelve «lo que el modelo sabe distinguir» y no un artefacto de la escala.
    if por_cuantiles:
        # rango PROMEDIO, no por posición: los empates tienen que caer en el mismo grupo. Con desempate por orden de aparición, un
        # pronóstico constante se partiría en diez grupos arbitrarios con frecuencias distintas y aparecería resolución donde no hay
        # ninguna. La prueba del pronóstico constante lo detectó.
        r = pd.Series(p).rank(method='average', pct=True).to_numpy()
        tramo = np.minimum((r * bins).astype(int), bins - 1)
    else:
        tramo = np.minimum((p * bins).astype(int), bins - 1)
    conf = res = 0.0
    agrupado = np.empty(len(y))
    for k in range(bins):
        m = tramo == k
        n = int(m.sum())
        if not n: continue
        pk, ok_k = float(p[m].mean()), float(y[m].mean())
        agrupado[m] = pk
        conf += n * (pk - ok_k) ** 2
        res += n * (ok_k - base) ** 2
    n = len(y)
    b, ba = float(np.mean((p - y) ** 2)), float(np.mean((agrupado - y) ** 2))
    return {'confiabilidad': conf / n, 'resolucion': res / n, 'incertidumbre': base * (1 - base),
            'brier': b, 'brier_agrupado': ba, 'residuo_agrupacion': b - ba, 'n': n, 'tramos': bins,
            'agrupacion': 'cuantiles del pronóstico' if por_cuantiles else 'tramos fijos de probabilidad'}


# ================================================================ calibración sin mirar el futuro
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def desplazamiento(p, y, limite=12.0, pasos=100):
    """Desplazamiento único en log-odds (UN parámetro) que iguala la probabilidad media predicha con la tasa realizada.

    Es la recalibración de solo intercepto que recomendó Mercado (M1.4 y su predicción del 2026-09-17): resuelve
    `sum_i sigma(logit(p_i) + delta) = sum_i y_i` por bisección, que funciona porque el lado izquierdo crece con delta.
    Se aplica al PRONÓSTICO del modelo, sea el mapa por decil o la probabilidad cruda: el objeto que hay que corregir es
    el que está mal calibrado, no el que resulte cómodo. Un desplazamiento es monótono, así que NO cambia el orden ni los
    grupos por cuantil: la resolución queda idéntica por álgebra y solo puede moverse la confiabilidad. None si no se puede."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if not len(p): return None
    objetivo = float(y.sum())
    if objetivo <= 0 or objetivo >= len(y): return None      # todo ceros o todo unos: el desplazamiento se va al infinito
    z = _logit(p)
    f = lambda dd: float((1 / (1 + np.exp(-np.clip(z + dd, -30, 30)))).sum() - objetivo)
    lo, hi = -limite, limite
    if f(lo) > 0 or f(hi) < 0: return None
    for _ in range(pasos):
        mid = (lo + hi) / 2
        if f(mid) < 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def calibrar(d, h, fechas_calendario, probabilidades=False):
    """Probabilidad por mes: decil del puntaje dentro del mes → frecuencia histórica de ese decil en meses cuyo desenlace ya se conocía
    (fecha del mes + h sesiones < fecha actual). Dos referencias ingenuas con la misma regla: tasa histórica general (p_base) y del régimen
    del mes (p_base_regimen). Sin 24 meses conocidos: nulo."""
    cal = pd.DatetimeIndex(sorted(pd.unique(fechas_calendario)))
    meses = sorted(d['fecha'].unique())
    pos = cal.searchsorted(pd.DatetimeIndex(meses))
    conocido = [cal[p + h] if p + h < len(cal) else pd.Timestamp.max for p in pos]
    d = d.copy()
    d['decil'] = np.minimum((d.groupby('fecha')['puntaje'].rank(pct=True, method='average') * 10).fillna(0).astype(int), 9)
    d['p_modelo'] = np.nan; d['p_base'] = np.nan; d['p_base_regimen'] = np.nan; d['p_platt'] = np.nan; d['p_regimen'] = np.nan
    d['p_nivel'] = np.nan; d['delta_nivel'] = np.nan
    grupos = {m: g.index for m, g in d.groupby('fecha')}
    dec_suma, dec_n = np.zeros(10), np.zeros(10)
    reg_suma, reg_n = {}, {}
    dec_reg = {}          # régimen -> [sumas por decil, conteos por decil], solo con meses ya conocidos de ESE régimen
    hist_x, hist_y = [], []          # M39: puntajes y desenlaces YA CONOCIDOS, para el Platt condicionado
    hist_d, hist_r = [], []          # decil y régimen de esos mismos meses, para el desplazamiento de nivel por régimen
    sig, n_conocidos = 0, 0
    for i, m in enumerate(meses):
        while sig < len(meses) and conocido[sig] < m:          # acumula meses cuyo desenlace ya se conocía en m
            g = d.loc[grupos[meses[sig]]]
            np.add.at(dec_suma, g['decil'].to_numpy(), g['evento'].to_numpy()); np.add.at(dec_n, g['decil'].to_numpy(), 1)
            for reg, gg in g.groupby('regimen'):
                reg_suma[reg] = reg_suma.get(reg, 0) + gg['evento'].sum(); reg_n[reg] = reg_n.get(reg, 0) + len(gg)
                # mapa decil → frecuencia SEPARADO por régimen (arreglo indicado por el desglose de Murphy; Mercado, M39 y su P1)
                if reg not in dec_reg: dec_reg[reg] = [np.zeros(10), np.zeros(10)]
                np.add.at(dec_reg[reg][0], gg['decil'].to_numpy(), gg['evento'].to_numpy())
                np.add.at(dec_reg[reg][1], gg['decil'].to_numpy(), 1)
            hist_x.append(g['puntaje'].to_numpy(float)); hist_y.append(g['evento'].to_numpy(float))
            hist_d.append(g['decil'].to_numpy()); hist_r.append(g['regimen'].to_numpy())
            sig += 1; n_conocidos += 1
        if n_conocidos < MESES_MIN_CALIBRAR: continue
        idx = grupos[m]
        tasa = dec_suma.sum() / dec_n.sum()
        reg = d.at[idx[0], 'regimen']
        d.loc[idx, 'p_base'] = tasa
        d.loc[idx, 'p_base_regimen'] = reg_suma[reg] / reg_n[reg] if reg_n.get(reg, 0) >= 200 else tasa
        p = (dec_suma + SUAVIZADO * tasa) / (dec_n + SUAVIZADO)   # mapa decil → frecuencia, común a p_modelo y al desplazamiento de nivel
        if probabilidades:
            d.loc[idx, 'p_modelo'] = d.loc[idx, 'puntaje'].clip(0, 1)
        else:
            d.loc[idx, 'p_modelo'] = p[d.loc[idx, 'decil'].to_numpy()]
        # calibración CONDICIONADA AL RÉGIMEN: el mismo decil se traduce a una probabilidad distinta según el régimen del mes. La etiqueta
        # es conocible en el momento de predecir —`regimen` es del grupo R y pasa la prueba de futuro alterado— así que condicionar en ella
        # no usa información del futuro, que era la comprobación previa que exigió Mercado antes de dar su predicción.
        if reg in dec_reg and dec_reg[reg][1].sum() >= 200:
            tasa_r = dec_reg[reg][0].sum() / dec_reg[reg][1].sum()
            pr = (dec_reg[reg][0] + SUAVIZADO * tasa_r) / (dec_reg[reg][1] + SUAVIZADO)
            d.loc[idx, 'p_regimen'] = pr[d.loc[idx, 'decil'].to_numpy()]
        # RECALIBRACIÓN DE SOLO NIVEL (Mercado, M1.4): un único desplazamiento en log-odds sobre el mapa general, estimado con los meses
        # YA CONOCIDOS del mismo régimen. Frente al mapa por régimen —que estima diez números por régimen— éste estima uno, y con 29
        # meses de estrés esa diferencia es la que decide si se puede aprender algo. Si el error en estrés es de NIVEL, esto lo arregla;
        # si es de FORMA, no se mueve, y ésa es justamente la refutación que Mercado escribió por adelantado.
        mr = np.concatenate(hist_r) == reg
        if mr.sum() >= 200:
            xh_all, dh_all, yh_all = np.concatenate(hist_x), np.concatenate(hist_d), np.concatenate(hist_y)
            # el pronóstico que el modelo habría dado en esos meses, con la MISMA regla que hoy: probabilidad cruda o mapa por decil
            p_hist = np.clip(xh_all[mr], 0, 1) if probabilidades else p[dh_all[mr]]
            dl = desplazamiento(p_hist, yh_all[mr])
            if dl is not None:
                z = _logit(d.loc[idx, 'p_modelo'].to_numpy(float)) + dl
                d.loc[idx, 'p_nivel'] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                d.loc[idx, 'delta_nivel'] = dl
        # M39: Platt condicionado. Logística sobre el puntaje ajustada SOLO con los meses cuyo desenlace ya se conocía en m. A diferencia
        # de la calibración por deciles, usa el puntaje continuo, así que no pierde el orden dentro del decil.
        xh, yh = np.concatenate(hist_x), np.concatenate(hist_y)
        fin = np.isfinite(xh) & np.isfinite(yh)
        xh, yh = xh[fin], yh[fin]
        if len(xh) >= 200 and 0 < yh.mean() < 1:
            transf = (lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))) if probabilidades else \
                     (lambda v, mu=xh.mean(), sd=(xh.std() or 1.0): (v - mu) / sd)
            b = _logistica(transf(xh), yh)
            xm = transf(d.loc[idx, 'puntaje'].to_numpy(float))
            d.loc[idx, 'p_platt'] = 1.0 / (1.0 + np.exp(-np.clip(b[0] + b[1] * xm, -30, 30)))
    # DIAGNÓSTICO, NO PRONÓSTICO: el mismo desplazamiento de nivel pero estimado con TODOS los meses del régimen, incluidos los futuros.
    # No se puede usar para decidir nada ni mostrar nada; sirve para una sola cosa que Mercado pidió: separar sus dos refutaciones. Si
    # dentro de muestra tampoco mejora, el error en estrés es de FORMA (caso a). Si mejora aquí y no condicionado, el desplazamiento
    # EXISTE pero no es estable entre episodios (caso b). La conclusión práctica es la misma; el diagnóstico no.
    c = d.dropna(subset=['p_modelo', 'evento'])
    for reg, g in c.groupby('regimen'):
        dl = desplazamiento(g['p_modelo'].to_numpy(float), g['evento'].to_numpy(float))
        if dl is None: continue
        z = _logit(g['p_modelo'].to_numpy(float)) + dl
        d.loc[g.index, 'p_nivel_dm'] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        d.loc[g.index, 'delta_nivel_dm'] = dl
    return d


# ================================================================ evaluación
def muestra_mensual(tabla, objetivo):
    col, h, _, f = OBJETIVOS[objetivo]
    fechas = EV.fechas_rebalanceo(tabla['fecha'])
    d = tabla[tabla['fecha'].isin(fechas)].copy()
    d['evento'] = np.where(d[col].notna(), f(d).astype(float), np.nan)
    if objetivo == 'idiosincratica_20_63':
        d.loc[d['ret_mercado_63'].isna(), 'evento'] = np.nan
    d = d.dropna(subset=['evento', 'puntaje'])
    n_mes = d.groupby('fecha')['ticker'].transform('size')
    d = d[n_mes >= MIN_EMPRESAS]
    idx = {f: i for i, f in enumerate(sorted(fechas))}
    d['mes_idx'] = d['fecha'].map(idx)
    return d, h


def metricas(d):
    y, s = d['evento'].to_numpy(), d['puntaje'].to_numpy()
    r = {'filas': int(len(d)), 'meses': int(d['fecha'].nunique()), 'positivos': int(y.sum()),
         'tasa_base': float(y.mean()) if len(y) else None, 'pr_auc': precision_promedio(y, s), 'roc_auc_referencia': roc_auc(y, s),
         'episodios_independientes': episodios(d)}
    r['pr_auc_lift'] = r['pr_auc'] / r['tasa_base'] if r['pr_auc'] is not None and r['tasa_base'] else None
    for k in K_RIESGO:
        marc = acie = 0.0
        for _, g in d.groupby('fecha'):
            a, b = aciertos_en_k(g['evento'], g['puntaje'], k); marc += a; acie += b
        prec = acie / marc if marc else None
        r[f'precision_{int(k * 100)}'] = prec
        r[f'cobertura_{int(k * 100)}'] = acie / y.sum() if y.sum() else None
        r[f'lift_{int(k * 100)}'] = prec / r['tasa_base'] if prec is not None and r['tasa_base'] else None
    c = d.dropna(subset=['p_modelo', 'p_base']) if 'p_modelo' in d else d.iloc[0:0]
    if len(c):
        bm, bg, br = brier(c['evento'], c['p_modelo']), brier(c['evento'], c['p_base']), brier(c['evento'], c['p_base_regimen'])
        bb = min(bg, br)   # contra la MEJOR referencia ingenua: un modelo no gana por elegir una referencia mala
        r.update({'brier': bm, 'brier_base_general': bg, 'brier_base_regimen': br, 'brier_skill_vs_mejor_referencia': 1 - bm / bb if bb else None,
                  'meses_calibrados': int(c['fecha'].nunique())})
        # M39: descomposición de Murphy del Brier del modelo, y el Platt condicionado al lado para ver qué parte era calibración
        r['murphy'] = murphy(c['evento'], c['p_modelo'])
        cr = d.dropna(subset=['p_regimen', 'evento']) if 'p_regimen' in d else d.iloc[0:0]
        if len(cr):
            br = brier(cr['evento'], cr['p_regimen'])
            r.update({'brier_por_regimen': br, 'brier_skill_por_regimen': 1 - br / bb if bb else None,
                      'murphy_por_regimen': murphy(cr['evento'], cr['p_regimen']), 'meses_por_regimen': int(cr['fecha'].nunique())})
        else:
            r.update({'brier_por_regimen': None, 'brier_skill_por_regimen': None, 'murphy_por_regimen': None, 'meses_por_regimen': 0})
        # recalibración de solo nivel: la honesta (p_nivel, solo meses previos) y la de diagnóstico (p_nivel_dm, dentro de muestra)
        for col, eti in (('p_nivel', 'nivel'), ('p_nivel_dm', 'nivel_dm')):
            cn = d.dropna(subset=[col, 'evento']) if col in d else d.iloc[0:0]
            if len(cn):
                bn = brier(cn['evento'], cn[col])
                r.update({f'brier_{eti}': bn, f'brier_skill_{eti}': 1 - bn / bb if bb else None,
                          f'murphy_{eti}': murphy(cn['evento'], cn[col]), f'meses_{eti}': int(cn['fecha'].nunique()),
                          f'delta_{eti}': float(cn[f'delta_{eti}'].mean()) if f'delta_{eti}' in cn else None})
            else:
                r.update({f'brier_{eti}': None, f'brier_skill_{eti}': None, f'murphy_{eti}': None, f'meses_{eti}': 0, f'delta_{eti}': None})
        cp = d.dropna(subset=['p_platt', 'evento']) if 'p_platt' in d else d.iloc[0:0]
        if len(cp):
            bp = brier(cp['evento'], cp['p_platt'])
            r.update({'brier_platt': bp, 'brier_skill_platt_vs_mejor_referencia': 1 - bp / bb if bb else None,
                      'murphy_platt': murphy(cp['evento'], cp['p_platt']), 'meses_platt': int(cp['fecha'].nunique())})
        else:
            r.update({'brier_platt': None, 'brier_skill_platt_vs_mejor_referencia': None, 'murphy_platt': None, 'meses_platt': 0})
    else:
        r.update({'brier': None, 'brier_base_general': None, 'brier_base_regimen': None, 'brier_skill_vs_mejor_referencia': None, 'meses_calibrados': 0,
                  'murphy': None, 'brier_platt': None, 'brier_skill_platt_vs_mejor_referencia': None, 'murphy_platt': None, 'meses_platt': 0})
    return r


def tabla_calibracion(d):
    c = d.dropna(subset=['p_modelo'])
    if not len(c): return []
    c = c.assign(bin=np.minimum((c['p_modelo'] * 10).astype(int), 9))
    g = c.groupby('bin').agg(n=('evento', 'size'), prob_media=('p_modelo', 'mean'), frecuencia=('evento', 'mean')).reset_index()
    return [{'rango': f"{int(b) * 10}–{int(b) * 10 + 10} %", 'n': int(n), 'prob_media': float(p), 'frecuencia_observada': float(f)}
            for b, n, p, f in g.itertuples(index=False)]


def bootstrap(d, semilla=SEMILLA, repeticiones=BOOTSTRAP, bloque=BLOQUE_MESES):
    """Remuestrea bloques circulares de meses consecutivos (la misma secuencia para todos los modelos: diferencias pareadas). Con objetivo a
    63 días los meses seguidos se traslapan y las caídas se agrupan en crisis: remuestrear meses sueltos subestimaba el error (tanda 4, M14).
    bloque=1 es el remuestreo de meses independientes (sirve para el efecto de diseño)."""
    ys, ss, t = [], [], []
    for m, g in d.groupby('fecha', sort=True):
        ys.append(g['evento'].to_numpy(float)); ss.append(g['puntaje'].to_numpy(float))
        a, b = aciertos_en_k(g['evento'], g['puntaje'], 0.10)
        c = g.dropna(subset=['p_modelo', 'p_base'])
        t.append((a, b, float(((c['p_modelo'] - c['evento']) ** 2).sum()), float(((c['p_base'] - c['evento']) ** 2).sum()),
                  float(((c['p_base_regimen'] - c['evento']) ** 2).sum())))
    t = np.array(t)
    rng = np.random.default_rng(semilla)
    muestras = {'pr_auc': [], 'precision_10': [], 'brier_skill_vs_mejor_referencia': []}
    n = len(ys)
    for _ in range(repeticiones):
        inicios = rng.integers(0, n, size=int(np.ceil(n / bloque)))
        e = ((inicios[:, None] + np.arange(bloque)[None, :]) % n).ravel()[:n]
        muestras['pr_auc'].append(precision_promedio(np.concatenate([ys[i] for i in e]), np.concatenate([ss[i] for i in e])) or np.nan)
        x = t[e].sum(axis=0)
        muestras['precision_10'].append(x[1] / x[0] if x[0] else np.nan)
        base = min(x[3], x[4])
        muestras['brier_skill_vs_mejor_referencia'].append(1 - x[2] / base if base else np.nan)
    return {k: np.array(v, float) for k, v in muestras.items()}


def intervalo(x):
    """IC 90 % del remuestreo. Con pocos meses los bloques circulares devuelven siempre la misma muestra y el intervalo degenera:
    en ese caso NO es un intervalo y se devuelve nulo, para que el reporte diga «no estimable» en lugar de un número (tanda 6, M34)."""
    x = x[~np.isnan(x)]
    if not len(x): return None
    lo, hi = float(np.percentile(x, 5)), float(np.percentile(x, 95))
    return None if hi - lo < 1e-12 else [lo, hi]


def evaluar(tabla, puntaje, nombre, variables, meta, descripcion='', ablacion=False, probabilidades=False, universo_csv=None,
            objetivos=tuple(OBJETIVOS), repeticiones=BOOTSTRAP):
    meta_cols = meta.get('columnas') or {}
    analistas = EV.revisar_grupos(meta_cols, variables, ablacion)
    t = tabla.assign(puntaje=np.asarray(puntaje, dtype=float))
    salida = {'version': VERSION, 'nombre': nombre, 'descripcion': descripcion, 'evaluado': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
              'variables': list(variables), 'ablacion': bool(ablacion), 'variables_analistas': analistas, 'probabilidades_propias': bool(probabilidades),
              'dataset': {k: meta.get(k) for k in ('version', 'generado', 'almacen_commit', 'filas', 'empresas', 'fecha_min', 'fecha_max')},
              'codigo': EV.commit_codigo(), 'semilla': SEMILLA, 'bootstrap': repeticiones, 'objetivos': {}, '_muestras': {}}
    for obj in objetivos:
        d, h = muestra_mensual(t, obj)
        if not len(d): continue
        d = calibrar(d, h, tabla['fecha'], probabilidades)
        r = {'descripcion': OBJETIVOS[obj][2], 'horizonte': h, 'total': metricas(d),
             'por_regimen': {reg: metricas(d[d['regimen'] == reg]) for reg in REGIMENES if (d['regimen'] == reg).any()},
             'por_anio': [], 'compuerta_publicacion': publicable.verificar(meta_cols, variables, OBJETIVOS[obj][0])}
        for anio, g in d.groupby(d['fecha'].dt.year):
            m = metricas(g)
            r['por_anio'].append({'anio': int(anio), 'tasa_base': m['tasa_base'], 'pr_auc': m['pr_auc'], 'precision_10': m['precision_10'],
                                  'lift_10': m['lift_10'], 'positivos': m['positivos']})
        if obj == PRINCIPAL:
            r['calibracion'] = tabla_calibracion(d)
            muestras = bootstrap(d, repeticiones=repeticiones)
            r['intervalos_90'] = {k: intervalo(v) for k, v in muestras.items()}
            iid = bootstrap(d, repeticiones=repeticiones, bloque=1)
            r['efecto_de_diseno'] = {k: (float(np.nanvar(muestras[k]) / np.nanvar(iid[k])) if np.nanvar(iid[k]) > 0 else None) for k in muestras}
            salida['_muestras'] = muestras
            salida['periodo'] = {'desde': str(d['fecha'].min().date()), 'hasta': str(d['fecha'].max().date()), 'meses': int(d['fecha'].nunique())}
            salida['cobertura_salidas'] = EV.cobertura_salidas(t, universo_csv or os.path.join(AQUI, 'universo.csv'), EV.fechas_rebalanceo(t['fecha']))
        salida['objetivos'][obj] = r
    return salida


def comparar_contra(resultados, referencia='compuesta'):
    """Diferencia pareada (mismos meses remuestreados) de cada modelo contra la regla compuesta en el objetivo principal."""
    ref = next((r for r in resultados if r['nombre'] == referencia), None)
    if not ref or not ref['_muestras']: return
    for r in resultados:
        if r is ref or not r['_muestras']: continue
        r['objetivos'][PRINCIPAL]['diferencia_vs_compuesta_90'] = {k: intervalo(r['_muestras'][k] - ref['_muestras'][k]) for k in r['_muestras']}


# ================================================================ salida
def _f(x, pct=False, dec=3):
    if x is None or (isinstance(x, float) and np.isnan(x)): return '—'
    return f'{x * 100:.1f} %' if pct else f'{x:.{dec}f}'


def _iv(iv, pct=False):
    return f'[{_f(iv[0], pct)}, {_f(iv[1], pct)}]' if iv else 'no estimable'


def reporte_md(resultados, titulo, intentos_previos):
    r0 = resultados[0]; ds = r0['dataset']; p = r0.get('periodo', {})
    principal = [r for r in resultados if PRINCIPAL in r['objetivos']]
    L = [f'# {titulo}', '',
         f"Evaluado {r0['evaluado']} · {VERSION} · código {r0['codigo']['commit']}{' (con cambios sin commit)' if r0['codigo']['laboratorio_con_cambios_sin_commit'] else ''} · "
         f"{ds.get('version')} (almacén {ds.get('almacen_commit')}) · muestra mensual {p.get('desde')} → {p.get('hasta')} ({p.get('meses')} meses) · "
         f"bootstrap de {r0['bootstrap']} remuestreos por meses · evaluaciones de caídas previas en resultados/: {intentos_previos}", '',
         f"## Objetivo principal: {OBJETIVOS[PRINCIPAL][2]}", '',
         '| Señal | Tasa base | PR-AUC (IC 90 %) | Lift PR | Precisión 10 % (IC 90 %) | Cobertura 10 % | Lift 10 % | Brier skill vs mejor referencia (IC 90 %) | Episodios | Δ PR-AUC vs compuesta |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for r in principal:
        o = r['objetivos'][PRINCIPAL]; m = o['total']; iv = o.get('intervalos_90', {}); dif = o.get('diferencia_vs_compuesta_90', {})
        L.append(f"| {r['nombre']} | {_f(m['tasa_base'], True)} | {_f(m['pr_auc'])} {_iv(iv.get('pr_auc'))} | {_f(m['pr_auc_lift'], dec=2)} | "
                 f"{_f(m['precision_10'], True)} {_iv(iv.get('precision_10'), True)} | {_f(m['cobertura_10'], True)} | {_f(m['lift_10'], dec=2)} | "
                 f"{_f(m['brier_skill_vs_mejor_referencia'])} {_iv(iv.get('brier_skill_vs_mejor_referencia'))} | {m['episodios_independientes']} | {_iv(dif.get('pr_auc')) if dif else '—'} |")
    L += ['', 'Efecto de diseño (varianza del bootstrap por bloques de 12 meses ÷ varianza con meses independientes; > 1 = los meses no son independientes): ' +
          '; '.join(f"{r['nombre']} PR-AUC {_f((r['objetivos'][PRINCIPAL].get('efecto_de_diseno') or {}).get('pr_auc'), dec=1)}, precisión 10 % "
                    f"{_f((r['objetivos'][PRINCIPAL].get('efecto_de_diseno') or {}).get('precision_10'), dec=1)}" for r in principal)]
    L += ['', '## Por régimen (objetivo principal)', '', '| Señal | Régimen | Meses | Tasa base | PR-AUC | Precisión 10 % | Lift 10 % | Brier skill vs mejor ref. |', '|---|---|---|---|---|---|---|---|']
    for r in principal:
        for reg, m in r['objetivos'][PRINCIPAL]['por_regimen'].items():
            L.append(f"| {r['nombre']} | {reg} | {m['meses']} | {_f(m['tasa_base'], True)} | {_f(m['pr_auc'])} | {_f(m['precision_10'], True)} | {_f(m['lift_10'], dec=2)} | {_f(m['brier_skill_vs_mejor_referencia'])} |")
    L += ['', '## Sensibilidad: lift en el 10 % de mayor riesgo por objetivo', '', '| Señal | ' + ' | '.join(OBJETIVOS) + ' |', '|---|' + '---|' * len(OBJETIVOS)]
    for r in resultados:
        L.append(f"| {r['nombre']} | " + ' | '.join(f"{_f(r['objetivos'][o]['total']['lift_10'], dec=2)} (base {_f(r['objetivos'][o]['total']['tasa_base'], True)})"
                                                   if o in r['objetivos'] else '—' for o in OBJETIVOS) + ' |')
    anios = sorted({a['anio'] for r in principal for a in r['objetivos'][PRINCIPAL]['por_anio']})
    L += ['', '## Por año (objetivo principal): tasa base y lift en el 10 %', '', '| Año | Tasa base | ' + ' | '.join(r['nombre'] for r in principal) + ' |',
          '|---|---|' + '---|' * len(principal)]
    for a in anios:
        filas = [next((x for x in r['objetivos'][PRINCIPAL]['por_anio'] if x['anio'] == a), None) for r in principal]
        base = next((x['tasa_base'] for x in filas if x), None)
        L.append(f'| {a} | {_f(base, True)} | ' + ' | '.join(_f(x['lift_10'], dec=2) if x else '—' for x in filas) + ' |')
    # Calibración de TODAS las señales que la tengan (antes solo salía la de la regla compuesta, que no es el candidato; tanda 6, M34)
    for r in principal:
        cal = r['objetivos'][PRINCIPAL].get('calibracion') or []
        if cal:
            L += ['', f"## Calibración de {r['nombre']} (probabilidad calibrada con meses ya conocidos)", '',
                  '| Probabilidad | Casos | Media predicha | Frecuencia observada |', '|---|---|---|---|']
            L += [f"| {c['rango']} | {c['n']} | {_f(c['prob_media'], True)} | {_f(c['frecuencia_observada'], True)} |" for c in cal]
    # M39: descomposición de Murphy y Platt condicionado. Contesta si un Brier malo es de calibración (arreglable) o de resolución (no).
    filas_m, filas_r = [], []
    for r in resultados:
        m = r['objetivos'].get(PRINCIPAL, {}).get('total', {})
        for etiqueta, mu, bs in (('calibrado por decil', m.get('murphy'), m.get('brier_skill_vs_mejor_referencia')),
                                 ('Platt condicionado', m.get('murphy_platt'), m.get('brier_skill_platt_vs_mejor_referencia'))):
            if mu:
                filas_m.append(f"| {r['nombre']} | {etiqueta} | {_f(mu['confiabilidad'], dec=5)} | {_f(mu['resolucion'], dec=5)} | "
                               f"{_f(mu['incertidumbre'], dec=5)} | {_f(mu['brier'], dec=5)} | {_f(bs)} |")
        # POR RÉGIMEN (Mercado, M39 y R5): un promedio se come el estrés, que es donde la calibración se rompe
        for reg, mr in (r['objetivos'].get(PRINCIPAL, {}).get('por_regimen') or {}).items():
            mu = mr.get('murphy')
            if mu:
                filas_r.append(f"| {r['nombre']} | {reg} | {mr['meses']} | {_f(mu['confiabilidad'], dec=5)} | {_f(mu['resolucion'], dec=5)} | "
                               f"{_f(mu['incertidumbre'], dec=5)} | {_f(mr.get('brier_skill_vs_mejor_referencia'))} |")
    if filas_m:
        L += ['', '## Descomposición de Murphy del Brier (M39)', '',
              'Confiabilidad − resolución + incertidumbre = Brier del pronóstico agrupado en diez tramos. **Confiabilidad** baja es buena y '
              'se arregla recalibrando sin tocar el modelo; **resolución** alta es buena y NO se arregla calibrando, porque es lo que el '
              'modelo sabe distinguir; **incertidumbre** es la varianza de la tasa base y no depende de nadie. El Platt condicionado ajusta '
              'una logística sobre el puntaje continuo con solo los meses cuyo desenlace ya se conocía.', '',
              '| Señal | Probabilidad | Confiabilidad | Resolución | Incertidumbre | Brier | Brier skill |', '|---|---|---|---|---|---|---|'] + filas_m
    if filas_r:
        L += ['', '### La misma descomposición POR RÉGIMEN', '',
              'Pedida por Science Mercado: un promedio se come el estrés. Aquí se ve si una señal que pierde Brier skill en estrés lo '
              'pierde porque **deja de distinguir** (resolución baja: no tiene arreglo) o porque **dice mal la probabilidad** '
              '(confiabilidad alta: se arregla recalibrando).', '',
              '| Señal | Régimen | Meses | Confiabilidad | Resolución | Incertidumbre | Brier skill |', '|---|---|---|---|---|---|---|'] + filas_r
    cs = r0.get('cobertura_salidas', {})
    L += ['', '## Veracidad (dictamen de Science §5) y compuertas', '',
          f"- Periodo: {p.get('desde')} a {p.get('hasta')}. Universo: todas las emisoras del dataset con desenlace conocido; {cs.get('conclusion', '')}.",
          '- Fuera de muestra: las reglas base no se ajustan con datos; las probabilidades se calibran solo con meses cuyo desenlace ya se conocía.',
          '- Periodos adversos incluidos: ver la tabla por régimen (crisis 2008–09, 2011, 2020, 2022).',
          '- Error en la unidad del usuario: la tabla de calibración (probabilidad media predicha contra frecuencia observada, en puntos porcentuales); Brier y Brier skill como resumen técnico (≤ 0 = no aporta). Tasa de acierto: precisión en el 10 % junto a la tasa base (pedido 113 de Science Legal).',
          '- Una caída de una emisora que dejó de cotizar a media ventana puede no verse (ver limitaciones de dataset_meta.json).']
    for r in resultados:
        b = r['objetivos'].get(PRINCIPAL, {}).get('compuerta_publicacion')
        if b: L.append(f"- **{r['nombre']}**: {'publicable' if b['publicable'] else 'NO publicable'} (objetivo publicable: {'sí' if b['objetivo_publicable'] else 'no'}).")
    return '\n'.join(L) + '\n'


def guardar(resultados, carpeta, titulo, base_md):
    os.makedirs(carpeta, exist_ok=True)
    previos = len(glob.glob(os.path.join(carpeta, '*_caidas_*.json')))
    hoy = dt.date.today().isoformat(); rutas = []
    for r in resultados:
        r['evaluaciones_previas_de_caidas'] = previos
        limpio = EV.limpiar_nan({k: v for k, v in r.items() if k != '_muestras'})
        ruta = EV.ruta_libre(carpeta, f"{hoy}_caidas_{r['nombre']}", '.json')
        with open(ruta, 'w', encoding='utf-8') as fh: json.dump(limpio, fh, ensure_ascii=False, indent=1, allow_nan=False)
        rutas.append(ruta)
    ruta = EV.ruta_libre(carpeta, f'{hoy}_caidas_{base_md}', '.md')
    with open(ruta, 'w', encoding='utf-8') as fh: fh.write(reporte_md([EV.limpiar_nan({k: v for k, v in r.items() if k != '_muestras'}) for r in resultados], titulo, previos))
    return rutas + [ruta]


def puntaje_regla(tabla, nombre):
    variables, f, _ = REGLAS_BASE[nombre]
    if f is None: return np.random.default_rng(SEMILLA).standard_normal(len(tabla))
    return np.asarray(f(tabla), dtype=float)


def main(argv=None):
    p = argparse.ArgumentParser(description='Evaluación de señales de caída fuerte de Brota')
    p.add_argument('--dataset', default=os.path.join(AQUI, 'datos')); p.add_argument('--resultados', default=os.path.join(AQUI, 'resultados'))
    p.add_argument('--reglas-base', action='store_true'); p.add_argument('--puntajes'); p.add_argument('--nombre'); p.add_argument('--variables', default='')
    p.add_argument('--descripcion', default=''); p.add_argument('--ablacion', action='store_true'); p.add_argument('--probabilidades', action='store_true')
    p.add_argument('--bootstrap', type=int, default=BOOTSTRAP)
    p.add_argument('--desde', help='primera fecha evaluada (incluida)'); p.add_argument('--hasta', help='última fecha evaluada (excluida)')
    a = p.parse_args(argv)
    if not a.reglas_base and not a.puntajes: p.error('usa --reglas-base y/o --puntajes')
    if a.puntajes and not a.nombre: p.error('--puntajes requiere --nombre')
    vars_modelo = [v for v in a.variables.split(',') if v]
    cols = ['fecha', 'ticker', 'pais', 'sector', 'en_app', 'regimen'] + COLUMNAS_OBJETIVO + [c for v, _, _ in REGLAS_BASE.values() for c in v] + vars_modelo
    tabla, meta = EV.leer_dataset(a.dataset, cols)
    if 'min_ret_63' not in tabla: raise SystemExit('El dataset no tiene objetivos de caída: regenera con dataset.py (dataset-v2)')
    tabla['regimen'] = tabla['regimen'].astype(object)
    if a.desde: tabla = tabla[pd.to_datetime(tabla['fecha']) >= pd.Timestamp(a.desde)]
    if a.hasta: tabla = tabla[pd.to_datetime(tabla['fecha']) < pd.Timestamp(a.hasta)]   # --hasta: el pliegue externo no entra a la selección
    tabla = tabla.sort_values(['fecha', 'ticker']).reset_index(drop=True)
    resultados = []
    if a.puntajes:
        pz = pd.read_parquet(a.puntajes, columns=['fecha', 'ticker', 'puntaje']); pz['ticker'] = pz['ticker'].astype(str); pz['fecha'] = pd.to_datetime(pz['fecha'])
        j = tabla.merge(pz, on=['fecha', 'ticker'], how='left')
        resultados.append(evaluar(tabla, j['puntaje'], a.nombre, vars_modelo, meta, a.descripcion, a.ablacion, a.probabilidades, repeticiones=a.bootstrap))
    for nombre, (variables, _, desc) in REGLAS_BASE.items():
        resultados.append(evaluar(tabla, puntaje_regla(tabla, nombre), nombre, variables, meta, desc, repeticiones=a.bootstrap))
    comparar_contra(resultados)
    titulo = f'Caídas fuertes: {a.nombre} contra reglas base' if a.puntajes else 'Caídas fuertes: reglas base'
    rutas = guardar(resultados, a.resultados, titulo, a.nombre or 'reglas-base')
    for r in resultados:
        m = r['objetivos'].get(PRINCIPAL, {}).get('total', {})
        print(f"{r['nombre']:20s} base {_f(m.get('tasa_base'), True)}  PR-AUC {_f(m.get('pr_auc'))}  precisión 10 % {_f(m.get('precision_10'), True)}  "
              f"lift {_f(m.get('lift_10'), dec=2)}  Brier skill {_f(m.get('brier_skill_vs_mejor_referencia'))}")
    print('Escrito:', *[os.path.relpath(x, EV.RAIZ) for x in rutas], sep='\n  ')


if __name__ == '__main__':
    main()
