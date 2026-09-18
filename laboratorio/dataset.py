#!/usr/bin/env python3
"""
laboratorio/dataset.py — tabla de entrenamiento de la IA de Brota a partir del almacén (alexallonson/brota-datos).

Uso:
    pip install pandas pyarrow
    python3 laboratorio/dataset.py --almacen ../brota-datos                  # escribe laboratorio/datos/dataset/anio=AAAA/parte.parquet
    python3 laboratorio/dataset.py --almacen ../brota-datos --desde 2016-01-01

Una fila por (empresa, día hábil) con variables que SOLO usan lo que se sabía antes de ese día (LABORATORIO_IA.md §2.1 y pedidos 50–53):
    - precios del mismo día y anteriores (la decisión se toma al cierre del día D);
    - eventos, estados y macro con `conocido < D` (lo publicado durante D, p. ej. un reporte tras el cierre, entra hasta D+1).
Grupos (PEDIDOS_FABLE §4 y CAIDAS_FABLE §5): A precios (incluye las de caída: volatilidad idiosincrática, MAX, asimetría, run-up),
B reportes y P/E reconstruido, E macro (entra al modelo por interacción, pedido 59), R régimen del mercado por fecha (pedido 82).
Objetivos (en pesos, rendimiento total propio, pedidos 53 y 81): y_21/y_63 punto a punto, su diferencia contra la mediana del día y su
percentil; min_ret_21/63 por trayectoria (peor punto de la ventana); rendimiento del mercado propio y del sector; y los ajustados.
Los binarios de caída (p. ej. min_ret_63 ≤ −20 %) se derivan al entrenar: el umbral no se fija aquí.
Cada columna lleva grupo, fuente y si es publicable en dataset_meta.json (pedido 42 de Science): entrenar.py lo usa.
La tabla es un derivado: se regenera y no se versiona (laboratorio/datos/ está en .gitignore).
"""
import argparse, datetime as dt, json, os, subprocess, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fundamentales as FU  # noqa: E402

VERSION = 'dataset-v7'   # v7: liquidez del espejo del SIC (dias_sic_12m, importe_sic_20d) para topar el libro extranjero con la capacidad real de un mexicano
#          v5: rangos centrados y y_h_rango_libro (el rango / n daba 1.0 a la única emisora de un país)
DESDE = '2006-01-01'                       # decisión del dueño (2026-09-14): usar historia desde 2006
HORIZONTES = (21, 63)
RELLENO_MAX = 5                            # días que se arrastra un precio en feriados de su bolsa
LOOKBACK_DIAS = 800                        # run-up de 2 años (504 sesiones) antes de la primera fila
DIAS_ENTRE_REPORTES = 91                   # próximo reporte estimado = último reporte conocido + 91 días
ANTIGUEDAD_MAX_ESTADO = 550                # días: un estado más viejo (la emisora dejó de presentar) no se usa
SECTORES_FINANCIEROS = {'fin'}             # bancos y aseguradoras: apalancamiento, devengos y cobertura no significan lo mismo (provisional, M1)
GRUPO_D = ['apalancamiento_libros', 'pasivos_mercado', 'libros_mercado', 'caja_activos', 'roa_4t', 'margen_neto_4t', 'devengos_4t',
           'flujo_libre_activos', 'crec_activos_1a', 'crec_ingresos_1a', 'emision_neta_1a', 'cobertura_intereses_4t', 'patrimonio_negativo',
           'antiguedad_estado_dias']
GRUPO_D += ['nimta', 'cashmta', 'rentabilidad_bruta', 'log_capitalizacion', 'dd_ingenuo', 'cap_dudosa']
GRUPO_D_CON_PRECIO = {'pasivos_mercado', 'libros_mercado', 'nimta', 'cashmta', 'log_capitalizacion', 'dd_ingenuo', 'cap_dudosa'}
RANGO_SECTOR_D = ['apalancamiento_libros', 'pasivos_mercado', 'caja_activos', 'roa_4t', 'devengos_4t', 'rentabilidad_bruta']   # pedido 109
# Rendimiento de deslistamiento (Shumway, 1997; pedido 101) según la causa de salida de universo: deterioro −30 %, oferta o fusión 0 %
DESLISTAMIENTO = [(r'concurso|quiebra|suspensi|incumpl', -0.30), (r'oferta|opa\b|adquisici|fusi|voluntari|privatiz', 0.0)]
GRUPO_L = ['importe_20d', 'log_importe_20d', 'amihud_60d', 'prop_ceros_60d', 'dias_sin_volumen_20d', 'dias_sin_dato_20d', 'medio_dif_ar_60d',
           'dias_sic_12m', 'importe_sic_20d']
SIC_SUFIJO = '.MX'                         # el espejo en pesos de una extranjera: AAPL -> AAPL.MX (mismo sufijo que almacen.py)
AUDITORIA = {}                             # diagnósticos que se calculan al construir y se escriben en dataset_meta.json
# Régimen del mercado conocido en D (pedido 82): VIX de FRED (conocido < D) y caída del IPC y del S&P 500 desde su máximo de 52 semanas
# Tramos de precio con error evidente de la fuente, revisados a mano (el almacén no se corrige: se excluyen al armar la tabla)
TRAMOS_EXCLUIDOS = [
    {'ticker': 'URBI.MX', 'desde': '2024-05-01', 'hasta': None,
     'motivo': 'en concurso y casi sin volumen: +970 % en un día (2024-05-10) y un «dividendo» de 111.88 sobre un precio de 110.50 (2024-05-29)'},
]
SALTO_EXTREMO = (0.80, -0.60)              # diagnóstico: movimientos diarios > +80 % o < −60 % se listan en dataset_meta.json
REGIMEN = {'crisis': {'vix': 30, 'caida': -0.20}, 'estres': {'vix': 22, 'caida': -0.10}}
FX = {'USD': 'MXN=X', 'EUR': 'EURMXN=X', 'CHF': 'CHFMXN=X', 'GBP': 'GBPMXN=X', 'GBp': 'GBPMXN=X', 'JPY': 'JPYMXN=X'}
SUBUNIDAD = {'GBp': 100.0}                 # precio en peniques, utilidades en libras
MACRO = {   # columna: (conjunto, serie, descripción, publicable)
    'cetes28': ('macro_banxico', 'SF60633', 'CETES 28 días (%)', True),
    'tasa_objetivo': ('macro_banxico', 'SF61745', 'Tasa objetivo de Banxico (%)', True),
    'inflacion_mx': ('macro_banxico', 'SP30578', 'Inflación general anual de México (%)', True),
    'igae_desest': ('macro_inegi', '737219', 'IGAE desestacionalizado (índice)', True),
    'confianza_mx': ('macro_inegi', '454168', 'Confianza del consumidor (México)', True),
    'tesoro_10a': ('macro_fred', 'DGS10', 'Tesoro EE. UU. 10 años (%)', True),
    'curva_10a_2a': ('macro_fred', 'T10Y2Y', 'Diferencia 10 años − 2 años (%)', True),
    'inflacion_esperada_us': ('macro_fred', 'T5YIE', 'Inflación esperada a 5 años EE. UU. (%)', True),
    'petroleo_wti': ('macro_fred', 'DCOILWTICO', 'Petróleo WTI (dólares)', True),
    'vix': ('macro_fred', 'VIXCLS', 'VIX (Cboe)', False),
}
FUENTE_PUBLICABLE = {'yahoo': False, 'banxico': True, 'inegi': True, 'fred': True, 'brota': True, 'calculado': True}


def leer(almacen, sub, columnas=None, filtro=None):
    import pyarrow.dataset as ds
    ruta = os.path.join(almacen, sub)
    if not os.path.exists(ruta): return None
    return ds.dataset(ruta, format='parquet', partitioning='hive').to_table(columns=columnas, filter=filtro).to_pandas()


# ================================================================ precios: rendimiento total propio y moneda
def indice_total(g):
    """Rendimiento total por día desde close y dividendo (Yahoo ajusta close y dividendos por splits): (close_t + div_t) / close_{t-1}."""
    c, d = g['close'].to_numpy(float), g['dividendo'].fillna(0).to_numpy(float)
    r = np.ones(len(c)); r[1:] = (c[1:] + d[1:]) / c[:-1]
    return pd.Series(np.cumprod(r), index=g.index)


def matrices(precios, desde_lookback):
    """Tablas anchas (fecha × ticker) sobre el calendario unión: close, máximo, mínimo, indice total, volumen, dividendo, y la máscara de días reales."""
    p = precios[precios['fecha'] >= desde_lookback].sort_values(['ticker', 'fecha']).copy()
    p['it'] = p.groupby('ticker', group_keys=False).apply(indice_total)
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(p.loc[p['tipo'] == 'empresa', 'fecha'].unique())))
    def ancha(col, rellenar=True):
        w = p.pivot(index='fecha', columns='ticker', values=col); w.index = pd.to_datetime(w.index); w = w.reindex(cal)
        return w.ffill(limit=RELLENO_MAX) if rellenar else w
    salida = {'close': ancha('close'), 'it': ancha('it'), 'vol': ancha('volumen', False).fillna(0), 'div': ancha('dividendo', False).fillna(0),
              'real': ancha('close', False).notna(), 'cal': cal}
    # máximo y mínimo del día SIN arrastrar: el estimador del diferencial necesita el rango de un día real, no el repetido de un feriado
    for col in ('high', 'low'):
        if col in p: salida[col] = ancha(col, False)
    return salida


def en_pesos(it, monedas, fx_close):
    """Índice total en pesos = índice nativo × tipo de cambio del día (la constante de subunidad no cambia rendimientos)."""
    salida = it.copy()
    for t in it.columns:
        par = FX.get(monedas.get(t))
        if par and par in fx_close: salida[t] = it[t] * fx_close[par]
    return salida


def rend_dias_reales(nivel, real):
    """Rendimiento desde el día real previo de cada emisora, solo en sus días reales (tanda 4, M15). En el calendario unión un feriado de
    su bolsa daba rendimiento 0 y bajaba volatilidades y betas de las extranjeras por el calendario, no por su riesgo. nivel: tabla ancha
    (emisoras) o serie (un índice: su rendimiento sobre el mismo tramo de cada emisora). Hueco mayor que el arrastre: nulo."""
    rv = real.to_numpy(bool); T, N = rv.shape
    pos = np.maximum.accumulate(np.where(rv, np.arange(T)[:, None], -1), axis=0)
    previo = np.vstack([np.full((1, N), -1), pos[:-1]])
    ok = rv & (previo >= 0) & (np.arange(T)[:, None] - previo <= RELLENO_MAX + 1)
    x = nivel.to_numpy(float)
    ahora = x if x.ndim == 2 else np.broadcast_to(x[:, None], (T, N))
    antes = np.take_along_axis(ahora, np.clip(previo, 0, None), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(ok, ahora / antes - 1, np.nan)
    return pd.DataFrame(r, index=real.index, columns=real.columns)


def momentos_pares(a, b, n, minimo):
    """Covarianza de a con b y varianza de b en ventanas móviles usando solo pares completos (ambos con dato), ddof = 1."""
    ok = a.notna() & b.notna()
    a, b = a.where(ok), b.where(ok)
    cnt = ok.astype(float).rolling(n, min_periods=1).sum()
    sa, sb = a.rolling(n, min_periods=1).sum(), b.rolling(n, min_periods=1).sum()
    sab, sbb = (a * b).rolling(n, min_periods=1).sum(), (b * b).rolling(n, min_periods=1).sum()
    valido = cnt >= minimo
    cov = ((sab - sa * sb / cnt) / (cnt - 1)).where(valido)
    var_b = ((sbb - sb * sb / cnt) / (cnt - 1)).where(valido)
    return cov, var_b


# ================================================================ grupo A
def grupo_a(m, it_mxn, indices):
    close, it, vol, real = m['close'], m['it'], m['vol'], m['real']
    r = rend_dias_reales(it, real); r_mxn = rend_dias_reales(it_mxn, real)
    ret = lambda x, n: x / x.shift(n) - 1
    f = {
        'ret_1m': ret(it, 21), 'ret_3m': ret(it, 63), 'ret_6m': ret(it, 126), 'ret_12m_ex1m': it.shift(21) / it.shift(252) - 1,
        'ret_mxn_1m': ret(it_mxn, 21), 'ret_mxn_6m': ret(it_mxn, 126), 'ret_mxn_12m_ex1m': it_mxn.shift(21) / it_mxn.shift(252) - 1,
        'dist_ma200': close / close.rolling(200, min_periods=150).mean() - 1,
        'dist_max_52s': close / close.rolling(252, min_periods=200).max() - 1,
        'vol_60d': r.rolling(60, min_periods=40).std() * np.sqrt(252),
        'vol_250d': r.rolling(250, min_periods=200).std() * np.sqrt(252),
        'max_dd_1a': close / close.rolling(252, min_periods=200).max() - 1,
        'vol_relativa': vol.rolling(20, min_periods=15).mean() / vol.rolling(250, min_periods=200).mean(),   # volumen anormal: Gervais, Kaniel y Mingelgrin (2001)
        'rend_dividendo_12m': m['div'].rolling(252, min_periods=1).sum() / close,
    }
    # máxima caída dentro del año (no solo contra el máximo actual): peor caída desde un máximo previo en 252 días
    # (antes usaba cummax desde el inicio del lookback: medía distancia al máximo de la muestra y dependía de la fecha `desde`; pedido 103)
    pico = close.rolling(252, min_periods=200).max()
    f['max_dd_1a'] = (close / pico - 1).rolling(252, min_periods=200).min()
    for nombre, serie in indices.items():
        if nombre == 'usdmxn': continue   # la exposición al dólar la mide corr_fx_1a
        cov, var = momentos_pares(r_mxn, rend_dias_reales(serie, real), 252, 200)
        f[f'beta_{nombre}_1a'] = cov / var
    if 'usdmxn' in indices:
        rfx = rend_dias_reales(indices['usdmxn'], real)
        cov, var_fx = momentos_pares(r, rfx, 252, 200)
        _, var_r = momentos_pares(rfx, r, 252, 200)
        f['corr_fx_1a'] = cov / np.sqrt(var_r * var_fx)
    f['vol_relativa'] = f['vol_relativa'].replace([np.inf, -np.inf], np.nan)
    return f


def grupo_caida(m, it, it_mxn, close_mxn, indices, es_mx, fa):
    """CAIDAS_FABLE §5.1: variables de precio con respaldo para caídas. El mercado propio es el IPC para emisoras mexicanas y el S&P 500
    en pesos para el resto (misma regla que beta_mercado_1a y que ret_mercado_h)."""
    close, real = m['close'], m['real']
    r = rend_dias_reales(it, real); r_mxn = rend_dias_reales(it_mxn, real)
    f = {
        'max_ret_1m': r.rolling(21, min_periods=15).max(),                       # Bali, Cakici y Whitelaw (2011)
        'min_ret_1m': r.rolling(21, min_periods=15).min(),
        'asimetria_6m': r.rolling(126, min_periods=100).skew(),                  # asimetría realizada: Amaya, Christoffersen, Jacobs y Vainberg (2015)
        'runup_1a': it / it.shift(252) - 1,                                      # Greenwood, Shleifer y You (2019) lo definen por industria; aquí por emisora
        'runup_2a': it / it.shift(504) - 1,
        'aceleracion': (it / it.shift(63) - 1) - (it.shift(63) / it.shift(126) - 1),
        'dist_ma50': close / close.rolling(50, min_periods=40).mean() - 1,
        'log_precio_mxn': np.log(close_mxn.where(close_mxn > 0)),                # Campbell, Hilscher y Szilagyi (2008): precio bajo
    }
    ivol = {}
    for nombre in ('ipc', 'spx'):
        if nombre not in indices: continue
        cov, var_m = momentos_pares(r_mxn, rend_dias_reales(indices[nombre], real), 60, 40)
        _, var_r = momentos_pares(rend_dias_reales(indices[nombre], real), r_mxn, 60, 40)
        # varianza del residuo de UN factor = var(r) − cov² / var(m), anualizada (en el espíritu de Ang, Hodrick, Xing y Zhang, 2006, que usan tres)
        ivol[nombre] = (var_r - cov.pow(2) / var_m).clip(lower=0).pow(0.5) * np.sqrt(252)
    def elegir(por_indice):
        return pd.DataFrame({c: por_indice['ipc' if es_mx.get(c) else 'spx'][c] if ('ipc' if es_mx.get(c) else 'spx') in por_indice else np.nan
                             for c in close.columns}, index=close.index)
    f['ivol_60d'] = elegir(ivol)
    f['beta_mercado_1a'] = elegir({k: fa[f'beta_{k}_1a'] for k in ('ipc', 'spx') if f'beta_{k}_1a' in fa})
    return f


def valor_previo_real(x, real):
    """Valor de x en el día real ANTERIOR de cada emisora (nulo si el hueco excede el arrastre). Misma aritmética que `rend_dias_reales`."""
    rv = real.to_numpy(bool); T, N = rv.shape
    fila = np.arange(T)[:, None]
    pos = np.maximum.accumulate(np.where(rv, fila, -1), axis=0)     # último día real ≤ t
    previo = np.vstack([np.full((1, N), -1), pos[:-1]])             # último día real < t
    ok = rv & (previo >= 0) & (fila - previo <= RELLENO_MAX + 1)
    tomado = np.take_along_axis(x.to_numpy(float), np.clip(previo, 0, None), axis=0)
    return pd.DataFrame(np.where(ok, tomado, np.nan), index=real.index, columns=real.columns)


def medio_diferencial(m, ventana=60, minimo=30):
    """Medio diferencial de compra-venta proporcional, estimador de Abdi y Ranaldo (2017), «A simple estimation of bid-ask spreads from daily
    close, high, and low prices», RFS 30(12). Con c = log(cierre) y η = (log máximo + log mínimo)/2:

        S = sqrt( max( 4 · E[(c_t − η_t)(c_t − η_{t+1})], 0 ) )      y el medio diferencial es S/2.

    t y t+1 son días REALES consecutivos de la emisora (pedido 99 / 184). PUNTO EN EL TIEMPO: el par (t, t+1) solo se conoce en t+1, así que
    cada producto se atribuye al SEGUNDO de sus dos días; el último par que entra al valor del día D es (D−1, D). Escrito de la forma
    directa —atribuyendo el producto a t— la variable del día D usaba el máximo y el mínimo del día siguiente, y la prueba de futuro
    alterado lo detectó. El estimador es el que menos supuestos pide de los que usan solo
    OHLC —no necesita volumen ni suponer que el precio no tiene tendencia dentro del día, como Corwin-Schultz (2012)— y en muestras con
    días sin operar sesga a la baja, no al alza: el costo que se cobra en la evaluación es conservador por el lado que importa."""
    if 'high' not in m or 'low' not in m: return {}
    # se excluyen los días SIN rango intradía (máximo = mínimo): ahí η = c por construcción y el producto vale 0, lo que arrastra el
    # estimador hacia abajo justo en las emisoras que menos operan. Eran el 32 % de los días mexicanos de 2006-2007 y el 4 % en 2013.
    real = m['real'] & (m['low'] > 0) & (m['high'] > m['low']) & (m['close'] > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        eta = (np.log(m['high'].where(real)) + np.log(m['low'].where(real))) / 2
        c = np.log(m['close'].where(real))
    c_ant, eta_ant = valor_previo_real(c, real), valor_previo_real(eta, real)
    s2 = 4 * (c_ant - eta_ant) * (c_ant - eta)      # el par (día real anterior, hoy): conocido al cierre de hoy
    media = s2.rolling(ventana, min_periods=minimo).mean()
    return {f'medio_dif_ar_{ventana}d': np.sqrt(media.clip(lower=0)) / 2}


def grupo_liquidez(m, close_mxn):
    """Grupo L (pedido 95 de Science Mercado): liquidez y precios rancios con datos ≤ D. El importe va en pesos para comparar emisoras.
    Los ceros se cuentan solo en días en que la emisora tuvo dato (un feriado de su bolsa no es precio rancio)."""
    real, vol, close = m['real'], m['vol'], m['close']
    importe = (close_mxn * vol).where(real)
    r = close.pct_change(fill_method=None).where(real)
    n60 = real.rolling(60, min_periods=1).sum()
    f = {
        # media sobre los días con dato (un día sin volumen cuenta como 0; un feriado de su bolsa no cuenta): la mediana degenera a 0 en la cola ilíquida
        'importe_20d': importe.rolling(20, min_periods=10).mean(),
        'amihud_60d': (r.abs() / importe.where(importe > 0) * 1e6).rolling(60, min_periods=30).mean(),   # Amihud (2002), por millón de pesos
        'prop_ceros_60d': ((r == 0) & real).rolling(60, min_periods=30).sum() / n60.where(n60 >= 30),     # Lesmond, Ogden y Trzcinka (1999); emergentes: Lesmond (2005)
        'dias_sin_volumen_20d': ((vol == 0) & real).rolling(20, min_periods=10).sum(),
        'dias_sin_dato_20d': (~real).rolling(20, min_periods=10).sum(),
    }
    f['log_importe_20d'] = np.log(f['importe_20d'].where(f['importe_20d'] > 0))
    f.update(medio_diferencial(m))
    return f


def liquidez_sic(m, columnas, ventana_dias=252, ventana_importe=20):
    """Liquidez del ESPEJO del SIC, atribuida a la emisora extranjera (pedido AT de Fable, `REGLA_TOPE_SIC.md`).

    Un mexicano compra las extranjeras en el SIC, no en su bolsa de origen, así que la capacidad que de verdad tiene es la del
    espejo. Medido el 2026-09-18: el importe de origen sobreestima la del SIC por una mediana de 24,922 veces.

    Las dos miran solo hacia atrás sobre datos conocidos al cierre de D, igual que `importe_20d`. Van NULAS —no cero— para las
    mexicanas y para las 15 extranjeras sin espejo: cero afirmaría que no se puede comprar, y lo que pasa es que no se sabe."""
    close, vol, real = m['close'], m['vol'], m['real']
    alto, bajo = m.get('high'), m.get('low')
    dias = pd.DataFrame(np.nan, index=close.index, columns=columnas)
    imp = pd.DataFrame(np.nan, index=close.index, columns=columnas)
    for t in columnas:
        espejo = f'{t}{SIC_SUFIJO}'
        if espejo not in close.columns: continue
        c, v = close[espejo], vol[espejo]
        opero = (v > 0) & real[espejo]
        if alto is not None and bajo is not None and espejo in alto:
            opero &= (alto[espejo] > bajo[espejo])          # un día con máximo igual a mínimo no es un día operado
        dias[t] = opero.rolling(ventana_dias, min_periods=60).mean()
        imp[t] = (c * v).where(opero).rolling(ventana_importe, min_periods=5).mean()   # ya viene en pesos: el SIC cotiza en pesos
    return {'dias_sic_12m': dias, 'importe_sic_20d': imp}


def auditoria_arrastre(m, pais):
    """Pedido 97: celdas del calendario unión con precio arrastrado (dentro de la vida de cada emisora) y días con dato pero sin volumen."""
    real, close = m['real'], m['close']
    vivo = real.cummax() & real[::-1].cummax()[::-1]
    arrastrado = close.notna() & ~real & vivo
    sin_vol = real & (m['vol'] == 0)
    anio = pd.Index(m['cal'].year, name='anio')
    salida = {}
    for nombre, x in (('celdas_arrastradas', arrastrado), ('dias_con_dato_sin_volumen', sin_vol)):
        por = x.groupby(anio).sum().T.groupby(lambda t: 'MX' if pais.get(t) == 'MX' else 'extranjero').sum().T
        salida[nombre] = {str(a): {k: int(v) for k, v in fila.items()} for a, fila in por.iterrows()}
    salida['dias_con_dato'] = {str(a): {k: int(v) for k, v in fila.items()} for a, fila in
                               real.groupby(anio).sum().T.groupby(lambda t: 'MX' if pais.get(t) == 'MX' else 'extranjero').sum().T.iterrows()}
    return salida


def regimen_mercado(close, macro):
    """Por fecha: caída del IPC y del S&P 500 (moneda propia) desde su máximo de 52 semanas con precios ≤ D, y VIX conocido antes de D."""
    salida = pd.DataFrame(index=close.index)
    for t, col in (('^MXX', 'dd_ipc_52s'), ('^GSPC', 'dd_spx_52s')):
        salida[col] = close[t] / close[t].rolling(252, min_periods=200).max() - 1 if t in close else np.nan
    vix = macro['vix'] if 'vix' in macro else pd.Series(np.nan, index=close.index)
    peor = np.fmin(salida['dd_ipc_52s'].to_numpy(float), salida['dd_spx_52s'].to_numpy(float))
    v = vix.to_numpy(float)
    with np.errstate(invalid='ignore'):
        crisis = (v > REGIMEN['crisis']['vix']) | (peor <= REGIMEN['crisis']['caida'])
        estres = ~crisis & ((v > REGIMEN['estres']['vix']) | (peor <= REGIMEN['estres']['caida']))
    reg = np.where(crisis, 'crisis', np.where(estres, 'estres', 'tranquilo')).astype(object)
    reg[np.isnan(v) & np.isnan(peor)] = None
    salida['regimen'] = reg
    return salida


def hacia_adelante_min(w, h):
    """Mínimo de w en las sesiones t+1 … t+h (peor punto de la trayectoria). Nulo si la ventana no cabe en el calendario; si la emisora
    deja de cotizar a media ventana se usa lo que hubo (con al menos 80 % de la ventana)."""
    adelante = w.shift(-1).iloc[::-1].rolling(h, min_periods=int(h * 0.8)).min().iloc[::-1]
    adelante.iloc[max(len(w) - h, 0):] = np.nan
    return adelante


# ================================================================ grupo B
def eventos_resultados(res, splits=None):
    """Un renglón por reporte OCURRIDO (no estimado), con su primera versión que trae utilidad reportada, y estadísticos acumulados.
    La utilidad por acción se pasa a la base de acciones de HOY (la de los precios de Yahoo): eps × I(capturado) / I(total). Yahoo reexpresa
    el EPS histórico con los splits hasta que se baja (verificado en NVDA, AAPL, AMZN, TSLA, WMT y NFLX), pero el almacén solo agrega:
    un renglón bajado antes de un split queda en la base vieja. Sin esto el P/E mezclaría bases (tanda 4, M21)."""
    if res is None or not len(res): return None
    e = res[(res['estimada'] == False) & res['eps_reportado'].notna()].copy()   # noqa: E712
    e['fecha_reporte'] = pd.to_datetime(e['fecha_reporte'], utc=True).astype('datetime64[ns, UTC]'); e['conocido'] = pd.to_datetime(e['conocido'], utc=True).astype('datetime64[ns, UTC]')
    e = e.sort_values(['ticker', 'fecha_reporte', 'conocido']).drop_duplicates(['ticker', 'fecha_reporte'], keep='first')
    if splits is not None and 'capturado' in e:
        total = e['ticker'].map(splits.groupby('ticker')['indice'].last()).fillna(1.0).to_numpy(float)
        capt = pd.to_datetime(e['capturado'], utc=True).dt.tz_localize(None).dt.normalize()
        e['eps_reportado'] = e['eps_reportado'].to_numpy(float) * indice_split_en(e['ticker'], capt, splits) / total
    salida = []
    for t, g in e.groupby('ticker', sort=False):
        g = g.sort_values('fecha_reporte').copy()
        eps = g['eps_reportado']
        g['eps_4t'] = eps.rolling(4, min_periods=4).sum()
        g['eps_4t_previo'] = g['eps_4t'].shift(4)
        g['sorpresa_ult'] = g['sorpresa_pct']
        g['sorpresa_prom_4'] = g['sorpresa_pct'].rolling(4, min_periods=2).mean()
        pos = (g['sorpresa_pct'] > 0).astype(int)
        g['racha_sorpresas'] = pos.groupby((pos == 0).cumsum()).cumsum()
        g['n_sorpresas_neg_4'] = (g['sorpresa_pct'] < 0).astype(float).rolling(4, min_periods=1).sum()
        # SUE (Foster, Olsen y Shevlin, 1984): camino aleatorio estacional CON deriva. Esperado = EPS de hace 4 trimestres + deriva (media de
        # los 8 cambios estacionales previos); escala = su desviación. Sin la deriva, una empresa en tendencia tenía «sorpresa» siempre positiva
        cambio = eps - eps.shift(4)
        previos = cambio.shift(1).rolling(8, min_periods=4)
        g['sue'] = (cambio - previos.mean()) / previos.std()
        salida.append(g)
    return pd.concat(salida, ignore_index=True)[['ticker', 'fecha_reporte', 'conocido', 'eps_4t', 'eps_4t_previo', 'sorpresa_ult', 'sorpresa_prom_4', 'racha_sorpresas', 'n_sorpresas_neg_4', 'sue']]


def grupo_b(filas, ev, fibras, precio_utilidad, sin_pe=()):
    """merge_asof estricto: para la fila del día D solo reportes con conocido < D (00:00 UTC)."""
    if ev is None:
        for c in ('sorpresa_ult', 'sorpresa_prom_4', 'racha_sorpresas', 'n_sorpresas_neg_4', 'sue', 'dias_desde_reporte', 'dias_al_proximo_est', 'ventana_reporte', 'reporte_atrasado',
                  'crec_eps_4t', 'pe_reconstruido', 'sin_utilidades'): filas[c] = np.nan
        return filas
    filas = filas.sort_values('ts')
    ev = ev.sort_values('conocido')
    j = pd.merge_asof(filas, ev, left_on='ts', right_on='conocido', by='ticker', direction='backward', allow_exact_matches=False)
    j['dias_desde_reporte'] = (j['ts'] - j['fecha_reporte']).dt.days
    # riesgo de evento (CAIDAS_FABLE §5.2): días al próximo reporte estimado con lo sabido en D; 0 si ya debió ocurrir
    j['dias_al_proximo_est'] = (j['fecha_reporte'] + pd.Timedelta(days=DIAS_ENTRE_REPORTES) - j['ts']).dt.days
    j['reporte_atrasado'] = np.where(j['fecha_reporte'].notna(), (j['dias_al_proximo_est'] < 0).astype(float), np.nan)   # pedido 107: retraso ≠ reporte mañana
    j['dias_al_proximo_est'] = j['dias_al_proximo_est'].clip(lower=0)
    j['ventana_reporte'] = np.where(j['fecha_reporte'].notna(), ((j['dias_al_proximo_est'] <= 10) | (j['dias_desde_reporte'] <= 5)).astype(float), np.nan)
    j['crec_eps_4t'] = np.where((j['eps_4t_previo'] > 0) & j['eps_4t'].notna(), j['eps_4t'] / j['eps_4t_previo'] - 1, np.nan)
    j['sin_utilidades'] = np.where(j['eps_4t'].notna(), (j['eps_4t'] <= 0).astype(float), np.nan)
    es_fibra = j['ticker'].isin(fibras)
    # P/E = precio / suma de las 4 utilidades por acción REPORTADAS (ajustadas, no GAAP). Vacío si hay pérdida, FIBRA o moneda distinta
    j['pe_reconstruido'] = np.where((j['eps_4t'] > 0) & ~es_fibra & ~j['ticker'].isin(sin_pe), j[precio_utilidad] / j['eps_4t'], np.nan)
    return j.drop(columns=['fecha_reporte', 'conocido', 'eps_4t', 'eps_4t_previo'])


def monedas_distintas(almacen, monedas):
    """Emisoras cuya moneda de reporte (financialCurrency de Yahoo) no es la del precio: su P/E reconstruido mezclaría monedas
    (p. ej. Novo Nordisk: utilidades en coronas, ADR en dólares). Es un atributo de la emisora, no un dato que cambie con el tiempo."""
    fotos = leer(almacen, 'fotos/valuacion', None)
    if fotos is None or 'financialCurrency' not in fotos: return set()
    fc = fotos.sort_values('capturado').drop_duplicates('ticker', keep='last').set_index('ticker')['financialCurrency'].dropna()
    base = lambda m: {'GBp': 'GBP'}.get(m, m)
    return {t for t, c in fc.items() if monedas.get(t) and base(monedas[t]) != c}


# ================================================================ grupo D: estados financieros de la SEC
def splits_acumulados(precios):
    """Índice acumulado de splits por emisora (producto de las razones hasta cada fecha): acciones de d1 / d0 atribuible a splits = I(d1)/I(d0)."""
    s = precios.loc[precios['split'].fillna(0) > 0, ['ticker', 'fecha', 'split']].copy() if 'split' in precios else precios.iloc[0:0]
    if not len(s): return None
    s['fecha'] = pd.to_datetime(s['fecha']).astype('datetime64[ns]'); s['ticker'] = s['ticker'].astype(str)   # pandas 3: misma resolución en merge_asof
    s = s.sort_values(['ticker', 'fecha'])
    s['indice'] = s.groupby('ticker')['split'].cumprod()
    return s[['ticker', 'fecha', 'indice']]


def factor_split_matriz(splits, cal, columnas):
    """Por día y emisora: I(total) / I(día). Multiplicar el cierre que Yahoo reexpresa con splits POSTERIORES por este factor devuelve el
    precio que se veía ese día (encontrado por la prueba de propiedades: sin esto, un split futuro cambia el nivel de precio del pasado)."""
    f = pd.DataFrame(1.0, index=cal, columns=columnas)
    if splits is None: return f
    for t, g in splits.groupby('ticker'):
        if t not in f.columns: continue
        idx = g.set_index('fecha')['indice'].astype(float)
        vigente = idx.reindex(idx.index.union(cal)).sort_index().ffill().reindex(cal).fillna(1.0)
        f[t] = float(idx.iloc[-1]) / vigente
    return f


def indice_split_en(tickers, fechas, splits):
    """Índice de splits vigente en cada (ticker, fecha); 1 antes del primer split."""
    salida = np.ones(len(tickers))
    if splits is None: return salida
    q = pd.DataFrame({'ticker': np.asarray(tickers).astype(str), 'fecha': pd.to_datetime(pd.Series(fechas).to_numpy()).astype('datetime64[ns]'),
                      'pos': np.arange(len(tickers))})
    ok = q['fecha'].notna()
    j = pd.merge_asof(q[ok].sort_values('fecha'), splits.assign(fecha=splits['fecha'].astype('datetime64[ns]')).sort_values('fecha'), on='fecha', by='ticker',
                      direction='backward')
    salida[j['pos'].to_numpy()] = j['indice'].fillna(1.0).to_numpy()
    return salida


def grupo_d(filas, almacen, splits, monedas):
    """Estados punto en el tiempo (conocido < D) y razones de quiebra, opacidad contable, crecimiento y emisión (CAIDAS_FABLE §5.3)."""
    sec = leer(almacen, 'fundamentales/sec')
    e = FU.estados_por_presentacion(sec)
    if e is None:
        for c in GRUPO_D: filas[c] = np.nan
        return filas
    e['conocido'] = pd.to_datetime(e['conocido'], utc=True).astype('datetime64[ns, UTC]')
    filas = filas.sort_values('ts')
    j = pd.merge_asof(filas, e.sort_values('conocido'), left_on='ts', right_on='conocido', by='ticker', direction='backward', allow_exact_matches=False)
    # precio comparable con los estados: solo emisoras listadas en EE. UU. con su propio símbolo y estados en la moneda de su precio
    base = lambda m: {'GBp': 'GBP'}.get(m, m)
    comparable = (j['simbolo_sec'] == j['ticker']) & (j['moneda_estados'] == j['ticker'].map(lambda t: base(monedas.get(t))))
    # Yahoo reexpresa TODO el historial de precios con los splits hasta hoy; las acciones de la portada están en la base de su fecha.
    # Capitalización real = acciones de portada × precio ajustado × I(total) / I(fecha de portada). No agrega información nueva: reconstruye el precio que se veía.
    total = splits.groupby('ticker')['indice'].last() if splits is not None else pd.Series(dtype=float)
    ajuste_split = j['ticker'].map(total).fillna(1.0).to_numpy() / indice_split_en(j['ticker'], j['fin_acciones'], splits)
    precio = np.where(comparable, j['precio_para_pe'] * ajuste_split, np.nan)
    r = FU.razones(j, j['fecha'], precio, j['sector'].isin(SECTORES_FINANCIEROS).to_numpy())
    # distancia al incumplimiento ingenua (Bharath y Shumway, 2008): E capitalización, F pasivo circulante + ½ deuda de largo plazo, σE de 1 año
    E, F = r['_capitalizacion'], r['_deuda_incumplimiento']
    sE = j['vol_250d'].to_numpy(float); ren = j['runup_1a'].to_numpy(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        w = E / (E + F)
        sV = w * sE + (1 - w) * (0.05 + 0.25 * sE)
        r['dd_ingenuo'] = np.where((F > 0) & (E > 0) & (sV > 0), (np.log((E + F) / F) + ren - 0.5 * sV ** 2) / sV, np.nan)
    viejo = ~(r['antiguedad_estado_dias'] <= ANTIGUEDAD_MAX_ESTADO)
    for c in GRUPO_D:
        j[c] = np.where(viejo, np.nan, r[c])
    for c in RANGO_SECTOR_D:   # comparar apalancamiento o rentabilidad entre sectores no tiene sentido: rango dentro del sector del día
        j[f'{c}_rango_sector'] = j.groupby(['fecha', 'sector'], observed=True)[c].rank(pct=True)
    return j[list(filas.columns) + GRUPO_D + [f'{c}_rango_sector' for c in RANGO_SECTOR_D]]


# ================================================================ relativos al sector sin la propia emisora (pedido 104)
def media_sin_propia(largo, c, grupos=('fecha', 'sector')):
    g = largo.groupby(list(grupos), observed=True)[c]
    suma, n = g.transform('sum'), g.transform('count')
    return np.where(largo[c].notna() & (n > 1), (suma - largo[c].fillna(0)) / (n - 1), np.nan)


def mediana_sin_propia(valores, claves):
    """Mediana de las DEMÁS observaciones no nulas del grupo, por renglón (vectorizada). claves: DataFrame con las columnas de grupo."""
    v = pd.Series(np.asarray(valores, float)); ok = v.notna().to_numpy()
    salida = np.full(len(v), np.nan)
    if not ok.any(): return salida
    d = claves[ok].reset_index(drop=True).assign(v=v[ok].to_numpy(), pos=np.nonzero(ok)[0])
    d = d.sort_values(list(claves.columns) + ['v'], kind='mergesort').reset_index(drop=True)
    gid = d.groupby(list(claves.columns), observed=True, sort=False).ngroup().to_numpy()
    rnk = d.groupby(gid).cumcount().to_numpy(); n = np.bincount(gid)[gid]
    inicio = np.r_[0, np.cumsum(np.bincount(gid))][:-1][gid]
    s = d['v'].to_numpy(); m = n - 1
    def t(jj):   # elemento jj (0..m−1) de la lista del grupo sin el propio renglón; índices acotados (los renglones con m = 0 se descartan abajo)
        jj = np.maximum(jj, 0)
        return s[np.minimum(np.where(jj < rnk, inicio + jj, inicio + jj + 1), len(s) - 1)]
    med = np.where(m % 2 == 1, t((m - 1) // 2), (t(m // 2 - 1) + t(m // 2)) / 2)
    salida[d['pos'].to_numpy()] = np.where(m >= 1, med, np.nan)
    return salida


# ================================================================ reacción al reporte (pedido 98)
def eventos_car(ev, it_mxn, indices, es_mx):
    """CAR de 3 sesiones alrededor de cada reporte: de la sesión anterior al día del reporte a dos sesiones después, contra el mercado propio
    en pesos. Se conoce al cierre de la última sesión de la ventana (entra al día siguiente)."""
    if ev is None or not len(ev): return None
    cal = it_mxn.index
    col = {c: k for k, c in enumerate(it_mxn.columns)}
    e = ev[ev['ticker'].isin(col)].copy()
    dia = pd.DatetimeIndex(e['fecha_reporte'].dt.tz_convert(None).dt.normalize())
    p = cal.searchsorted(dia, side='right') - 1
    ok = (p >= 1) & (p + 2 < len(cal))
    e, p = e[ok], p[ok]
    it = it_mxn.to_numpy(float); jj = e['ticker'].map(col).to_numpy()
    mercado = np.column_stack([(indices.get('ipc') if es_mx.get(c) else indices.get('spx')).reindex(cal).to_numpy(float)
                               if (indices.get('ipc') is not None and indices.get('spx') is not None) else np.full(len(cal), np.nan) for c in it_mxn.columns])
    e['car_3d'] = it[p + 2, jj] / it[p - 1, jj] - mercado[p + 2, jj] / mercado[p - 1, jj]
    e['conocido'] = (pd.DatetimeIndex(cal[p + 2]).tz_localize('UTC') + pd.Timedelta(hours=23, minutes=59)).astype('datetime64[ns, UTC]')
    return e.dropna(subset=['car_3d'])[['ticker', 'conocido', 'car_3d']]


def aplicar_deslistamiento(largo, uni, it_mxn, real, horizontes=HORIZONTES):
    """Pedido 101: en emisoras que dejaron de cotizar (salida con causa en universo), las filas cuya ventana cruza la última sesión reciben el
    rendimiento hasta esa sesión compuesto con el de deslistamiento; la trayectoria incluye ese punto. Devuelve la tabla y un registro."""
    import re
    cal = it_mxn.index; registro = []
    salidas = uni[uni['salida'].fillna('').str.strip() != ''] if 'salida' in uni else uni.iloc[0:0]
    for _, u in salidas.iterrows():
        t = u['ticker']
        if t not in it_mxn.columns: continue
        R = next((v for patron, v in DESLISTAMIENTO if re.search(patron, str(u.get('causa') or ''), re.I)), None)
        vivos = np.nonzero(real[t].to_numpy())[0]
        if R is None or not len(vivos) or vivos[-1] >= len(cal) - 6: continue   # causa desconocida o todavía cotiza al final de la muestra
        L = vivos[-1]; serie = it_mxn[t].to_numpy(float)
        filas = largo.index[(largo['ticker'] == t)]
        pos = cal.get_indexer(largo.loc[filas, 'fecha'])
        n = 0
        for h in horizontes:
            cruza = (pos + h > L) & (pos <= L)
            for fila, q in zip(filas[cruza], pos[cruza]):
                final = serie[L] / serie[q] * (1 + R) - 1
                trayecto = np.nanmin(serie[q + 1:L + 1] / serie[q] - 1) if L > q else np.nan
                largo.at[fila, f'y_{h}'] = final
                largo.at[fila, f'min_ret_{h}'] = np.nanmin([final, trayecto])
                n += 1
        registro.append({'ticker': t, 'ultima_sesion': str(cal[L].date()), 'rendimiento_deslistamiento': R, 'filas_ajustadas': n})
    return largo, registro


# ================================================================ grupo E: macro tal como se sabía
def macro_al_dia(almacen, cal):
    """Para cada serie y día D: valor de la observación MÁS RECIENTE conocida antes de D, en su versión vigente antes de D."""
    salida = pd.DataFrame(index=cal)
    ts = pd.DatetimeIndex(cal).tz_localize('UTC').astype('datetime64[ns, UTC]')
    cache = {}
    for col, (conj, serie, _, _) in MACRO.items():
        if conj not in cache: cache[conj] = leer(almacen, conj.replace('macro_', 'macro/'))
        df = cache[conj]
        if df is None or 'conocido' not in df: salida[col] = np.nan; continue
        s = df[df['serie'] == serie][['fecha', 'valor', 'conocido']].copy()
        if not len(s): salida[col] = np.nan; continue
        s['conocido'] = pd.to_datetime(s['conocido'], utc=True).astype('datetime64[ns, UTC]'); s['fecha'] = pd.to_datetime(s['fecha'])
        s = s.sort_values('conocido')
        # estado vigente: la observación de fecha más reciente conocida hasta ese momento (y su última versión)
        vals, mejor_fecha, mejor_valor, momentos = [], None, None, []
        for fe, va, co in zip(s['fecha'], s['valor'], s['conocido']):
            if mejor_fecha is None or fe >= mejor_fecha: mejor_fecha, mejor_valor = fe, va
            momentos.append(co); vals.append(mejor_valor)
        linea = pd.DataFrame({'conocido': momentos, 'v': vals}).drop_duplicates('conocido', keep='last')
        pos = np.searchsorted(linea['conocido'].to_numpy(), ts.to_numpy(), side='left') - 1   # estrictamente antes de D
        salida[col] = np.where(pos >= 0, linea['v'].to_numpy()[np.clip(pos, 0, None)], np.nan)
    return salida


# ================================================================ armado
def excluir_tramos(precios):
    for tr in TRAMOS_EXCLUIDOS:
        fuera = (precios['ticker'] == tr['ticker']) & (precios['fecha'] >= tr['desde'])
        if tr['hasta']: fuera &= precios['fecha'] <= tr['hasta']
        precios = precios[~fuera]
    return precios


def saltos_extremos(almacen, desde):
    """Movimientos diarios de cierre fuera de SALTO_EXTREMO desde `desde`: candidatos a revisar (quiebras reales o errores de la fuente)."""
    import pyarrow.dataset as pads
    p = leer(almacen, 'precios/diarios', ['fecha', 'ticker', 'tipo', 'close'],
             (pads.field('fecha') >= desde) & (pads.field('tipo') == 'empresa'))
    p = excluir_tramos(p[p['tipo'] == 'empresa']).sort_values(['ticker', 'fecha'])
    p['r'] = p.groupby('ticker')['close'].pct_change()
    x = p[(p['r'] > SALTO_EXTREMO[0]) | (p['r'] < SALTO_EXTREMO[1])]
    return [{'ticker': str(a), 'fecha': str(b), 'rendimiento': round(float(c), 3)} for a, b, c in zip(x['ticker'], x['fecha'], x['r'])]


def construir(almacen, desde=DESDE, hasta=None):
    uni = leer(almacen, 'universo/universo.parquet')
    import pyarrow.dataset as pads
    lookback = (pd.Timestamp(desde) - pd.Timedelta(days=LOOKBACK_DIAS)).strftime('%Y-%m-%d')
    filtro = pads.field('fecha') >= lookback   # memoria: solo el rango que se usa
    if hasta: filtro = filtro & (pads.field('fecha') <= hasta)
    # las claves de tipo 'archivo' se guardan pero NO se analizan (pedido 205): se descartan aquí, antes de leerlas, porque
    # matrices() pivotea todo lo que llegue y una tabla ancha de 1,600 columnas revienta la memoria sin cambiar ningún número.
    filtro = filtro & (pads.field('tipo') != 'archivo')
    import pyarrow.dataset as _pds
    hay = set(_pds.dataset(os.path.join(almacen, 'precios/diarios'), format='parquet', partitioning='hive').schema.names)
    ohlc = [c for c in ('high', 'low') if c in hay]
    if len(ohlc) < 2: print('AVISO: el almacén no trae máximo y mínimo diarios; no se estima el medio diferencial de compra-venta')
    precios = leer(almacen, 'precios/diarios', ['fecha', 'ticker', 'tipo', 'close'] + ohlc + ['volumen', 'dividendo', 'split', 'moneda'], filtro)
    if uni is None or precios is None: raise SystemExit('Falta el universo o los precios en el almacén')
    precios = excluir_tramos(precios)
    splits = splits_acumulados(precios)
    precios['ticker'] = precios['ticker'].astype('category')
    m = matrices(precios, lookback)
    monedas = precios.sort_values('fecha').groupby('ticker')['moneda'].last().to_dict()
    empresas = [t for t in uni.loc[uni['tipo'] == 'empresa', 'ticker'] if t in m['close'].columns]
    fx = {par: m['close'][par] for par in set(FX.values()) if par in m['close']}
    it_mxn = en_pesos(m['it'][empresas], monedas, fx)
    indices = {}
    if '^MXX' in m['close']: indices['ipc'] = m['close']['^MXX']
    if '^GSPC' in m['close'] and 'MXN=X' in m['close']: indices['spx'] = m['close']['^GSPC'] * m['close']['MXN=X']
    if 'MXN=X' in m['close']: indices['usdmxn'] = m['close']['MXN=X']
    mm = {k: (v[empresas] if isinstance(v, pd.DataFrame) else v) for k, v in m.items()}
    fa = grupo_a(mm, it_mxn, indices)
    es_mx = {t: str(p) == 'MX' for t, p in uni.set_index('ticker')['pais'].items()}
    subunidad = pd.Series({t: SUBUNIDAD.get(monedas.get(t), 1.0) for t in empresas})
    close_mxn = en_pesos(mm['close'].div(subunidad, axis=1), monedas, fx)
    # nivel de precio real del día (Yahoo reexpresa el historial con splits futuros); el importe usa cierre y volumen reexpresados, que se cancelan
    precio_real_mxn = close_mxn * factor_split_matriz(splits, mm['cal'], empresas)
    fa.update(grupo_caida(mm, mm['it'], it_mxn, precio_real_mxn, indices, es_mx, fa))
    del precio_real_mxn
    fa.update(grupo_liquidez(mm, close_mxn))
    fa.update(liquidez_sic(m, empresas))   # con `m` sin restringir: los espejos del SIC no están entre las columnas de empresa
    del close_mxn
    AUDITORIA['arrastre'] = auditoria_arrastre(mm, {t: p for t, p in uni.set_index('ticker')['pais'].items()})

    # objetivo: rendimiento total en pesos hacia adelante (fechas > D)
    objetivos = {}
    for h in HORIZONTES:
        objetivos[f'y_{h}'] = it_mxn.shift(-h) / it_mxn - 1
        objetivos[f'y_{h}_nativo'] = mm['it'].shift(-h) / mm['it'] - 1
        objetivos[f'min_ret_{h}'] = hacia_adelante_min(it_mxn, h) / it_mxn - 1
        # mercado propio en pesos (índices de precio: sin dividendos)
        fwd = {k: indices[k].shift(-h) / indices[k] - 1 for k in ('ipc', 'spx') if k in indices}
        objetivos[f'ret_mercado_{h}'] = pd.DataFrame({c: fwd.get('ipc' if es_mx.get(c) else 'spx', pd.Series(np.nan, index=mm['cal'])) for c in empresas}, index=mm['cal'])

    # a formato largo, solo días en que la empresa cotizó (por posición en las tablas anchas: sin copias intermedias)
    real = mm['real'].to_numpy(copy=True)   # copia: con copy-on-write (pandas 3) la vista es de solo lectura
    real[np.asarray(mm['cal'] < pd.Timestamp(desde))] = False
    ii, jj = np.nonzero(real)
    columnas_w = list(mm['close'].columns)
    largo = pd.DataFrame({'fecha': mm['cal'][ii], 'ticker': pd.Categorical.from_codes(jj, categories=columnas_w)})
    for nombre, w in {**fa, **objetivos, 'close': mm['close']}.items():
        largo[nombre] = w[columnas_w].to_numpy(dtype=np.float32)[ii, jj]
    del fa, objetivos
    largo['ticker'] = largo['ticker'].astype(str)
    largo['vol_ratio_60_250'] = largo['vol_60d'] / largo['vol_250d']
    largo['precio_para_pe'] = largo['close'] / largo['ticker'].map(lambda t: SUBUNIDAD.get(monedas.get(t), 1.0))
    info = uni.set_index('ticker')
    largo['sector'] = largo['ticker'].map(info['sector']); largo['pais'] = largo['ticker'].map(info['pais'])
    largo['moneda'] = largo['ticker'].map(monedas); largo['en_app'] = largo['ticker'].map(info['en_app']).astype(bool)
    largo['tipo_valor'] = largo['ticker'].map(info['tipo_valor'])
    largo['en_ipc'] = np.nan   # pendiente de ipc_miembros con historia (pedido 55)

    # relativos al sector (corte transversal del mismo día)
    for c in ('ret_1m', 'ret_6m'):
        med = media_sin_propia(largo, c)
        largo[f'{c}_sector'] = med; largo[f'{c}_rel_sector'] = largo[c] - med

    # grupo B
    largo['ts'] = largo['fecha'].dt.tz_localize('UTC').astype('datetime64[ns, UTC]')
    ev = eventos_resultados(leer(almacen, 'eventos/resultados'), splits)
    fibras = set(info.index[info['tipo_valor'] == 'FIBRA'])
    sin_pe = monedas_distintas(almacen, monedas)
    car = eventos_car(ev, it_mxn, indices, es_mx)
    largo = grupo_b(largo, ev, fibras, 'precio_para_pe', sin_pe)
    if car is not None:
        largo = pd.merge_asof(largo.sort_values('ts'), car.sort_values('conocido'), left_on='ts', right_on='conocido', by='ticker',
                              direction='backward', allow_exact_matches=False).drop(columns=['conocido'])
    else:
        largo['car_3d'] = np.nan
    largo = grupo_d(largo, almacen, splits, monedas)
    pe_ok = largo['pe_reconstruido'].where(largo['pe_reconstruido'].between(0, 1000))
    largo['pe_rel_sector'] = pe_ok / mediana_sin_propia(pe_ok, largo[['fecha', 'sector']])
    largo = largo.sort_values(['ticker', 'fecha'])
    largo['pe_rel_propio_3a'] = pe_ok / largo.assign(p=pe_ok).groupby('ticker')['p'].transform(lambda s: s.rolling(756, min_periods=250).median())

    # grupo E
    macro = macro_al_dia(almacen, mm['cal'])
    largo = largo.join(macro, on='fecha')
    largo = largo.join(regimen_mercado(m['close'], macro), on='fecha')
    ri = {k: v.pct_change(21, fill_method=None) for k, v in indices.items()}
    for k, col in (('usdmxn', 'fx_ret_1m'), ('ipc', 'ret_ipc_1m'), ('spx', 'ret_spx_1m')):
        largo[col] = largo['fecha'].map(ri[k]) if k in ri else np.nan

    # precio bajo topado (Campbell, Hilscher y Szilagyi): log(min(precio en dólares, 15))
    if 'MXN=X' in m['close']:
        usd = np.exp(largo['log_precio_mxn']) / largo['fecha'].map(m['close']['MXN=X']).to_numpy(float)
        largo['log_precio_usd_topado'] = np.log(np.minimum(usd, 15.0))
    else:
        largo['log_precio_usd_topado'] = np.nan
    largo, AUDITORIA['deslistamiento'] = aplicar_deslistamiento(largo.reset_index(drop=True), uni, it_mxn, mm['real'])

    # objetivos relativos al día
    for h in HORIZONTES:
        y = f'y_{h}'
        largo[f'{y}_rel'] = largo[y] - largo.groupby('fecha')[y].transform('median')
        largo[f'{y}_rango'] = rango_centrado(largo[y], [largo['fecha']])
        largo[f'{y}_rango_pais'] = rango_centrado(largo[y], [largo['fecha'], largo['pais']])   # pedido 92: selección sin apuesta de país ni moneda
        largo[f'{y}_rango_libro'] = rango_centrado(largo[y], [largo['fecha'], np.where(largo['pais'] == 'MX', 'MX', 'extranjero')])   # tanda 4B: dos libros
        # sector sin la propia emisora (en sectores chicos la media con ella misma se parece a su propio rendimiento)
        g = largo.groupby(['fecha', 'sector'], observed=True)[y]
        suma, n = g.transform('sum'), g.transform('count')
        largo[f'ret_sector_{h}'] = np.where(largo[y].notna() & (n > 1), (suma - largo[y].fillna(0)) / (n - 1), np.nan)
        beta = largo['beta_mercado_1a'].clip(0, 2).fillna(1.0)
        largo[f'{y}_aj_mercado'] = largo[y] - beta * largo[f'ret_mercado_{h}']
        largo[f'{y}_aj_sector'] = largo[y] - largo[f'ret_sector_{h}']

    largo = largo.drop(columns=['ts', 'precio_para_pe']).sort_values(['fecha', 'ticker']).reset_index(drop=True)
    for c in largo.columns:
        if largo[c].dtype == 'float64': largo[c] = largo[c].astype('float32')
    largo['ticker'] = largo['ticker'].astype('category'); largo['sector'] = largo['sector'].astype('category')
    largo['regimen'] = largo['regimen'].astype('category')
    return largo


def meta_columnas():
    """grupo, fuente y publicable de cada columna (el dataset es derivado; la fuente es la de los datos de entrada)."""
    A = ['ret_1m', 'ret_3m', 'ret_6m', 'ret_12m_ex1m', 'ret_mxn_1m', 'ret_mxn_6m', 'ret_mxn_12m_ex1m', 'dist_ma200', 'dist_max_52s', 'vol_60d',
         'vol_250d', 'max_dd_1a', 'vol_relativa', 'rend_dividendo_12m', 'beta_ipc_1a', 'beta_spx_1a', 'corr_fx_1a',
         'ret_1m_sector', 'ret_1m_rel_sector', 'ret_6m_sector', 'ret_6m_rel_sector', 'close',
         'max_ret_1m', 'min_ret_1m', 'asimetria_6m', 'runup_1a', 'runup_2a', 'aceleracion', 'dist_ma50', 'log_precio_mxn', 'ivol_60d',
         'beta_mercado_1a', 'vol_ratio_60_250', 'log_precio_usd_topado']
    B = ['sorpresa_ult', 'sorpresa_prom_4', 'racha_sorpresas', 'n_sorpresas_neg_4', 'sue', 'car_3d', 'reporte_atrasado', 'dias_al_proximo_est', 'ventana_reporte', 'dias_desde_reporte', 'crec_eps_4t', 'pe_reconstruido', 'sin_utilidades', 'pe_rel_sector', 'pe_rel_propio_3a']
    meta = {c: {'grupo': 'A', 'fuente': 'yahoo', 'publicable': False} for c in A}
    meta.update({c: {'grupo': 'B', 'fuente': 'yahoo', 'publicable': False} for c in B})
    for col, (conj, serie, desc, pub) in MACRO.items():
        meta[col] = {'grupo': 'E', 'fuente': conj.replace('macro_', ''), 'serie': serie, 'descripcion': desc, 'publicable': pub}
    meta.update({c: {'grupo': 'E', 'fuente': 'yahoo', 'publicable': False} for c in ('fx_ret_1m', 'ret_ipc_1m', 'ret_spx_1m')})
    meta.update({c: {'grupo': 'identidad', 'fuente': 'brota', 'publicable': True} for c in ('fecha', 'ticker', 'sector', 'pais', 'moneda', 'en_app', 'tipo_valor', 'en_ipc')})
    meta.update({c: {'grupo': 'L', 'fuente': 'yahoo', 'publicable': False} for c in GRUPO_L})
    meta.update({c: {'grupo': 'D', 'fuente': 'sec+yahoo' if c in GRUPO_D_CON_PRECIO else 'sec', 'publicable': c not in GRUPO_D_CON_PRECIO} for c in GRUPO_D})
    meta.update({f'{c}_rango_sector': dict(meta[c]) for c in RANGO_SECTOR_D})
    meta.update({c: {'grupo': 'R', 'fuente': 'yahoo', 'publicable': False} for c in ('dd_ipc_52s', 'dd_spx_52s')})
    meta['regimen'] = {'grupo': 'R', 'fuente': 'yahoo+fred:VIXCLS', 'publicable': False, 'regla': REGIMEN}
    for h in HORIZONTES:
        for s in ('', '_nativo', '_rel', '_rango', '_rango_pais', '_rango_libro', '_aj_mercado', '_aj_sector'):
            meta[f'y_{h}{s}'] = {'grupo': 'objetivo', 'fuente': 'yahoo', 'publicable': False}
        for c in (f'min_ret_{h}', f'ret_mercado_{h}', f'ret_sector_{h}'):
            meta[c] = {'grupo': 'objetivo', 'fuente': 'yahoo', 'publicable': False}
    return meta


def diagnostico_pe(tabla, almacen, tolerancia=0.20):
    """Pedido 52: el P/E reconstruido del último día contra trailingPE de la foto de Yahoo; diferencias > 20 % delatan razón ADR o moneda."""
    fotos = leer(almacen, 'fotos/valuacion', None)
    if fotos is None or 'trailingPE' not in fotos: return []
    ultimo = fotos.sort_values('capturado').drop_duplicates('ticker', keep='last').set_index('ticker')['trailingPE']
    hoy = tabla.sort_values('fecha').drop_duplicates('ticker', keep='last').set_index('ticker')['pe_reconstruido']
    salida = []
    for t, pe in hoy.dropna().items():
        ref = pd.to_numeric(ultimo.get(t), errors='coerce')
        if ref and ref > 0 and abs(pe / ref - 1) > tolerancia: salida.append({'ticker': str(t), 'pe_reconstruido': round(float(pe), 2), 'trailingPE_yahoo': round(float(ref), 2)})
    return salida


def rango_centrado(valores, grupos):
    """(rango − 0.5) / n dentro de cada grupo: promedio 0.5 en CUALQUIER grupo. El percentil de pandas (rango / n) da promedio (n + 1) / 2n y
    1.0 a la única emisora de un país: con rango por país el modelo aprendía «emisora extranjera sola en su país» (IC falso de 0.2–0.29)."""
    g = pd.Series(np.asarray(valores, dtype=float)).groupby([pd.Series(np.asarray(x)) for x in grupos], observed=True)
    return ((g.rank(method='average') - 0.5) / g.transform('count')).to_numpy()


def columnas_vacias(tabla):
    """Variables del modelo sin un solo dato: casi siempre un concepto que el almacén no tiene (dd_ingenuo estuvo vacía en dataset-v3 porque
    LiabilitiesCurrent nunca se había bajado). Se imprime al construir para que no pase en silencio."""
    meta = meta_columnas()
    return sorted(c for c in tabla.columns if meta.get(c, {}).get('grupo') in ('A', 'B', 'D', 'E', 'L', 'R') and tabla[c].isna().all())


def escribir(tabla, salida, almacen, desde):
    base = os.path.join(salida, 'dataset')
    if os.path.isdir(base):
        import shutil; shutil.rmtree(base)
    for anio, g in tabla.groupby(tabla['fecha'].dt.year):
        ruta = os.path.join(base, f'anio={anio}'); os.makedirs(ruta, exist_ok=True)
        g.to_parquet(os.path.join(ruta, 'parte.parquet'), index=False, compression='zstd')
    try: commit = subprocess.run(['git', '-C', almacen, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    except Exception: commit = None
    meta = {'version': VERSION, 'generado': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'almacen_commit': commit, 'desde': desde,
            'filas': int(len(tabla)), 'empresas': int(tabla['ticker'].nunique()), 'fecha_min': str(tabla['fecha'].min().date()), 'fecha_max': str(tabla['fecha'].max().date()),
            'regla': 'variables con precios ≤ D y datos con conocido < D; objetivos con fechas > D',
            'filas_por_anio': {str(k): int(v) for k, v in tabla.groupby(tabla['fecha'].dt.year).size().items()},
            'filas_por_sector': {str(k): int(v) for k, v in tabla.groupby('sector', observed=True).size().items()},
            'fechas_por_regimen': {str(k): int(v) for k, v in tabla.drop_duplicates('fecha')['regimen'].value_counts().items()},
            'tasa_caida_min_ret_63_20': round(float((tabla['min_ret_63'] <= -0.20).sum() / max(tabla['min_ret_63'].notna().sum(), 1)), 4),
            'nulos': {c: round(float(tabla[c].isna().mean()), 3) for c in tabla.columns if tabla[c].isna().any()},
            'columnas_vacias': columnas_vacias(tabla),
            'columnas': meta_columnas(), 'diagnostico_pe': diagnostico_pe(tabla, almacen),
            'tramos_excluidos': TRAMOS_EXCLUIDOS, 'saltos_extremos': saltos_extremos(almacen, desde), 'arrastre': AUDITORIA.get('arrastre'),
            'deslistamiento': AUDITORIA.get('deslistamiento'),
            'limitaciones': ['sesgo de supervivencia: casi todas las emisoras que salieron de bolsa no tienen historia en Yahoo',
                             'en_ipc sin historia antes de 2026-03', 'objetivo nulo en los últimos 21/63 días y tras la salida de una emisora',
                             'calendario unión de bolsas: 21 filas ≈ 21 días hábiles',
                             'pe_reconstruido usa utilidad por acción ajustada (la que reporta la empresa), no la contable: difiere del trailingPE de Yahoo en emisoras con partidas extraordinarias',
                             'pe_reconstruido vacío en emisoras que reportan en otra moneda que la de su precio (ADR)',
                             'ret_mercado_h usa índices de precio (IPC, S&P 500 en pesos) sin dividendos: y_h_aj_mercado queda sesgado AL ALZA ~0.2–0.3 % mensual; no cambia el orden del día, pero cualquier cifra de nivel que se muestre debe corregirlo (pedido 108)',
                             'rendimiento de deslistamiento (−30 % por deterioro, 0 % por oferta o fusión) solo en emisoras con precios y causa conocida; y_h_nativo no se ajusta',
                             'min_ret_h de una emisora que deja de cotizar usa la parte de la ventana que cotizó (≥ 80 %); si quebró después, la caída no se ve',
                             'dias_al_proximo_est supone reportes cada 91 días desde el último conocido; las fechas anunciadas solo se archivan desde 2026-09',
                             'regimen usa el VIX (serie de Cboe, tercero): solo laboratorio',
                             'grupo D solo existe para emisoras con reportes en la SEC (87; mexicanas únicamente con ADR) y desde 2009; fórmulas provisionales hasta la consulta M1',
                             'pasivos_mercado y libros_mercado solo para emisoras listadas en EE. UU. con estados en la moneda de su precio'],
            'sin_pe_por_moneda': sorted(monedas_distintas(almacen, tabla.drop_duplicates('ticker').set_index('ticker')['moneda'].astype(str).to_dict()))}
    with open(os.path.join(salida, 'dataset_meta.json'), 'w', encoding='utf-8') as fh: json.dump(meta, fh, ensure_ascii=False, indent=1)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description='Tabla de entrenamiento de Brota')
    p.add_argument('--almacen', required=True); p.add_argument('--salida', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datos'))
    p.add_argument('--desde', default=DESDE); p.add_argument('--hasta')
    p.add_argument('--deslistamiento-deterioro', type=float, help='rendimiento al salir de bolsa por deterioro (por omisión −0.30, Shumway 1997; −0.55 es Shumway y Warther 1999)')
    a = p.parse_args(argv)
    if a.deslistamiento_deterioro is not None:   # sensibilidad del supuesto (tanda 6, M33): solo cambia el rendimiento por deterioro
        DESLISTAMIENTO[0] = (DESLISTAMIENTO[0][0], float(a.deslistamiento_deterioro))
    tabla = construir(a.almacen, a.desde, a.hasta)
    meta = escribir(tabla, a.salida, a.almacen, a.desde)
    print(json.dumps({k: meta[k] for k in ('filas', 'empresas', 'fecha_min', 'fecha_max', 'almacen_commit')}, ensure_ascii=False))
    print('P/E con diferencia > 20 % contra Yahoo:', meta['diagnostico_pe'][:12])
    if meta['columnas_vacias']: print('⚠ variables sin ningún dato (revisar el almacén):', meta['columnas_vacias'])


if __name__ == '__main__':
    main()
