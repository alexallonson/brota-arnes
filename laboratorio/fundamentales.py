"""
laboratorio/fundamentales.py — estados financieros PUNTO EN EL TIEMPO desde los hechos XBRL de la SEC (almacén: fundamentales/sec).

Para cada emisora y cada fecha en que presentó un reporte (10-K, 10-Q, 20-F, 40-F y enmiendas) reconstruye lo que se sabía justo después:
  - saldos (activos, pasivos, patrimonio, caja, deuda de largo plazo, acciones en circulación) al cierre más reciente y hace un año;
  - flujos de los últimos 12 meses (utilidad neta, flujo operativo, ingresos, utilidad operativa, gasto por intereses, inversión en activo fijo):
    el anual si el último periodo es un cierre fiscal; si no, anual previo + acumulado del año − acumulado del año anterior (los 10-Q traen
    el acumulado y su comparativo). Años fiscales de 52/53 semanas: tolerancia de 20 días.
Cada hecho cuenta con su versión más reciente CONOCIDA en ese momento (una reexpresión posterior no cambia el pasado).
dataset.py lo une a cada día con conocido < D (grupo D, pedidos 65 y 85). Las fórmulas de las razones son provisionales hasta la
respuesta M1 de Science Mercado (`CONSULTAS_MERCADO.md`).
"""
import numpy as np
import pandas as pd

SALDOS = {
    'activos': [('us-gaap', 'Assets'), ('ifrs-full', 'Assets')],
    'pasivos': [('us-gaap', 'Liabilities'), ('ifrs-full', 'Liabilities')],
    'patrimonio': [('us-gaap', 'StockholdersEquity'), ('us-gaap', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'),
                   ('ifrs-full', 'EquityAttributableToOwnersOfParent'), ('ifrs-full', 'Equity')],
    'caja': [('us-gaap', 'CashAndCashEquivalentsAtCarryingValue'), ('us-gaap', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'),
             ('ifrs-full', 'CashAndCashEquivalents')],
    'deuda_lp': [('us-gaap', 'LongTermDebtNoncurrent'), ('us-gaap', 'LongTermDebt'), ('us-gaap', 'LongTermDebtAndCapitalLeaseObligations'),
                 ('ifrs-full', 'NoncurrentPortionOfNoncurrentBorrowings')],
    'acciones': [('dei', 'EntityCommonStockSharesOutstanding')],
    'pasivos_circulantes': [('us-gaap', 'LiabilitiesCurrent'), ('ifrs-full', 'CurrentLiabilities')],
    # deuda en el pasivo circulante (DLC de Compustat) para la F de Bharath y Shumway (2008): el total si se reporta; si no, sus partes
    'deuda_cp_total': [('us-gaap', 'DebtCurrent'), ('ifrs-full', 'CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings')],
    'deuda_cp_porcion_lp': [('us-gaap', 'LongTermDebtCurrent'), ('us-gaap', 'LongTermDebtAndCapitalLeaseObligationsCurrent'),
                            ('ifrs-full', 'CurrentPortionOfLongtermBorrowings')],
    'deuda_cp_prestamos': [('us-gaap', 'ShortTermBorrowings'), ('us-gaap', 'CommercialPaper'), ('ifrs-full', 'ShorttermBorrowings')],
}
FLUJOS = {
    'utilidad_neta': [('us-gaap', 'NetIncomeLoss'), ('us-gaap', 'ProfitLoss'), ('ifrs-full', 'ProfitLossAttributableToOwnersOfParent'), ('ifrs-full', 'ProfitLoss')],
    'flujo_operativo': [('us-gaap', 'NetCashProvidedByUsedInOperatingActivities'), ('ifrs-full', 'CashFlowsFromUsedInOperatingActivities')],
    'ingresos': [('us-gaap', 'Revenues'), ('us-gaap', 'RevenueFromContractWithCustomerExcludingAssessedTax'),
                 ('us-gaap', 'RevenueFromContractWithCustomerIncludingAssessedTax'), ('us-gaap', 'SalesRevenueNet'),
                 ('ifrs-full', 'Revenue'), ('ifrs-full', 'RevenueFromContractsWithCustomers')],
    'utilidad_operativa': [('us-gaap', 'OperatingIncomeLoss'), ('ifrs-full', 'ProfitLossFromOperatingActivities')],
    'gasto_intereses': [('us-gaap', 'InterestExpense'), ('us-gaap', 'InterestExpenseNonoperating'), ('us-gaap', 'InterestExpenseDebt'),
                        ('ifrs-full', 'InterestExpense'), ('ifrs-full', 'FinanceCosts')],
    'utilidad_bruta': [('us-gaap', 'GrossProfit'), ('ifrs-full', 'GrossProfit')],
    'capex': [('us-gaap', 'PaymentsToAcquirePropertyPlantAndEquipment'), ('ifrs-full', 'PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities')],
}
PROMEDIOS = {   # acciones promedio del periodo (se reexpresan por splits en los comparativos: sirven para medir emisión)
    'acciones_diluidas': [('us-gaap', 'WeightedAverageNumberOfDilutedSharesOutstanding'), ('ifrs-full', 'AdjustedWeightedAverageShares'),
                          ('ifrs-full', 'WeightedAverageShares')],
}
RANGO_ACCIONES = (0.5, 2.0)     # acciones de portada / diluidas fuera de este rango: otra clase, error de escala o emisión posterior al periodo
ESCALA_ACCIONES = (0.05, 20.0)  # cambio anual de acciones diluidas fuera de este rango = error de escala en el XBRL
ANUAL = (340, 390)            # días de un año fiscal
ACUMULADO = (80, 300)         # trimestre, semestre o nueve meses
TOLERANCIA = 20               # años de 52/53 semanas
HACE_UN_ANIO = (365, 45)      # saldo comparable: cierre a 365 ± 45 días


class Serie:
    """Hechos de un concepto de una emisora en arreglos ordenados por conocido (prefijo = lo sabido hasta un momento)."""
    def __init__(self, g):
        # a igual momento, el concepto preferido queda al final (gana en _ultimo); una presentación posterior gana siempre
        g = g.sort_values(['conocido', 'preferencia'], ascending=[True, False], kind='mergesort')
        self.con = g['conocido'].to_numpy()
        self.ini = pd.to_datetime(g['inicio'], errors='coerce').to_numpy('datetime64[D]')
        self.fin = pd.to_datetime(g['fin'], errors='coerce').to_numpy('datetime64[D]')
        self.val = g['valor'].to_numpy(float)
        self.dur = np.where(np.isnat(self.ini) | np.isnat(self.fin), np.nan,
                            (self.fin - self.ini).astype('timedelta64[D]').astype('int64').astype(float))   # NaN en saldos (sin inicio)

    def hasta(self, momento):
        return int(np.searchsorted(self.con, momento, side='right'))


def _ultimo(mascara, val):
    i = np.nonzero(mascara)[0]
    return val[i[-1]] if len(i) else np.nan   # la versión conocida más reciente (orden por conocido)


def saldo(s, p, fin_objetivo=None):
    """Saldo al cierre más reciente (o al más cercano a fin_objetivo ± 45 días) entre los primeros p hechos. Devuelve (valor, fin)."""
    es = np.isnan(s.dur[:p]) & ~np.isnat(s.fin[:p])
    if not es.any(): return np.nan, None
    if fin_objetivo is not None: fin_objetivo = np.datetime64(pd.Timestamp(fin_objetivo).date(), 'D')   # acepta Timestamp o datetime64
    fins = s.fin[:p][es]
    if fin_objetivo is None:
        F = fins.max()
    else:
        dist = np.abs((fins - fin_objetivo).astype(float))
        if dist.min() > HACE_UN_ANIO[1]: return np.nan, None
        F = fins[np.argmin(dist)]
    return _ultimo(es & (s.fin[:p] == F), s.val[:p]), F


def doce_meses(s, p, F=None):
    """Flujo de los 12 meses que terminan en F ± 20 días (por omisión, el periodo más reciente). Devuelve (valor, F)."""
    dur, fin, val = s.dur[:p], s.fin[:p], s.val[:p]
    flujo = ~np.isnan(dur)
    if not flujo.any(): return np.nan, None
    if F is None:
        F = fin[flujo].max()
    else:
        F = np.datetime64(pd.Timestamp(F).date(), 'D')   # acepta Timestamp o datetime64
        dist = np.abs((fin[flujo] - F).astype(float))
        if dist.min() > TOLERANCIA: return np.nan, None
        F = fin[flujo][np.argmin(dist)]
    anual = flujo & (dur >= ANUAL[0]) & (dur <= ANUAL[1])
    v = _ultimo(anual & (fin == F), val)
    if not np.isnan(v): return v, F
    acum = flujo & (fin == F) & (dur >= ACUMULADO[0]) & (dur <= ACUMULADO[1])
    if not acum.any(): return np.nan, F
    d = dur[acum].max()
    ytd = _ultimo(acum & (dur == d), val)
    previos = anual & (fin < F) & (fin >= F - np.timedelta64(400, 'D'))
    if not previos.any(): return np.nan, F
    fy = _ultimo(previos & (fin == fin[previos].max()), val)
    comparable = flujo & (np.abs((fin - (F - np.timedelta64(364, 'D'))).astype(float)) <= TOLERANCIA) & (np.abs(dur - d) <= TOLERANCIA)
    ytd_prev = _ultimo(comparable, val)
    return (fy + ytd - ytd_prev, F) if not np.isnan(ytd_prev) else (np.nan, F)


def promedio(s, p, F=None):
    """Promedio del periodo más largo que termina en F (por omisión, el más reciente) y su comparativo de un año antes con la misma duración,
    ambos en su versión conocida más reciente (los comparativos vienen reexpresados por splits). Devuelve (actual, hace_1a, F)."""
    dur, fin, val = s.dur[:p], s.fin[:p], s.val[:p]
    flujo = ~np.isnan(dur)
    if not flujo.any(): return np.nan, np.nan, None
    F = fin[flujo].max() if F is None else np.datetime64(pd.Timestamp(F).date(), 'D')
    en_f = flujo & (fin == F)
    d = dur[en_f].max()
    actual = _ultimo(en_f & (dur == d), val)
    comparable = flujo & (np.abs((fin - (F - np.timedelta64(364, 'D'))).astype(float)) <= TOLERANCIA) & (np.abs(dur - d) <= TOLERANCIA)
    return actual, _ultimo(comparable, val), F


def _moneda(g):
    m = g.loc[~g['unidad'].str.contains('shares|pure', regex=True), 'unidad']
    return m.mode().iloc[0] if len(m) else None


def estados_por_presentacion(sec):
    """Una fila por (ticker, conocido): lo que se sabía de sus estados financieros tras cada presentación."""
    if sec is None or not len(sec): return None
    sec = sec.copy()
    sec['conocido'] = pd.to_datetime(sec['conocido'], utc=True).astype('datetime64[ns, UTC]')
    filas = []
    for t, g in sec.groupby('ticker', sort=False):
        moneda = _moneda(g)
        series = {}
        for nombre, alternativas in {**SALDOS, **FLUJOS, **PROMEDIOS}.items():
            # conceptos equivalentes en UNA serie: al cambiar de etiqueta (p. ej. ASC 606 en 2018) el anual viejo y el acumulado nuevo se combinan
            partes = []
            for k, (tax, con) in enumerate(alternativas):
                h = g[(g['taxonomia'] == tax) & (g['concepto'] == con) & (g['unidad'] == ('shares' if nombre in ('acciones', 'acciones_diluidas') else moneda))]
                if len(h): partes.append(h.assign(preferencia=k))
            series[nombre] = Serie(pd.concat(partes)) if partes else None
        for momento in np.sort(g['conocido'].unique()):
            fila = {'ticker': t, 'conocido': momento, 'moneda_estados': moneda, 'simbolo_sec': g['simbolo_sec'].iloc[0]}
            fines = {}
            for nombre in list(SALDOS) + list(FLUJOS):
                s = series[nombre]
                p = s.hasta(momento) if s else 0
                v, F = (saldo if nombre in SALDOS else doce_meses)(s, p) if p else (np.nan, None)
                fines[nombre] = F
                if nombre in SALDOS:
                    fila[nombre] = v
                    if nombre in ('activos', 'acciones'):
                        previo = saldo(s, p, F - np.timedelta64(HACE_UN_ANIO[0], 'D')) if F is not None else (np.nan, None)
                        fila[f'{nombre}_hace_1a'], fila[f'fin_{nombre}_hace_1a'] = previo
                        fila[f'fin_{nombre}'] = F
                else:
                    fila[f'{nombre}_4t'] = v
                    if nombre == 'ingresos':
                        fila['ingresos_4t_hace_1a'] = doce_meses(s, p, F - np.timedelta64(365, 'D'))[0] if F is not None else np.nan
            s = series['acciones_diluidas']
            p = s.hasta(momento) if s else 0
            fila['acciones_diluidas'], fila['acciones_diluidas_hace_1a'], fines['acciones_diluidas'] = promedio(s, p) if p else (np.nan, np.nan, None)
            # periodo de referencia: el cierre más reciente de activos o utilidad. Un concepto que la emisora dejó de reportar no se arrastra
            ref = max([f for f in (fines['activos'], fines['utilidad_neta']) if f is not None], default=None)
            fila['fin_periodo'] = ref
            for nombre, F in fines.items():
                if nombre == 'acciones' or F is None or ref is None: continue
                if F < ref - np.timedelta64(45, 'D'):
                    col = nombre if nombre in SALDOS or nombre in PROMEDIOS else f'{nombre}_4t'
                    fila[col] = np.nan
                    if nombre == 'ingresos': fila['ingresos_4t_hace_1a'] = np.nan
                    if nombre == 'activos': fila['activos_hace_1a'] = np.nan
                    if nombre == 'acciones_diluidas': fila['acciones_diluidas'] = fila['acciones_diluidas_hace_1a'] = np.nan
            filas.append(fila)
    e = pd.DataFrame(filas)
    for c in [c for c in e.columns if c.startswith('fin_')]:
        e[c] = pd.to_datetime(e[c])
    return e


def razones(filas, fin_ref, precio, sin_razones_financieras):
    """Grupo D sobre filas ya unidas por punto en el tiempo. precio: precio en la moneda de los estados y en la base de acciones de la
    portada (NaN si no es comparable). sin_razones_financieras: bancos y aseguradoras."""
    f = filas.copy()
    for c in ('utilidad_bruta_4t', 'pasivos_circulantes', 'deuda_lp', 'capex_4t', 'gasto_intereses_4t', 'utilidad_operativa_4t',
              'deuda_cp_total', 'deuda_cp_porcion_lp', 'deuda_cp_prestamos', 'activos_hace_1a'):
        if c not in f: f[c] = np.nan   # conceptos que una emisora (o una prueba) no trae
    f['pasivos'] = f['pasivos'].fillna(f['activos'] - f['patrimonio'])   # muchas emisoras no reportan el total de pasivos
    div = lambda a, b: np.where((b != 0) & np.isfinite(b), a / b, np.nan)
    salida = {
        'apalancamiento_libros': div(f['pasivos'], f['activos']),
        'caja_activos': div(f['caja'], f['activos']),
        'roa_4t': div(f['utilidad_neta_4t'], f['activos']),
        'margen_neto_4t': div(f['utilidad_neta_4t'], f['ingresos_4t']),
        # Sloan (1996) divide entre activos PROMEDIO: con activos al cierre el orden se mezcla con el crecimiento de activos (tanda 4, M16)
        'devengos_4t': div(f['utilidad_neta_4t'] - f['flujo_operativo_4t'], (f['activos'] + f['activos_hace_1a']) / 2),
        'flujo_libre_activos': div(f['flujo_operativo_4t'] - f['capex_4t'].fillna(0), f['activos']),
        'crec_activos_1a': div(f['activos'], f['activos_hace_1a']) - 1,                                      # Cooper, Gulen y Schill (2008)
        'crec_ingresos_1a': div(f['ingresos_4t'], f['ingresos_4t_hace_1a']) - 1,
        'emision_neta_1a': div(f['acciones_diluidas'], f['acciones_diluidas_hace_1a']) - 1,                # Pontiff y Woodgate (2008)
        'cobertura_intereses_4t': div(f['utilidad_operativa_4t'], f['gasto_intereses_4t']),
        'patrimonio_negativo': np.where(f['patrimonio'].notna(), (f['patrimonio'] < 0).astype(float), np.nan),
        'antiguedad_estado_dias': (fin_ref - f['fin_periodo']).dt.days.astype(float),
    }
    razon = div(f['acciones_diluidas'], f['acciones_diluidas_hace_1a'])
    salida['emision_neta_1a'] = np.where((razon > ESCALA_ACCIONES[0]) & (razon < ESCALA_ACCIONES[1]), razon - 1, np.nan)   # fuera: error de escala
    # Fuera del rango (revisado en los datos, tanda 4 M17: clases múltiples como MA o NKE, portada o diluidas en miles o millones como NVDA o CRM,
    # emisión tras el periodo como la OPI de UBER) la cifra correcta es la MAYOR de las dos; la fila queda marcada en cap_dudosa en vez de perderse
    razon_acc = div(f['acciones'], f['acciones_diluidas'])
    congruentes = pd.Series(razon_acc).between(*RANGO_ACCIONES).to_numpy()
    acciones_cap = np.where(congruentes, f['acciones'], np.fmax(f['acciones'], f['acciones_diluidas']))
    capitalizacion = np.where(np.isfinite(razon_acc) & (acciones_cap > 0), acciones_cap * precio, np.nan)
    salida['cap_dudosa'] = np.where(np.isfinite(razon_acc) & np.isfinite(precio), (~congruentes).astype(float), np.nan)
    mta = f['pasivos'] + capitalizacion                                                                    # valor de mercado de los activos (CHS)
    salida['pasivos_mercado'] = div(f['pasivos'], mta)                                                     # tlmta, Campbell, Hilscher y Szilagyi (2008)
    salida['nimta'] = div(f['utilidad_neta_4t'], mta)
    salida['cashmta'] = div(f['caja'], mta)
    salida['libros_mercado'] = div(f['patrimonio'], capitalizacion)
    salida['rentabilidad_bruta'] = div(f['utilidad_bruta_4t'], f['activos'])                              # Novy-Marx (2013)
    salida['log_capitalizacion'] = np.log(np.where(capitalizacion > 0, capitalizacion, np.nan))
    salida['_capitalizacion'] = capitalizacion
    # F de Bharath y Shumway (2008): deuda en el pasivo circulante + ½ deuda de largo plazo. El pasivo circulante total (proveedores,
    # gastos acumulados) no es deuda y bajaba la distancia al incumplimiento de minoristas y consumo (tanda 4, M17). Sin partida reportada = 0
    deuda_cp = f['deuda_cp_total'].fillna(f['deuda_cp_porcion_lp'].fillna(0) + f['deuda_cp_prestamos'].fillna(0))
    salida['_deuda_incumplimiento'] = deuda_cp + 0.5 * f['deuda_lp'].fillna(0)
    for c in ('apalancamiento_libros', 'pasivos_mercado', 'cashmta', 'devengos_4t', 'flujo_libre_activos', 'cobertura_intereses_4t', '_deuda_incumplimiento'):
        salida[c] = np.where(sin_razones_financieras, np.nan, salida[c])
    return salida
