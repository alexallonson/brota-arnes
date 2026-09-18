"""Pruebas de laboratorio/dataset.py con un almacén sintético: punto en el tiempo, rendimiento total en pesos, P/E, macro y metadatos."""
import json, os, sys, tempfile, unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'laboratorio'))
try:
    import numpy as np
    import pandas as pd
    import pyarrow  # noqa: F401
    import dataset as DS
    HAY = True
except ImportError:
    HAY = False


def escribir(df, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True); df.to_parquet(ruta, index=False)


def almacen_sintetico(d, alterar_futuro=False):
    dias = pd.bdate_range('2019-01-01', '2021-06-30')
    rng = np.random.default_rng(7)
    filas = []
    def serie(t, tipo, moneda, inicio, divs=None, vol=0.01):
        precio = inicio * np.cumprod(1 + rng.normal(0.0003, vol, len(dias)))
        if alterar_futuro: precio = np.where(dias > pd.Timestamp('2020-06-30'), precio * 1.5, precio)   # el futuro cambia; el pasado no debe
        # El volumen TAMBIÉN tiene que cambiar en el futuro. Hasta el 2026-09-18 solo se alteraba el precio, así que ninguna columna
        # derivada del volumen —importe, Amihud, días sin volumen, y las dos del SIC— estaba protegida de verdad: se comprobó
        # inyectando un `.shift(-5)` en `dias_sic_12m` y la prueba pasaba igual.
        volumen = np.arange(len(dias)) + 1000
        if alterar_futuro: volumen = np.where(dias > pd.Timestamp('2020-06-30'), 0, volumen)
        for i, f in enumerate(dias):
            filas.append({'fecha': f.strftime('%Y-%m-%d'), 'ticker': t, 'tipo': tipo, 'close': float(precio[i]), 'volumen': int(volumen[i]),
                          'high': float(precio[i]) * 1.01, 'low': float(precio[i]) * 0.99,
                          'dividendo': (divs or {}).get(f.strftime('%Y-%m-%d'), 0.0), 'split': 2.0 if (t, f.strftime('%Y-%m-%d')) == ('AAA', '2020-06-01') else 0.0,
                          'moneda': moneda})
    serie('AAA', 'empresa', 'USD', 50, {'2020-03-02': 1.0}); serie('BBB.MX', 'empresa', 'MXN', 30); serie('FIB.MX', 'empresa', 'MXN', 20)
    serie('GBR.L', 'empresa', 'GBp', 1000); serie('ADR', 'empresa', 'USD', 40)
    serie('^MXX', 'indice', 'MXN', 40000); serie('^GSPC', 'indice', 'USD', 3000); serie('MXN=X', 'divisa', 'MXN', 20, vol=0.005); serie('GBPMXN=X', 'divisa', 'MXN', 25, vol=0.005)
    # espejos del SIC: sin ellos, `dias_sic_12m` e `importe_sic_20d` salen todas nulas y la prueba de punto en el tiempo
    # pasaría EN VACÍO. AAA tiene espejo y ADR no, para que se pruebe también el caso sin espejo (pedido AT).
    serie('AAA.MX', 'sic', 'MXN', 1000)
    p = pd.DataFrame(filas)
    for (anio, mes), g in p.groupby([p['fecha'].str[:4], p['fecha'].str[5:7]]): escribir(g, os.path.join(d, f'precios/diarios/anio={anio}/mes={mes}.parquet'))
    uni = pd.DataFrame([{'ticker': 'AAA', 'tipo': 'empresa', 'sector': 'tec', 'pais': 'US', 'en_app': True, 'tipo_valor': 'accion'},
                        {'ticker': 'BBB.MX', 'tipo': 'empresa', 'sector': 'tec', 'pais': 'MX', 'en_app': True, 'tipo_valor': 'accion'},
                        {'ticker': 'FIB.MX', 'tipo': 'empresa', 'sector': 'inm', 'pais': 'MX', 'en_app': True, 'tipo_valor': 'FIBRA'},
                        {'ticker': 'GBR.L', 'tipo': 'empresa', 'sector': 'lujo', 'pais': 'GB', 'en_app': True, 'tipo_valor': 'accion'},
                        {'ticker': 'ADR', 'tipo': 'empresa', 'sector': 'salud', 'pais': 'DK', 'en_app': True, 'tipo_valor': 'accion'},
                        {'ticker': '^MXX', 'tipo': 'indice', 'sector': None, 'pais': None, 'en_app': False, 'tipo_valor': None}])
    escribir(uni, os.path.join(d, 'universo/universo.parquet'))
    utc = lambda s: pd.Timestamp(s, tz='UTC')
    ev = []
    for t in ('AAA', 'FIB.MX', 'GBR.L', 'ADR'):
        for k, f in enumerate(['2019-04-25', '2019-07-25', '2019-10-24', '2020-01-30', '2020-04-30']):
            eps = {'AAA': 0.5, 'FIB.MX': 0.3, 'GBR.L': 0.1, 'ADR': 5.0}[t]
            ev.append({'ticker': t, 'fecha_reporte': utc(f + 'T20:30'), 'eps_estimado': eps * 0.9, 'eps_reportado': eps, 'sorpresa_pct': 10.0, 'estimada': False, 'conocido': utc(f + 'T20:30'), 'capturado': utc('2026-09-14')})
    ev.append({'ticker': 'BBB.MX', 'fecha_reporte': utc('2020-02-20T21:00'), 'eps_estimado': 1.0, 'eps_reportado': -2.0, 'sorpresa_pct': -300.0, 'estimada': False, 'conocido': utc('2020-02-20T21:00'), 'capturado': utc('2026-09-14')})
    ev.append({'ticker': 'AAA', 'fecha_reporte': utc('2021-07-29T20:30'), 'eps_estimado': 0.6, 'eps_reportado': None, 'sorpresa_pct': None, 'estimada': True, 'conocido': utc('2021-06-01'), 'capturado': utc('2021-06-01')})
    escribir(pd.DataFrame(ev), os.path.join(d, 'eventos/resultados/anio=2020.parquet'))
    fred = pd.DataFrame([{'fecha': '2020-02-03', 'serie': 'DGS10', 'valor': 1.5, 'conocido': utc('2020-02-04T23:59:59')},
                         {'fecha': '2020-02-04', 'serie': 'DGS10', 'valor': 1.6, 'conocido': utc('2020-02-05T23:59:59')},
                         {'fecha': '2020-02-03', 'serie': 'DGS10', 'valor': 1.55, 'conocido': utc('2020-02-10T10:00')}])   # revisión de un dato viejo: no cambia el vigente
    escribir(fred, os.path.join(d, 'macro/fred/anio=2020.parquet'))
    fotos = pd.DataFrame([{'ticker': 'ADR', 'capturado': utc('2026-09-14'), 'financialCurrency': 'DKK', 'trailingPE': 10.0},
                          {'ticker': 'GBR.L', 'capturado': utc('2026-09-14'), 'financialCurrency': 'GBP', 'trailingPE': 25.0},
                          {'ticker': 'AAA', 'capturado': utc('2026-09-14'), 'financialCurrency': 'USD', 'trailingPE': 20.0}])
    escribir(fotos, os.path.join(d, 'fotos/valuacion/anio=2026/mes=09/dia=14.parquet'))
    def hecho(concepto, inicio, fin, valor, presentado, tax='us-gaap', unidad='USD'):
        return {'ticker': 'AAA', 'cik': 1, 'simbolo_sec': 'AAA', 'taxonomia': tax, 'concepto': concepto, 'unidad': unidad, 'inicio': inicio, 'fin': fin,
                'valor': float(valor), 'forma': '10-K', 'accn': 'x', 'presentado': presentado, 'conocido': utc(presentado + 'T23:59:59'), 'fuente': 'sec'}
    sec = [hecho('Assets', None, '2019-12-31', 2000, '2020-02-20'), hecho('Assets', None, '2018-12-31', 1600, '2020-02-20'),
           hecho('Liabilities', None, '2019-12-31', 1200, '2020-02-20'), hecho('StockholdersEquity', None, '2019-12-31', 800, '2020-02-20'),
           hecho('NetIncomeLoss', '2019-01-01', '2019-12-31', 100, '2020-02-20'), hecho('NetCashProvidedByUsedInOperatingActivities', '2019-01-01', '2019-12-31', 60, '2020-02-20'),
           hecho('EntityCommonStockSharesOutstanding', None, '2020-02-10', 10, '2020-02-20', tax='dei', unidad='shares'),
           hecho('WeightedAverageNumberOfDilutedSharesOutstanding', '2019-01-01', '2019-12-31', 10, '2020-02-20', unidad='shares')]
    escribir(pd.DataFrame(sec), os.path.join(d, 'fundamentales/sec/anio_presentado=2020.parquet'))


@unittest.skipUnless(HAY, 'requiere pandas y pyarrow')
class Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.d = cls.tmp.name
        almacen_sintetico(cls.d)
        cls.t = DS.construir(cls.d, desde='2020-01-01')

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def fila(self, t, f):
        r = self.t[(self.t['ticker'] == t) & (self.t['fecha'] == pd.Timestamp(f))]
        self.assertEqual(len(r), 1, f'{t} {f}'); return r.iloc[0]

    def test_indice_total_con_dividendo(self):
        g = pd.DataFrame({'close': [100.0, 100.0, 99.0], 'dividendo': [0, 0, 1.0]})
        self.assertEqual(list(DS.indice_total(g).round(6)), [1.0, 1.0, 1.0], 'caer 1 por pagar 1 de dividendo no es pérdida')

    def test_sin_informacion_del_futuro(self):
        """Cambiar precios posteriores al 2020-06-30 no puede cambiar ninguna variable de días anteriores (solo los objetivos)."""
        with tempfile.TemporaryDirectory() as d2:
            almacen_sintetico(d2, alterar_futuro=True)
            t2 = DS.construir(d2, desde='2020-01-01')
        corte = pd.Timestamp('2020-06-30')
        meta = DS.meta_columnas()
        variables = [c for c in self.t.columns if meta.get(c, {}).get('grupo') in ('A', 'B', 'D', 'E', 'L', 'R') and c != 'close']
        a = self.t[self.t['fecha'] <= corte].set_index(['fecha', 'ticker'])[variables]
        b = t2[t2['fecha'] <= corte].set_index(['fecha', 'ticker'])[variables]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-5)
        cambia = self.t[self.t['fecha'] == pd.Timestamp('2020-06-25')]['y_21'].to_numpy() != t2[t2['fecha'] == pd.Timestamp('2020-06-25')]['y_21'].to_numpy()
        self.assertTrue(cambia.any(), 'los objetivos sí miran al futuro')

    def test_reportes_entran_el_dia_siguiente(self):
        self.assertEqual(self.fila('AAA', '2020-01-30')['dias_desde_reporte'], 97, 'el reporte del 30-ene tras el cierre no se usa ese día')
        self.assertEqual(self.fila('AAA', '2020-01-31')['dias_desde_reporte'], 0)
        self.assertEqual(self.fila('AAA', '2021-06-30')['dias_desde_reporte'], 425, 'un reporte estimado nunca cuenta como ocurrido')

    def test_pe_reconstruido(self):
        a = self.fila('AAA', '2020-02-03')
        self.assertAlmostEqual(a['pe_reconstruido'], a['close'] / 2.0, places=3, msg='precio / suma de 4 utilidades')
        self.assertTrue(np.isnan(self.fila('AAA', '2020-01-02')['pe_reconstruido']), 'el 2-ene-2020 solo había 3 trimestres reportados')
        self.assertTrue(np.isnan(self.fila('BBB.MX', '2020-03-02')['pe_reconstruido']), 'con menos de 4 trimestres no hay P/E')
        self.assertTrue(np.isnan(self.fila('FIB.MX', '2020-06-01')['pe_reconstruido']), 'FIBRA sin P/E')
        self.assertTrue(np.isnan(self.fila('ADR', '2020-06-01')['pe_reconstruido']), 'reporta en otra moneda que su precio')
        g = self.fila('GBR.L', '2020-06-01')
        self.assertAlmostEqual(g['pe_reconstruido'], (g['close'] / 100) / 0.4, places=3, msg='peniques → libras')

    def test_objetivo_en_pesos_y_rango(self):
        r = self.fila('AAA', '2020-03-02')
        d = self.t[self.t['fecha'] == pd.Timestamp('2020-03-02')]
        self.assertTrue(((d['y_21_rango'] > 0) & (d['y_21_rango'] <= 1)).all())
        self.assertAlmostEqual(float(d['y_21_rel'].median()), 0.0, places=6)
        self.assertNotAlmostEqual(float(r['y_21']), float(r['y_21_nativo']), places=6, msg='en pesos incluye el tipo de cambio')

    def test_macro_tal_como_se_sabia(self):
        v = lambda f: float(self.fila('AAA', f)['tesoro_10a'])
        self.assertTrue(np.isnan(self.fila('AAA', '2020-02-04')['tesoro_10a']), 'el dato del 3 se conoce hasta el final del 4: no antes del día 5')
        self.assertAlmostEqual(v('2020-02-05'), 1.5); self.assertAlmostEqual(v('2020-02-06'), 1.6)
        self.assertAlmostEqual(v('2020-02-11'), 1.6, msg='revisar un dato viejo no cambia la observación más reciente')

    def test_minimo_hacia_adelante(self):
        w = pd.Series([1.0, 2.0, 0.5, 3.0, 4.0])
        r = DS.hacia_adelante_min(w, 2)
        self.assertEqual(list(r[:3]), [0.5, 0.5, 3.0], 'peor punto de t+1 … t+h, sin incluir el día t')
        self.assertTrue(r[3:].isna().all(), 'la ventana no cabe en el calendario')

    def test_objetivos_de_caida(self):
        t = self.t.dropna(subset=['y_21', 'min_ret_21', 'min_ret_63'])
        self.assertTrue(len(t) > 500)
        self.assertTrue((t['min_ret_21'] <= t['y_21'] + 1e-6).all(), 'la trayectoria incluye el día t+21')
        self.assertTrue((t['min_ret_63'] <= t['min_ret_21'] + 1e-6).all())
        ultimo = self.t['fecha'].max()
        self.assertTrue(self.t[self.t['fecha'] == ultimo]['min_ret_21'].isna().all())
        a, b, g = self.fila('AAA', '2020-03-02'), self.fila('BBB.MX', '2020-03-02'), self.fila('GBR.L', '2020-03-02')
        self.assertAlmostEqual(a['ret_sector_21'], b['y_21'], places=5, msg='sector sin la propia emisora')
        self.assertAlmostEqual(a['y_21_aj_sector'], a['y_21'] - b['y_21'], places=5)
        self.assertAlmostEqual(a['ret_mercado_21'], g['ret_mercado_21'], places=6, msg='extranjeras: S&P 500 en pesos')
        self.assertNotAlmostEqual(a['ret_mercado_21'], b['ret_mercado_21'], places=6, msg='mexicanas: IPC')
        beta = min(max(a['beta_mercado_1a'], 0), 2) if not np.isnan(a['beta_mercado_1a']) else 1.0
        self.assertAlmostEqual(a['y_21_aj_mercado'], a['y_21'] - beta * a['ret_mercado_21'], places=4)

    def test_variables_de_caida(self):
        b = self.t[(self.t['ticker'] == 'BBB.MX')].dropna(subset=['ivol_60d', 'vol_60d'])
        self.assertTrue(len(b) > 100)
        self.assertTrue((b['ivol_60d'] <= b['vol_60d'] * 1.001).all(), 'la volatilidad idiosincrática no supera la total (misma moneda)')
        r = self.fila('AAA', '2020-06-01')
        self.assertGreaterEqual(r['max_ret_1m'], r['min_ret_1m'])
        self.assertTrue(np.isnan(r['runup_2a']), 'el almacén sintético no tiene 2 años de historia antes de 2020-06')
        self.assertFalse(np.isnan(r['runup_1a']))
        self.assertEqual(self.fila('BBB.MX', '2020-02-24')['n_sorpresas_neg_4'], 1)
        s = self.fila('AAA', '2020-01-31')
        self.assertEqual(s['dias_al_proximo_est'], 90); self.assertEqual(s['ventana_reporte'], 1)
        self.assertEqual(self.fila('AAA', '2020-03-16')['ventana_reporte'], 0)
        self.assertEqual(self.fila('AAA', '2021-06-30')['dias_al_proximo_est'], 0, 'reporte vencido: riesgo de evento inminente')

    def test_sector_sin_la_propia_emisora(self):
        claves = pd.DataFrame({'fecha': [1, 1, 1, 1, 2, 2, 3], 'sector': ['a'] * 7})
        v = [10.0, 20.0, 30.0, np.nan, 5.0, 7.0, 9.0]
        med = DS.mediana_sin_propia(v, claves)
        self.assertEqual(list(med[:3]), [25.0, 20.0, 15.0], 'mediana de las otras dos')
        self.assertTrue(np.isnan(med[3])); self.assertEqual(list(med[4:6]), [7.0, 5.0]); self.assertTrue(np.isnan(med[6]), 'sola en su grupo')
        g = pd.DataFrame({'fecha': [1, 1, 1], 'sector': ['a'] * 3, 'x': [1.0, 2.0, 6.0]})
        self.assertEqual(list(DS.media_sin_propia(g, 'x')), [4.0, 3.5, 1.5])

    def test_nuevas_de_v3(self):
        t = self.t
        self.assertIn('y_21_rango_pais', t); self.assertIn('sue', t); self.assertIn('car_3d', t); self.assertIn('reporte_atrasado', t)
        self.assertEqual(self.fila('AAA', '2021-06-30')['reporte_atrasado'], 1)
        self.assertEqual(self.fila('AAA', '2020-01-31')['reporte_atrasado'], 0)
        d = t[t['fecha'] == pd.Timestamp('2020-03-02')]
        self.assertTrue(((d['y_21_rango_pais'] > 0) & (d['y_21_rango_pais'] < 1)).all())
        self.assertEqual(float(d.loc[d['ticker'] == 'GBR.L', 'y_21_rango_pais'].iloc[0]), 0.5, 'única emisora de su país: neutral, no 1.0')
        self.assertAlmostEqual(float(d['y_21_rango_libro'].mean()), 0.5, places=5)
        # CAR del reporte del 30-abr-2020 (tras el cierre): ventana de la sesión previa a dos sesiones después; se conoce al cierre del 4-may
        self.assertNotEqual(self.fila('AAA', '2020-05-04')['car_3d'], self.fila('AAA', '2020-05-05')['car_3d'])
        r = self.fila('AAA', '2021-06-30')
        self.assertLessEqual(r['max_dd_1a'], r['dist_max_52s'] + 1e-6, 'la peor caída del año no es menor que la caída de hoy')

    def test_deslistamiento(self):
        cal = pd.bdate_range('2020-01-01', periods=40)
        it = pd.DataFrame({'X.MX': np.r_[np.linspace(1, 0.8, 20), [np.nan] * 20]}, index=cal)
        real = it.notna()
        largo = pd.DataFrame({'fecha': cal[:20], 'ticker': 'X.MX', 'y_21': np.nan, 'min_ret_21': np.nan, 'y_63': np.nan, 'min_ret_63': np.nan})
        uni = pd.DataFrame([{'ticker': 'X.MX', 'salida': '2020-01', 'causa': 'concurso mercantil'}])
        out, reg = DS.aplicar_deslistamiento(largo, uni, it, real)
        self.assertAlmostEqual(out.loc[0, 'y_21'], 0.8 * 0.7 - 1, places=6)
        self.assertAlmostEqual(out.loc[0, 'min_ret_21'], 0.8 * 0.7 - 1, places=6, msg='la trayectoria incluye el −30 % de la salida')
        self.assertEqual(reg[0]['rendimiento_deslistamiento'], -0.30)
        oferta = DS.aplicar_deslistamiento(largo.copy(), uni.assign(causa='oferta de su controladora'), it, real)[0]
        self.assertAlmostEqual(oferta.loc[0, 'y_21'], -0.2, places=6)

    def test_tramos_excluidos(self):
        p = pd.DataFrame({'ticker': ['URBI.MX', 'URBI.MX', 'AAA'], 'fecha': ['2024-04-30', '2024-05-10', '2024-05-10']})
        q = DS.excluir_tramos(p)
        self.assertEqual(list(zip(q['ticker'], q['fecha'])), [('URBI.MX', '2024-04-30'), ('AAA', '2024-05-10')])

    def test_grupo_d_estados_de_la_sec(self):
        self.assertTrue(np.isnan(self.fila('AAA', '2020-02-20')['apalancamiento_libros']), 'presentado el 20-feb: entra hasta el día siguiente')
        r = self.fila('AAA', '2020-02-21')
        self.assertAlmostEqual(r['apalancamiento_libros'], 0.6, places=5); self.assertAlmostEqual(r['devengos_4t'], 40 / 1800, places=6)
        self.assertAlmostEqual(r['crec_activos_1a'], 0.25, places=5)
        # Yahoo ajusta el historial con el split 2:1 del 2020-06-01: el precio ajustado del 21-feb es la mitad del que se veía ese día
        self.assertAlmostEqual(r['pasivos_mercado'], 1200 / (1200 + 10 * r['close'] * 2), places=4)
        s = self.fila('AAA', '2020-06-02')
        self.assertAlmostEqual(s['libros_mercado'], 800 / (10 * s['close'] * 2), places=4, msg='acciones de portada (antes del split) × precio en la base de hoy × 2')
        self.assertTrue(np.isnan(self.fila('BBB.MX', '2020-06-02')['roa_4t']), 'sin reportes en la SEC')
        self.assertEqual(self.fila('AAA', '2021-06-30')['antiguedad_estado_dias'], 547)
        original = DS.ANTIGUEDAD_MAX_ESTADO
        try:
            DS.ANTIGUEDAD_MAX_ESTADO = 400
            t = DS.construir(self.d, desde='2021-06-01')
        finally:
            DS.ANTIGUEDAD_MAX_ESTADO = original
        self.assertTrue(t.loc[t['ticker'] == 'AAA', 'roa_4t'].isna().all(), 'un estado más viejo que el máximo no se usa')
        meta = DS.meta_columnas()
        self.assertTrue(meta['roa_4t']['publicable']); self.assertFalse(meta['pasivos_mercado']['publicable'])

    def test_medio_diferencial_recupera_una_amplitud_conocida(self):
        """Pedidos 99 y 184: se simula un mercado con amplitud conocida —el precio eficiente sigue una caminata y el cierre cae en la
        punta de compra o en la de venta al azar— y el estimador de Abdi-Ranaldo tiene que devolver la mitad de esa amplitud."""
        rng = np.random.default_rng(11)
        n, amplitud = 4000, 0.008                                    # 0.8 % de amplitud: el medio diferencial es 0.4 %
        cal = pd.bdate_range('2006-01-02', periods=n)
        eficiente = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        lado = rng.choice([-1.0, 1.0], n)                              # el cierre pega en una punta o en la otra
        close = eficiente * (1 + lado * amplitud / 2)
        rango = np.abs(rng.normal(0, 0.012, n))                        # el máximo y el mínimo del día rodean al precio eficiente
        m = {'close': pd.DataFrame({'X': close}, index=cal), 'real': pd.DataFrame({'X': True}, index=cal),
             'high': pd.DataFrame({'X': eficiente * (1 + rango)}, index=cal), 'low': pd.DataFrame({'X': eficiente * (1 - rango)}, index=cal)}
        est = DS.medio_diferencial(m, ventana=750, minimo=250)['medio_dif_ar_750d']['X'].dropna()
        self.assertAlmostEqual(float(est.median()), amplitud / 2, delta=0.0008,
                               msg=f'el estimador debe devolver la mitad de la amplitud; dio {float(est.median()):.5f}')
        # sin diferencial (el cierre es el precio eficiente) el estimador no puede inventar uno
        m['close'] = pd.DataFrame({'X': eficiente}, index=cal)
        cero = DS.medio_diferencial(m, ventana=750, minimo=250)['medio_dif_ar_750d']['X'].dropna()
        self.assertLess(float(cero.median()), 0.0008, 'sin diferencial el estimador debe quedar cerca de cero')

    def test_medio_diferencial_solo_empareja_dias_reales(self):
        """El par es (día real anterior, hoy): un feriado de su bolsa no es un día de operación, y un hueco más largo que el arrastre
        deja el par sin valor (mismo criterio que rend_dias_reales)."""
        cal = pd.bdate_range('2020-01-01', periods=10)
        x = pd.Series([1.0, 2, np.nan, 4, 5, 6, 7, 8, 9, 10], index=cal)
        real = pd.DataFrame({'X': x.notna()})
        ant = DS.valor_previo_real(pd.DataFrame({'X': x}), real)['X']
        self.assertEqual(ant.iloc[3], 2.0, 'salta el feriado y toma el día real anterior')
        self.assertTrue(pd.isna(ant.iloc[2]), 'un día que no es real no tiene par')
        self.assertTrue(pd.isna(ant.iloc[0]), 'el primer día no tiene anterior')
        largo = pd.DataFrame({'X': [True, True] + [False] * 7 + [True]}, index=cal)
        self.assertTrue(pd.isna(DS.valor_previo_real(pd.DataFrame({'X': np.arange(10.0)}), largo)['X'].iloc[-1]),
                        f'un hueco de más de {DS.RELLENO_MAX + 1} días no se empareja')

    def test_momentos_en_dias_reales(self):
        """Un feriado de la bolsa de la emisora no es un rendimiento 0: volatilidad y beta iguales a las de su propio calendario (M15)."""
        cal = pd.bdate_range('2020-01-01', periods=12)
        mercado = pd.Series([100, 101, 99, 102, 104, 103, 101, 105, 107, 106, 108, 110], index=cal, dtype=float)
        precio = pd.Series([10, 10.2, np.nan, 10.5, 10.9, 10.7, np.nan, np.nan, 11.4, 11.2, 11.5, 11.9], index=cal)
        real = pd.DataFrame({'X': precio.notna()})
        nivel = pd.DataFrame({'X': precio.ffill()})
        r = DS.rend_dias_reales(nivel, real)['X']
        propio = precio.dropna()
        pd.testing.assert_series_equal(r.dropna(), propio.pct_change().dropna(), check_names=False)
        rm = DS.rend_dias_reales(mercado, real)['X'].dropna()
        np.testing.assert_allclose(rm.to_numpy(), mercado[propio.index].pct_change().dropna().to_numpy(), err_msg='el índice sobre el mismo tramo')
        cov, var = DS.momentos_pares(pd.DataFrame({'X': r}), pd.DataFrame({'X': DS.rend_dias_reales(mercado, real)['X']}), 12, 3)
        rp, rmp = propio.pct_change().dropna(), mercado[propio.index].pct_change().dropna()
        self.assertAlmostEqual(cov['X'].iloc[-1], np.cov(rp, rmp)[0, 1], places=12)
        self.assertAlmostEqual(var['X'].iloc[-1], rmp.var(), places=12)
        largo = pd.DataFrame({'X': [1.0] + [np.nan] * (DS.RELLENO_MAX + 1) + [2.0]}, index=pd.bdate_range('2020-01-01', periods=DS.RELLENO_MAX + 3))
        self.assertTrue(DS.rend_dias_reales(largo.ffill(limit=DS.RELLENO_MAX), largo.notna())['X'].isna().all(), 'hueco mayor que el arrastre: nulo')

    def test_columnas_vacias(self):
        t = pd.DataFrame({'vol_60d': [np.nan, np.nan], 'ret_1m': [0.1, np.nan], 'y_21': [np.nan, np.nan]})
        self.assertEqual(DS.columnas_vacias(t), ['vol_60d'], 'solo variables del modelo; un objetivo vacío no es alarma')

    def test_eps_en_base_de_hoy(self):
        """Un reporte bajado ANTES de un split queda en la base vieja en el almacén: se pasa a la base de hoy (M21)."""
        splits = pd.DataFrame({'ticker': ['X'], 'fecha': [pd.Timestamp('2021-01-04')], 'indice': [4.0]})
        base = dict(ticker='X', estimada=False, eps_estimado=1.0, sorpresa_pct=1.0)
        res = pd.DataFrame([{**base, 'fecha_reporte': f'2020-{m:02d}-15', 'eps_reportado': 4.0, 'conocido': f'2020-{m:02d}-15', 'capturado': '2020-12-01'}
                            for m in (1, 4, 7, 10)] + [{**base, 'fecha_reporte': '2021-01-20', 'eps_reportado': 1.0, 'conocido': '2021-01-20', 'capturado': '2021-02-01'}])
        e = DS.eventos_resultados(res, splits)
        self.assertAlmostEqual(e['eps_4t'].iloc[3], 4.0, msg='cuatro trimestres de 4.0 en base vieja = 1.0 cada uno en base de hoy')
        self.assertAlmostEqual(e['eps_4t'].iloc[4], 4.0, msg='sin salto artificial al cruzar el split')

    def test_liquidez(self):
        r = self.fila('BBB.MX', '2020-06-01')
        self.assertGreater(r['importe_20d'], 0); self.assertAlmostEqual(r['log_importe_20d'], np.log(r['importe_20d']), places=4)
        self.assertEqual(r['prop_ceros_60d'], 0); self.assertEqual(r['dias_sin_volumen_20d'], 0)
        m = {'real': pd.DataFrame({'X': [True] * 40 + [False, True] * 10}), 'vol': pd.DataFrame({'X': [0.0] * 30 + [100.0] * 30}),
             'close': pd.DataFrame({'X': [10.0] * 35 + list(np.linspace(10, 12, 25))})}
        f = DS.grupo_liquidez(m, m['close'])
        self.assertEqual(f['dias_sin_dato_20d']['X'].iloc[-1], 10, 'días sin dato en las últimas 20 filas')
        self.assertEqual(f['importe_20d']['X'].iloc[25], 0, 'sin volumen el importe operado es cero')
        self.assertTrue(np.isnan(f['log_importe_20d']['X'].iloc[25]))
        self.assertAlmostEqual(f['importe_20d']['X'].iloc[39], m['close']['X'].iloc[30:40].sum() * 100 / 20, msg='media de 20 días con dato: 10 con volumen y 10 sin volumen (la mediana daría 0 o arbitrario)')
        self.assertGreater(f['prop_ceros_60d']['X'].iloc[40], 0.8, 'precio rancio: rendimientos cero en días con dato')

    def test_regimen(self):
        dias = pd.bdate_range('2019-01-01', '2020-12-31')
        ipc = pd.Series(100.0, index=dias); spx = pd.Series(100.0, index=dias)
        ipc[dias >= pd.Timestamp('2020-03-02')] = 85.0      # −15 %: estrés
        spx[dias >= pd.Timestamp('2020-06-01')] = 75.0      # −25 %: crisis
        close = pd.DataFrame({'^MXX': ipc, '^GSPC': spx})
        vix = pd.Series(15.0, index=dias); vix[dias == pd.Timestamp('2020-01-15')] = 35.0
        r = DS.regimen_mercado(close, pd.DataFrame({'vix': vix}))['regimen']
        self.assertEqual(r[pd.Timestamp('2020-01-14')], 'tranquilo'); self.assertEqual(r[pd.Timestamp('2020-01-15')], 'crisis')
        self.assertEqual(r[pd.Timestamp('2020-03-02')], 'estres'); self.assertEqual(r[pd.Timestamp('2020-06-01')], 'crisis')
        self.assertIn(self.t['regimen'].dropna().iloc[0], ('tranquilo', 'estres', 'crisis'))

    def test_meta_cubre_todas_las_columnas(self):
        meta = DS.meta_columnas()
        faltan = [c for c in self.t.columns if c not in meta]
        self.assertEqual(faltan, [])
        self.assertFalse(meta['vix']['publicable']); self.assertTrue(meta['cetes28']['publicable']); self.assertFalse(meta['ret_1m']['publicable'])
        with tempfile.TemporaryDirectory() as sal:
            m = DS.escribir(self.t, sal, self.d, '2020-01-01')
            self.assertEqual(m['sin_pe_por_moneda'], ['ADR'])
            self.assertTrue(os.path.exists(os.path.join(sal, 'dataset/anio=2020/parte.parquet')))


if __name__ == '__main__':
    unittest.main()
