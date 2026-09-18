"""Pruebas de laboratorio/fundamentales.py: 12 meses desde acumulados, años de 52/53 semanas, reexpresiones que no cambian el pasado,
conceptos alternativos, conceptos abandonados y razones."""
import os, sys, unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'laboratorio'))
try:
    import numpy as np
    import pandas as pd
    import fundamentales as FU
    HAY = True
except ImportError:
    HAY = False


def hecho(concepto, inicio, fin, valor, presentado, tax='us-gaap', unidad='USD', ticker='AAA'):
    return {'ticker': ticker, 'simbolo_sec': ticker, 'taxonomia': tax, 'concepto': concepto, 'unidad': unidad, 'inicio': inicio, 'fin': fin,
            'valor': float(valor), 'conocido': pd.Timestamp(presentado + 'T23:59:59', tz='UTC')}


def sec_sintetico():
    h = []
    # 10-K 2019 (presentado 2020-02-20): anual 2019 y 2018, saldos 2019 y 2018
    h += [hecho('NetIncomeLoss', '2019-01-01', '2019-12-31', 100, '2020-02-20'), hecho('NetIncomeLoss', '2018-01-01', '2018-12-31', 80, '2020-02-20'),
          hecho('Revenues', '2019-01-01', '2019-12-31', 1000, '2020-02-20'), hecho('Revenues', '2018-01-01', '2018-12-31', 900, '2020-02-20'),
          hecho('Assets', None, '2019-12-31', 2000, '2020-02-20'), hecho('Assets', None, '2018-12-31', 1800, '2020-02-20'),
          hecho('StockholdersEquity', None, '2019-12-31', 800, '2020-02-20'),
          hecho('EntityCommonStockSharesOutstanding', None, '2020-02-10', 50, '2020-02-20', tax='dei', unidad='shares'),
          hecho('WeightedAverageNumberOfDilutedSharesOutstanding', '2019-01-01', '2019-12-31', 48, '2020-02-20', unidad='shares'),
          hecho('WeightedAverageNumberOfDilutedSharesOutstanding', '2018-01-01', '2018-12-31', 45, '2020-02-20', unidad='shares')]
    # 10-Q Q1 2020 (presentado 2020-05-01, trimestre de 52 semanas que cierra el 28-mar): acumulado 2020 y comparativo 2019
    h += [hecho('NetIncomeLoss', '2020-01-01', '2020-03-28', 30, '2020-05-01'), hecho('NetIncomeLoss', '2019-01-01', '2019-03-30', 20, '2020-05-01'),
          hecho('RevenueFromContractWithCustomerExcludingAssessedTax', '2020-01-01', '2020-03-28', 260, '2020-05-01'),
          hecho('RevenueFromContractWithCustomerExcludingAssessedTax', '2019-01-01', '2019-03-30', 240, '2020-05-01'),
          hecho('Assets', None, '2020-03-28', 2100, '2020-05-01')]
    # split 2:1 en 2020: el 10-Q del Q3 trae el acumulado de 9 meses y su comparativo YA reexpresado
    h += [hecho('WeightedAverageNumberOfDilutedSharesOutstanding', '2020-01-01', '2020-09-26', 100, '2020-11-01', unidad='shares'),
          hecho('WeightedAverageNumberOfDilutedSharesOutstanding', '2019-01-01', '2019-09-28', 94, '2020-11-01', unidad='shares')]
    # reexpresión del anual 2019 presentada en 2021: no puede cambiar lo que se sabía en 2020
    h += [hecho('NetIncomeLoss', '2019-01-01', '2019-12-31', 90, '2021-02-20')]
    return pd.DataFrame(h)


@unittest.skipUnless(HAY, 'faltan numpy/pandas')
class PruebasFundamentales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = FU.estados_por_presentacion(sec_sintetico()).set_index('conocido')

    def fila(self, f):
        return self.e.loc[pd.Timestamp(f + 'T23:59:59', tz='UTC')]

    def test_anual(self):
        r = self.fila('2020-02-20')
        self.assertEqual(r['utilidad_neta_4t'], 100); self.assertEqual(r['ingresos_4t'], 1000); self.assertEqual(r['ingresos_4t_hace_1a'], 900)
        self.assertEqual(r['activos'], 2000); self.assertEqual(r['activos_hace_1a'], 1800)
        self.assertEqual(r['fin_periodo'], pd.Timestamp('2019-12-31'))

    def test_doce_meses_desde_acumulado_y_52_semanas(self):
        r = self.fila('2020-05-01')
        self.assertEqual(r['utilidad_neta_4t'], 100 + 30 - 20, 'anual previo + acumulado − acumulado del año anterior')
        self.assertEqual(r['ingresos_4t'], 1000 + 260 - 240, 'concepto alternativo con el periodo más reciente')
        self.assertEqual(r['activos'], 2100); self.assertEqual(r['fin_periodo'], pd.Timestamp('2020-03-28'))

    def test_acciones_diluidas_con_split(self):
        r = self.fila('2020-02-20')
        self.assertEqual((r['acciones_diluidas'], r['acciones_diluidas_hace_1a']), (48, 45))
        q = self.fila('2020-11-01')
        self.assertEqual((q['acciones_diluidas'], q['acciones_diluidas_hace_1a']), (100, 94), 'comparativo reexpresado de la misma presentación')

    def test_reexpresion_no_cambia_el_pasado(self):
        self.assertEqual(self.fila('2020-05-01')['utilidad_neta_4t'], 110)
        self.assertEqual(self.fila('2021-02-20')['utilidad_neta_4t'], 90 + 30 - 20, 'con la reexpresión conocida se usa la versión nueva')

    def test_concepto_abandonado_no_se_arrastra(self):
        h = sec_sintetico()
        h = h[h['concepto'] != 'RevenueFromContractWithCustomerExcludingAssessedTax']
        h = pd.concat([h, pd.DataFrame([hecho('Assets', None, '2021-12-31', 2500, '2022-02-20'), hecho('NetIncomeLoss', '2021-01-01', '2021-12-31', 120, '2022-02-20')])])
        e = FU.estados_por_presentacion(h).set_index('conocido')
        r = e.loc[pd.Timestamp('2022-02-20T23:59:59', tz='UTC')]
        self.assertEqual(r['utilidad_neta_4t'], 120)
        self.assertTrue(np.isnan(r['ingresos_4t']), 'ingresos de 2019 no valen como ingresos de 2021')

    def test_razones(self):
        f = pd.DataFrame([{'pasivos': np.nan, 'activos': 2000.0, 'patrimonio': 800.0, 'caja': 100.0, 'utilidad_neta_4t': 100.0, 'ingresos_4t': 1000.0,
                           'flujo_operativo_4t': 60.0, 'capex_4t': 20.0, 'activos_hace_1a': 1600.0, 'ingresos_4t_hace_1a': 800.0, 'acciones': 110.0,
                           'acciones_hace_1a': 50.0, 'acciones_diluidas': 105.0, 'acciones_diluidas_hace_1a': 100.0, 'utilidad_operativa_4t': 150.0, 'gasto_intereses_4t': 30.0, 'fin_periodo': pd.Timestamp('2019-12-31')}])
        r = FU.razones(f, pd.Series([pd.Timestamp('2020-03-01')]), np.array([10.0]), np.array([False]))
        self.assertAlmostEqual(r['apalancamiento_libros'][0], 1200 / 2000, msg='pasivos = activos − patrimonio si no se reporta el total')
        self.assertAlmostEqual(r['devengos_4t'][0], 40 / 1800, msg='entre activos promedio (Sloan, 1996)'); self.assertAlmostEqual(r['crec_activos_1a'][0], 0.25)
        self.assertAlmostEqual(r['emision_neta_1a'][0], 0.05, msg='acciones diluidas contra su comparativo reexpresado')
        self.assertAlmostEqual(r['pasivos_mercado'][0], 1200 / (1200 + 1100)); self.assertAlmostEqual(r['libros_mercado'][0], 800 / 1100)
        self.assertEqual(r['cobertura_intereses_4t'][0], 5.0); self.assertEqual(r['antiguedad_estado_dias'][0], 61)
        self.assertAlmostEqual(r['nimta'][0], 100 / 2300); self.assertAlmostEqual(r['cashmta'][0], 100 / 2300)
        self.assertTrue(np.isnan(r['rentabilidad_bruta'][0]), 'sin utilidad bruta reportada')
        g = FU.razones(f.assign(utilidad_bruta_4t=400.0, pasivos_circulantes=300.0, deuda_lp=500.0), pd.Series([pd.Timestamp('2020-03-01')]), np.array([10.0]), np.array([False]))
        self.assertAlmostEqual(g['rentabilidad_bruta'][0], 0.2); self.assertAlmostEqual(g['_deuda_incumplimiento'][0], 250.0, msg='el pasivo circulante total no es deuda')
        base = f.assign(deuda_lp=500.0)
        con_total = FU.razones(base.assign(deuda_cp_total=120.0, deuda_cp_porcion_lp=50.0), pd.Series([pd.Timestamp('2020-03-01')]), np.array([10.0]), np.array([False]))
        self.assertAlmostEqual(con_total['_deuda_incumplimiento'][0], 370.0, msg='deuda de corto plazo total + ½ largo plazo')
        por_partes = FU.razones(base.assign(deuda_cp_porcion_lp=50.0, deuda_cp_prestamos=30.0), pd.Series([pd.Timestamp('2020-03-01')]), np.array([10.0]), np.array([False]))
        self.assertAlmostEqual(por_partes['_deuda_incumplimiento'][0], 330.0, msg='sin total: porción circulante del largo plazo + préstamos')
        self.assertEqual(r['cap_dudosa'][0], 0.0)
        fin = FU.razones(f, pd.Series([pd.Timestamp('2020-03-01')]), np.array([10.0]), np.array([True]))
        otra_clase = FU.razones(f.assign(acciones=11.0), pd.Series([pd.Timestamp('2020-03-01')]), np.array([10.0]), np.array([False]))
        self.assertAlmostEqual(otra_clase['libros_mercado'][0], 800 / 1050, msg='portada de una sola clase o en miles: se usa la mayor (diluidas)')
        self.assertEqual(otra_clase['cap_dudosa'][0], 1.0, 'la fila queda marcada, no se pierde')
        escala = FU.razones(f.assign(acciones_diluidas=105000.0), pd.Series([pd.Timestamp('2020-03-01')]), np.array([10.0]), np.array([False]))
        self.assertTrue(np.isnan(escala['emision_neta_1a'][0]), 'multiplicar las acciones por mil es un error de escala, no emisión')
        self.assertEqual(escala['cap_dudosa'][0], 1.0)
        sin_activos_previos = FU.razones(f.assign(activos_hace_1a=np.nan), pd.Series([pd.Timestamp('2020-03-01')]), np.array([10.0]), np.array([False]))
        self.assertTrue(np.isnan(sin_activos_previos['devengos_4t'][0]), 'sin activos de hace un año no hay promedio')
        self.assertTrue(np.isnan(fin['apalancamiento_libros'][0])); self.assertAlmostEqual(fin['roa_4t'][0], 0.05)


if __name__ == '__main__':
    unittest.main()
