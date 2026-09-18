"""Pruebas de laboratorio/entrenar.py: purga del walk-forward, pliegue externo sellado y un modelo que encuentra una señal plantada."""
import json, os, sys, tempfile, unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'laboratorio'))
try:
    import numpy as np
    import pandas as pd
    import sklearn  # noqa: F401
    import entrenar as EN
    import evaluar as EV
    import evaluar_caidas as EC
    HAY = True
except ImportError:
    HAY = False


def tabla_sintetica(semilla=7, emisoras=30, desde='2008-01-01', hasta='2016-12-31'):
    """Días hábiles × emisoras. y_21_rango_libro sigue a `senal` con ruido; la caída fuerte es más probable con `senal` baja; `macro` es de fecha."""
    rng = np.random.default_rng(semilla)
    cal = pd.bdate_range(desde, hasta)
    f = pd.DataFrame({'fecha': np.repeat(cal, emisoras), 'ticker': np.tile([f'E{i:02d}' for i in range(emisoras)], len(cal))})
    f['pais'] = 'MX'
    f['senal'] = rng.standard_normal(len(f))
    f['ruido'] = rng.standard_normal(len(f))
    f['macro'] = f['fecha'].map(dict(zip(cal, rng.standard_normal(len(cal)))))
    latente = f['senal'] + 0.8 * rng.standard_normal(len(f))
    f['y_21_rango_libro'] = (latente.groupby(f['fecha']).rank() - 0.5) / latente.groupby(f['fecha']).transform('count')
    prob = 1 / (1 + np.exp(2.0 + 1.5 * f['senal']))
    f['min_ret_63'] = np.where(rng.random(len(f)) < prob, -0.3, -0.05)
    return f


@unittest.skipUnless(HAY, 'requiere numpy, pandas y scikit-learn')
class PruebasEntrenar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orig = (EN.PRIMER_ANIO_PRUEBA, EN.PLIEGUE_EXTERNO_DESDE, dict(EN.HIPER), EN.MESES_VALIDACION, EN.BOLSAS, EN.REPETICIONES_PERMUTACION)
        EN.PRIMER_ANIO_PRUEBA, EN.PLIEGUE_EXTERNO_DESDE, EN.MESES_VALIDACION = 2011, '2015-01-01', 12
        EN.BOLSAS, EN.REPETICIONES_PERMUTACION = 1, 2
        EN.HIPER.update(min_samples_leaf=20, max_iter=60, learning_rate=0.1)
        cls.t = tabla_sintetica()
        cls.variables = ['senal', 'ruido', 'macro']

    @classmethod
    def tearDownClass(cls):
        EN.PRIMER_ANIO_PRUEBA, EN.PLIEGUE_EXTERNO_DESDE, hiper, EN.MESES_VALIDACION, EN.BOLSAS, EN.REPETICIONES_PERMUTACION = cls.orig
        EN.HIPER.clear(); EN.HIPER.update(hiper)

    def test_particiones_con_purga_y_sin_pliegue(self):
        cal = EN.calendario(self.t)
        d = EN.muestra(self.t, 'ranking')
        partes = EN.particiones(d['fecha'], cal, 21, purga=63, primer_anio=2011, pliegue_desde='2015-01-01')
        self.assertEqual([p['anio'] for p in partes], [2011, 2012, 2013, 2014])
        inicio = cal.searchsorted(pd.Timestamp('2015-01-01'))
        for p in partes:
            self.assertLessEqual(EN.fin_objetivo(cal, p['entrenar'], 21).max() + 63, cal.searchsorted(p['prueba'][0]), 'purga de 63 sesiones')
            self.assertLess(EN.fin_objetivo(cal, p['prueba'], 21).max(), inicio, 'ningún objetivo de prueba entra al pliegue externo')
            self.assertTrue((p['entrenar'] < p['prueba'][0]).all())
        self.assertLess(partes[-1]['prueba'][-1], pd.Timestamp('2014-12-15'), 'diciembre de 2014 queda fuera: su objetivo cruza a 2015')

    def test_ranking_encuentra_la_senal_fuera_de_muestra(self):
        pz, rep = EN.walk_forward(self.t, 'ranking', self.variables)
        self.assertTrue((pz['fecha'] < pd.Timestamp('2015-01-01')).all())
        self.assertGreaterEqual(pz['fecha'].min(), pd.Timestamp('2011-01-01'))
        self.assertGreater(rep['metrica_fuera_de_muestra'], 0.3, 'IC medio con señal plantada')
        imp = rep['importancia_permutacion']
        self.assertEqual(next(iter(imp)), 'senal')
        self.assertTrue(imp['macro']['de_fecha']); self.assertFalse(imp['senal']['de_fecha'])
        self.assertLess(abs(imp['ruido']['caida_metrica']), 0.05)
        self.assertEqual(rep['inicio_evaluacion'][:4], '2011')

    def test_caidas_ordena_mejor_que_la_tasa_base(self):
        pz, rep = EN.walk_forward(self.t, 'caidas', self.variables, con_importancia=False)
        j = self.t.merge(pz, on=['fecha', 'ticker'])
        y = (j['min_ret_63'] <= -0.2).astype(float)
        self.assertGreater(EC.precision_promedio(y, j['puntaje']), 1.5 * y.mean())
        self.assertTrue(((j['puntaje'] >= 0) & (j['puntaje'] <= 1)).all(), 'probabilidades')

    def test_bolsas_por_mes(self):
        cal = EN.calendario(self.t); d = EN.muestra(self.t, 'ranking')
        meses = pd.DatetimeIndex(sorted(d['fecha'].unique()))[:48]
        modelos, iters = EN.ajustar(d, meses, self.variables, 'regresion', cal, 21, bolsas=3)
        self.assertEqual(len(modelos), 3); self.assertEqual(len(iters), 3)

    def test_pliegue_externo_sellado(self):
        pz, info = EN.pliegue_externo(self.t, 'ranking', self.variables)
        self.assertGreaterEqual(pz['fecha'].min(), pd.Timestamp('2015-01-01'))
        cal = EN.calendario(self.t)
        self.assertLess(pd.Timestamp(info['ultimo_mes_entrenar']), pd.Timestamp('2015-01-01'))
        self.assertLessEqual(EN.fin_objetivo(cal, [pd.Timestamp(info['ultimo_mes_entrenar'])], 21)[0] + EN.PURGA, cal.searchsorted(pd.Timestamp('2015-01-01')))
        with tempfile.TemporaryDirectory() as d:
            e = EN.registrar_pliegue(d, {'nombre': 'uno', 'tarea': 'ranking'})
            self.assertEqual(e['apertura'], 1)
            with self.assertRaises(SystemExit):
                EN.registrar_pliegue(d, {'nombre': 'dos', 'tarea': 'ranking'})
            self.assertEqual(EN.registrar_pliegue(d, {'nombre': 'uno', 'tarea': 'caidas'})['apertura'], 1, 'cada tarea tiene su pliegue')
            self.assertEqual(EN.registrar_pliegue(d, {'nombre': 'dos', 'tarea': 'ranking'}, permitir_otra=True)['apertura'], 2)
            with open(os.path.join(d, EN.REGISTRO_PLIEGUE), encoding='utf-8') as fh:
                self.assertEqual(len(fh.readlines()), 3)

    def test_objetivo_con_promedio_por_grupo_se_rechaza(self):
        t = self.t.copy(); t['y_21_rango_libro'] = t.groupby('fecha')['senal'].rank(pct=True)
        t.loc[t['ticker'] == 'E00', 'y_21_rango_libro'] = 1.0   # como la única emisora de un país con rango / n
        t['y_21_rango_libro'] = t['y_21_rango_libro'] + 0.05
        with self.assertRaises(ValueError):
            EN.muestra(t, 'ranking')

    def test_arboles_un_error_estandar(self):
        rng = np.random.default_rng(3)
        meses = 24
        verdad = np.r_[np.linspace(0, 0.05, 50), np.full(550, 0.05)]           # la señal se satura en 50 árboles
        curvas = verdad[:, None] + rng.normal(0, 0.03, size=(1, meses)) + rng.normal(0, 0.002, size=(600, meses))
        n = EN.arboles_un_error_estandar(curvas)
        self.assertLessEqual(n, 90, 'dentro de un error estándar del máximo gana el modelo más chico')
        self.assertGreater(n, 5, 'con el error estándar de la diferencia emparejada no colapsa a 1 árbol (tanda 6, M32)')
        self.assertGreaterEqual(n, 1)
        self.assertEqual(EN.arboles_un_error_estandar(np.array([[0.0] * 5, [1.0] * 5])), 2, 'sin empate estadístico gana el mejor')

    def test_importancia_con_repeticiones(self):
        pz, rep = EN.walk_forward(self.t, 'ranking', self.variables)
        x = rep['importancia_permutacion']['senal']
        self.assertEqual(x['repeticiones'], 2); self.assertIsNotNone(x['sd'])

    def test_variables_del_modelo(self):
        meta = {'ret_1m': {'grupo': 'A'}, 'close': {'grupo': 'A'}, 'eps_est': {'grupo': 'C'}, 'y_21': {'grupo': 'objetivo'},
                'ticker': {'grupo': 'identidad'}, 'regimen': {'grupo': 'R'}, 'dd_ipc_52s': {'grupo': 'R'}, 'cetes28': {'grupo': 'E'}}
        self.assertEqual(EN.variables_modelo(meta), ['ret_1m', 'dd_ipc_52s', 'cetes28'], 'sin precio crudo, analistas, objetivos, identidad ni régimen categórico')


if __name__ == '__main__':
    unittest.main()
