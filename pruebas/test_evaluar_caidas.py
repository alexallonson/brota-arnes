"""Pruebas de laboratorio/evaluar_caidas.py: PR-AUC, precisión en k con empates, episodios, calibración sin mirar el futuro,
Brier skill contra la mejor referencia, compuertas y resultados que no se sobrescriben."""
import os, sys, tempfile, unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'laboratorio'))
try:
    import numpy as np
    import pandas as pd
    import evaluar_caidas as EC
    HAY = True
except ImportError:
    HAY = False

META = {'version': 'dataset-prueba', 'columnas': {
    'riesgo': {'grupo': 'A', 'fuente': 'yahoo', 'publicable': False},
    'consenso': {'grupo': 'C', 'fuente': 'yahoo', 'publicable': False},
    'min_ret_63': {'grupo': 'objetivo', 'fuente': 'yahoo', 'publicable': False}}}


def tabla_sintetica(n_emp=60, desde='2010-01-01', hasta='2016-12-31'):
    rng = np.random.default_rng(5)
    dias = pd.bdate_range(desde, hasta)
    tickers = [f'E{i:02d}' for i in range(n_emp)]
    riesgo = rng.standard_normal((len(dias), n_emp))
    prob = 1 / (1 + np.exp(-(riesgo * 1.5 - 2.5)))                    # ~10 % de caídas, más probables con riesgo alto
    cae = rng.random((len(dias), n_emp)) < prob
    min_ret = np.where(cae, -0.30, -0.05)
    t = pd.DataFrame({'fecha': np.repeat(dias, n_emp), 'ticker': np.tile(tickers, len(dias)), 'riesgo': riesgo.ravel(), 'min_ret_63': min_ret.ravel()})
    t['min_ret_21'] = t['min_ret_63']; t['y_63'] = t['min_ret_63'] / 2; t['y_21'] = t['y_63']
    t['y_63_aj_mercado'] = t['y_63']; t['ret_mercado_63'] = 0.0
    t['regimen'] = np.where(t['fecha'].dt.year == 2012, 'crisis', 'tranquilo')
    t['pais'] = 'US'; t['sector'] = 'x'; t['en_app'] = True
    t.loc[t['fecha'] > pd.Timestamp(hasta) - pd.Timedelta(days=90), ['min_ret_63', 'min_ret_21']] = np.nan   # sin desenlace al final
    return t


@unittest.skipUnless(HAY, 'faltan numpy/pandas')
class PruebasEvaluarCaidas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tabla_sintetica()
        kw = dict(universo_csv='/no/existe', objetivos=(EC.PRINCIPAL,), repeticiones=30)
        cls.buena = EC.evaluar(cls.t, cls.t['riesgo'], 'buena', ['riesgo'], META, **kw)
        cls.placebo = EC.evaluar(cls.t, np.random.default_rng(1).standard_normal(len(cls.t)), 'placebo', [], META, **kw)

    def test_precision_promedio(self):
        self.assertAlmostEqual(EC.precision_promedio([1, 0, 1, 0], [4, 3, 2, 1]), 0.5 * 1 + 0.5 * (2 / 3))
        self.assertAlmostEqual(EC.precision_promedio([1, 0, 1, 0, 0], [1, 1, 1, 1, 1]), 0.4, msg='todo empatado = tasa base')
        self.assertIsNone(EC.precision_promedio([0, 0], [1, 2]))
        self.assertAlmostEqual(EC.roc_auc([1, 1, 0, 0], [4, 3, 2, 1]), 1.0)

    def test_aciertos_en_k_con_empates(self):
        y = [1, 0, 1] + [0] * 7
        self.assertEqual(EC.aciertos_en_k(y, [0] * 10, 0.10), (1.0, 0.2), 'empate total: la marcada acierta en fracción de la tasa')
        self.assertEqual(EC.aciertos_en_k(y, [9, 1, 8] + [0] * 7, 0.20), (2.0, 2.0))

    def test_bootstrap_por_bloques(self):
        d = EC.muestra_mensual(self.t.assign(puntaje=self.t['riesgo']), EC.PRINCIPAL)[0]
        d = EC.calibrar(d, 63, self.t['fecha'], False)
        n = d['fecha'].nunique()
        rotaciones = EC.bootstrap(d, repeticiones=20, bloque=n)
        self.assertLess(np.nanstd(rotaciones['pr_auc']), 1e-12, 'un bloque del tamaño de la muestra solo la rota: siempre la misma muestra')
        self.assertIn('efecto_de_diseno', self.buena['objetivos'][EC.PRINCIPAL])

    def test_episodios(self):
        f = pd.DataFrame({'ticker': ['A'] * 5 + ['B'], 'mes_idx': [1, 2, 3, 10, 11, 1], 'evento': [1, 1, 1, 1, 0, 1]})
        self.assertEqual(EC.episodios(f), 3)

    def test_senal_real_supera_y_placebo_no(self):
        b, p = self.buena['objetivos'][EC.PRINCIPAL], self.placebo['objetivos'][EC.PRINCIPAL]
        base = b['total']['tasa_base']
        self.assertTrue(0.05 < base < 0.25)
        self.assertGreater(b['total']['pr_auc'], 1.5 * base); self.assertGreater(b['total']['lift_10'], 2)
        self.assertGreater(b['total']['brier_skill_vs_mejor_referencia'], 0.02)
        self.assertLess(abs(p['total']['lift_10'] - 1), 0.25)
        self.assertLess(p['total']['brier_skill_vs_mejor_referencia'], 0.01, 'el azar no aporta sobre la mejor referencia')
        lo, hi = b['intervalos_90']['pr_auc']; self.assertLess(lo, b['total']['pr_auc']); self.assertGreater(hi, b['total']['pr_auc'])
        self.assertIn('crisis', b['por_regimen']); self.assertEqual(len(b['por_anio']), 7)

    def test_calibracion_no_mira_el_futuro(self):
        d, h = EC.muestra_mensual(self.t.assign(puntaje=self.t['riesgo']), EC.PRINCIPAL)
        c1 = EC.calibrar(d, h, self.t['fecha'])
        corte = pd.Timestamp('2014-06-01')
        d2 = d.copy(); d2.loc[d2['fecha'] >= corte, 'evento'] = 1.0      # cambiar desenlaces desde el corte
        c2 = EC.calibrar(d2, h, self.t['fecha'])
        antes = c1['fecha'] <= corte
        pd.testing.assert_series_equal(c1.loc[antes, 'p_modelo'], c2.loc[antes, 'p_modelo'])
        despues = c1['fecha'] > corte + pd.Timedelta(days=150)
        self.assertFalse(np.allclose(c1.loc[despues, 'p_modelo'], c2.loc[despues, 'p_modelo']), 'los desenlaces ya conocidos sí se usan')
        self.assertTrue(c1.loc[c1['fecha'] < pd.Timestamp('2012-01-01'), 'p_modelo'].isna().all(), 'sin 24 meses conocidos no hay probabilidad')

    def test_murphy_suma_exactamente_el_brier(self):
        """M39: la descomposición de Murphy solo sirve si es una identidad. confiabilidad − resolución + incertidumbre = Brier."""
        rng = np.random.default_rng(5)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype(float)      # pronóstico perfectamente calibrado
        m = EC.murphy(y, p)
        self.assertAlmostEqual(m['confiabilidad'] - m['resolucion'] + m['incertidumbre'], m['brier_agrupado'], places=12,
                               msg='la identidad tiene que cerrar exacto sobre el pronóstico agrupado')
        self.assertLess(abs(m['residuo_agrupacion']), 0.01, 'y el residuo de agrupar tiene que ser chico y estar declarado')
        self.assertLess(m['confiabilidad'], 0.002, 'un pronóstico calibrado tiene confiabilidad ~0')
        self.assertGreater(m['resolucion'], 0.05, 'y resolución alta, porque sí distingue')
        # un pronóstico constante en la tasa base: no distingue nada, así que resolución 0 y Brier = incertidumbre
        cte = EC.murphy(y, np.full(len(y), y.mean()))
        self.assertAlmostEqual(cte['resolucion'], 0.0, places=12)
        self.assertAlmostEqual(cte['brier_agrupado'], cte['incertidumbre'], places=12)
        # y uno sistemáticamente corrido: misma resolución, peor confiabilidad
        corrido = EC.murphy(y, np.clip(p + 0.2, 0, 1))
        self.assertGreater(corrido['confiabilidad'], m['confiabilidad'])
        self.assertIsNone(EC.murphy([1.0, 0.0], [0.5, 0.5]), 'con menos casos que tramos no se descompone')

    def test_resolucion_invariante_a_transformaciones_monotonas(self):
        """Corrección de Mercado (M1): la resolución tiene que medir lo que el modelo distingue, no la escala en que lo dice. Con grupos
        por CUANTILES del pronóstico es invariante a cualquier transformación monótona; con tramos fijos de probabilidad, no."""
        rng = np.random.default_rng(5)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype(float)
        monotona = 1 / (1 + np.exp(-(3 * p - 1.5)))          # estrictamente creciente: no reordena nada
        a, b = EC.murphy(y, p), EC.murphy(y, monotona)
        self.assertAlmostEqual(a['resolucion'], b['resolucion'], places=12,
                               msg='con cuantiles del pronóstico la resolución no puede moverse por una transformación monótona')
        self.assertEqual(a['agrupacion'], 'cuantiles del pronóstico')
        # y con tramos fijos sí se mueve, que es el defecto que la corrección arregla
        c, d = EC.murphy(y, p, por_cuantiles=False), EC.murphy(y, monotona, por_cuantiles=False)
        self.assertGreater(abs(c['resolucion'] - d['resolucion']), 1e-5, 'con tramos fijos la escala contamina la resolución')

    def test_platt_condicionado_no_mira_el_futuro(self):
        """M39: el Platt se ajusta solo con meses cuyo desenlace ya se conocía. Cambiar el futuro no puede mover el pasado."""
        d, h = EC.muestra_mensual(self.t.assign(puntaje=self.t['riesgo']), EC.PRINCIPAL)
        c1 = EC.calibrar(d, h, self.t['fecha'])
        corte = pd.Timestamp('2014-06-01')
        d2 = d.copy(); d2.loc[d2['fecha'] >= corte, 'evento'] = 1.0
        c2 = EC.calibrar(d2, h, self.t['fecha'])
        antes = c1['fecha'] <= corte
        pd.testing.assert_series_equal(c1.loc[antes, 'p_platt'], c2.loc[antes, 'p_platt'])
        p = c1['p_platt'].dropna()
        self.assertTrue(len(p), 'se calcula en algún mes')
        self.assertTrue(((p >= 0) & (p <= 1)).all(), 'es una probabilidad')

    def test_nivel_no_mira_el_futuro_y_conserva_el_orden(self):
        """M1.4 de Mercado: la recalibración de solo nivel se ajusta con meses ya conocidos del mismo régimen, y al ser un
        desplazamiento monótono NO puede cambiar el orden. Si cambiara el orden, la predicción de resolución exacta sería falsa."""
        d, h = EC.muestra_mensual(self.t.assign(puntaje=self.t['riesgo']), EC.PRINCIPAL)
        c1 = EC.calibrar(d, h, self.t['fecha'])
        corte = pd.Timestamp('2014-06-01')
        d2 = d.copy(); d2.loc[d2['fecha'] >= corte, 'evento'] = 1.0
        c2 = EC.calibrar(d2, h, self.t['fecha'])
        antes = c1['fecha'] <= corte
        pd.testing.assert_series_equal(c1.loc[antes, 'p_nivel'], c2.loc[antes, 'p_nivel'])
        p = c1.dropna(subset=['p_nivel', 'p_modelo'])
        self.assertTrue(len(p), 'se calcula en algún mes')
        self.assertTrue(((p['p_nivel'] >= 0) & (p['p_nivel'] <= 1)).all(), 'es una probabilidad')
        for _, g in p.groupby(['fecha']):   # dentro de un mes el desplazamiento es único: el orden se conserva exactamente
            a = g['p_modelo'].rank(method='average').to_numpy()
            b = g['p_nivel'].rank(method='average').to_numpy()
            self.assertTrue((a == b).all(), 'un desplazamiento en log-odds no reordena')
        # p_nivel_dm es el diagnóstico declarado dentro de muestra: SÍ mira el futuro a propósito y por eso no se prueba aquí
        self.assertIn('p_nivel_dm', c1.columns)

    def test_desplazamiento_iguala_la_tasa(self):
        """El desplazamiento resuelve una ecuación con una sola raíz: la probabilidad media predicha tiene que quedar en la tasa real."""
        azar = np.random.default_rng(7)
        p = azar.uniform(0.01, 0.4, 400)
        for tasa in (0.02, 0.0925, 0.30):
            y = (azar.random(400) < tasa).astype(float)
            dl = EC.desplazamiento(p, y)
            z = EC._logit(p) + dl
            self.assertAlmostEqual(float((1 / (1 + np.exp(-z))).sum()), float(y.sum()), places=4)
        self.assertIsNone(EC.desplazamiento(p, np.zeros(400)), 'sin positivos no hay desplazamiento finito')
        self.assertIsNone(EC.desplazamiento(p, np.ones(400)), 'todo positivos tampoco')

    def test_compuertas(self):
        c = self.buena['objetivos'][EC.PRINCIPAL]['compuerta_publicacion']
        self.assertFalse(c['publicable']); self.assertFalse(c['objetivo_publicable'])
        with self.assertRaises(ValueError):
            EC.evaluar(self.t, self.t['riesgo'], 'x', ['consenso'], META, universo_csv='/no/existe', objetivos=(EC.PRINCIPAL,), repeticiones=5)

    def test_reglas_base_y_guardar(self):
        t = self.t.assign(dist_ma200=self.t['riesgo'] * -0.1, dist_max_52s=-0.3 * (self.t['riesgo'] > 1), vol_60d=self.t['riesgo'],
                          runup_2a=0.0, sorpresa_ult=np.nan)
        s = EC.puntaje_regla(t, 'compuesta')
        self.assertEqual(float(s.max()), 3.0)
        r = [EC.evaluar(t, EC.puntaje_regla(t, n), n, v, META, universo_csv='/no/existe', objetivos=(EC.PRINCIPAL,), repeticiones=10)
             for n, (v, _, _) in EC.REGLAS_BASE.items() if n in ('compuesta', 'placebo')]
        EC.comparar_contra(r)
        self.assertIn('diferencia_vs_compuesta_90', r[1]['objetivos'][EC.PRINCIPAL])
        with tempfile.TemporaryDirectory() as d:
            a = EC.guardar(r, d, 'Prueba', 'reglas'); b = EC.guardar(r, d, 'Prueba', 'reglas')
            self.assertEqual(len(set(a) | set(b)), 6)
            with open(b[-1], encoding='utf-8') as fh: self.assertIn('evaluaciones de caídas previas en resultados/: 2', fh.read())


if __name__ == '__main__':
    unittest.main()
