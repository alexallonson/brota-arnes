"""Pruebas de `laboratorio/sobreextension.py` (pedido AM de Fable).

Lo que hay que fijar no es el resultado —ése depende de los datos— sino que las cuatro pruebas midan
lo que la regla dice que miden: la población correcta, los grupos correctos, y que una señal falsa
NO pase mientras una verdadera SÍ.
"""
import os
import sys
import unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'laboratorio'))

try:
    import numpy as np
    import pandas as pd
    HAY = True
except ImportError:
    HAY = False


@unittest.skipUnless(HAY, 'requiere pandas y numpy')
class Sobreextension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global S
        import sobreextension as S

    def tabla(self, n_meses=48, n_emisoras=30, semilla=3, senal=0.0):
        """Panel sintético: `senal` es cuánto sube la probabilidad de caída para el puntaje alto."""
        azar = np.random.default_rng(semilla)
        fechas = pd.date_range('2012-01-31', periods=n_meses, freq='ME')
        filas = []
        for f in fechas:
            for i in range(n_emisoras):
                p = azar.random()
                filas.append({'fecha': f, 'ticker': f'E{i:02d}.MX', 'pais': 'MX', 'en_app': True,
                              'ret_12m_ex1m': azar.normal(), 'puntaje': p,
                              'runup_2a': azar.normal(), 'max_ret_1m': azar.normal(), 'vol_60d': azar.normal(),
                              'y_63': azar.normal(0, .1), 'y_63_rel': azar.normal(0, .1),
                              'min_ret_63': -0.30 if azar.random() < 0.08 + senal * (p > 2 / 3) else 0.05})
        return pd.DataFrame(filas)

    def test_poblacion_es_solo_el_tercil_superior_mexicano(self):
        t = self.tabla()
        d = S.poblacion(t, t['puntaje'])
        self.assertTrue((d['pais'] == 'MX').all(), 'solo libro mexicano')
        self.assertTrue((d['fecha'] >= S.DESDE).all() and (d['fecha'] < S.HASTA).all(), 'solo 2012-2020')
        # dentro de cada mes, ninguno de los elegidos puede tener impulso por debajo de los no elegidos
        for f, g in d.groupby('fecha'):
            todos = t[t['fecha'] == f]['ret_12m_ex1m']
            self.assertGreater(g['ret_12m_ex1m'].min(), todos.quantile(2 / 3) - 1e-9, 'es el tercil superior')
        self.assertLess(len(d), len(t) / 2, 'un tercio, no la tabla entera')

    def test_grupos_parten_el_mes_en_un_tercio_y_dos(self):
        t = self.tabla()
        d = S.poblacion(t, t['puntaje'])
        a = S.grupos(d, 'puntaje')
        frac = a.sum() / len(a)
        self.assertGreater(frac, 0.25); self.assertLess(frac, 0.42)
        for f, g in d.assign(_a=a).groupby('fecha'):   # A siempre por encima de B dentro del mes
            if g['_a'].any() and (~g['_a']).any():
                self.assertGreaterEqual(g.loc[g['_a'], 'puntaje'].min(), g.loc[~g['_a'], 'puntaje'].max() - 1e-9)

    def test_senal_falsa_no_pasa_y_senal_real_si(self):
        """La prueba de que la prueba sirve: con puntaje puramente aleatorio el intervalo tiene que tocar
        cero; con una señal fuerte de verdad tiene que excluirlo. Si no, la prueba 1 no mide nada."""
        azar = np.random.default_rng(7)
        falsa = self.tabla(n_meses=72, n_emisoras=40, semilla=11, senal=0.0)
        d = S.poblacion(falsa, falsa['puntaje'])
        r = S.prueba_1(d, S.grupos(d, 'puntaje'), azar)
        self.assertFalse(r['pasa'], 'una señal inventada no puede pasar')

        real = self.tabla(n_meses=72, n_emisoras=40, semilla=11, senal=0.35)
        d2 = S.poblacion(real, real['puntaje'])
        r2 = S.prueba_1(d2, S.grupos(d2, 'puntaje'), azar)
        self.assertTrue(r2['pasa'], 'una señal fuerte de verdad sí tiene que pasar')
        self.assertGreater(r2['X_menos_Y'], r['X_menos_Y'])

    def test_episodios_cuenta_meses_y_no_emisoras(self):
        t = self.tabla(n_meses=24, n_emisoras=20, semilla=5)
        d = S.poblacion(t, t['puntaje'])
        ep = S.episodios(d)
        self.assertLessEqual(ep, d['fecha'].nunique(), 'nunca más episodios que meses')
        self.assertLessEqual(ep, int(d['evento'].sum()), 'nunca más episodios que eventos')

    def test_la_regla_fija_no_usa_el_puntaje(self):
        """La regla del paso 4 tiene que ser independiente del modelo: si cambia el puntaje, no cambia."""
        t = self.tabla()
        d = S.poblacion(t, t['puntaje'])
        r1 = sum(S.rango_mes(d, c) for c in S.REGLA_FIJA) / len(S.REGLA_FIJA)
        d2 = d.copy(); d2['puntaje'] = np.random.default_rng(99).random(len(d2))
        r2 = sum(S.rango_mes(d2, c) for c in S.REGLA_FIJA) / len(S.REGLA_FIJA)
        pd.testing.assert_series_equal(r1, r2, check_names=False)

    def test_prueba_2_detecta_orden_invertido(self):
        """Si la frecuencia observada baja cuando el puntaje sube, la calibración tiene que fallar."""
        t = self.tabla(n_meses=60, n_emisoras=40, semilla=2, senal=0.35)
        d = S.poblacion(t, t['puntaje'])
        al_derecho = S.prueba_2(d)
        al_reves = S.prueba_2(d.assign(puntaje=-d['puntaje']))
        self.assertGreater(al_derecho['spearman'], al_reves['spearman'])
        self.assertFalse(al_reves['pasa'], 'con el orden invertido no puede pasar')


if __name__ == '__main__':
    unittest.main()
