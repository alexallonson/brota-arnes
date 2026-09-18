"""Pruebas de laboratorio/evaluar.py con una tabla sintética: IC, t de Newey-West, quintiles, rotación, costos de la app, criterio,
compuerta de publicación, analistas solo en ablación, cobertura de salidas y resultados que no se sobrescriben."""
import json, os, sys, tempfile, unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'laboratorio'))
try:
    import numpy as np
    import pandas as pd
    import evaluar as EV
    HAY = True
except ImportError:
    HAY = False

META = {'version': 'dataset-prueba', 'columnas': {
    'senal': {'grupo': 'A', 'fuente': 'yahoo', 'publicable': False},
    'macro_ok': {'grupo': 'E', 'fuente': 'banxico', 'publicable': True},
    'consenso': {'grupo': 'C', 'fuente': 'yahoo', 'publicable': False},
    'y_21': {'grupo': 'objetivo', 'fuente': 'yahoo', 'publicable': False},
    'y_63': {'grupo': 'objetivo', 'fuente': 'yahoo', 'publicable': False}}}
COSTOS = {'comision': 0.0025, 'iva': 0.16, 'costo_por_peso_operado': 0.0029, 'fuente': 'prueba'}


def tabla_sintetica(n_emp=60, desde='2011-01-01', hasta='2016-12-31', fuerza=1.0):
    rng = np.random.default_rng(3)
    dias = pd.bdate_range(desde, hasta)
    tickers = [f'E{i:02d}' + ('.MX' if i % 2 else '') for i in range(n_emp)]
    senal = rng.standard_normal((len(dias), n_emp))
    ruido = rng.standard_normal((len(dias), n_emp))
    y21 = 0.02 * (fuerza * senal + ruido)
    filas = pd.DataFrame({'fecha': np.repeat(dias, n_emp), 'ticker': np.tile(tickers, len(dias)),
                          'senal': senal.ravel(), 'y_21': y21.ravel(), 'y_63': (y21 * 2).ravel()})
    filas['pais'] = np.where(filas['ticker'].str.endswith('.MX'), 'MX', 'US')
    filas['sector'] = 'x'; filas['en_app'] = True
    return filas


@unittest.skipUnless(HAY, 'faltan numpy/pandas')
class PruebasEvaluar(unittest.TestCase):
    def test_senal_real_cumple_y_placebo_no(self):
        t = tabla_sintetica()
        debil = EV.evaluar(t, t['senal'] + 3 * np.random.default_rng(2).standard_normal(len(t)), 'impulso_12m_ex1m', ['senal'], META, costos=COSTOS, universo_csv='/no/existe')
        ref = debil['subconjuntos']['todas']['resumen']
        with tempfile.TemporaryDirectory() as d:
            r = EV.evaluar(t, t['senal'], 'buena', ['senal'], META, costos=COSTOS, universo_csv='/no/existe', referencia=ref, carpeta_intentos=d, tipo='modelo')
            self.assertEqual(len(EV.leer_intentos(d)), 1)
        s = r['subconjuntos']['todas']['resumen']
        self.assertGreater(s['ic_21']['medio'], 0.5)
        self.assertGreater(s['ic_21']['t_newey_west'], 3)
        self.assertEqual(s['ic_21']['meses'], 72)
        self.assertTrue(r['criterio']['cumple'], r['criterio']['detalle']); self.assertTrue(r['candidato_a_app'])
        self.assertGreaterEqual(r['criterio_extra']['desinflado_q1_topado_n1']['probabilidad_dsr'], 0.95)
        self.assertEqual(r['criterio']['regla']['referencia'], 'impulso_12m_ex1m')
        self.assertIsNone(debil['criterio']['detalle']['vence_referencia'], 'la referencia no se compara contra sí misma')
        sin_ref = EV.evaluar(t, t['senal'], 'buena', ['senal'], META, costos=COSTOS, universo_csv='/no/existe')
        self.assertFalse(sin_ref['criterio']['detalle']['vence_referencia']); self.assertFalse(sin_ref['candidato_a_app'], 'sin referencia no hay candidato')
        self.assertGreater(s['quintiles_21']['q1_medio'], s['quintiles_21']['q5_medio'])
        self.assertIn('mexico', r['subconjuntos']); self.assertIn('extranjero', r['subconjuntos'])
        p = EV.evaluar(t, np.random.default_rng(1).standard_normal(len(t)), 'placebo', [], META, costos=COSTOS, universo_csv='/no/existe')
        sp = p['subconjuntos']['todas']['resumen']
        self.assertLess(abs(sp['ic_21']['medio']), 0.05)
        self.assertFalse(p['criterio']['cumple'])
        self.assertFalse(p['criterio']['detalle']['ic90_excluye_cero'])
        self.assertLess(sp['estabilidad_media'], 0.2)

    def test_senal_invertida_no_cumple(self):
        t = tabla_sintetica()
        r = EV.evaluar(t, -t['senal'], 'mala', ['senal'], META, costos=COSTOS, universo_csv='/no/existe')
        self.assertLess(r['subconjuntos']['todas']['resumen']['ic_21']['medio'], -0.5)
        self.assertFalse(r['criterio']['detalle']['ic_medio']); self.assertFalse(r['candidato_a_app'])

    def test_costos_restan_rotacion(self):
        t = tabla_sintetica()
        # sobrecosto=0: esta prueba mide que la rotación se cobre, así que apaga el del SIC para dejar un solo costo en juego
        sin = EV.evaluar(t, t['senal'], 's', ['senal'], META, costos={**COSTOS, 'costo_por_peso_operado': 0.0}, universo_csv='/no/existe', sic=0)
        con = EV.evaluar(t, t['senal'], 's', ['senal'], META, costos={**COSTOS, 'costo_por_peso_operado': 0.01}, universo_csv='/no/existe', sic=0)
        q0, q1 = sin['subconjuntos']['todas']['resumen']['quintiles_21'], con['subconjuntos']['todas']['resumen']['quintiles_21']
        self.assertAlmostEqual(q0['q1q5_neto_medio'], q0['q1q5_bruto_medio'], places=9)
        esperado = q1['q1q5_bruto_medio'] - 0.01 * (q1['rotacion_q1_media'] + q1['rotacion_q5_media'])
        self.assertAlmostEqual(q1['q1q5_neto_medio'], esperado, places=9)

    def test_bootstrap_sharpe_desinflado_y_peor_anio(self):
        rng = np.random.default_rng(4)
        x = 0.01 + 0.02 * rng.standard_normal(240)
        lo, hi = EV.bootstrap_bloques(x, 12, 500); self.assertLess(lo, x.mean()); self.assertGreater(hi, x.mean()); self.assertGreater(lo, 0)
        self.assertIsNone(EV.bootstrap_bloques(x[:10], 12, 100))
        uno = EV.sharpe_desinflado(x, [EV.sharpe_mensual(x)])
        self.assertEqual(uno['sharpe0_anual'], 0.0); self.assertAlmostEqual(uno['sharpe_anual'], EV.sharpe_mensual(x) * np.sqrt(12))
        muchos = EV.sharpe_desinflado(x, list(np.linspace(-0.3, 0.3, 30)))
        self.assertGreater(muchos['sharpe0_anual'], 0.3, 'con 30 intentos dispersos el azar da un Sharpe alto')
        self.assertLess(muchos['probabilidad_dsr'], uno['probabilidad_dsr'])
        m = pd.DataFrame({'fecha': [f'2020-{i:02d}-01' for i in range(1, 13)] + [f'2021-{i:02d}-01' for i in range(1, 4)],
                          'q1_21': [0.01] * 12 + [-0.05] * 3, 'universo_21': [0.0] * 15, 'rot_q1': [0.0] * 15, 'rot_universo': [0.0] * 15})
        peor, anio, todos = EV.peor_anio_relativo(m, 0.0029, 6)
        self.assertEqual(anio, '2020'); self.assertNotIn('2021', todos, 'un año con menos de 6 meses no cuenta')
        self.assertAlmostEqual(peor, 1.01 ** 12 - 1)

    def test_por_libro_combina_con_riesgo_activo_igual(self):
        rng = np.random.default_rng(4)
        fechas = pd.bdate_range('2015-01-01', periods=36, freq='BMS')
        filas = []
        for f in fechas:
            for i in range(60):
                pais = 'MX' if i < 30 else 'US'
                senal = rng.standard_normal()
                ruido = 0.02 if pais == 'MX' else 0.08          # el libro extranjero es cuatro veces más volátil
                filas.append({'fecha': f, 'ticker': f'{pais}{i}', 'pais': pais, 'sector': 's', 'en_app': True,
                              'puntaje': senal, 'y_21': 0.01 * senal + ruido * rng.standard_normal(), 'y_63': np.nan,
                              'importe_20d': 1e9})
        d = pd.DataFrame(filas)
        comb, libros, info = EV.meses_por_libro(d, 0.0, pesos=None)
        self.assertEqual(set(libros), {'MX', 'extranjero'})
        self.assertEqual(int(libros['MX']['cuantiles'].iloc[0]), 3, 'terciles en México')
        self.assertEqual(int(libros['extranjero']['cuantiles'].iloc[0]), 5)
        self.assertGreater(info['pesos']['MX'], info['pesos']['extranjero'], 'el libro menos volátil pesa más con riesgo activo igual')
        self.assertAlmostEqual(sum(info['pesos'].values()), 1.0, places=9)
        riesgo = {b: info['pesos'][b] * info['error_seguimiento_mensual_medido'][b] for b in info['pesos']}
        self.assertAlmostEqual(riesgo['MX'], riesgo['extranjero'], places=9, msg='cada libro aporta el mismo riesgo activo')
        f0 = comb['fecha'].iloc[5]
        esperado = sum(info['pesos'][b] * libros[b].set_index('fecha').at[f0, 'q1_topado_vs_universo_neto'] for b in info['pesos'])
        self.assertAlmostEqual(float(comb.set_index('fecha').at[f0, 'q1_topado_vs_universo_neto']), esperado, places=9)

    def test_dsr_precomprometido(self):
        x = np.random.default_rng(2).normal(0.006, 0.015, 120)
        n1 = EV.sharpe_desinflado(x, [None])
        n2 = EV.sharpe_desinflado(x, [None], n_intentos=2, sd_intentos=0.213)
        self.assertEqual(n1['intentos'], 0); self.assertEqual(n2['intentos'], 2)
        self.assertGreater(n2['sharpe0_anual'], 0); self.assertAlmostEqual(n1['sharpe0_anual'], 0.0, places=9)
        self.assertLess(n2['probabilidad_dsr'], n1['probabilidad_dsr'], 'con dos candidatos la barra sube')

    def test_condicion_6_por_sharpe_del_margen(self):
        """Decisión del dueño (2026-09-16): la condición 6 compara el Sharpe del margen, no el IC ni la media."""
        r = {'ic_21': {'medio': 0.04, 't_newey_west': 3.5, 'meses': 100}, 'quintiles_21': {'q1_topado_vs_universo_neto_medio': 0.004},
             'sharpe_anual_margen': 1.10}
        ref = {'ic_21': {'medio': 0.05, 'meses': 100}, 'quintiles_21': {'q1_topado_vs_universo_neto_medio': 0.006}, 'sharpe_anual_margen': 0.95}
        extra = {'ic90_q1_topado_vs_universo_neto': [0.001, 0.007], 'desinflado_q1_topado_n1': {'probabilidad_dsr': 0.99}, 'peor_anio_topado': -0.05}
        c = EV.criterio(r, extra, ref)
        self.assertTrue(c['detalle']['vence_referencia'], 'gana en Sharpe aunque pierda en IC y en media')
        self.assertTrue(c['cumple'])
        peor = EV.criterio(dict(r, sharpe_anual_margen=0.80), extra, ref)
        self.assertFalse(peor['detalle']['vence_referencia'])
        self.assertIsNone(EV.criterio(r, extra, ref, es_referencia=True)['detalle']['vence_referencia'])

    def test_pesos_topados_con_objetivo(self):
        sin_tope = EV.pesos_topados([np.nan] * 3, objetivo=[1, 2, 1])
        np.testing.assert_allclose(sin_tope, [0.25, 0.5, 0.25], atol=1e-12, err_msg='sin tope, proporcionales al objetivo')
        iguales = EV.pesos_topados([np.nan] * 4)
        np.testing.assert_allclose(iguales, [0.25] * 4, atol=1e-12)
        # la primera topa en 10 % y el resto se reparte 2:1 entre las otras dos
        con_tope = EV.pesos_topados([0.1 * 1e6 / 0.01, np.nan, np.nan], objetivo=[5, 2, 1])
        self.assertAlmostEqual(con_tope[0], 0.10, places=9)
        self.assertAlmostEqual(con_tope[1] / con_tope[2], 2.0, places=9)
        self.assertAlmostEqual(con_tope.sum(), 1.0, places=9)

    def test_pesos_por_libro_congelados(self):
        """Los pesos del pliegue externo no se pueden recalcular con los datos que se están midiendo (tanda 6, B1)."""
        rng = np.random.default_rng(11)
        fechas = pd.bdate_range('2021-01-01', periods=18, freq='BMS')
        filas = [{'fecha': f, 'ticker': f'{p}{i}', 'pais': p, 'sector': 's', 'en_app': True, 'puntaje': rng.standard_normal(),
                  'y_21': rng.standard_normal() * 0.05, 'y_63': np.nan, 'importe_20d': 1e9}
                 for f in fechas for p in ('MX', 'US') for i in range(30)]
        d = pd.DataFrame(filas)
        comb, libros, info = EV.meses_por_libro(d, 0.0)
        self.assertTrue(info['pesos_congelados'])
        self.assertEqual({k: round(v, 6) for k, v in info['pesos'].items()}, EV.PESOS_LIBRO, 'los pesos de salida son los congelados')
        f0 = comb['fecha'].iloc[3]
        esperado = sum(EV.PESOS_LIBRO[b] * libros[b].set_index('fecha').at[f0, 'q1_topado_vs_universo_neto'] for b in EV.PESOS_LIBRO)
        self.assertAlmostEqual(float(comb.set_index('fecha').at[f0, 'q1_topado_vs_universo_neto']), esperado, places=9)
        libres = EV.meses_por_libro(d, 0.0, pesos=None)[2]
        self.assertFalse(libres['pesos_congelados'])
        self.assertNotEqual({k: round(v, 6) for k, v in libres['pesos'].items()}, EV.PESOS_LIBRO, 'sin congelar, los pesos salen de estos datos')

    def test_compuesto_eje_faltante_neutral(self):
        t = pd.DataFrame({'fecha': [pd.Timestamp('2020-01-01')] * 3, 'a': [1.0, 2.0, 3.0], 'b': [3.0, 2.0, 1.0], 'c': [1.0, np.nan, 3.0], 'd': [np.nan, np.nan, 5.0]})
        ejes = {'uno': [('a', 1)], 'dos': [('b', -1)], 'tres': [('c', 1)], 'cuatro': [('d', 1)]}
        p = EV.puntaje_compuesto(t, ejes, min_ejes=2)
        # rangos del día − 0.5: a [−1/6, 1/6, 1/2]; −b [−1/2, −1/6, 1/6]; c [0, —, 1/2]; d [—, —, 1/2]
        self.assertAlmostEqual(p[0], (-1 / 6 - 1 / 2 + 0) / 4, msg='ejes sin dato valen 0 y se divide entre los cuatro ejes')
        self.assertAlmostEqual(p[1], 0.0)
        self.assertAlmostEqual(p[2], (1 / 2 + 1 / 6 + 1 / 2 + 1 / 2) / 4)
        self.assertTrue(np.isnan(EV.puntaje_compuesto(t, ejes, min_ejes=3)[1]), 'con menos ejes que el mínimo: sin puntaje')

    def test_t_newey_west(self):
        x = np.array([0.1, -0.05, 0.2, 0.0, 0.07, 0.12, -0.02, 0.05])
        self.assertAlmostEqual(EV.t_newey_west(x, 0), x.mean() / (x.std() / np.sqrt(len(x))), places=10)
        rng = np.random.default_rng(0)
        ar = np.convolve(rng.standard_normal(300), np.ones(3), 'valid') * 0.1 + 0.02   # autocorrelada, como IC a 63 días con meses traslapados
        self.assertLess(abs(EV.t_newey_west(ar, 2)), abs(EV.t_newey_west(ar, 0)))
        self.assertIsNone(EV.t_newey_west([0.1, 0.2], 0))

    def test_rotacion_y_quintiles(self):
        self.assertEqual(EV.rotacion({'A', 'B'}, {'A', 'B'}), 0.0)
        self.assertAlmostEqual(EV.rotacion({'A', 'B'}, {'C', 'D'}), 2.0)
        self.assertAlmostEqual(EV.rotacion(None, {'A'}), 1.0)
        self.assertAlmostEqual(EV.rotacion({'A', 'B'}, {'A', 'C'}), 1.0)
        q = EV.quintiles([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        self.assertEqual(list(q), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_fechas_rebalanceo_primer_dia_habil(self):
        f = EV.fechas_rebalanceo(pd.to_datetime(['2020-02-03', '2020-01-31', '2020-02-04', '2020-01-02', '2020-01-03']))
        self.assertEqual([str(x.date()) for x in f], ['2020-01-02', '2020-02-03'])

    def test_rebalanceo_evita_feriados_y_cartera_topada(self):
        f = pd.Series(pd.to_datetime(['2021-02-01'] * 3 + ['2021-02-02'] * 5 + ['2021-03-01'] * 5))
        self.assertEqual([str(x.date()) for x in EV.fechas_rebalanceo(f)], ['2021-02-02', '2021-03-01'], 'el 1-feb solo abrió una bolsa')
        w = EV.pesos_topados([1e9, 1e9, 1e9, 1e6], capital=1e6, participacion=0.01)
        self.assertAlmostEqual(w[3], 0.01); self.assertAlmostEqual(w[:3].sum(), 0.99); self.assertAlmostEqual(w[0], 0.33)
        todo_topado = EV.pesos_topados([1e6, 1e6], capital=1e6, participacion=0.01)
        self.assertAlmostEqual(todo_topado.sum(), 0.02, msg='lo que no cabe queda en efectivo')
        self.assertAlmostEqual(EV.rotacion_pesos({'A': 0.5, 'B': 0.5}, {'A': 1.0}), 1.0)

    def test_dsr_es_probabilidad(self):
        rng = np.random.default_rng(9); x = 0.006 + 0.05 * rng.standard_normal(248)
        psr = EV.sharpe_desinflado(x, [None]); dsr = EV.sharpe_desinflado(x, [0.11, 0.08, -0.06, -0.19, -0.21])
        self.assertTrue(0 < dsr['probabilidad_dsr'] < psr['probabilidad_dsr'] < 1)
        self.assertEqual(psr['sharpe0_anual'], 0.0)

    def test_costos_leidos_de_la_app(self):
        c = EV.costos_app()
        self.assertEqual(c['comision'], 0.0025); self.assertEqual(c['iva'], 0.16)
        self.assertAlmostEqual(c['costo_por_peso_operado'], 0.0029)

    def test_costo_unitario_cobra_diferencial_por_emisora_y_sic_solo_a_extranjeras(self):
        """Pedidos 99 y 184: cada emisora paga su propio medio diferencial, y el sobrecosto del SIC solo lo pagan las de fuera."""
        f = pd.DataFrame({'pais': ['MX', 'MX', 'US'], EV.COL_DIFERENCIAL: [0.001, np.nan, 0.004]})
        c = EV.costo_unitario(f, k=0.0029, sic=0.005).to_numpy()
        self.assertAlmostEqual(c[0], 0.0029 + 0.001)
        self.assertAlmostEqual(c[1], 0.0029 + 0.0025, msg='sin dato se cobra la mediana del día, no cero')
        # corrección de Mercado (D1): la extranjera paga el diferencial del SIC EN LUGAR del de su bolsa de origen, no los dos
        self.assertAlmostEqual(c[2], 0.0029 + 0.005, msg='el del SIC sustituye, no se suma')
        sin = EV.costo_unitario(f, k=0.0029, sic=0).to_numpy()
        self.assertAlmostEqual(sin[2], 0.0029 + 0.004, msg='con sic=0 se usa el diferencial de la bolsa de origen')
        # sin la columna del diferencial el costo vuelve a ser el de antes de los pedidos 99 y 184
        self.assertTrue(np.allclose(EV.costo_unitario(f[['pais']], k=0.0029, sic=0).to_numpy(), 0.0029))

    def test_factor_de_costos_al_doble_cobra_exactamente_el_doble(self):
        """La corrida obligatoria al doble (pedido 184) tiene que duplicar el costo, no aproximarlo."""
        t = tabla_sintetica(desde='2013-01-01', hasta='2015-12-31')
        t[EV.COL_DIFERENCIAL] = 0.002
        t = t.assign(puntaje=t['senal'])
        d = t[t['fecha'].isin(EV.fechas_rebalanceo(t['fecha']))]
        m1 = EV.evaluar_meses(d, 0.0029, sic=0.005)
        m2 = EV.evaluar_meses(d, 0.0029, sic=0.005, factor_costos=2)
        self.assertTrue(np.allclose(m2['costo_q1'], 2 * m1['costo_q1']))
        self.assertTrue(np.allclose(m2['costo_universo'], 2 * m1['costo_universo']))
        # el bruto no se toca y el neto empeora exactamente en el costo extra
        self.assertTrue(np.allclose(m2['q1q5_bruto'], m1['q1q5_bruto']))
        extra = m1['costo_q1'] * m1['rot_q1'] + m1['costo_q5'] * m1['rot_q5']
        self.assertTrue(np.allclose(m2['q1q5_neto'], m1['q1q5_neto'] - extra))
        # una cartera que no rota no paga: el costo entra multiplicado por la rotación, no como comisión fija
        quietas = m1[m1['rot_q1'] == 0]
        if len(quietas): self.assertTrue(np.allclose(m1.loc[quietas.index, 'q1_21'], m2.loc[quietas.index, 'q1_21']))

    def test_control_al_azar_rota_igual_que_q1(self):
        """Pedido 193: el control tiene que cambiar tantos nombres como Q1 y conservar el resto. Si rotara más pagaría costos que Q1 no
        paga y la señal saldría inflada, que es exactamente el defecto del placebo por puntaje al azar (147 % contra 47 %)."""
        azar = np.random.default_rng(1)
        disp = [f'T{i:03d}' for i in range(60)]
        anterior = disp[:20]
        c = EV.control_igual_rotacion(anterior, disp, tamano=20, cambios=3, azar=azar)
        self.assertEqual(len(c), 20)
        self.assertEqual(len(set(c) - set(anterior)), 3, 'cambia exactamente los que cambió Q1')
        self.assertEqual(len(set(c)), 20, 'sin nombres repetidos')
        # sin cambios en Q1 el control tampoco se mueve
        self.assertEqual(set(EV.control_igual_rotacion(anterior, disp, 20, 0, azar)), set(anterior))
        # si una emisora deja de estar disponible, se reemplaza aunque Q1 no haya cambiado nada: es rotación forzada, la misma que sufre Q1
        menos = [t for t in disp if t != anterior[0]]
        f = EV.control_igual_rotacion(anterior, menos, 20, 0, azar)
        self.assertNotIn(anterior[0], f); self.assertEqual(len(f), 20)
        # primer mes: sin cartera previa se arma entera al azar
        self.assertEqual(len(EV.control_igual_rotacion(None, disp, 20, 0, azar)), 20)
        # y en la evaluación completa la rotación del control queda pegada a la de Q1
        t = tabla_sintetica(desde='2013-01-01', hasta='2016-12-31')
        t = t.assign(puntaje=t['senal'], importe_20d=1e9)
        m = EV.evaluar_meses(t[t['fecha'].isin(EV.fechas_rebalanceo(t['fecha']))], 0.0029, sic=0)
        self.assertIn('rot_control', m)
        r_q1, r_c = m['rot_q1_topado'].dropna(), m['rot_control'].dropna()
        self.assertLess(abs(r_c.mean() - r_q1.mean()), 0.15 * r_q1.mean(),
                        f'el control rota {r_c.mean():.3f} y Q1 {r_q1.mean():.3f}: tienen que parecerse')

    def test_mecanismos_de_rotacion(self):
        """REGLA_ROTACION.md: el trimestral solo rearma en enero, abril, julio y octubre; la histéresis conserva al que sigue arriba del
        percentil de salida y deja entrar solo al que llega al cuantil superior. Los dos tienen que rotar MENOS que el mensual."""
        v = pd.DataFrame({'ticker': list('ABCDEFGHIJ'), 'puntaje': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]})
        q = EV.quintiles(v['puntaje'], 5)                      # quintil superior = A, B
        jul, ago = pd.Timestamp('2020-07-01'), pd.Timestamp('2020-08-03')
        self.assertEqual(EV.nombres_cartera('mensual', 0.5, ago, v, q, ['C', 'D']), ['A', 'B'])
        self.assertEqual(EV.nombres_cartera('trimestral', 0.5, jul, v, q, ['C', 'D']), ['A', 'B'], 'julio sí rearma')
        self.assertEqual(EV.nombres_cartera('trimestral', 0.5, ago, v, q, ['C', 'D']), ['C', 'D'], 'agosto conserva')
        self.assertEqual(EV.nombres_cartera('trimestral', 0.5, ago, v, q, ['C', 'ZZZ']), ['C'], 'la que dejó de cotizar sale')
        # histéresis: C y D siguen arriba de la mediana, así que se quedan; H está abajo y sale; A y B entran por el cuantil
        self.assertEqual(set(EV.nombres_cartera('histeresis', 0.5, ago, v, q, ['C', 'D', 'H'])), {'A', 'B', 'C', 'D'})
        self.assertEqual(EV.nombres_cartera('histeresis', 0.5, ago, v, q, None), ['A', 'B'], 'el primer mes es el corte del cuantil')
        # y en una evaluación completa los dos rotan menos que el mensual
        t = tabla_sintetica(desde='2012-01-01', hasta='2016-12-31').assign(importe_20d=1e9)
        t = t.assign(puntaje=t['senal'])
        d = t[t['fecha'].isin(EV.fechas_rebalanceo(t['fecha']))]
        rot = {}
        for mec in ('mensual', 'trimestral', 'histeresis'):
            m = EV.evaluar_meses(d, 0.0029, sic=0, mecanismo=mec)
            rot[mec] = float(m['rot_q1_topado'].dropna().mean())
        self.assertLess(rot['trimestral'], rot['mensual'], f'trimestral {rot["trimestral"]:.3f} vs mensual {rot["mensual"]:.3f}')
        self.assertLess(rot['histeresis'], rot['mensual'], f'histéresis {rot["histeresis"]:.3f} vs mensual {rot["mensual"]:.3f}')

    def test_compuerta_de_tres_valores(self):
        """Pedido AO de Fable: con potencia baja, un intervalo que incluye el cero NO es «falla» —es que la muestra no alcanza—.
        El criterio no se relaja; lo que cambia es qué significa un «no», y todo veredicto lleva su potencia al lado."""
        self.assertEqual(EV.veredicto([0.001, 0.006]), 'pasa')
        self.assertEqual(EV.veredicto([-0.006, -0.001]), 'falla', 'intervalo entero bajo cero sí es falla, sin importar la potencia')
        self.assertEqual(EV.veredicto([-0.003, 0.006], 0.40), 'indeterminado', 'toca el cero con potencia baja: no se puede concluir')
        self.assertEqual(EV.veredicto([-0.003, 0.006], 0.85), 'falla', 'toca el cero con potencia alta: ahí sí es evidencia de que no hay')
        self.assertEqual(EV.veredicto(None), 'indeterminado', 'sin intervalo no se afirma nada')
        # un candidato que pasa el intervalo pero falla otra condición no puede salir «pasa»
        self.assertEqual(EV.veredicto([0.001, 0.006], None, cumple_lo_demas=False), 'falla')

    def test_compuerta_cubre_objetivo(self):
        t = tabla_sintetica(desde='2017-01-01', hasta='2018-12-31')
        r = EV.evaluar(t, t['senal'], 'macro', ['macro_ok'], META, costos=COSTOS, universo_csv='/no/existe')
        c = r['compuerta_publicacion']['y_21']
        self.assertFalse(c['publicable']); self.assertFalse(c['objetivo_publicable']); self.assertEqual(c['variables_no_publicables'], [])
        r2 = EV.evaluar(t, t['senal'], 'sin_meta', ['no_documentada'], META, costos=COSTOS, universo_csv='/no/existe')
        self.assertIn('no_documentada', r2['compuerta_publicacion']['y_63']['variables_no_publicables'])

    def test_analistas_solo_en_ablacion(self):
        t = tabla_sintetica(desde='2017-01-01', hasta='2018-12-31')
        with self.assertRaises(ValueError):
            EV.evaluar(t, t['senal'], 'con_consenso', ['senal', 'consenso'], META, costos=COSTOS, universo_csv='/no/existe')
        r = EV.evaluar(t, t['senal'], 'con_consenso', ['senal', 'consenso'], META, ablacion=True, costos=COSTOS, universo_csv='/no/existe')
        self.assertEqual(r['variables_analistas'], ['consenso']); self.assertFalse(r['candidato_a_app'])

    def test_cobertura_salidas(self):
        t = tabla_sintetica(n_emp=30, desde='2017-01-01', hasta='2018-12-31')
        t.loc[(t['ticker'] == 'E01.MX') & (t['fecha'] > '2017-06-30'), 'y_21'] = np.nan   # dejó de cotizar a mitad de 2017
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, 'universo.csv')
            pd.DataFrame([{'ticker': 'E01.MX', 'salida': '2017-07', 'causa': 'x'}, {'ticker': 'NUNCA.MX', 'salida': 'por verificar', 'causa': 'x'},
                          {'ticker': 'E03.MX', 'salida': '', 'causa': ''}]).to_csv(ruta, index=False)
            c = EV.cobertura_salidas(t, ruta, EV.fechas_rebalanceo(t['fecha']))
        self.assertEqual(c['salidas_en_universo'], 2)
        self.assertEqual(c['con_historia_utilizable'], ['E01.MX']); self.assertEqual(c['sin_historia'], ['NUNCA.MX'])
        por = {p['anio']: p['n'] for p in c['por_anio']}
        self.assertEqual(por['2017'], 1); self.assertEqual(por['2018'], 0)

    def test_resultados_no_se_sobrescriben(self):
        t = tabla_sintetica(desde='2017-01-01', hasta='2018-12-31')
        r = EV.limpiar_nan(EV.evaluar(t, t['senal'], 'buena', ['senal'], META, costos=COSTOS, universo_csv='/no/existe'))
        with tempfile.TemporaryDirectory() as d:
            a = EV.guardar([dict(r)], d, 'Prueba', 'prueba')
            b = EV.guardar([dict(r)], d, 'Prueba', 'prueba')
            self.assertEqual(len(set(a) | set(b)), 4)
            self.assertTrue(b[0].endswith('_buena_2.json'))
            with open(a[0], encoding='utf-8') as fh: self.assertEqual(json.load(fh)['nombre'], 'buena')
            with open(a[1], encoding='utf-8') as fh: self.assertIn('| buena |', fh.read())


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(HAY, 'requiere pandas')
class TopeSIC(unittest.TestCase):
    """Pedido AT: el tope del libro extranjero con la liquidez del SIC (`REGLA_TOPE_SIC.md`)."""

    def filas(self):
        return pd.DataFrame({'ticker': ['MX1.MX', 'EX1', 'EX2', 'EX3'], 'pais': ['MX', 'US', 'US', 'US'],
                             'importe_20d': [1e6, 9e9, 9e9, 9e9],
                             'importe_sic_20d': [np.nan, 2e6, 5e5, np.nan],
                             'dias_sic_12m': [np.nan, 0.95, 0.70, np.nan]})

    def test_el_libro_mexicano_no_cambia(self):
        f = self.filas()
        a = EV.importe_para_topar(f, False); b = EV.importe_para_topar(f, True)
        self.assertEqual(a[0], b[0], 'la mexicana conserva el importe de la BMV')
        self.assertEqual(b[0], 1e6)

    def test_las_extranjeras_pasan_al_importe_del_sic(self):
        b = EV.importe_para_topar(self.filas(), True)
        self.assertEqual(list(b[1:3]), [2e6, 5e5], 'usan el del SIC, no el de origen')
        self.assertTrue(np.isnan(b[3]), 'sin espejo: nulo, no el de origen ni cero')

    def test_sin_la_bandera_todo_sigue_igual(self):
        f = self.filas()
        self.assertEqual(list(EV.importe_para_topar(f, False)), list(f['importe_20d']))
        self.assertTrue(EV.elegibles_sic(f, False).all(), 'sin bandera no se excluye a nadie')

    def test_elegibilidad_deja_fuera_lo_que_no_se_opera_en_el_sic(self):
        e = EV.elegibles_sic(self.filas(), True)
        self.assertTrue(e[0], 'la mexicana siempre es elegible')
        self.assertTrue(e[1], '0.95 pasa el corte de 0.60')
        self.assertTrue(e[2], '0.70 pasa el corte de 0.60')
        self.assertFalse(e[3], 'sin espejo no es comprable desde México')
        f = self.filas(); f.loc[2, 'dias_sic_12m'] = 0.40
        self.assertFalse(EV.elegibles_sic(f, True)[2], '0.40 no pasa el corte')

    def test_sin_las_columnas_falla_en_vez_de_no_hacer_nada(self):
        """El 2026-09-18 la primera corrida con --tope-sic reporto 'cero cambio' porque las columnas no se leian del dataset
        y los ayudantes caian al comportamiento viejo EN SILENCIO. Ahora tiene que reventar."""
        f = self.filas().drop(columns=['importe_sic_20d'])
        with self.assertRaises(ValueError): EV.importe_para_topar(f, True)
        g = self.filas().drop(columns=['dias_sic_12m'])
        with self.assertRaises(ValueError): EV.elegibles_sic(g, True)
        # y sin la bandera no debe reventar: lo viejo sigue funcionando sin las columnas
        self.assertEqual(len(EV.importe_para_topar(f, False)), 4)
        self.assertTrue(EV.elegibles_sic(g, False).all())
