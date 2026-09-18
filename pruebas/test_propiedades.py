"""Pruebas de propiedades (Hypothesis) del laboratorio: cada función se compara contra una referencia lenta y obvia (oráculo) o se le exige
una regla que vale para TODA entrada del dominio, no solo para los ejemplos escritos a mano.

Perfiles: `HYPOTHESIS_PROFILE=ci` (lo usa GitHub) o `dev` (por omisión, más rápido). Si falla, Hypothesis imprime la entrada mínima que
lo rompe: antes de «arreglar» la prueba, decidir si la propiedad está mal, la especificación es ambigua o el código tiene un error.
"""
import datetime as _dt, itertools, math, os, sys, unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'laboratorio'))
try:
    import numpy as np
    import pandas as pd
    import tempfile as _tmp
    from hypothesis.configuration import set_hypothesis_home_dir
    set_hypothesis_home_dir(os.path.join(_tmp.gettempdir(), 'hypothesis-brota'))   # cachés de Hypothesis fuera del repo
    from hypothesis import Phase, example, given, settings, strategies as st
    import dataset as DS
    import evaluar as EV
    import evaluar_caidas as EC
    import fundamentales as FU
    import entrenar as EN
    HAY = True
    # sin base de ejemplos en disco (no deja carpetas .hypothesis en el repo); cada falla imprime su ejemplo y es reproducible con él
    settings.register_profile('ci', max_examples=150, deadline=None, database=None)
    settings.register_profile('dev', max_examples=40, deadline=None, database=None)
    settings.load_profile(os.environ.get('HYPOTHESIS_PROFILE', 'dev'))
except ImportError:   # sin numpy/pandas/hypothesis (p. ej. el python3 del sistema): el módulo carga y las clases se saltan
    HAY = False
    def given(*a, **k): return lambda f: f
    example = settings = given
    class _Fase:
        explicit = generate = None
    Phase = _Fase

    class _Estrategias:
        def composite(self, f): return lambda *a, **k: None
        def __getattr__(self, _): return lambda *a, **k: None
    st = _Estrategias()

finitos = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
positivos = st.floats(min_value=1e-3, max_value=1e9, allow_nan=False, allow_infinity=False)


@st.composite
def grupos_con_valores(draw):
    """Filas (grupo, valor) con valores repetidos y nulos: el caso real de sectores chicos."""
    n = draw(st.integers(1, 25))
    grupos = draw(st.lists(st.integers(0, 3), min_size=n, max_size=n))
    valores = draw(st.lists(st.one_of(st.none(), st.integers(-5, 5).map(float), finitos), min_size=n, max_size=n))
    return grupos, [np.nan if v is None else v for v in valores]


@unittest.skipUnless(HAY, 'requiere numpy, pandas y hypothesis')
class PropiedadesSector(unittest.TestCase):
    @given(grupos_con_valores())
    @example(([0, 0, 0], [1.0, 1.0, 1.0]))
    @example(([0], [3.0]))
    def test_mediana_sin_propia_es_la_mediana_de_las_demas(self, datos):
        grupos, valores = datos
        claves = pd.DataFrame({'fecha': [1] * len(grupos), 'sector': grupos})
        obtenido = DS.mediana_sin_propia(valores, claves)
        for i, (g, v) in enumerate(zip(grupos, valores)):
            otras = [x for j, (h, x) in enumerate(zip(grupos, valores)) if j != i and h == g and not np.isnan(x)]
            if np.isnan(v) or not otras:
                self.assertTrue(np.isnan(obtenido[i]), (i, grupos, valores, obtenido))
            else:
                self.assertAlmostEqual(obtenido[i], float(np.median(otras)), delta=1e-9 * max(1.0, abs(np.median(otras))))

    @given(grupos_con_valores())
    def test_media_sin_propia_es_la_media_de_las_demas(self, datos):
        grupos, valores = datos
        t = pd.DataFrame({'fecha': 1, 'sector': grupos, 'x': valores})
        obtenido = DS.media_sin_propia(t, 'x')
        for i, (g, v) in enumerate(zip(grupos, valores)):
            otras = [x for j, (h, x) in enumerate(zip(grupos, valores)) if j != i and h == g and not np.isnan(x)]
            if np.isnan(v) or not otras:
                self.assertTrue(np.isnan(obtenido[i]))
            else:
                self.assertAlmostEqual(obtenido[i], float(np.mean(otras)), delta=1e-6 * max(1.0, abs(np.mean(otras))))


@unittest.skipUnless(HAY, 'requiere numpy, pandas y hypothesis')
class PropiedadesCartera(unittest.TestCase):
    @given(st.lists(st.one_of(st.none(), positivos), min_size=1, max_size=40),
           st.floats(min_value=1e3, max_value=1e9), st.floats(min_value=1e-4, max_value=0.5))
    @example([1e6, 1e6], 1e6, 0.01)
    @example([None, None, 1.0], 1e6, 0.01)
    def test_pesos_topados(self, importes, capital, participacion):
        imp = [np.nan if x is None else x for x in importes]
        w = EV.pesos_topados(imp, capital=capital, participacion=participacion)
        tope = np.array([np.inf if np.isnan(x) else participacion * x / capital for x in imp])
        self.assertTrue((w >= -1e-12).all())
        self.assertTrue((w <= tope + 1e-9).all(), 'ninguna posición pasa su tope')
        self.assertLessEqual(w.sum(), 1 + 1e-9)
        # llenado por nivel: sin topes que alcancen el 100 %, se invierte todo; y toda emisora no topada pesa lo mismo y no menos que las topadas
        if tope.sum() >= 1 - 1e-9:
            self.assertAlmostEqual(w.sum(), 1.0, places=6)
        libres = w < tope - 1e-9
        if libres.any():
            self.assertLess(np.ptp(w[libres]), 1e-9)
            self.assertTrue((w[~libres] <= w[libres].max() + 1e-9).all())

    @given(st.dictionaries(st.sampled_from('ABCDEFG'), st.floats(0, 1)), st.dictionaries(st.sampled_from('ABCDEFG'), st.floats(0, 1)))
    def test_rotacion_es_una_distancia(self, a, b):
        self.assertAlmostEqual(EV.rotacion_pesos(a, b), EV.rotacion_pesos(b, a))
        self.assertEqual(EV.rotacion_pesos(a, dict(a)), 0.0)
        self.assertLessEqual(EV.rotacion_pesos(a, b), sum(a.values()) + sum(b.values()) + 1e-12)

    @given(st.lists(finitos, min_size=1, max_size=60))
    def test_quintiles_balanceados_y_monotonos(self, puntajes):
        q = EV.quintiles(puntajes)
        tam = np.bincount(q, minlength=6)[1:]
        self.assertLessEqual(tam.max() - tam[tam > 0].min() if (tam > 0).any() else 0, 1 + (len(puntajes) < 5))
        for i, j in itertools.combinations(range(len(puntajes)), 2):
            if puntajes[i] > puntajes[j]:
                self.assertLessEqual(q[i], q[j], 'un puntaje mayor nunca cae en un quintil peor')


def ap_referencia(y, s):
    pos = sum(y)
    if pos == 0: return None
    ap, r_prev = 0.0, 0.0
    for u in sorted(set(s), reverse=True):
        sel = [yy for yy, ss in zip(y, s) if ss >= u]
        p, r = sum(sel) / len(sel), sum(sel) / pos
        ap += (r - r_prev) * p; r_prev = r
    return ap


@unittest.skipUnless(HAY, 'requiere numpy, pandas y hypothesis')
class PropiedadesMetricas(unittest.TestCase):
    @given(st.lists(st.tuples(st.booleans(), st.integers(0, 6)), min_size=1, max_size=40))
    def test_precision_promedio_contra_referencia_e_invariante(self, pares):
        y = [int(a) for a, _ in pares]; s = [float(b) for _, b in pares]
        ref = ap_referencia(y, s); obt = EC.precision_promedio(y, s)
        if ref is None:
            self.assertIsNone(obt)
        else:
            self.assertAlmostEqual(obt, ref, places=9)
            self.assertAlmostEqual(EC.precision_promedio(y, [math.exp(x / 3) * 7 - 2 for x in s]), obt, places=9, msg='transformación monótona')

    @given(st.lists(st.tuples(st.booleans(), st.integers(0, 6)), min_size=2, max_size=40))
    def test_roc_auc_contra_conteo_de_pares(self, pares):
        pos = [b for a, b in pares if a]; neg = [b for a, b in pares if not a]
        obt = EC.roc_auc([a for a, _ in pares], [float(b) for _, b in pares])
        if not pos or not neg:
            self.assertIsNone(obt); return
        ref = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg) / (len(pos) * len(neg))
        self.assertAlmostEqual(obt, ref, places=9)

    @given(st.lists(st.tuples(st.booleans(), st.integers(0, 3)), min_size=1, max_size=6), st.floats(0.01, 1.0))
    def test_aciertos_en_k_es_el_promedio_sobre_desempates(self, pares, k):
        y = [int(a) for a, _ in pares]; s = [float(b) for _, b in pares]; n = len(y); c = math.ceil(k * n)
        marcadas, aciertos = EC.aciertos_en_k(y, s, k)
        self.assertEqual(marcadas, c)
        totales = []
        for orden in itertools.permutations(range(n)):   # cada forma de romper empates, igual de probable
            top = sorted(orden, key=lambda i: -s[i])[:c]
            totales.append(sum(y[i] for i in top))
        self.assertAlmostEqual(aciertos, float(np.mean(totales)), places=9)

    @given(st.lists(finitos, min_size=3, max_size=80), st.floats(0.01, 100))
    def test_t_newey_west_invariante_a_escala(self, x, c):
        a, b = EV.t_newey_west(x, 2), EV.t_newey_west([c * v for v in x], 2)
        if a is None or b is None:
            self.assertEqual(a is None, b is None); return
        self.assertAlmostEqual(a, b, delta=1e-6 * max(1, abs(a)))

    @given(st.lists(st.tuples(finitos, finitos), min_size=3, max_size=40))
    def test_spearman_acotado_simetrico_e_invariante(self, pares):
        a = [p for p, _ in pares]; b = [q for _, q in pares]
        r = EV.spearman(a, b)
        if np.isnan(r): return
        self.assertLessEqual(abs(r), 1 + 1e-9)
        self.assertAlmostEqual(r, EV.spearman(b, a), places=9)
        t = [math.atan(v) for v in a]
        if len(set(t)) == len(set(a)):   # atan satura en punto flotante cerca de ±1e6: solo es estricta si no crea empates en la muestra
            self.assertAlmostEqual(r, EV.spearman(t, b), places=9, msg='transformación monótona estricta')

    @given(st.lists(st.floats(-0.2, 0.2), min_size=24, max_size=120), st.lists(st.floats(-0.3, 0.3), min_size=2, max_size=6))
    def test_dsr_es_probabilidad_y_castiga_la_busqueda(self, x, intentos):
        base = EV.sharpe_desinflado(x, [None])
        if base is None or base['probabilidad_dsr'] is None: return
        con = EV.sharpe_desinflado(x, intentos)
        self.assertTrue(0 <= base['probabilidad_dsr'] <= 1)
        if con['probabilidad_dsr'] is not None and con['sharpe0_anual'] >= 0:
            self.assertLessEqual(con['probabilidad_dsr'], base['probabilidad_dsr'] + 1e-12, 'probar más cosas nunca vuelve más creíble el resultado')


@unittest.skipUnless(HAY, 'requiere numpy, pandas y hypothesis')
class PropiedadesTiempo(unittest.TestCase):
    @given(st.lists(st.one_of(st.none(), positivos), min_size=1, max_size=40), st.integers(1, 10))
    def test_minimo_hacia_adelante_contra_bucle(self, valores, h):
        w = pd.Series([np.nan if v is None else v for v in valores], dtype=float)
        obt = DS.hacia_adelante_min(w, h)
        for t in range(len(w)):
            if t + h >= len(w):
                self.assertTrue(np.isnan(obt.iloc[t])); continue
            ventana = w.iloc[t + 1:t + h + 1].dropna()
            if len(ventana) < max(1, int(h * 0.8)):   # ventana vacía: no hay mínimo
                self.assertTrue(np.isnan(obt.iloc[t]))
            else:
                self.assertAlmostEqual(obt.iloc[t], ventana.min(), places=9)

    @given(st.lists(st.tuples(st.dates(min_value=_dt.date(2020, 1, 1), max_value=_dt.date(2021, 12, 31)), st.integers(1, 30)),
                    min_size=1, max_size=60))
    def test_fechas_rebalanceo_una_por_mes_y_con_cobertura(self, dias):
        fechas = pd.Series([pd.Timestamp(d) for d, n in dias for _ in range(n)])
        r = EV.fechas_rebalanceo(fechas)
        cuenta = fechas.value_counts()
        meses = {f.to_period('M') for f in fechas}
        self.assertEqual(len(r), len(meses)); self.assertEqual(len({f.to_period('M') for f in r}), len(r))
        for f in r:
            del_mes = cuenta[cuenta.index.to_period('M') == f.to_period('M')]
            self.assertGreaterEqual(cuenta[f], 0.9 * del_mes.max())
            self.assertEqual(f, del_mes[del_mes >= 0.9 * del_mes.max()].index.min(), 'el primero que cumple, no uno posterior')


@st.composite
def reportes_trimestrales(draw):
    """Años fiscales con trimestres (ruido de 52/53 semanas) presentados como 10-Q acumulados con comparativo y 10-K anual con comparativo."""
    anios = draw(st.integers(2, 4)); inicio = 2015
    q = {(inicio + a, k): draw(st.integers(-50, 100)) for a in range(anios) for k in range(1, 5)}
    ruido = {(inicio + a, k): draw(st.integers(-3, 3)) for a in range(anios) for k in range(1, 5)}
    fin_q = lambda y, k: pd.Timestamp(y, 3 * k, 1) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=ruido[(y, k)])
    hechos = []
    for a in range(anios):
        y = inicio + a
        for k in range(1, 5):
            fin = fin_q(y, k); presentado = fin + pd.Timedelta(days=60 if k == 4 else 40)
            filas = [(y, k)] + ([(y - 1, k)] if a > 0 else [])
            for yy, kk in filas:
                hechos.append({'inicio': pd.Timestamp(yy, 1, 1), 'fin': fin_q(yy, kk), 'valor': float(sum(q[(yy, j)] for j in range(1, kk + 1))),
                               'conocido': pd.Timestamp(presentado, tz='UTC')})
    return q, fin_q, anios, inicio, pd.DataFrame(hechos)


@unittest.skipUnless(HAY, 'requiere numpy, pandas y hypothesis')
class PropiedadesFundamentales(unittest.TestCase):
    @given(reportes_trimestrales())
    def test_doce_meses_es_la_suma_de_los_ultimos_cuatro_trimestres(self, datos):
        q, fin_q, anios, inicio, hechos = datos
        s = FU.Serie(hechos.assign(preferencia=0))
        orden = [(inicio + a, k) for a in range(anios) for k in range(1, 5)]
        for idx, (y, k) in enumerate(orden):
            momento = pd.Timestamp(fin_q(y, k) + pd.Timedelta(days=60 if k == 4 else 40), tz='UTC')
            v, F = FU.doce_meses(s, s.hasta(momento))
            self.assertEqual(pd.Timestamp(F), fin_q(y, k))
            if idx >= 3 and (k == 4 or y > inicio):
                self.assertAlmostEqual(v, float(sum(q[orden[j]] for j in range(idx - 3, idx + 1))), places=6, msg=(y, k))
            elif k != 4:
                self.assertTrue(np.isnan(v), 'sin el anual previo no hay 12 meses')

    @given(reportes_trimestrales(), st.integers(-200, 200))
    def test_una_reexpresion_posterior_no_cambia_el_pasado(self, datos, nuevo):
        q, fin_q, anios, inicio, hechos = datos
        y = inicio
        reexp = {'inicio': pd.Timestamp(y, 1, 1), 'fin': fin_q(y, 4), 'valor': float(nuevo), 'conocido': pd.Timestamp('2030-01-01', tz='UTC')}
        s0 = FU.Serie(hechos.assign(preferencia=0))
        s1 = FU.Serie(pd.concat([hechos, pd.DataFrame([reexp])], ignore_index=True).assign(preferencia=0))
        for m in hechos['conocido'].unique():
            a, b = FU.doce_meses(s0, s0.hasta(m)), FU.doce_meses(s1, s1.hasta(m))
            self.assertTrue((np.isnan(a[0]) and np.isnan(b[0])) or a[0] == b[0])
        self.assertEqual(FU.doce_meses(s1, s1.hasta(pd.Timestamp('2030-01-02', tz='UTC')), fin_q(y, 4))[0], float(nuevo),
                         'con la reexpresión ya conocida se usa la versión nueva')


@unittest.skipUnless(HAY, 'requiere numpy, pandas y hypothesis')
class PropiedadesEntrenar(unittest.TestCase):
    @given(st.lists(st.tuples(st.integers(0, 6), st.floats(-1, 1, allow_nan=False)), min_size=1, max_size=80))
    def test_rango_centrado_promedia_medio_en_cualquier_grupo(self, filas):
        grupos = [g for g, _ in filas]; valores = [v for _, v in filas]
        r = pd.Series(DS.rango_centrado(valores, [grupos]))
        medias = r.groupby(grupos).mean()
        np.testing.assert_allclose(medias.to_numpy(), 0.5, atol=1e-12, err_msg='un grupo de cualquier tamaño promedia 0.5: el tamaño no se puede aprender')
        self.assertTrue(((r > 0) & (r < 1)).all())
        for g in set(grupos):   # monótono dentro del grupo
            v = np.array(valores)[np.array(grupos) == g]; rr = r[np.array(grupos) == g].to_numpy()
            self.assertTrue(all(rr[i] <= rr[j] for i in range(len(v)) for j in range(len(v)) if v[i] < v[j]))

    @given(st.integers(0, 2500), st.integers(800, 3000), st.sampled_from([21, 63]), st.integers(0, 130), st.integers(0, 6), st.integers(200, 2900),
           st.integers(0, 30))
    def test_walk_forward_nunca_entrena_con_objetivos_que_tocan_la_prueba_ni_el_pliegue(self, desfase, n, h, purga, anios, pos_pliegue, huecos):
        cal = pd.bdate_range(pd.Timestamp('2006-01-02') + pd.Timedelta(days=desfase), periods=n)
        cal = cal.delete(sorted(set(np.random.default_rng(huecos).integers(0, n, size=huecos))))   # feriados
        meses = pd.DatetimeIndex(EV.fechas_rebalanceo(pd.Series(cal)))
        pliegue = cal[min(pos_pliegue, len(cal) - 1)]
        primer = cal[0].year + anios
        partes = EN.particiones(meses, cal, h, purga=purga, primer_anio=primer, pliegue_desde=pliegue)
        inicio = cal.searchsorted(pliegue)
        vistos = set()
        for p in partes:
            self.assertGreaterEqual(p['anio'], primer)
            self.assertTrue(all(m.year == p['anio'] for m in p['prueba']))
            self.assertLessEqual(int(EN.fin_objetivo(cal, p['entrenar'], h).max()) + purga, int(cal.searchsorted(p['prueba'][0])))
            self.assertLess(int(EN.fin_objetivo(cal, p['prueba'], h).max()), inicio)
            self.assertFalse(vistos & set(p['prueba']), 'cada mes se prueba una sola vez'); vistos |= set(p['prueba'])
        ext = EN.particion_pliegue(meses, cal, h, purga=purga, pliegue_desde=pliegue)
        if len(ext['entrenar']):
            self.assertLessEqual(int(EN.fin_objetivo(cal, ext['entrenar'], h).max()) + purga, inicio)
        self.assertTrue((ext['prueba'] >= pliegue).all())


if __name__ == '__main__':
    unittest.main()


# ================================================================ punto en el tiempo: ningún futuro alternativo cambia el pasado
sys.path.insert(0, os.path.join(RAIZ, 'pruebas'))
GRUPOS_VARIABLES = ('A', 'B', 'D', 'E', 'L', 'R')


def _reescribir_precios(d, fn):
    import glob
    for ruta in glob.glob(os.path.join(d, 'precios/diarios/*/*.parquet')):
        p = pd.read_parquet(ruta); p = fn(p); p.to_parquet(ruta, index=False)


@st.composite
def futuros(draw):
    corte = draw(st.dates(min_value=_dt.date(2020, 2, 3), max_value=_dt.date(2021, 3, 31)))
    tickers = draw(st.lists(st.sampled_from(['AAA', 'AAA.MX', 'BBB.MX', 'FIB.MX', 'GBR.L', 'ADR', '^MXX', '^GSPC', 'MXN=X', 'GBPMXN=X']), min_size=1, max_size=4, unique=True))
    factor = draw(st.floats(0.2, 5.0))
    dividendo = draw(st.floats(0.0, 3.0))
    return pd.Timestamp(corte), tickers, factor, dividendo, draw(st.booleans()), draw(st.booleans()), draw(st.booleans())


@unittest.skipUnless(HAY, 'requiere numpy, pandas y hypothesis')
class PropiedadPuntoEnElTiempo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile, test_dataset as TD
        cls.TD = TD
        cls.tmp = tempfile.TemporaryDirectory()
        TD.almacen_sintetico(cls.tmp.name)
        cls.base = DS.construir(cls.tmp.name, desde='2020-01-01').set_index(['fecha', 'ticker']).sort_index()
        meta = DS.meta_columnas()
        cls.variables = [c for c in cls.base.columns if meta.get(c, {}).get('grupo') in GRUPOS_VARIABLES and c != 'close']

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def comparar_hasta(self, otra, corte, cuales=None):
        cols = cuales or self.variables
        a = self.base.loc[self.base.index.get_level_values('fecha') <= corte, cols]
        b = otra.set_index(['fecha', 'ticker']).sort_index()
        b = b.loc[b.index.get_level_values('fecha') <= corte, cols]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-5, check_categorical=False)

    def test_las_columnas_del_sic_no_miran_el_futuro(self):
        """Determinista, y existe porque la prueba aleatoria NO bastaba: con 3 ejemplos y diez tickers, Hypothesis puede no
        sortear nunca el espejo del SIC, y entonces `dias_sic_12m` e `importe_sic_20d` pasan sin ser probadas. Se comprobó
        el 2026-09-18 inyectando un `.shift(-5)`: con 3 ejemplos pasaba y con 15 fallaba. Esta prueba no depende del sorteo."""
        import tempfile
        corte = pd.Timestamp('2020-09-30')
        with tempfile.TemporaryDirectory() as d:
            self.TD.almacen_sintetico(d)
            def perturbar(p):
                m = (pd.to_datetime(p['fecha']) > corte) & (p['ticker'] == 'AAA.MX')
                p.loc[m, 'close'] = p.loc[m, 'close'] * 4.0
                p.loc[m, 'volumen'] = 0                     # el espejo deja de operar: `opero` cambia en el futuro
                return p
            _reescribir_precios(d, perturbar)
            otra = DS.construir(d, desde='2020-01-01')
            self.comparar_hasta(otra, corte, cuales=['dias_sic_12m', 'importe_sic_20d'])
            # y la prueba tiene que estar mirando algo: sin valores, comparar no prueba nada
            self.assertGreater(int(self.base.loc[self.base.index.get_level_values('fecha') <= corte, 'dias_sic_12m'].notna().sum()), 0,
                               'si no hay valores antes del corte, esta prueba pasa en vacío')

    @settings(max_examples=int(os.environ.get('PBT_FUTUROS', '3')), deadline=None, database=None,
              phases=[Phase.explicit, Phase.generate])   # cada ejemplo arma el dataset: sin reducción de la falla (tardaría minutos); el ejemplo fallido se imprime
    @given(futuros())
    def test_cambiar_el_futuro_no_cambia_ninguna_variable_pasada(self, fut):
        import tempfile
        corte, tickers, factor, dividendo, con_reporte, con_sec, con_macro = fut
        utc = lambda s: pd.Timestamp(s, tz='UTC')
        with tempfile.TemporaryDirectory() as d:
            self.TD.almacen_sintetico(d)
            despues = lambda p: pd.to_datetime(p['fecha']) > corte
            def perturbar(p):
                m = despues(p) & p['ticker'].isin(tickers)
                p.loc[m, 'close'] = p.loc[m, 'close'] * factor
                p.loc[m, 'volumen'] = p.loc[m, 'volumen'] * 3 + 1
                # apagar el volumen algunos días futuros: sin esto, `opero` (volumen > 0) nunca cambia y las columnas que dependen
                # de ÉL —no del importe— quedan sin probar. Se descubrió el 2026-09-18 inyectando una fuga en `dias_sic_12m`.
                p.loc[m & (pd.to_datetime(p['fecha']).dt.day % 3 == 0), 'volumen'] = 0
                p.loc[m & (pd.to_datetime(p['fecha']).dt.day == 15), 'dividendo'] = dividendo
                return p
            _reescribir_precios(d, perturbar)
            fecha_futura = (corte + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
            if con_reporte:
                ruta = os.path.join(d, 'eventos/resultados/anio=2020.parquet'); ev = pd.read_parquet(ruta)
                ev = pd.concat([ev, pd.DataFrame([{'ticker': 'AAA', 'fecha_reporte': utc(fecha_futura + 'T20:30'), 'eps_estimado': 9.0, 'eps_reportado': -7.0,
                                                   'sorpresa_pct': -900.0, 'estimada': False, 'conocido': utc(fecha_futura + 'T20:30'), 'capturado': utc('2026-09-14')}])])
                ev.to_parquet(ruta, index=False)
            if con_sec:
                ruta = os.path.join(d, 'fundamentales/sec/anio_presentado=2020.parquet'); sec = pd.read_parquet(ruta)
                nuevo = sec[sec['concepto'] == 'Assets'].head(1).assign(valor=1.0, presentado=fecha_futura, conocido=utc(fecha_futura + 'T23:59:59'), accn='futuro')
                pd.concat([sec, nuevo]).to_parquet(ruta, index=False)
            if con_macro:
                ruta = os.path.join(d, 'macro/fred/anio=2020.parquet'); fr = pd.read_parquet(ruta)
                fr = pd.concat([fr, pd.DataFrame([{'fecha': '2020-02-04', 'serie': 'DGS10', 'valor': 99.0, 'conocido': utc(fecha_futura + 'T10:00')}])])
                fr.to_parquet(ruta, index=False)
            otra = DS.construir(d, desde='2020-01-01')
        self.comparar_hasta(otra, corte)

    @settings(max_examples=int(os.environ.get('PBT_FUTUROS', '3')), deadline=None, database=None,
              phases=[Phase.explicit, Phase.generate])   # cada ejemplo arma el dataset: sin reducción de la falla (tardaría minutos); el ejemplo fallido se imprime
    @given(st.dates(min_value=_dt.date(2020, 3, 2), max_value=_dt.date(2021, 3, 31)), st.sampled_from([2.0, 3.0, 4.0, 10.0]),
           st.sampled_from(['AAA', 'BBB.MX', 'GBR.L']), st.booleans())
    @example(dia=_dt.date(2020, 4, 3), razon=3.0, ticker='AAA', eps_bajado_antes=True)   # hallado en CI: AAA ya tenía un split posterior
    def test_un_split_futuro_con_el_historial_reajustado_como_yahoo_no_cambia_el_pasado(self, dia, razon, ticker, eps_bajado_antes):
        """Yahoo reexpresa TODO el historial al ocurrir un split: precios y dividendos anteriores ÷ razón, volumen × razón, utilidades por
        acción ÷ razón. Ninguna variable de un día anterior al split puede cambiar por eso (sería información del futuro).
        eps_bajado_antes: el almacén solo agrega, así que los reportes bajados ANTES del split conservan su EPS en la base vieja (tanda 4, M21)."""
        import tempfile
        corte = pd.Timestamp(dia)
        with tempfile.TemporaryDirectory() as d:
            self.TD.almacen_sintetico(d)
            fecha_split = (corte + pd.offsets.BDay(5)).strftime('%Y-%m-%d')   # día hábil: en fin de semana no habría fila donde anotar el split
            def reajustar(p):
                antes = (p['ticker'] == ticker) & (p['fecha'] < fecha_split)
                for col in ('close', 'high', 'low'):   # Yahoo reexpresa las tres: el máximo y el mínimo son precios como el cierre
                    if col in p: p.loc[antes, col] = p.loc[antes, col] / razon
                p.loc[antes, 'dividendo'] = p.loc[antes, 'dividendo'] / razon
                p.loc[antes, 'volumen'] = p.loc[antes, 'volumen'] * razon
                el_dia = (p['ticker'] == ticker) & (p['fecha'] == fecha_split)
                p.loc[el_dia, 'split'] = razon
                return p
            import glob
            originales = DS.splits_acumulados(pd.concat([pd.read_parquet(r) for r in glob.glob(os.path.join(d, 'precios/diarios/*/*.parquet'))]))
            _reescribir_precios(d, reajustar)
            ruta = os.path.join(d, 'eventos/resultados/anio=2020.parquet'); ev = pd.read_parquet(ruta)
            m = (ev['ticker'] == ticker) & (ev['fecha_reporte'] < pd.Timestamp(fecha_split, tz='UTC'))
            if eps_bajado_antes:
                # bajado el día previo al split: en la base de acciones de ESE día, así que se deshacen los splits que ya había después
                # (AAA tiene un 2:1 en 2020-06: Yahoo lo reexpresó en la captura original de 2026)
                nueva = pd.Timestamp(fecha_split) - pd.Timedelta(days=1)
                if originales is not None and (originales['ticker'] == ticker).any():
                    total = float(originales.loc[originales['ticker'] == ticker, 'indice'].iloc[-1])
                    ev.loc[m, 'eps_reportado'] = ev.loc[m, 'eps_reportado'] * total / DS.indice_split_en([ticker], [nueva], originales)[0]
                ev['capturado'] = pd.to_datetime(ev['capturado'], utc=True)
                ev.loc[m, 'capturado'] = nueva.tz_localize('UTC')
            else:
                ev.loc[m, ['eps_estimado', 'eps_reportado']] = ev.loc[m, ['eps_estimado', 'eps_reportado']] / razon
            ev.to_parquet(ruta, index=False)
            otra = DS.construir(d, desde='2020-01-01')
        base_t = self.base.loc[(self.base.index.get_level_values('ticker') == ticker) & (self.base.index.get_level_values('fecha') <= corte), self.variables]
        otra_t = otra[otra['ticker'] == ticker].set_index(['fecha', 'ticker']).sort_index()
        otra_t = otra_t.loc[otra_t.index.get_level_values('fecha') <= corte, self.variables]
        def iguales(a, b):
            if a.dtype.kind in 'fiub' and b.dtype.kind in 'fiub':
                return np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-5, equal_nan=True)
            return list(a.astype(object).fillna('∅')) == list(b.astype(object).fillna('∅'))
        distintas = [c for c in self.variables if not iguales(base_t[c], otra_t[c])]
        self.assertEqual(distintas, [], f'variables que cambian por un split futuro de {ticker} ({razon}:1)')
