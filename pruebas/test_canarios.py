"""El arnés probándose a sí mismo: cada canario tiene que ser RECHAZADO.

Una prueba normal verifica el código. Ésta verifica **la prueba**: mete una fuga a propósito y exige que
el arnés la vea. Si un canario pasa, la suite falla con el nombre del agujero que se abrió, aunque todo
lo demás esté en verde.

Nace de un hecho concreto: el 2026-09-18 la prueba de punto en el tiempo **pasaba** con una fuga
deliberada en `dias_sic_12m`. Estaba en verde y no servía.
"""
import os
import sys
import tempfile
import unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'laboratorio'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas as pd
    import dataset as DS
    import canarios as C
    from test_dataset import almacen_sintetico, escribir
    HAY = True
except ImportError:
    HAY = False


def _reescribir(d, fn):
    import glob
    for ruta in glob.glob(os.path.join(d, 'precios/diarios/*/*.parquet')):
        p = pd.read_parquet(ruta)
        p = fn(p)
        p.to_parquet(ruta, index=False)


@unittest.skipUnless(HAY, 'requiere pandas y el laboratorio')
class BancoDeCanarios(unittest.TestCase):
    """Cada canario es una fuga deliberada; el arnés tiene que verla."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        almacen_sintetico(cls.tmp.name)
        cls.base_limpia = DS.construir(cls.tmp.name, desde='2020-01-01').set_index(['fecha', 'ticker']).sort_index()
        # SOLO las variables. Los objetivos (y_21, min_ret_63…) miran al futuro POR DISEÑO, así que
        # perturbarlo tiene que moverlos: incluirlos haría que cualquier perturbación pareciera una fuga.
        # Lo detectó el propio control de esta suite en su primera corrida.
        meta = DS.meta_columnas()
        GRUPOS = ('A', 'B', 'D', 'E', 'L', 'R')
        cls.cols = [c for c in cls.base_limpia.columns
                    if meta.get(c, {}).get('grupo') in GRUPOS and c != 'close'
                    and pd.api.types.is_numeric_dtype(cls.base_limpia[c])]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _pasado_cambia(self, parche, perturbacion, columnas):
        """Construye con el parche puesto, perturba el futuro, y devuelve si alguna fila ≤ corte se movió."""
        restaurar = parche(DS)
        if restaurar is None:
            self.skipTest('el canario no aplica a esta versión del dataset')
        try:
            with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
                almacen_sintetico(d1)
                almacen_sintetico(d2)
                _reescribir(d2, C.PERTURBACIONES[perturbacion])
                a = DS.construir(d1, desde='2020-01-01').set_index(['fecha', 'ticker']).sort_index()
                b = DS.construir(d2, desde='2020-01-01').set_index(['fecha', 'ticker']).sort_index()
        finally:
            restaurar()
        cols = [c for c in (columnas or self.cols) if c in a.columns and c in b.columns]
        ix = a.index.intersection(b.index)
        ix = ix[ix.get_level_values('fecha') <= C.CORTE]
        self.assertGreater(len(ix), 0, 'no hay filas antes del corte que comparar')
        for c in cols:
            x, y = a.loc[ix, c].fillna(-9.99), b.loc[ix, c].fillna(-9.99)
            if not x.equals(y):
                return True, c
        return False, None

    def test_cada_canario_es_rechazado(self):
        fallaron = []
        for can in C.CANARIOS:
            with self.subTest(canario=can['nombre']):
                try:
                    visto, col = self._pasado_cambia(can['parche'], can['perturbacion'], can['columnas'])
                except unittest.SkipTest:
                    continue
                if not visto:
                    fallaron.append(f"{can['nombre']} (agujero {can['agujero']})")
                self.assertTrue(visto, f"EL ARNÉS NO VIO LA FUGA «{can['nombre']}» — agujero: {can['agujero']}. "
                                       'Una prueba en verde no significa nada si no ve esto.')
        self.assertFalse(fallaron, f'canarios que pasaron sin ser vistos: {fallaron}')

    def test_sin_canario_el_pasado_no_cambia(self):
        """El control del control: sin fuga, ninguna perturbación puede mover el pasado. Si esto falla,
        el arnés estaría marcando fugas donde no las hay y los canarios no probarían nada."""
        for nombre in ('precio', 'volumen', 'nulos'):
            with self.subTest(perturbacion=nombre):
                visto, col = self._pasado_cambia(lambda _DS: (lambda: None), nombre, None)
                self.assertFalse(visto, f'sin fuga, la perturbación «{nombre}» movió {col}')


if __name__ == '__main__':
    unittest.main()
