"""Ficha de sobreextensión: las cuatro pruebas del pedido AM de Fable (`REGLA_SOBREEXTENSION.md`).

Se corren EN ORDEN y la primera que falla manda el detector a diagnóstico. La regla fija del paso 4 quedó
escrita en la regla antes de correr el paso 1; aquí solo se implementa.

    python3 sobreextension.py --puntajes datos/puntajes/gbm_caidas_v2.parquet
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import evaluar as EV
import evaluar_caidas as EC

DESDE, HASTA = '2012-01-01', '2021-01-01'     # población declarada: 2012-2020
UMBRAL = -0.20                                 # tocó −20 % en trayectoria
BLOQUE = 12                                    # bootstrap por bloques de 12 meses
REPS = 2000
NIVEL = 90                                     # intervalo al 90 %
REGLA_FIJA = ('runup_2a', 'max_ret_1m', 'vol_60d')   # pesos iguales, definida en la regla, sin entrenar


def poblacion(tabla, puntaje):
    """Cierres de mes 2012-2020, libro mexicano, tercil SUPERIOR de impulso 12-1. Nada fuera de ahí entra."""
    fechas = EV.fechas_rebalanceo(tabla['fecha'])
    d = tabla[tabla['fecha'].isin(fechas)].copy()
    d['puntaje'] = puntaje
    d = d[(d['pais'] == 'MX') & (d['fecha'] >= DESDE) & (d['fecha'] < HASTA)]
    d = d.dropna(subset=['min_ret_63', 'ret_12m_ex1m', 'puntaje'])
    # tercil superior de impulso DENTRO del mes: el tercil se define en el libro mexicano de esa fecha
    t = d.groupby('fecha')['ret_12m_ex1m'].rank(pct=True, method='average')
    d = d[t > 2 / 3].copy()
    d['evento'] = (d['min_ret_63'] <= UMBRAL).astype(float)
    return d


def rango_mes(d, col):
    return d.groupby('fecha')[col].rank(pct=True, method='average')


def grupos(d, col_orden):
    """A = tercio con mayor `col_orden` dentro del mes (dentro del tercil superior); B = los otros dos tercios."""
    r = d.groupby('fecha')[col_orden].rank(pct=True, method='average')
    return (r > 2 / 3).to_numpy()


def xy(d, en_a):
    a, b = d.loc[en_a, 'evento'], d.loc[~en_a, 'evento']
    X = float(a.mean()) if len(a) else np.nan
    Y = float(b.mean()) if len(b) else np.nan
    return X, Y


def bloques_de(d):
    """Índices agrupados en bloques contiguos de 12 meses, para remuestrear sin romper la dependencia temporal."""
    meses = sorted(d['fecha'].unique())
    ini = {m: i // BLOQUE for i, m in enumerate(meses)}
    d = d.assign(_b=d['fecha'].map(ini))
    return [g.index.to_numpy() for _, g in d.groupby('_b')]


def bootstrap(d, en_a, azar, otro=None):
    """Distribución de X−Y (o de la diferencia contra `otro`) remuestreando bloques de 12 meses."""
    bl = bloques_de(d)
    ea = pd.Series(en_a, index=d.index)
    salida = []
    for _ in range(REPS):
        elegidos = [bl[i] for i in azar.integers(0, len(bl), len(bl))]
        idx = np.concatenate(elegidos)
        g = d.loc[idx]
        m = ea.loc[idx].to_numpy()
        if m.sum() < 5 or (~m).sum() < 5: continue
        v = float(g.loc[m, 'evento'].mean()) - float(g.loc[~m, 'evento'].mean())
        if otro is not None:
            o = pd.Series(otro, index=d.index).loc[idx].to_numpy()
            if o.sum() < 5 or (~o).sum() < 5: continue
            v -= float(g.loc[o, 'evento'].mean()) - float(g.loc[~o, 'evento'].mean())
        salida.append(v)
    s = np.array(salida)
    lo, hi = (100 - NIVEL) / 2, 100 - (100 - NIVEL) / 2
    return float(np.percentile(s, lo)), float(np.percentile(s, hi)), s


def episodios(d):
    """Meses con al menos un evento: la unidad independiente de verdad, no la emisora-mes."""
    return int(d.groupby('fecha')['evento'].max().sum())


def prueba_1(d, en_a, azar, etiqueta='modelo'):
    X, Y = xy(d, en_a)
    lo, hi, _ = bootstrap(d, en_a, azar)
    return {'prueba': f'1 separacion ({etiqueta})', 'X': X, 'Y': Y, 'X_menos_Y': X - Y,
            'cociente': X / Y if Y else None, 'ic90': [lo, hi], 'episodios': episodios(d),
            'emisoras_mes': int(len(d)), 'pasa': bool(lo > 0)}


def prueba_2(d):
    """Quintiles del puntaje DENTRO de la población; monotonía y Brier skill contra la base del tercil."""
    q = np.minimum((d.groupby('fecha')['puntaje'].rank(pct=True, method='average') * 5).fillna(0).astype(int), 4)
    frec = d.groupby(q)['evento'].mean()
    base = float(d['evento'].mean())
    sp = float(pd.Series(frec.index, dtype=float).corr(pd.Series(frec.to_numpy()), method='spearman'))
    p = d.groupby(q)['evento'].transform('mean')
    brier = float(((p - d['evento']) ** 2).mean())
    brier_base = float(((base - d['evento']) ** 2).mean())
    skill = 1 - brier / brier_base if brier_base else None
    mu = EC.murphy(d['evento'].to_numpy(), p.to_numpy())
    return {'prueba': '2 calibracion dentro del tercil', 'tasa_base_tercil': base,
            'frecuencia_por_quintil': [float(x) for x in frec.to_numpy()], 'spearman': sp,
            'brier_dentro': brier, 'brier_base_tercil': brier_base, 'brier_skill_dentro': skill,
            'confiabilidad_murphy': mu.get('confiabilidad'),
            'pasa': bool(sp >= 0.8 and skill is not None and skill > 0)}


def prueba_3(d, en_a, azar):
    d = d.assign(_a=en_a)
    anios = sorted(d['fecha'].dt.year.unique())
    por_anio, fuera = {}, {}
    for a in anios:
        g = d[d['fecha'].dt.year == a]
        X, Y = xy(g, g['_a'].to_numpy())
        por_anio[int(a)] = None if np.isnan(X) or np.isnan(Y) else X - Y
        h = d[d['fecha'].dt.year != a]
        lo, hi, _ = bootstrap(h, h['_a'].to_numpy(), azar)
        fuera[int(a)] = [lo, hi]
    positivos = sum(1 for v in por_anio.values() if v is not None and v > 0)
    todos_excluyen = all(v[0] > 0 for v in fuera.values())
    return {'prueba': '3 no es un anio', 'x_menos_y_por_anio': por_anio, 'ic90_sin_cada_anio': fuera,
            'anios_positivos': positivos, 'de': len(anios), 'todas_excluyen_cero': todos_excluyen,
            'pasa': bool(positivos >= 6 and todos_excluyen)}


def prueba_4(d, en_a, azar):
    """Regla fija sin entrenar: promedio simple del rango percentil dentro del mes de tres variables."""
    faltan = [c for c in REGLA_FIJA if c not in d.columns]
    if faltan: return {'prueba': '4 regla simple contra modelo', 'error': f'faltan columnas: {faltan}', 'pasa': None}
    d = d.copy()
    d['regla'] = sum(rango_mes(d, c) for c in REGLA_FIJA) / len(REGLA_FIJA)
    en_a2 = grupos(d, 'regla')
    r_reg = prueba_1(d, en_a2, azar, etiqueta='regla fija')
    lo, hi, _ = bootstrap(d, en_a, azar, otro=en_a2)
    usa_modelo = lo > 0
    return {'prueba': '4 regla simple contra modelo', 'regla': list(REGLA_FIJA),
            'regla_X': r_reg['X'], 'regla_Y': r_reg['Y'], 'regla_X_menos_Y': r_reg['X_menos_Y'],
            'diferencia_modelo_menos_regla': None, 'ic90_diferencia': [lo, hi],
            'la_ficha_usa': 'modelo' if usa_modelo else 'regla', 'pasa': bool(usa_modelo),
            'prueba_1_de_la_regla': r_reg}


def honestos(d, en_a):
    """Los dos números que van en la ficha aunque todo pase."""
    a = d.loc[en_a]
    sup = None
    if 'y_63_rel' in d.columns:
        sup = float((a['y_63_rel'] > 0).mean())
    elif 'y_63' in d.columns:
        med = d.groupby('fecha')['y_63'].transform('mean')
        sup = float((a['y_63'] > med.loc[a.index]).mean())
    ep = episodios(d)
    return {'fraccion_de_A_que_supero_al_universo': sup, 'episodios_independientes': ep,
            'pocos_episodios': bool(ep < 30)}


def main(argv=None):
    p = argparse.ArgumentParser(description='Ficha de sobreextensión: las cuatro pruebas del pedido AM')
    p.add_argument('--dataset', default=os.path.join(AQUI, 'datos'))
    p.add_argument('--puntajes', required=True)
    p.add_argument('--salida', default=os.path.join(AQUI, 'resultados'))
    p.add_argument('--semilla', type=int, default=20260918)
    a = p.parse_args(argv)

    cols = ['fecha', 'ticker', 'pais', 'en_app', 'ret_12m_ex1m', 'min_ret_63', 'y_63', 'y_63_rel'] + list(REGLA_FIJA)
    tabla, _ = EV.leer_dataset(a.dataset, cols)
    tabla = tabla.sort_values(['fecha', 'ticker']).reset_index(drop=True)
    pz = pd.read_parquet(a.puntajes, columns=['fecha', 'ticker', 'puntaje'])
    pz['ticker'] = pz['ticker'].astype(str); pz['fecha'] = pd.to_datetime(pz['fecha'])
    j = tabla.merge(pz, on=['fecha', 'ticker'], how='left')

    d = poblacion(tabla, j['puntaje'])
    azar = np.random.default_rng(a.semilla)
    en_a = grupos(d, 'puntaje')

    r = {'regla': 'REGLA_SOBREEXTENSION.md', 'poblacion': {'desde': DESDE, 'hasta': HASTA, 'libro': 'MX',
         'tercil': 'superior de ret_12m_ex1m', 'emisoras_mes': int(len(d)), 'meses': int(d['fecha'].nunique()),
         'emisoras': int(d['ticker'].nunique()), 'tasa_base': float(d['evento'].mean())}, 'pruebas': []}

    for fn in (lambda: prueba_1(d, en_a, azar), lambda: prueba_2(d),
               lambda: prueba_3(d, en_a, azar), lambda: prueba_4(d, en_a, azar)):
        res = fn()
        r['pruebas'].append(res)
        if res.get('pasa') is False and res['prueba'].startswith(('1', '2', '3')):
            r['veredicto'] = f"FALLA en la prueba {res['prueba']}: el detector va a DIAGNOSTICO"
            break
    else:
        ult = r['pruebas'][-1]
        r['veredicto'] = f"pasa las cuatro; la ficha usa el {ult.get('la_ficha_usa', '?')}"
    r['numeros_honestos'] = honestos(d, en_a)

    os.makedirs(a.salida, exist_ok=True)
    ruta = os.path.join(a.salida, '2026-09-18_sobreextension.json')
    n = 1
    while os.path.exists(ruta):
        n += 1; ruta = os.path.join(a.salida, f'2026-09-18_sobreextension_{n}.json')
    with open(ruta, 'w', encoding='utf-8') as fh: json.dump(r, fh, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print('\nEscrito:', os.path.relpath(ruta, EV.RAIZ))


if __name__ == '__main__':
    main()
