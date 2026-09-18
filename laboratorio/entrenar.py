#!/usr/bin/env python3
"""
laboratorio/entrenar.py — primer modelo de la IA de Brota: árboles de gradiente regularizados, entrenados por tiempo.

Uso:
    pip install pandas pyarrow scikit-learn
    python3 laboratorio/entrenar.py --tarea ranking --nombre gbm_ranking_v1            # walk-forward antes del pliegue externo
    python3 laboratorio/evaluar.py --puntajes laboratorio/datos/puntajes/gbm_ranking_v1.parquet --nombre gbm_ranking_v1 --hasta <corte que imprime>
    python3 laboratorio/entrenar.py --tarea caidas --nombre gbm_caidas_v1
    python3 laboratorio/evaluar_caidas.py --puntajes … --nombre gbm_caidas_v1 --probabilidades --hasta <corte>
    python3 laboratorio/entrenar.py --tarea ranking --nombre gbm_ranking_v1 --pliegue-externo   # UNA vez, con el modelo ya elegido

Diseño (FORMULAS_LABORATORIO.md §8, tanda 4 de Science Mercado M18 y pedidos 149 y 151):
  - Muestra: una fecha por mes (fechas_rebalanceo de evaluar.py); las filas diarias se traslapan en el objetivo.
  - Walk-forward anual expansivo desde PRIMER_ANIO_PRUEBA: para probar el año Y se entrena con los meses cuya ventana del objetivo termina
    PURGA sesiones antes del primer mes de Y. Los meses de prueba cuyo objetivo entraría al pliegue externo se descartan.
  - Pliegue externo (desde 2021-01, decisión del dueño): no se usa para elegir nada. --pliegue-externo entrena UN modelo final con todo lo
    anterior y predice el pliegue; cada apertura queda en resultados/pliegue_externo.jsonl y el DSR ahí debe usar N = número de aperturas.
  - Modelo: HistGradientBoosting de scikit-learn (mismo algoritmo de histogramas que LightGBM, sin dependencias del sistema) con los rangos
    de M18: hojas pocas y grandes, tasa baja, submuestreo de variables, L2. Número de árboles elegido con los últimos meses de entrenamiento
    (separados por la purga) con la regla de un error estándar, y reentrenado con todos. BOLSAS = 10 bolsas por MES, no por fila (las filas
    de un mes comparten el factor común); cada una elige sus árboles y las predicciones se promedian (tanda 5 M26).
  - Ranking: regresión sobre y_21_rango_libro (rango centrado dentro de México o extranjero; tanda 4B). Caídas: clasificación de min_ret_63 ≤ −20 %.
  - Variables: grupos A, B, D, L, R y E del dataset (sin analistas, sin identidad, sin objetivos). Toda variable ya pasó las pruebas de punto
    en el tiempo. Se declara que las variables se eligieron mirando toda la muestra (Mercado, tanda 4 §5).
  - Importancia por permutación sobre las predicciones fuera de muestra: variables de emisora permutadas dentro del mes; variables de fecha
    (macro, régimen) permutando meses completos, porque dentro del mes son constantes.
"""
import argparse, datetime as dt, gc, hashlib, json, os, sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import evaluar as EV  # noqa: E402

VERSION = 'entrenar-v1'
SEMILLA = 20260915
GRUPOS = ('A', 'B', 'D', 'L', 'R', 'E')
EXCLUIR = {'close'}                                   # nivel de precio crudo reexpresado por splits: no es variable
PURGA = 63                                            # sesiones entre el fin del objetivo de entrenamiento y la prueba
PRIMER_ANIO_PRUEBA = 2012                             # 2006–2011 = seis años para el primer entrenamiento
PLIEGUE_EXTERNO_DESDE = EV.CRITERIO['pliegue_externo_desde']
MESES_VALIDACION = 24                                 # últimos meses del entrenamiento para elegir el número de árboles
MIN_EMPRESAS_MES = 25
TAREAS = {
    'ranking': {'objetivo': 'y_21_rango_libro', 'horizonte': 21, 'tipo': 'regresion'},
    'caidas': {'objetivo': 'min_ret_63', 'umbral': -0.20, 'horizonte': 63, 'tipo': 'clasificacion'},
}
HIPER = {'learning_rate': 0.03, 'max_leaf_nodes': 8, 'min_samples_leaf': 500, 'max_depth': 4, 'l2_regularization': 10.0,
         'max_features': 0.5, 'max_iter': 600}
BOLSAS = 10                                           # tanda 5 M26: bolsas por mes amortiguan el ruido de la selección de árboles
FRACCION_MESES_BOLSA = 0.8
REGISTRO_PLIEGUE = 'pliegue_externo.jsonl'
REPETICIONES_PERMUTACION = 10                       # tanda 5 M26: una sola permutación no ordena variables con caídas parecidas


# ================================================================ datos
def variables_modelo(meta_cols, sin=()):
    """Variables del modelo. `sin` deja fuera columnas concretas: sirve para ABLACIONES (misma configuración, misma base, una variable
    menos), que son diagnóstico y no intento —no seleccionan nada, contestan a qué se debió una diferencia ya observada."""
    sin = set(sin)
    return [c for c, m in meta_cols.items() if m.get('grupo') in GRUPOS and c not in EXCLUIR and c != 'regimen' and c not in sin]


def muestra(tabla, tarea):
    """Filas de las fechas de rebalanceo con objetivo conocido y al menos MIN_EMPRESAS_MES emisoras. y: objetivo de la tarea."""
    cfg = TAREAS[tarea]
    fechas = EV.fechas_rebalanceo(tabla['fecha'])
    d = tabla[tabla['fecha'].isin(fechas) & tabla[cfg['objetivo']].notna()].copy()
    d['y'] = (d[cfg['objetivo']] <= cfg['umbral']).astype(float) if cfg['tipo'] == 'clasificacion' else d[cfg['objetivo']].astype(float)
    d = d[d.groupby('fecha')['ticker'].transform('size') >= MIN_EMPRESAS_MES]
    if cfg['tipo'] == 'regresion':
        # un rango cuyo promedio depende del grupo le enseña al modelo a reconocer el grupo, no a ordenar (y_21_rango_pais de dataset-v4)
        medias = d.groupby('fecha')['y'].mean()
        if not np.allclose(medias, 0.5, atol=0.02):
            raise ValueError(f"El objetivo {cfg['objetivo']} no promedia 0.5 por fecha (hasta {float((medias - 0.5).abs().max()):.3f}): regenera el dataset")
    return d.sort_values(['fecha', 'ticker']).reset_index(drop=True)


def calendario(tabla):
    return pd.DatetimeIndex(sorted(pd.unique(tabla['fecha'])))


def fin_objetivo(cal, fechas, horizonte):
    """Posición en el calendario del último día que usa el objetivo de cada fecha (fecha + horizonte sesiones)."""
    return cal.searchsorted(pd.DatetimeIndex(fechas)) + horizonte


def particiones(meses, cal, horizonte, purga=None, primer_anio=None, pliegue_desde=None):
    """Walk-forward anual. Para cada año de prueba: meses de entrenamiento cuyo objetivo termina ≥ purga sesiones antes del primer mes de
    prueba, y meses de prueba cuyo objetivo termina antes del pliegue externo. Devuelve [{anio, entrenar, prueba}].
    Sin argumentos se usan las constantes del módulo en el momento de llamar (no al importar)."""
    purga = PURGA if purga is None else purga
    primer_anio = PRIMER_ANIO_PRUEBA if primer_anio is None else primer_anio
    pliegue_desde = pliegue_desde or PLIEGUE_EXTERNO_DESDE
    meses = pd.DatetimeIndex(sorted(pd.unique(meses)))
    inicio_pliegue = cal.searchsorted(pd.Timestamp(pliegue_desde))
    fin = fin_objetivo(cal, meses, horizonte)
    salida = []
    for anio in sorted({m.year for m in meses if m.year >= primer_anio}):
        prueba = meses[(meses.year == anio) & (fin < inicio_pliegue)]
        if not len(prueba): continue
        corte = cal.searchsorted(prueba[0])
        entrenar = meses[fin + purga <= corte]
        if len(entrenar): salida.append({'anio': anio, 'entrenar': entrenar, 'prueba': prueba})
    return salida


def particion_pliegue(meses, cal, horizonte, purga=None, pliegue_desde=None):
    purga = PURGA if purga is None else purga
    pliegue_desde = pliegue_desde or PLIEGUE_EXTERNO_DESDE
    meses = pd.DatetimeIndex(sorted(pd.unique(meses)))
    inicio = cal.searchsorted(pd.Timestamp(pliegue_desde))
    return {'anio': 'pliegue_externo', 'entrenar': meses[fin_objetivo(cal, meses, horizonte) + purga <= inicio], 'prueba': meses[meses >= pd.Timestamp(pliegue_desde)]}


# ================================================================ modelo
def _estimador(tipo, hiper, semilla, max_iter=None):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    h = dict(hiper); h['max_iter'] = max_iter or h['max_iter']
    clase = HistGradientBoostingClassifier if tipo == 'clasificacion' else HistGradientBoostingRegressor
    return clase(early_stopping=False, random_state=semilla, **h)


def metrica_por_mes(d, pred, tipo):
    """Mayor = mejor, un valor por mes. Ranking: IC de Spearman. Caídas: −log loss medio del mes (elige probabilidades bien calibradas,
    no el orden en el decil superior; declarado en el reporte)."""
    d = d.reset_index(drop=True)
    salida = []
    for _, g in d.groupby('fecha'):
        idx = g.index.to_numpy()
        if tipo == 'clasificacion':
            p = np.clip(pred[idx], 1e-6, 1 - 1e-6); y = g['y'].to_numpy()
            salida.append(float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
        else:
            salida.append(EV.spearman(g['y'].to_numpy(), pred[idx]))
    x = np.array(salida, float)
    return x[~np.isnan(x)]


def metrica(d, pred, tipo):
    x = metrica_por_mes(d, pred, tipo)
    return float(x.mean()) if len(x) else np.nan


def arboles_un_error_estandar(curvas):
    """curvas: etapas × meses. Regla de un error estándar (Breiman, Friedman, Olshen y Stone, 1984) con el error estándar de la DIFERENCIA
    EMPAREJADA contra la mejor etapa: todas las etapas se miden en los mismos meses, así que el efecto común del mes se cancela. Con el error
    estándar del NIVEL —el del libro, pensado para pliegues independientes— la regla resulta 2 a 7 veces más estricta según la correlación
    entre etapas y elegía 1 árbol en el 40 % de las bolsas (tanda 6, M32)."""
    medias = np.nanmean(curvas, axis=1)
    mejor = int(np.nanargmax(medias))
    dif = curvas - curvas[mejor]
    n = np.sum(~np.isnan(dif), axis=1)
    with np.errstate(invalid='ignore'):
        ee = np.nanstd(dif, axis=1, ddof=1) / np.sqrt(np.maximum(n, 1))
    ee = np.where(np.isfinite(ee), ee, 0.0)
    return int(np.argmax(medias >= medias[mejor] - ee)) + 1


def _predecir(modelo, X, tipo):
    return modelo.predict_proba(X)[:, 1] if tipo == 'clasificacion' else modelo.predict(X)


def ajustar(d, meses, variables, tipo, cal, horizonte, hiper=None, bolsas=None, semilla=SEMILLA):
    """Elige el número de árboles con los últimos MESES_VALIDACION meses (separados por la purga) y reentrena con todos los meses.
    Con bolsas > 1, cada bolsa usa una fracción de MESES (sin reemplazo) y las predicciones se promedian."""
    hiper = HIPER if hiper is None else hiper
    bolsas = BOLSAS if bolsas is None else bolsas
    rng = np.random.default_rng(semilla)
    modelos, iteraciones = [], []
    for b in range(bolsas):
        ms = pd.DatetimeIndex(sorted(rng.choice(meses, size=max(1, int(round(FRACCION_MESES_BOLSA * len(meses)))), replace=False))) if bolsas > 1 else meses
        n_iter = hiper['max_iter']
        if len(ms) > MESES_VALIDACION + 12:
            val = ms[-MESES_VALIDACION:]
            base = ms[fin_objetivo(cal, ms, horizonte) + PURGA <= cal.searchsorted(val[0])]
            dv, db = d[d['fecha'].isin(val)].reset_index(drop=True), d[d['fecha'].isin(base)]
            if len(base) and len(db) and len(dv):
                m = _estimador(tipo, hiper, semilla + b).fit(db[variables], db['y'])
                etapas = m.staged_predict_proba(dv[variables]) if tipo == 'clasificacion' else m.staged_predict(dv[variables])
                curvas = [metrica_por_mes(dv, (p[:, 1] if tipo == 'clasificacion' else p), tipo) for p in etapas]
                largo = min(len(c) for c in curvas)
                n_iter = arboles_un_error_estandar(np.array([c[:largo] for c in curvas]))
        dt_ = d[d['fecha'].isin(ms)]
        modelos.append(_estimador(tipo, hiper, semilla + b, max_iter=n_iter).fit(dt_[variables], dt_['y']))
        iteraciones.append(n_iter)
    return modelos, iteraciones


def predecir(modelos, X, tipo):
    return np.mean([_predecir(m, X, tipo) for m in modelos], axis=0)


# ================================================================ importancia
def importancia_permutacion(pruebas, variables, tipo, semilla=SEMILLA, repeticiones=None):
    """pruebas: [(modelos, d_prueba)] fuera de muestra. Caída de la métrica al permutar cada variable (media y desviación de `repeticiones`
    permutaciones); las de fecha (constantes dentro del mes) se permutan entre meses completos."""
    repeticiones = REPETICIONES_PERMUTACION if repeticiones is None else repeticiones
    rng = np.random.default_rng(semilla)
    todo = pd.concat([d for _, d in pruebas], ignore_index=True)
    pred = np.concatenate([predecir(m, d[variables], tipo) for m, d in pruebas])
    base = metrica(todo, pred, tipo)
    salida = {}
    for v in variables:
        if todo[v].isna().all(): continue
        de_fecha = todo.groupby('fecha')[v].nunique(dropna=False).max() <= 1
        caidas = []
        for _ in range(repeticiones):
            preds = []
            for modelos, d in pruebas:
                X = d[variables].copy()
                if de_fecha:
                    valores = d.groupby('fecha')[v].first()
                    X[v] = d['fecha'].map(dict(zip(valores.index, rng.permutation(valores.to_numpy())))).to_numpy()
                else:
                    X[v] = d.groupby('fecha')[v].transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index)).to_numpy()
                preds.append(predecir(modelos, X, tipo))
            caidas.append(base - metrica(todo, np.concatenate(preds), tipo))
        salida[v] = {'caida_metrica': float(np.mean(caidas)), 'sd': float(np.std(caidas, ddof=1)) if len(caidas) > 1 else None,
                     'repeticiones': repeticiones, 'de_fecha': bool(de_fecha)}
    return base, dict(sorted(salida.items(), key=lambda kv: -kv[1]['caida_metrica']))


# ================================================================ corrida
def huella(tarea, variables, hiper, bolsas):
    return hashlib.sha256(json.dumps({'tarea': tarea, 'variables': sorted(variables), 'hiper': hiper, 'bolsas': bolsas, 'purga': PURGA,
                                      'objetivo': TAREAS[tarea]}, sort_keys=True).encode()).hexdigest()[:12]


def walk_forward(tabla, tarea, variables, hiper=None, bolsas=None, con_importancia=True):
    """Envoltura: arma la muestra mensual y el calendario y llama a walk_forward_muestra."""
    return walk_forward_muestra(muestra(tabla, tarea), calendario(tabla), tarea, variables, hiper, bolsas, con_importancia)


def walk_forward_muestra(d, cal, tarea, variables, hiper=None, bolsas=None, con_importancia=True):
    """d: muestra mensual (la tabla diaria completa ya se puede liberar; con 693 mil filas y 95 variables el proceso se quedaba sin memoria)."""
    cfg = TAREAS[tarea]
    partes = particiones(d['fecha'], cal, cfg['horizonte'])
    puntajes, pruebas, pliegues = [], [], []
    for p in partes:
        modelos, iters = ajustar(d, p['entrenar'], variables, cfg['tipo'], cal, cfg['horizonte'], hiper, bolsas)
        dp = d[d['fecha'].isin(p['prueba'])].reset_index(drop=True)
        pred = predecir(modelos, dp[variables], cfg['tipo'])
        puntajes.append(pd.DataFrame({'fecha': dp['fecha'], 'ticker': dp['ticker'], 'puntaje': pred}))
        pruebas.append((modelos, dp))
        pliegues.append({'anio': int(p['anio']), 'meses_entrenar': len(p['entrenar']), 'filas_entrenar': int(d['fecha'].isin(p['entrenar']).sum()),
                         'ultimo_mes_entrenar': str(p['entrenar'][-1].date()), 'primer_mes_prueba': str(p['prueba'][0].date()),
                         'meses_prueba': len(p['prueba']), 'arboles': iters, 'metrica_prueba': metrica(dp, pred, cfg['tipo'])})
    reporte = {'pliegues': pliegues}
    if pruebas:
        ultimo = max(pd.Timestamp(p['prueba'][-1]) for p in partes)
        reporte['corte_evaluacion'] = str((cal[min(cal.searchsorted(ultimo) + 1, len(cal) - 1)]).date())
        reporte['inicio_evaluacion'] = str(min(pd.Timestamp(p['prueba'][0]) for p in partes).date())
        if con_importancia:
            reporte['metrica_fuera_de_muestra'], reporte['importancia_permutacion'] = importancia_permutacion(pruebas, variables, cfg['tipo'])
    return (pd.concat(puntajes, ignore_index=True) if puntajes else pd.DataFrame(columns=['fecha', 'ticker', 'puntaje'])), reporte


def registrar_pliegue(carpeta, entrada, permitir_otra=False):
    """El pliegue externo se abre una vez por tarea. Si ya se abrió, solo con permitir_otra, y el N del DSR pasa a ser el número de aperturas."""
    ruta = os.path.join(carpeta, REGISTRO_PLIEGUE)
    previas = []
    if os.path.exists(ruta):
        with open(ruta, encoding='utf-8') as fh: previas = [json.loads(l) for l in fh if l.strip()]
    mismas = [p for p in previas if p['tarea'] == entrada['tarea']]
    if mismas and not permitir_otra:
        raise SystemExit(f"El pliegue externo de «{entrada['tarea']}» ya se abrió con {[p['nombre'] for p in mismas]}. Abrirlo otra vez convierte "
                         'el pliegue en selección: usa --otra-apertura y evalúa con N = número de aperturas (pedido 149).')
    entrada = dict(entrada, apertura=len(mismas) + 1)
    os.makedirs(carpeta, exist_ok=True)
    with open(ruta, 'a', encoding='utf-8') as fh: fh.write(json.dumps(entrada, ensure_ascii=False) + '\n')
    return entrada


def pliegue_externo(tabla, tarea, variables, hiper=None, bolsas=None):
    cal = calendario(tabla)
    fechas = EV.fechas_rebalanceo(tabla['fecha'])
    # predice TODAS las filas del pliegue en fechas de rebalanceo, incluso sin objetivo todavía (los últimos meses)
    dp = tabla[tabla['fecha'].isin(fechas) & (tabla['fecha'] >= pd.Timestamp(PLIEGUE_EXTERNO_DESDE))].reset_index(drop=True)
    return pliegue_externo_muestra(muestra(tabla, tarea), cal, dp, tarea, variables, hiper, bolsas)


def pliegue_externo_muestra(d, cal, dp, tarea, variables, hiper=None, bolsas=None):
    cfg = TAREAS[tarea]
    p = particion_pliegue(d['fecha'], cal, cfg['horizonte'])
    modelos, iters = ajustar(d, p['entrenar'], variables, cfg['tipo'], cal, cfg['horizonte'], hiper, bolsas)
    pz = pd.DataFrame({'fecha': dp['fecha'], 'ticker': dp['ticker'], 'puntaje': predecir(modelos, dp[variables], cfg['tipo'])})
    return pz, {'meses_entrenar': len(p['entrenar']), 'ultimo_mes_entrenar': str(p['entrenar'][-1].date()), 'arboles': iters,
                'meses_pliegue': int(dp['fecha'].nunique())}


def main(argv=None):
    a = argparse.ArgumentParser(description='Entrenamiento por tiempo de la IA de Brota')
    a.add_argument('--dataset', default=os.path.join(AQUI, 'datos')); a.add_argument('--resultados', default=os.path.join(AQUI, 'resultados'))
    a.add_argument('--tarea', choices=list(TAREAS), required=True); a.add_argument('--nombre', required=True)
    a.add_argument('--bolsas', type=int, default=BOLSAS); a.add_argument('--sin-importancia', action='store_true')
    a.add_argument('--sin-variables', default='', help='columnas a dejar fuera, separadas por coma: ablación (diagnóstico, no intento)')
    a.add_argument('--pliegue-externo', action='store_true'); a.add_argument('--otra-apertura', action='store_true')
    args = a.parse_args(argv)
    meta_ruta = os.path.join(args.dataset, 'dataset_meta.json')
    with open(meta_ruta, encoding='utf-8') as fh: meta = json.load(fh)
    variables = variables_modelo(meta.get('columnas') or {}, [v for v in (args.sin_variables or '').split(',') if v])
    EV.revisar_grupos(meta.get('columnas') or {}, variables, False)
    cfg = TAREAS[args.tarea]
    tabla, _ = EV.leer_dataset(args.dataset, ['fecha', 'ticker', 'pais', cfg['objetivo']] + variables)
    variables = [v for v in variables if v in tabla and not tabla[v].isna().all()]
    tabla['fecha'] = pd.to_datetime(tabla['fecha'])
    tabla = tabla.sort_values(['fecha', 'ticker']).reset_index(drop=True)
    for v in variables: tabla[v] = tabla[v].astype('float32')
    salida_pz = os.path.join(args.dataset, 'puntajes'); os.makedirs(salida_pz, exist_ok=True)
    comun = {'version': VERSION, 'nombre': args.nombre, 'tarea': args.tarea, 'objetivo': cfg, 'modelo': 'sklearn HistGradientBoosting',
             'hiper': HIPER, 'bolsas': args.bolsas, 'purga_sesiones': PURGA, 'variables': variables, 'huella': huella(args.tarea, variables, HIPER, args.bolsas),
             'dataset': {k: meta.get(k) for k in ('version', 'generado', 'almacen_commit', 'filas', 'fecha_min', 'fecha_max')},
             'codigo': EV.commit_codigo(), 'semilla': SEMILLA, 'entrenado': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
             'limitaciones': ['las variables y sus grupos se eligieron mirando líneas base de toda la muestra, incluido el pliegue externo (tanda 4 §5): '
                              'el pliegue protege contra el ajuste de hiperparámetros, no contra la selección de variables',
                              'precios de Yahoo: nada es publicable (compuerta_publicacion)']}
    cal = calendario(tabla)
    muestra_mensual = muestra(tabla, args.tarea)
    filas_pliegue = None
    if args.pliegue_externo:
        fechas = EV.fechas_rebalanceo(tabla['fecha'])
        filas_pliegue = tabla[tabla['fecha'].isin(fechas) & (tabla['fecha'] >= pd.Timestamp(PLIEGUE_EXTERNO_DESDE))].reset_index(drop=True)
    del tabla; gc.collect()   # la tabla diaria completa ya no hace falta: solo la muestra mensual
    if args.pliegue_externo:
        entrada = registrar_pliegue(args.resultados, {'nombre': args.nombre, 'tarea': args.tarea, 'huella': comun['huella'], 'abierto': comun['entrenado'],
                                                      'codigo': comun['codigo']}, args.otra_apertura)
        pz, info = pliegue_externo_muestra(muestra_mensual, cal, filas_pliegue, args.tarea, variables, bolsas=args.bolsas)
        ruta = os.path.join(salida_pz, f'{args.nombre}_pliegue_externo.parquet'); pz.to_parquet(ruta, index=False)
        rep = dict(comun, pliegue_externo=dict(info, apertura=entrada['apertura']))
        print(f"Pliegue externo abierto (apertura {entrada['apertura']}): {len(pz)} filas → {ruta}")
        print(f"Evalúa con --desde {PLIEGUE_EXTERNO_DESDE} y N = {entrada['apertura']} aperturas.")
    else:
        pz, info = walk_forward_muestra(muestra_mensual, cal, args.tarea, variables, bolsas=args.bolsas, con_importancia=not args.sin_importancia)
        ruta = os.path.join(salida_pz, f'{args.nombre}.parquet'); pz.to_parquet(ruta, index=False)
        rep = dict(comun, walk_forward=info)
        for p in info['pliegues']:
            print(f"{p['anio']}: entrena {p['meses_entrenar']} meses (hasta {p['ultimo_mes_entrenar']}), árboles {p['arboles']}, métrica {p['metrica_prueba']:.4f}")
        if 'importancia_permutacion' in info:
            print('Más importantes:', ', '.join(f"{k} {v['caida_metrica']:.4f}" for k, v in list(info['importancia_permutacion'].items())[:10]))
        print(f"Puntajes fuera de muestra: {ruta}\nEvalúa con --desde {info.get('inicio_evaluacion')} --hasta {info.get('corte_evaluacion')} (mismos meses para la referencia; sin el pliegue externo).")
    rep_ruta = EV.ruta_libre(args.resultados, f"{dt.date.today().isoformat()}_entrenamiento_{args.nombre}", '.json')
    with open(rep_ruta, 'w', encoding='utf-8') as fh: json.dump(EV.limpiar_nan(rep), fh, ensure_ascii=False, indent=1)
    print('Reporte:', os.path.relpath(rep_ruta, EV.RAIZ))


if __name__ == '__main__':
    main()
