"""
laboratorio/publicable.py — compuerta de publicación de un modelo (pedidos 42 y 70 de Claude Science).

Un modelo solo es publicable si TODAS sus variables y TAMBIÉN su objetivo lo son: las etiquetas con las que se entrena cuentan igual que
las variables (un modelo con etiquetas de datos restringidos no se limpia quitando columnas). entrenar.py y evaluar.py llaman a
`verificar()` y guardan el resultado junto con cada modelo; mientras `publicable` sea False, el modelo vive solo en el laboratorio.
"""


def verificar(meta_columnas, variables, objetivo):
    bloqueos = []
    for c in list(variables) + [objetivo]:
        m = meta_columnas.get(c)
        rol = 'objetivo' if c == objetivo else 'variable'
        if m is None:
            bloqueos.append({'columna': c, 'rol': rol, 'motivo': 'sin metadatos (falla cerrado)'})
        elif not m.get('publicable'):
            bloqueos.append({'columna': c, 'rol': rol, 'motivo': f"fuente {m.get('fuente')} no publicable"})
    return {'publicable': not bloqueos, 'bloqueos': bloqueos,
            'objetivo_publicable': not any(b['rol'] == 'objetivo' for b in bloqueos),
            'variables_no_publicables': sorted({b['columna'] for b in bloqueos if b['rol'] == 'variable'})}
