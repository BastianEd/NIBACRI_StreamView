"""
ETL para el dashboard de Gerencia de Marketing (StreamView Analytics)
----------------------------------------------------------------------
"""

from pathlib import Path
import numpy as np
import pandas as pd

EXPORT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
ARCHIVO_LIMPIO = EXPORT_DIR / 'catalogo_marketing_limpio.csv'

# -----------------------------------------------------------------------------
# 1. Carga de la base ya limpia (salida del EDA, NO datos crudos ni aleatorios)
# -----------------------------------------------------------------------------
base = pd.read_csv(ARCHIVO_LIMPIO, low_memory=False)

columnas_esperadas = {
    'show_id', 'type', 'title', 'country', 'genres', 'language', 'idioma',
    'popularity', 'vote_count', 'vote_average', 'budget', 'revenue',
    'year_added', 'tipo_es',
}
faltantes = columnas_esperadas - set(base.columns)
if faltantes:
    raise ValueError(f"Faltan columnas esperadas en {ARCHIVO_LIMPIO.name}: {faltantes}")

# La columna 'idioma' que trae el catálogo limpio solo tradujo una parte de
# los códigos ISO (quedaban ~1.000 filas con 'cn', 'ta', 'ur', 'fa', etc. sin
# traducir). Se recalcula completa aquí para que Marketing siempre vea el
# nombre del idioma (Inglés, Japonés, Coreano...) y nunca el código.
MAPA_IDIOMAS = {
    'af': 'Afrikáans', 'am': 'Amárico', 'ar': 'Árabe', 'as': 'Asamés', 'az': 'Azerí',
    'bg': 'Búlgaro', 'bn': 'Bengalí', 'bs': 'Bosnio', 'ca': 'Catalán', 'cn': 'Chino',
    'cs': 'Checo', 'cy': 'Galés', 'da': 'Danés', 'de': 'Alemán', 'dz': 'Dzongkha',
    'el': 'Griego', 'en': 'Inglés', 'es': 'Español', 'et': 'Estonio', 'eu': 'Euskera',
    'fa': 'Persa', 'fi': 'Finés', 'fr': 'Francés', 'ga': 'Irlandés', 'gl': 'Gallego',
    'he': 'Hebreo', 'hi': 'Hindi', 'hr': 'Croata', 'ht': 'Criollo haitiano',
    'hu': 'Húngaro', 'hy': 'Armenio', 'id': 'Indonesio', 'is': 'Islandés',
    'it': 'Italiano', 'ja': 'Japonés', 'ka': 'Georgiano', 'kk': 'Kazajo',
    'kl': 'Groenlandés', 'km': 'Jemer', 'kn': 'Canarés', 'ko': 'Coreano',
    'ku': 'Kurdo', 'ky': 'Kirguís', 'la': 'Latín', 'lb': 'Luxemburgués',
    'lt': 'Lituano', 'lv': 'Letón', 'mi': 'Maorí', 'mk': 'Macedonio',
    'ml': 'Malayalam', 'mn': 'Mongol', 'mr': 'Maratí', 'ms': 'Malayo',
    'mt': 'Maltés', 'nb': 'Noruego', 'ne': 'Nepalí', 'nl': 'Neerlandés',
    'no': 'Noruego', 'or': 'Oriya', 'pa': 'Panyabí', 'pl': 'Polaco',
    'ps': 'Pastú', 'pt': 'Portugués', 'ro': 'Rumano', 'ru': 'Ruso',
    'si': 'Cingalés', 'sk': 'Eslovaco', 'sl': 'Esloveno', 'sr': 'Serbio',
    'sv': 'Sueco', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Tailandés',
    'tl': 'Tagalo', 'tr': 'Turco', 'uk': 'Ucraniano', 'ur': 'Urdu',
    'vi': 'Vietnamita', 'xx': 'Sin idioma', 'yo': 'Yoruba', 'za': 'Zhuang',
    'zh': 'Chino', 'zu': 'Zulú',
}
base['idioma'] = base['language'].map(MAPA_IDIOMAS).fillna(base['language'])

codigos_sin_traducir = set(base.loc[base['idioma'] == base['language'], 'language']) - {'xx'}
if codigos_sin_traducir:
    print(f"[aviso] códigos de idioma sin traducción, agregar a MAPA_IDIOMAS: {codigos_sin_traducir}")

# -----------------------------------------------------------------------------
# 2. Universo de "puntuables": para calcular percentiles y un score de
#    campaña con sentido, se necesita un mínimo de evidencia real de
#    recepción de audiencia. Se excluyen títulos sin ningún voto
#    (vote_count == 0) o con calificación en 0 (vote_average == 0): son
#    ~4.500 títulos (mayormente series) sin señal real, y si se incluyeran
#    distorsionarían los percentiles y podrían aparecer como "candidatos"
#    solo por ausencia de datos, no por desempeño real.
#    La evolución temporal del catálogo (sección 4.6) SÍ usa el catálogo
#    completo, porque ahí interesa cuánto contenido se fue incorporando,
#    tenga o no votos todavía.
# -----------------------------------------------------------------------------
con_votos = base[(base['vote_count'] > 0) & (base['vote_average'] > 0)].copy()

con_votos['p_popularidad'] = con_votos['popularity'].rank(pct=True)
con_votos['p_valoracion'] = con_votos['vote_average'].rank(pct=True)
con_votos['p_votos'] = np.log1p(con_votos['vote_count']).rank(pct=True)

con_votos['score_campania'] = (
    0.45 * con_votos['p_popularidad'] +
    0.30 * con_votos['p_valoracion'] +
    0.25 * con_votos['p_votos']
)

umbral_pop = con_votos['popularity'].median()
umbral_val = con_votos['vote_average'].median()

condiciones = [
    (con_votos['popularity'] >= umbral_pop) & (con_votos['vote_average'] >= umbral_val),
    (con_votos['popularity'] >= umbral_pop) & (con_votos['vote_average'] < umbral_val),
    (con_votos['popularity'] < umbral_pop) & (con_votos['vote_average'] >= umbral_val),
]
opciones = [
    'Promocionar: alto alcance y alta valoración',
    'Awareness: alto alcance; validar mensaje y segmentación',
    'Descubrir: alta valoración; aumentar visibilidad',
]
con_votos['cuadrante_campania'] = np.select(condiciones, opciones, default='No priorizar: bajo alcance y valoración')

# -----------------------------------------------------------------------------
# 3. Oportunidad comercial (budget / revenue) — NUEVO
#    Antes esta información se descartaba (el script viejo ni siquiera la traía
#    desde los CSV crudos), y sin embargo es justo lo que pide Marketing
#    ("oportunidades comerciales"). Solo existe budget y revenue reales para
#    películas (~22% de las películas los tiene; las series prácticamente no
#    los reportan en TMDB), así que el cálculo se limita a esos casos y el
#    resto queda marcado explícitamente como "Sin datos de presupuesto" en vez
#    de mostrarse como si tuviera rentabilidad cero. Se calcula sobre
#    'con_votos' para poder cruzar rentabilidad con popularidad/score real.
# -----------------------------------------------------------------------------
tiene_datos_financieros = (con_votos['budget'] > 0) & (con_votos['revenue'] > 0)

con_votos['roi'] = np.where(
    tiene_datos_financieros,
    (con_votos['revenue'] - con_votos['budget']) / con_votos['budget'],
    np.nan,
)

# 'temporada' (NUEVO) — a pedido de Marketing, para poder filtrar el cruce
# presupuesto/recaudación por época del año de incorporación a la plataforma
# (fecha real disponible: date_added; no existe mes de estreno en la fuente).
# Se usan estaciones del hemisferio sur (Chile), ya que StreamView reporta a
# Gerencia de Marketing local. Si el negocio opera también con calendario de
# hemisferio norte, este mapeo se ajusta cambiando MESES_TEMPORADA.
MESES_TEMPORADA = {
    12: 'Verano', 1: 'Verano', 2: 'Verano',
    3: 'Otoño', 4: 'Otoño', 5: 'Otoño',
    6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
    9: 'Primavera', 10: 'Primavera', 11: 'Primavera',
}
con_votos['date_added'] = pd.to_datetime(con_votos['date_added'], errors='coerce')
con_votos['temporada'] = con_votos['date_added'].dt.month.map(MESES_TEMPORADA)

mediana_roi = con_votos.loc[tiene_datos_financieros, 'roi'].median()
mediana_pop_fin = con_votos.loc[tiene_datos_financieros, 'popularity'].median()

condiciones_oportunidad = [
    ~tiene_datos_financieros,
    tiene_datos_financieros & (con_votos['roi'] >= mediana_roi) & (con_votos['popularity'] >= mediana_pop_fin),
    tiene_datos_financieros & (con_votos['roi'] >= mediana_roi) & (con_votos['popularity'] < mediana_pop_fin),
    tiene_datos_financieros & (con_votos['roi'] < mediana_roi) & (con_votos['popularity'] >= mediana_pop_fin),
]
opciones_oportunidad = [
    'Sin datos de presupuesto',
    'Ya explotado: alta rentabilidad y alta popularidad',
    'Joya oculta: alta rentabilidad, poca exposición — potenciar',
    'Popular pero poco rentable: revisar costo de campaña',
]
con_votos['oportunidad_comercial'] = np.select(
    condiciones_oportunidad, opciones_oportunidad, default='Bajo retorno y baja popularidad'
)

# -----------------------------------------------------------------------------
# 4. Vistas analíticas para el dashboard
# -----------------------------------------------------------------------------

# 4.1 Catálogo con score (solo títulos con votos reales, ver sección 2)
catalogo_marketing_score = con_votos.copy()

# 4.2 Top candidatos para campaña (además se exige un mínimo de votos más
#     alto para evitar títulos con score alto solo por ser nuevos/poco vistos)
#     Se usa 'idioma' (Inglés, Japonés, Coreano...) en vez de 'language'
#     (en, ja, ko...): Marketing necesita el nombre legible, no el código ISO.
top_candidatos_campania = (
    con_votos[con_votos['vote_count'] >= 100]
    .nlargest(15, 'score_campania')
    [['title', 'tipo_es', 'genres', 'idioma', 'popularity',
      'vote_average', 'vote_count', 'score_campania', 'cuadrante_campania']]
    .copy()
)

# 4.3 Explosión de géneros -> familia de género (mapeo curado a español)
MAPA_FAMILIA_GENERO = {
    'Action': 'Acción y aventura',
    'Action & Adventure': 'Acción y aventura',
    'Adventure': 'Acción y aventura',
    'Animation': 'Animación',
    'Comedy': 'Comedia',
    'Crime': 'Crimen',
    'Documentary': 'Documental',
    'Drama': 'Drama',
    'Family': 'Familiar',
    'Fantasy': 'Ciencia ficción y fantasía',
    'History': 'Historia',
    'Horror': 'Terror',
    'Kids': 'Kids',
    'Music': 'Música',
    'Mystery': 'Suspenso y misterio',
    'News': 'Noticias',
    'Reality': 'Reality',
    'Romance': 'Romance',
    'Sci-Fi & Fantasy': 'Ciencia ficción y fantasía',
    'Science Fiction': 'Ciencia ficción y fantasía',
    'Soap': 'Soap',
    'TV Movie': 'Película TV',
    'Talk': 'Conversación',
    'Thriller': 'Suspenso y misterio',
    'War': 'War',
    'War & Politics': 'Guerra y política',
    'Western': 'Western',
    'Unknown': 'Unknown',
}
# Nota: 'War' queda sin traducir a propósito para que coincida con la familia
# tal como está en los datos ya validados por el EDA (distinta de "Guerra y
# política"). Si prefieres unificarla, avísame y ajusto el mapeo.

# Los títulos sin género registrado (genres nulo) se excluyen de esta vista
# en vez de agruparlos artificialmente en "Sin género": no aportan a la
# pregunta de "categorías con mayor potencial".
# IMPORTANTE: show_id NO es único entre películas y series (324 casos), por lo que
# se une por (type, show_id); de lo contrario las filas se duplican y se mezclan
# géneros/países de títulos distintos.
cat_genero = con_votos.loc[con_votos['genres'].notna(), ['type', 'show_id', 'genres']].copy()
cat_genero = cat_genero.assign(genero=cat_genero['genres'].str.split(',')).explode('genero')
cat_genero['genero'] = cat_genero['genero'].str.strip()
cat_genero['familia_genero'] = cat_genero['genero'].map(MAPA_FAMILIA_GENERO).fillna('Unknown')

sin_mapeo = set(cat_genero.loc[cat_genero['familia_genero'] == 'Unknown', 'genero']) - {'Unknown'}
if sin_mapeo:
    print(f"[aviso] géneros sin mapeo explícito, revisar MAPA_FAMILIA_GENERO: {sin_mapeo}")

catalogo_por_genero = con_votos.drop(columns=['genres']).merge(
    cat_genero[['type', 'show_id', 'genero', 'familia_genero']], on=['type', 'show_id'], how='inner'
)
# Identificador único de título (tipo + show_id)
catalogo_por_genero['uid'] = catalogo_por_genero['type'] + '_' + catalogo_por_genero['show_id'].astype(str)

# 4.4 Rendimiento por familia de género (se excluyen 'Unknown' y 'Noticias':
#     no son categorías de contenido que Marketing pueda usar para planificar
#     una campaña, y solo agregan ruido a la vista de "categorías con mayor
#     potencial")
FAMILIAS_NO_PROMOCIONABLES = {'Unknown', 'Noticias'}
rendimiento_familias = (
    catalogo_por_genero[~catalogo_por_genero['familia_genero'].isin(FAMILIAS_NO_PROMOCIONABLES)]
    .drop_duplicates(['uid', 'familia_genero'])  # un título cuenta una vez por familia
    .groupby('familia_genero')
    .agg(
        contenidos=('uid', 'nunique'),
        popularidad_mediana=('popularity', 'median'),
        calificacion_promedio=('vote_average', 'mean'),
        votos_medianos=('vote_count', 'median'),
    )
    .reset_index()
    .sort_values('contenidos', ascending=False)
)

# 4.5 Explosión de país (país de origen). Igual que con género, los títulos
# sin país registrado se excluyen en vez de rellenarse con "Desconocido".
cat_pais = con_votos.loc[con_votos['country'].notna(), ['type', 'show_id', 'country']].copy()
cat_pais = cat_pais.assign(pais=cat_pais['country'].str.split(',')).explode('pais')
cat_pais['pais'] = cat_pais['pais'].str.strip()
catalogo_por_pais = con_votos.drop(columns=['country']).merge(
    cat_pais[['type', 'show_id', 'pais']], on=['type', 'show_id'], how='inner'
)

# 4.6 Evolución anual de incorporación
evolucion_incorporacion = (
    base.groupby(['year_added', 'tipo_es']).size()
    .unstack(fill_value=0)
    .reset_index()
    .rename(columns={'Película': 'Películas', 'Serie': 'Series'})
)

# 4.7 Oportunidades comerciales (solo títulos con budget y revenue reales)
oportunidades_comerciales = (
    con_votos[tiene_datos_financieros]
    [['show_id', 'title', 'tipo_es', 'genres', 'popularity', 'vote_average',
      'date_added', 'temporada', 'budget', 'revenue', 'roi', 'oportunidad_comercial']]
    .sort_values('roi', ascending=False)
    .copy()
)

# -----------------------------------------------------------------------------
# 5. Exportar CSV para el dashboard
# -----------------------------------------------------------------------------
catalogo_marketing_score.to_csv(EXPORT_DIR / 'catalogo_marketing_score.csv', index=False)
top_candidatos_campania.to_csv(EXPORT_DIR / 'top_candidatos_campania.csv', index=False)
catalogo_por_genero.to_csv(EXPORT_DIR / 'catalogo_por_genero.csv', index=False)
rendimiento_familias.to_csv(EXPORT_DIR / 'rendimiento_familias.csv', index=False)
catalogo_por_pais.to_csv(EXPORT_DIR / 'catalogo_por_pais.csv', index=False)
evolucion_incorporacion.to_csv(EXPORT_DIR / 'evolucion_incorporacion.csv', index=False)
oportunidades_comerciales.to_csv(EXPORT_DIR / 'oportunidades_comerciales.csv', index=False)

print("ETL completo. Filas por archivo:")
for nombre, df in [
    ('catalogo_marketing_score', catalogo_marketing_score),
    ('top_candidatos_campania', top_candidatos_campania),
    ('catalogo_por_genero', catalogo_por_genero),
    ('rendimiento_familias', rendimiento_familias),
    ('catalogo_por_pais', catalogo_por_pais),
    ('evolucion_incorporacion', evolucion_incorporacion),
    ('oportunidades_comerciales', oportunidades_comerciales),
]:
    print(f"  {nombre}: {len(df):,}")
