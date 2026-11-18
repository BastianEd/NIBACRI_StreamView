import pandas as pd
from pathlib import Path
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


"""
StreamView Analytics - Dashboard estratégico (Gerencia de Marketing).
Ejecución:  python dashboard/app.py   ->  http://127.0.0.1:8050
Lee los CSV de ../data/processed/ (generados por el EDA y por src/etl_marketing.py).
"""
# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS Y PALETA
# -----------------------------------------------------------------------------
ROJO = '#E50914'
AZUL = '#1769E0'
VERDE = '#16805C'
AMBAR = '#D48500'
MORADO = '#7C3AED'
GRIS = '#6B7280'
OSCURO = '#111827'
GRIS_CLARO = '#F8F9FA'

# Tema único para todos los gráficos (tipografía, rejilla suave, paleta corporativa)
_tpl = pio.templates['plotly_white'].layout
_tpl.font.family = "Lato, 'Segoe UI', Arial, sans-serif"
_tpl.font.size = 15
_tpl.font.color = '#1F2937'
_tpl.title.font.size = 19
_tpl.title.font.color = OSCURO
_tpl.title.x = 0.01
_tpl.xaxis.gridcolor = '#EEF1F5'
_tpl.yaxis.gridcolor = '#EEF1F5'
_tpl.xaxis.linecolor = '#D1D5DB'
_tpl.yaxis.linecolor = '#D1D5DB'
_tpl.hoverlabel.bgcolor = OSCURO
_tpl.hoverlabel.font.color = 'white'
_tpl.hoverlabel.font.size = 14
_tpl.colorway = [AZUL, ROJO, VERDE, AMBAR, MORADO, '#0891B2', '#DB2777', '#65A30D', GRIS, '#EA580C']

BASE = Path(__file__).resolve().parent.parent / 'data' / 'processed'
_REQUERIDOS = ['top_candidatos_campania.csv', 'catalogo_marketing_score.csv', 'evolucion_incorporacion.csv',
               'catalogo_por_pais.csv', 'catalogo_por_genero.csv', 'oportunidades_comerciales.csv']
_faltan = [f for f in _REQUERIDOS if not (BASE / f).exists()]
if _faltan:
    raise FileNotFoundError(f"Faltan archivos en {BASE}: {_faltan}. Ejecute primero: python src/etl_marketing.py")

top_campanias = pd.read_csv(BASE / 'top_candidatos_campania.csv', low_memory=False)
cat_score = pd.read_csv(BASE / 'catalogo_marketing_score.csv', low_memory=False)
evolucion = pd.read_csv(BASE / 'evolucion_incorporacion.csv', low_memory=False)
cat_pais = pd.read_csv(BASE / 'catalogo_por_pais.csv', low_memory=False)
oportunidades = pd.read_csv(BASE / 'oportunidades_comerciales.csv', low_memory=False)

# UMBRAL_BUDGET_MINIMO_TAB: TMDB trae presupuestos placeholder (\$1, \$5, \$84...)
# que no son datos financieros reales y, con un denominador tan chico, inflan el ROI
# a millones de veces (ej. \$5 de presupuesto -> ROI de 2.456.759x). Se restringe SOLO
# dentro de la ventana 'Oportunidades Comerciales' (KPIs de la pestaña, tabla Top ROI
# y gráfico de dispersión); el KPI 'Joyas Ocultas' del encabezado general sigue usando
# 'oportunidades' sin este filtro adicional.
UMBRAL_BUDGET_MINIMO_TAB = 100_000
oportunidades_tab = oportunidades[oportunidades['budget'] >= UMBRAL_BUDGET_MINIMO_TAB].copy()
# Nota: ya no hace falta el merge manual de tipo_es a cat_pais — el ETL
# corregido conserva esa columna directamente en catalogo_por_pais.csv.

CUADRANTE_PROMOCIONAR = 'Promocionar: alto alcance y alta valoración'
JOYA_OCULTA = 'Joya oculta: alta rentabilidad, poca exposición — potenciar'

ORDEN_OPORTUNIDAD = [
    'Ya explotado: alta rentabilidad y alta popularidad',
    JOYA_OCULTA,
    'Popular pero poco rentable: revisar costo de campaña',
    'Bajo retorno y baja popularidad',
]
COLOR_OPORTUNIDAD = {
    'Ya explotado: alta rentabilidad y alta popularidad': VERDE,
    JOYA_OCULTA: MORADO,
    'Popular pero poco rentable: revisar costo de campaña': AMBAR,
    'Bajo retorno y baja popularidad': GRIS,
}


def calcular_rendimiento_familias(df_genero):
    """Recalcula la agregación por familia de género para un subconjunto.
    Se define aquí (en vez de leer un CSV estático) porque el resultado cambia
    según el filtro de Tipo seleccionado. Excluye 'Unknown' y 'Noticias':
    no son categorías que Marketing pueda usar para planificar una campaña."""
    df = df_genero[~df_genero['familia_genero'].isin(['Unknown', 'Noticias'])]
    df = df.drop_duplicates(['uid', 'familia_genero'])  # un título cuenta una vez por familia
    return (df.groupby('familia_genero')
        .agg(contenidos=('uid', 'nunique'),
             popularidad_mediana=('popularity', 'median'),
             calificacion_promedio=('vote_average', 'mean'),
             votos_medianos=('vote_count', 'median'))
        .reset_index()
    )


cat_genero = pd.read_csv(BASE / 'catalogo_por_genero.csv', low_memory=False)

# --- Identificador único de título (show_id se repite entre películas y series) ---
cat_score['uid'] = cat_score['type'] + '_' + cat_score['show_id'].astype(str)
cat_pais['uid'] = cat_pais['type'] + '_' + cat_pais['show_id'].astype(str)
oportunidades['uid'] = 'Movie_' + oportunidades['show_id'].astype(str)
oportunidades_tab['uid'] = 'Movie_' + oportunidades_tab['show_id'].astype(str)
cat_score['cuadrante_corto'] = cat_score['cuadrante_campania'].str.split(':').str[0]

FAMILIAS_NO_PROMOCIONABLES = ['Unknown', 'Noticias']
familias_disp = sorted(f for f in cat_genero['familia_genero'].dropna().unique() if f not in FAMILIAS_NO_PROMOCIONABLES)
UIDS_POR_FAMILIA = cat_genero.groupby('familia_genero')['uid'].agg(set).to_dict()
idiomas_disp = cat_score['idioma'].value_counts().head(15).index.tolist()
MIN_VOTOS_CANDIDATO = 100          # mismo criterio del score del informe
UMBRAL_POP = cat_score['popularity'].median()      # umbrales globales de la matriz 2x2
UMBRAL_VAL = cat_score['vote_average'].median()

PAISES_ES = {
    'United States of America': 'Estados Unidos', 'Japan': 'Japón', 'United Kingdom': 'Reino Unido', 'France': 'Francia',
    'South Korea': 'Corea del Sur', 'China': 'China', 'Canada': 'Canadá', 'Germany': 'Alemania', 'Spain': 'España',
    'India': 'India', 'Italy': 'Italia', 'Mexico': 'México', 'Australia': 'Australia', 'Brazil': 'Brasil',
    'Turkey': 'Turquía', 'Argentina': 'Argentina', 'Thailand': 'Tailandia', 'Russia': 'Rusia', 'Sweden': 'Suecia',
    'Denmark': 'Dinamarca', 'Taiwan': 'Taiwán', 'Ireland': 'Irlanda', 'Philippines': 'Filipinas',
    'Netherlands': 'Países Bajos', 'Belgium': 'Bélgica', 'Poland': 'Polonia', 'Norway': 'Noruega',
    'South Africa': 'Sudáfrica', 'Egypt': 'Egipto', 'Switzerland': 'Suiza', 'New Zealand': 'Nueva Zelanda',
}


def aplicar_filtros(df, tipo=None, idioma=None, familia=None):
    """Filtros globales (tipo, idioma, familia de género) sobre cualquier tabla con uid/tipo_es/idioma."""
    if tipo and tipo != 'TODOS':
        df = df[df['tipo_es'] == tipo]
    if idioma and idioma != 'TODOS':
        df = df[df['idioma'] == idioma]
    if familia and familia != 'TODOS':
        df = df[df['uid'].isin(UIDS_POR_FAMILIA.get(familia, set()))]
    return df


def figura_vacia(titulo):
    fig = go.Figure()
    fig.update_layout(template='plotly_white', title=titulo,
                      annotations=[dict(text="Sin datos para la combinación de filtros seleccionada", showarrow=False,
                                        xref='paper', yref='paper', x=0.5, y=0.5, font=dict(size=14, color=GRIS))])
    return fig


# -----------------------------------------------------------------------------
# 2. INICIALIZACIÓN E INTERFAZ
# -----------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME])
ESTILOS = r"""
<style>
body{font-size:16px; background:#F1F4F9}
.fs-7{font-size:1rem!important} .fs-8{font-size:.95rem!important}
small,.small{font-size:.95rem!important}
.Select-value-label,.Select-option{font-size:15px!important}

/* Encabezado */
.app-header{background:linear-gradient(120deg,#0B1220 0%,#17213A 60%,#2B1219 100%);
            border-bottom:3px solid #E50914}
.app-header .brand-icon{width:54px;height:54px;border-radius:14px;background:rgba(229,9,20,.15);
            border:1px solid rgba(229,9,20,.45);display:flex;align-items:center;justify-content:center;
            color:#FF4B55;font-size:1.6rem}

/* Tarjetas */
.card{border-radius:14px!important}
.kpi-card{border-left:5px solid var(--c)!important; transition:transform .18s ease, box-shadow .18s ease}
.kpi-card:hover{transform:translateY(-3px); box-shadow:0 10px 24px rgba(17,24,39,.10)!important}
.kpi-icon{width:58px;height:58px;border-radius:16px;display:flex;align-items:center;
          justify-content:center;font-size:1.55rem;color:var(--c);background:var(--bg)}
.kpi-label{letter-spacing:.04em;text-transform:uppercase;font-size:.78rem!important;color:#6B7280}
.hallazgo-icon{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;
               justify-content:center;font-size:1.15rem;color:var(--c);background:var(--bg)}
.section-title{color:#111827}
.section-title i{color:#E50914}

/* Pestañas */
.nav-tabs{border-bottom:2px solid #E3E8EF; margin-bottom:1rem}
.nav-tabs .nav-link{color:#4B5563;font-weight:700;border:none;border-bottom:3px solid transparent;
                    margin-bottom:-2px;padding:.85rem 1.2rem;background:transparent;border-radius:0}
.nav-tabs .nav-link:hover{color:#E50914;border-bottom-color:#F4A6AB}
.nav-tabs .nav-link.active{color:#E50914;background:transparent;border-bottom:3px solid #E50914}
.nav-tabs .nav-link::before{font-family:'Font Awesome 6 Free';font-weight:900;margin-right:.55rem}
.tab-matriz::before{content:'\f140'}
.tab-hallazgos::before{content:'\f0eb'}
.tab-familias::before{content:'\f080'}
.tab-geo::before{content:'\f57d'}
.tab-comercial::before{content:'\f51e'}

/* Botones y acordeones */
.btn-danger{background:#E50914;border-color:#E50914}
.btn-danger:hover{background:#B8070F;border-color:#B8070F}
.btn-outline-secondary:hover{background:#111827;border-color:#111827}
.accordion-button:not(.collapsed){color:#E50914;background:#FDF2F2;box-shadow:none}
.list-group-item{border-color:#EEF1F5}
</style>
"""
app.index_string = app.index_string.replace('</head>', ESTILOS + '</head>')

header = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Div(html.I(className="fa-solid fa-circle-play"), className="brand-icon"), width="auto"),
            dbc.Col(html.Div([
                html.H2("StreamView Analytics", className="mb-0 text-white fw-bold", style={'whiteSpace': 'nowrap', 'letterSpacing': '.01em'}),
                html.Div("Dashboard Estratégico — Gerencia de Marketing", className="fs-6", style={'whiteSpace': 'nowrap', 'color': '#AAB4C8'}),
            ], className="text-center")),
            dbc.Col(html.Div(className="brand-icon", style={'visibility': 'hidden'}), width="auto"),  # equilibra el ícono para centrar el texto
        ], align="center", className="g-0 w-100 flex-nowrap")
    ], fluid=True),
    color=OSCURO, dark=True, className="mb-4 shadow-sm py-3 app-header"
)

def _tinte(hex_color, alpha=0.12):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def make_kpi_card(title, value, subtitle, icon, color=AZUL):
    return dbc.Card(dbc.CardBody(
        dbc.Row([
            dbc.Col([html.Div(title, className="kpi-label fw-bold mb-1"),
                     html.H3(value, className="card-title fw-bold mb-0", style={'fontSize': '2rem', 'color': OSCURO}),
                     html.Small(subtitle, className="text-muted fs-8")], width=9),
            dbc.Col(html.Div(html.I(className=f"fa-solid {icon}"), className="kpi-icon"),
                    width=3, className="d-flex align-items-center justify-content-end")
        ])
    ), className="shadow-sm border-0 h-100 kpi-card", style={'--c': color, '--bg': _tinte(color)})

# KPIs dinámicos: se recalculan con los filtros globales (callback update_kpis)
kpis = dbc.Row(id='kpi-row', className="mb-4 g-3")

filtros_globales = dbc.Card(dbc.CardBody([
    html.Div([html.I(className="fa-solid fa-sliders me-2", style={'color': ROJO}), "Filtros globales"],
             className="fw-bold mb-2", style={'color': OSCURO}),
    dbc.Row([
        dbc.Col([html.Label("Tipo de contenido", className="fw-semibold fs-7 mb-1"),
                 dcc.Dropdown(id='filter-tipo', options=[{'label': 'Todos', 'value': 'TODOS'}] + [{'label': i, 'value': i} for i in sorted(cat_score['tipo_es'].dropna().unique())], value='TODOS', clearable=False)], width=12, md=5),
        dbc.Col([html.Label("Idioma original", className="fw-semibold fs-7 mb-1"),
                 dcc.Dropdown(id='filter-idioma', options=[{'label': 'Todos', 'value': 'TODOS'}] + [{'label': i, 'value': i} for i in idiomas_disp], value='TODOS', clearable=False)], width=12, md=5),
        dbc.Col(dbc.Button([html.I(className="fa-solid fa-rotate-left me-2"), "Restablecer"], id='btn-reset', color='secondary', outline=True, className="w-100"),
                width=12, md=2, className="d-flex align-items-end"),
    ], className="g-2"),
    dbc.Accordion(dbc.AccordionItem(
        dbc.Row([dbc.Col([html.Label("Familia de género", className="fw-semibold fs-7 mb-1"),
                          dcc.Dropdown(id='filter-familia', options=[{'label': 'Todas', 'value': 'TODOS'}] + [{'label': i, 'value': i} for i in familias_disp], value='TODOS', clearable=False)], width=12, md=6)]),
        title="Más filtros"), start_collapsed=True, flush=True, className="mt-2"),
    html.Small("Los filtros actualizan los indicadores y las pestañas Qué promocionar, Géneros y Geografía. "
               "Oportunidades Comerciales (solo películas) tiene sus propios filtros.",
               className="text-muted d-block mt-2"),
]), className="border-0 shadow-sm rounded-3 mb-3")

tab_matriz = dbc.Card(dbc.CardBody([
    html.H5([html.I(className="fa-solid fa-bullseye me-2"), "¿Qué promocionar? Matriz de contenidos"], className="fw-bold mb-1 section-title"),
    html.P("Filtra y explora los cuadrantes para extraer títulos de pauta publicitaria.", className="text-muted mb-3 fs-7"),
    dbc.Row([
        dbc.Col([html.Label("Filtrar por Cuadrante:", className="fw-semibold fs-7 mb-1"),
                 dcc.Dropdown(id='filter-cuadrante', options=[{'label': 'Todos', 'value': 'TODOS'}] + [{'label': i, 'value': i} for i in sorted(cat_score['cuadrante_campania'].dropna().unique())], value='TODOS', clearable=False)], width=12, md=6),
        dbc.Col([html.Label("Búsqueda por Título:", className="fw-semibold fs-7 mb-1"),
                 dbc.Input(id='search-title', placeholder="Ej: Stranger Things...", type="text")], width=12, md=6),
    ], className="mb-4 g-2 bg-light p-3 rounded-3"),
    dbc.Accordion(dbc.AccordionItem(
        dbc.Row([dbc.Col([html.Label("Muestra de datos (%): solo aligera el gráfico", className="fw-semibold fs-7 mb-1"),
                          dcc.Slider(id='sample-slider', min=10, max=100, step=None,
                                     marks={10: '10%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
                                     value=100)], width=12)]),
        title="Opciones avanzadas"), start_collapsed=True, flush=True, className="mb-2"),
    html.Div(id='matrix-points-count', className="text-primary fs-8 mb-3 fw-bold text-end"),
    dcc.Graph(id='scatter-matriz-campania', style={'height': '550px'}),
    html.Hr(className="my-4"),
    dbc.Row([
        dbc.Col(html.H6("Tabla de Candidatos Destacados para Campañas", className="fw-bold mb-1"), width=12, md=8),
        dbc.Col(dbc.Button([html.I(className="fa-solid fa-file-arrow-down me-2"), "Exportar candidatos (CSV)"], id='btn-export', color='danger', size='sm', className="float-md-end"), width=12, md=4),
    ], className="align-items-center"),
    html.Div(id='tabla-count', className="text-muted fs-8 mb-2"),
    dash_table.DataTable(
        id='tabla-top-campanias',
        columns=[
            {"name": "Título", "id": "title"}, {"name": "Tipo", "id": "tipo_es"},
            {"name": "Géneros", "id": "genres"}, {"name": "Idioma", "id": "idioma"},
            {"name": "Popularidad", "id": "popularity", "type": "numeric", "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Calificación", "id": "vote_average", "type": "numeric", "format": dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Votos", "id": "vote_count", "type": "numeric", "format": dash_table.Format.Format(group=',')},
            {"name": "Score Marketing", "id": "score_campania", "type": "numeric", "format": dash_table.Format.Format(precision=3, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Cuadrante", "id": "cuadrante_corto"},
        ],
        data=[],
        page_size=10, sort_action='native', filter_action='native',
        style_table={'overflowX': 'auto'}, style_header={'backgroundColor': OSCURO, 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '15px', 'maxWidth': '260px', 'overflow': 'hidden', 'textOverflow': 'ellipsis'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': GRIS_CLARO}]
    )
]), className="border-0 shadow-sm rounded-3 mb-4")

tab_familias = dbc.Card(dbc.CardBody([
    html.H5([html.I(className="fa-solid fa-chart-column me-2"), "Rendimiento por Familias de Géneros"], className="fw-bold mb-1 section-title"),
    html.P("Qué categorías tienen mayor potencial: volumen disponible vs. desempeño real de audiencia.", className="text-muted mb-3 fs-7"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='bar-volumen-familias', style={'height': '450px'}), width=12, lg=6),
        dbc.Col(dcc.Graph(id='scatter-rendimiento-familias', style={'height': '450px'}), width=12, lg=6),
    ])
]), className="border-0 shadow-sm rounded-3 mb-4")

tab_geografia = dbc.Card(dbc.CardBody([
    html.H5([html.I(className="fa-solid fa-earth-americas me-2"), "Segmentación Geográfica, Idiomas y Ritmo de Incorporación"], className="fw-bold mb-1 section-title"),
    html.P("Países e idiomas reaccionan a los filtros globales. El ritmo de incorporación solo responde al tipo de contenido.", className="text-muted mb-3 fs-7"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='bar-top-paises', style={'height': '420px'}), width=12, lg=6),
        dbc.Col(dcc.Graph(id='bar-top-idiomas', style={'height': '420px'}), width=12, lg=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='line-evolucion-incorporacion', style={'height': '420px'}), width=12),
    ])
]), className="border-0 shadow-sm rounded-3 mb-4")

# --- NUEVA PESTAÑA: Oportunidades Comerciales ------------------------------
# budget/revenue de TMDB solo existen (y de forma confiable) para películas,
# así que esta pestaña no lleva el filtro de Tipo: no aplicaría a series.
kpi_oportunidad = dbc.Row([
    dbc.Col(make_kpi_card(
        "Películas Analizadas", f"{len(oportunidades_tab):,}",
        "Con presupuesto y recaudación reales", "fa-sack-dollar", VERDE
    ), width=12, md=4),
    dbc.Col(make_kpi_card(
        "ROI Mediano", f"{oportunidades_tab['roi'].median():.1f}x",
        "Retorno sobre presupuesto invertido", "fa-chart-line", AZUL
    ), width=12, md=4),
    dbc.Col(make_kpi_card(
        "Joyas Ocultas", f"{(oportunidades_tab['oportunidad_comercial'] == JOYA_OCULTA).sum():,}",
        "Alta rentabilidad, poca exposición aún", "fa-gem", MORADO
    ), width=12, md=4),
], className="mb-4 g-3")

tab_comercial = dbc.Card(dbc.CardBody([
    html.H5([html.I(className="fa-solid fa-coins me-2"), "Oportunidades Comerciales"], className="fw-bold mb-1 section-title"),
    html.P("Cruce entre rentabilidad (presupuesto vs. recaudación) y popularidad, solo para "
           "películas con datos financieros reales en TMDB. Las 'joyas ocultas' son las mejores "
           "candidatas a impulsar con campaña: ya son rentables pero todavía tienen poco alcance.",
           className="text-muted mb-3 fs-7"),
    kpi_oportunidad,
    dbc.Row([
        dbc.Col([html.Label("Búsqueda por título:", className="fw-semibold fs-7 mb-1"),
                 dbc.Input(id='search-title-comercial', placeholder="Ej: Interstellar...", type="text")],
                width=12, md=6),
        dbc.Col([html.Label("Filtrar por temporada de incorporación:", className="fw-semibold fs-7 mb-1"),
                 dcc.Dropdown(
                     id='filter-temporada',
                     options=[{'label': 'Todas', 'value': 'TODAS'}] +
                             [{'label': t, 'value': t} for t in ['Verano', 'Otoño', 'Invierno', 'Primavera']],
                     value='TODAS', clearable=False)],
                width=12, md=6),
    ], className="mb-3 g-2 bg-light p-3 rounded-3"),
    html.Div(id='comercial-points-count', className="text-primary fs-8 mb-2 fw-bold text-end"),
    dcc.Graph(id='scatter-oportunidad-comercial', style={'height': '520px'}),
    html.Hr(className="my-4"),
    html.H6("Top 10 Películas por Retorno sobre Presupuesto (ROI)", className="fw-bold mb-3"),
    dash_table.DataTable(
        id='tabla-top-roi',
        columns=[
            {"name": "Título", "id": "title"},
            {"name": "Temporada", "id": "temporada"},
            {"name": "Presupuesto (USD)", "id": "budget", "type": "numeric", "format": dash_table.Format.Format(group=',')},
            {"name": "Recaudación (USD)", "id": "revenue", "type": "numeric", "format": dash_table.Format.Format(group=',')},
            {"name": "ROI", "id": "roi", "type": "numeric", "format": dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Popularidad", "id": "popularity", "type": "numeric", "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Oportunidad", "id": "oportunidad_comercial"},
        ],
        data=oportunidades_tab.nlargest(10, 'roi').to_dict('records'),
        page_size=10, sort_action='native',
        style_table={'overflowX': 'auto'}, style_header={'backgroundColor': OSCURO, 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '15px'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': GRIS_CLARO}]
    )
]), className="border-0 shadow-sm rounded-3 mb-4")

# --- PESTAÑA: Hallazgos y recomendaciones (infografía / data storytelling) -------
def hallazgo_card(numero, titulo, detalle, color, icono="fa-circle-info"):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(html.I(className=f"fa-solid {icono}"), className="hallazgo-icon mb-2", style={'--c': color, '--bg': _tinte(color)}),
        html.Div(numero, className="fw-bold", style={'fontSize': '2.4rem', 'color': color, 'lineHeight': '1.1'}),
        html.Div(titulo, className="fw-semibold mb-1"),
        html.Small(detalle, className="text-muted"),
    ]), className="shadow-sm border-0 rounded-3 h-100", style={'borderTop': f'5px solid {color}'}), width=12, md=6, lg=3)


_n_total = len(cat_score)
_n_prom = int((cat_score['cuadrante_campania'] == CUADRANTE_PROMOCIONAR).sum())
_top3 = ', '.join(top_campanias['title'].head(3))
_rf = calcular_rendimiento_familias(cat_genero)
_mejores = _rf[_rf['contenidos'] >= 1000].nlargest(2, 'calificacion_promedio')['familia_genero'].tolist()
_n_joyas = int((oportunidades_tab['oportunidad_comercial'] == JOYA_OCULTA).sum())
_top_joya = oportunidades_tab[oportunidades_tab['oportunidad_comercial'] == JOYA_OCULTA].nlargest(1, 'roi')
_n_series_top = int((top_campanias['tipo_es'] == 'Serie').sum())

tab_hallazgos = dbc.Card(dbc.CardBody([
    html.H5([html.I(className="fa-solid fa-lightbulb me-2"), "Hallazgos clave y recomendaciones"], className="fw-bold mb-1 section-title"),
    html.P("Resumen narrativo del análisis sobre el catálogo completo (sin filtros). Datos de catálogo: no incluyen reproducciones, clics ni conversión.", className="text-muted mb-3 fs-7"),
    dbc.Row([
        hallazgo_card(f"{_n_prom / _n_total:.0%}", "del catálogo es “Promocionar”",
                      f"{_n_prom:,} de {_n_total:,} títulos combinan alto alcance y alta valoración: base de la campaña principal.", VERDE, "fa-bullseye"),
        hallazgo_card(f"{_n_series_top}/{len(top_campanias)}", "candidatos del Top son series",
                      f"Lideran {_top3}. Estrategia sugerida: series destacadas sin excluir películas de alto score.", AZUL, "fa-tv"),
        hallazgo_card(' y '.join(_mejores), "mejor valoración entre familias grandes",
                      "Volumen no es potencial: Drama y Comedia dominan el inventario, pero no la calidad percibida.", AMBAR, "fa-masks-theater"),
        hallazgo_card(f"{_n_joyas}", "películas “joya oculta”",
                      f"Alta rentabilidad y poca exposición (presupuesto ≥ USD 100.000). Mayor ROI: {_top_joya['title'].iloc[0] if len(_top_joya) else '—'}.", MORADO, "fa-gem"),
    ], className="g-3 mb-4"),
    html.H6("Recomendaciones para Marketing", className="fw-bold mb-2"),
    dbc.ListGroup([
        dbc.ListGroupItem("1 · Activar campañas principales para el cuadrante Promocionar y el ranking de candidatos disponibles en el territorio objetivo."),
        dbc.ListGroupItem("2 · Curar colecciones de descubrimiento con contenido bien evaluado y poca visibilidad (cuadrante Descubrir)."),
        dbc.ListGroupItem("3 · Validar mensajes y audiencias de los títulos Awareness antes de asignar alto presupuesto."),
        dbc.ListGroupItem("4 · Segmentar creatividades por familia de género, idioma y país de producción."),
        dbc.ListGroupItem("5 · Potenciar las joyas ocultas y revisar el costo de campaña de películas populares poco rentables."),
        dbc.ListGroupItem("6 · Integrar métricas internas (reproducciones, clics, conversión, costos) para validar el score."),
    ], flush=True, className="mb-3"),
    html.Small("El score de campaña es descriptivo (45% popularidad, 30% valoración, 25% votos) y no predice conversiones.", className="text-muted"),
]), className="border-0 shadow-sm rounded-3 mb-4")

app.layout = html.Div([
    header,
    dbc.Container([
        filtros_globales,
        kpis,
        dbc.Tabs([
            dbc.Tab(tab_matriz, label="Qué promocionar", tab_id="tab-matriz", label_class_name="tab-matriz"),
            dbc.Tab(tab_hallazgos, label="Hallazgos y recomendaciones", tab_id="tab-hallazgos", label_class_name="tab-hallazgos"),
            dbc.Tab(tab_familias, label="Géneros", tab_id="tab-familias", label_class_name="tab-familias"),
            dbc.Tab(tab_geografia, label="Geografía y Tiempos", tab_id="tab-geo", label_class_name="tab-geo"),
            dbc.Tab(tab_comercial, label="Oportunidades", tab_id="tab-comercial", label_class_name="tab-comercial"),
        ], active_tab="tab-matriz"),
        dcc.Download(id='download-candidatos'),
        html.Footer(html.P("StreamView Analytics © 2026 — Gerencia de Marketing.", className="text-center text-muted fs-8 my-4"))
    ], fluid=True)
], style={'backgroundColor': '#F1F4F9', 'minHeight': '100vh'})

# -----------------------------------------------------------------------------
# 3. CALLBACKS
# -----------------------------------------------------------------------------
def _candidatos(tipo, idioma, familia, cuadrante, search):
    """Candidatos a campaña: títulos con >= MIN_VOTOS_CANDIDATO votos ordenados por score."""
    df = aplicar_filtros(cat_score, tipo, idioma, familia)
    df = df[df['vote_count'] >= MIN_VOTOS_CANDIDATO]
    if cuadrante and cuadrante != 'TODOS':
        df = df[df['cuadrante_campania'] == cuadrante]
    if search:
        df = df[df['title'].astype(str).str.contains(search, case=False, na=False, regex=False)]
    return df.sort_values('score_campania', ascending=False)


COLUMNAS_TABLA = ['title', 'tipo_es', 'genres', 'idioma', 'popularity', 'vote_average', 'vote_count', 'score_campania', 'cuadrante_corto']


@app.callback(
    [Output('filter-tipo', 'value'), Output('filter-idioma', 'value'), Output('filter-familia', 'value'),
     Output('filter-cuadrante', 'value'), Output('search-title', 'value')],
    Input('btn-reset', 'n_clicks'), prevent_initial_call=True
)
def reset_filtros(_):
    return 'TODOS', 'TODOS', 'TODOS', 'TODOS', ''


@app.callback(
    Output('kpi-row', 'children'),
    [Input('filter-tipo', 'value'), Input('filter-idioma', 'value'), Input('filter-familia', 'value')]
)
def update_kpis(tipo, idioma, familia):
    df = aplicar_filtros(cat_score, tipo, idioma, familia)
    n = len(df)
    n_prom = int((df['cuadrante_campania'] == CUADRANTE_PROMOCIONAR).sum())
    joyas = int(((oportunidades_tab['oportunidad_comercial'] == JOYA_OCULTA) & oportunidades_tab['uid'].isin(df['uid'])).sum())
    nd = '—'
    tarjetas = [
        make_kpi_card("Catálogo Evaluado", f"{n:,}", "Con votos y valoración válidos", "fa-clapperboard", AZUL),
        make_kpi_card("Candidatos Promoción", f"{n_prom:,}", f"{n_prom / n:.0%} del catálogo filtrado" if n else "Alto alcance y alta valoración", "fa-bullhorn", ROJO),
        make_kpi_card("Score Promedio", f"{df['score_campania'].mean():.2f}" if n else nd, "Escala estandarizada (0-1)", "fa-ranking-star", AMBAR),
        make_kpi_card("Joyas Ocultas", f"{joyas:,}", "Alta rentabilidad, poca exposición", "fa-gem", MORADO),
    ]
    return [dbc.Col(t, width=12, md=6, lg=3) for t in tarjetas]


@app.callback(
    [Output('scatter-matriz-campania', 'figure'), Output('matrix-points-count', 'children')],
    [Input('filter-tipo', 'value'), Input('filter-idioma', 'value'), Input('filter-familia', 'value'),
     Input('filter-cuadrante', 'value'), Input('search-title', 'value'), Input('sample-slider', 'value')]
)
def update_matriz(tipo, idioma, familia, cuadrante, search, muestra_pct):
    df = aplicar_filtros(cat_score, tipo, idioma, familia)
    if cuadrante and cuadrante != 'TODOS':
        df = df[df['cuadrante_campania'] == cuadrante]
    if search:
        df = df[df['title'].astype(str).str.contains(search, case=False, na=False, regex=False)]

    total_filtrado = len(df)
    # La muestra solo aligera el scatter; KPIs y tabla usan el dato completo.
    if muestra_pct and muestra_pct < 100 and total_filtrado > 0:
        df = df.sample(frac=muestra_pct / 100, random_state=42)
    conteo_texto = f"Mostrando {len(df):,} de {total_filtrado:,} títulos ({muestra_pct}% de la muestra)"

    if df.empty:
        return figura_vacia("Matriz de Priorización Estratégica (2x2)"), conteo_texto

    fig = px.scatter(
        df, x='popularity', y='vote_average', color='cuadrante_campania', hover_name='title',
        hover_data={'tipo_es': True, 'idioma': True, 'score_campania': ':.3f', 'vote_count': True, 'cuadrante_campania': False},
        title="Matriz de Priorización Estratégica (2x2)",
        color_discrete_map={
            'Promocionar: alto alcance y alta valoración': VERDE,
            'Awareness: alto alcance; validar mensaje y segmentación': AMBAR,
            'Descubrir: alta valoración; aumentar visibilidad': AZUL,
            'No priorizar: bajo alcance y valoración': GRIS
        },
        log_x=True, opacity=0.7
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color='DarkSlateGrey')), selector=dict(type='scatter'))
    fig.add_vline(x=UMBRAL_POP, line_width=2, line_dash="dash", line_color=OSCURO)
    fig.add_hline(y=UMBRAL_VAL, line_width=2, line_dash="dash", line_color=OSCURO)
    for x, y, txt, col in [(0.98, 0.98, " PROMOCIONAR", VERDE), (0.02, 0.98, " DESCUBRIR", AZUL),
                           (0.98, 0.02, " AWARENESS", AMBAR), (0.02, 0.02, " NO PRIORIZAR", GRIS)]:
        fig.add_annotation(x=x, y=y, xref="paper", yref="paper", text=txt, showarrow=False,
                           font=dict(color=col, size=14), bgcolor="rgba(255,255,255,0.8)")
    fig.update_layout(
        template='plotly_white', showlegend=False, margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title="Popularidad TMDB (Escala Log) ➔ MAYOR ALCANCE",
        yaxis_title="Calificación Promedio (1-10) ➔ MAYOR CALIDAD"
    )
    return fig, conteo_texto


@app.callback(
    [Output('tabla-top-campanias', 'data'), Output('tabla-count', 'children')],
    [Input('filter-tipo', 'value'), Input('filter-idioma', 'value'), Input('filter-familia', 'value'),
     Input('filter-cuadrante', 'value'), Input('search-title', 'value')]
)
def update_tabla_candidatos(tipo, idioma, familia, cuadrante, search):
    df = _candidatos(tipo, idioma, familia, cuadrante, search)
    texto = (f"{len(df):,} candidatos con ≥ {MIN_VOTOS_CANDIDATO} votos, ordenados por score "
             f"(se muestran los 200 primeros; el CSV exporta la lista completa).")
    return df.head(200)[COLUMNAS_TABLA].to_dict('records'), texto


@app.callback(
    Output('download-candidatos', 'data'),
    Input('btn-export', 'n_clicks'),
    [State('filter-tipo', 'value'), State('filter-idioma', 'value'), State('filter-familia', 'value'),
     State('filter-cuadrante', 'value'), State('search-title', 'value')],
    prevent_initial_call=True
)
def exportar_candidatos(_, tipo, idioma, familia, cuadrante, search):
    df = _candidatos(tipo, idioma, familia, cuadrante, search)[COLUMNAS_TABLA]
    # BOM para que Excel lea bien las tildes
    return dcc.send_string('\ufeff' + df.to_csv(index=False), 'candidatos_campania.csv')


@app.callback(
    Output('bar-volumen-familias', 'figure'),
    [Input('filter-tipo', 'value'), Input('filter-idioma', 'value')]
)
def update_bar_familias(tipo, idioma):
    df_base = aplicar_filtros(cat_genero, tipo, idioma)
    rend = calcular_rendimiento_familias(df_base)
    if rend.empty:
        return figura_vacia("Volumen de Contenidos por Familia")
    df = rend.sort_values(by='contenidos', ascending=True)
    fig = px.bar(df, y='familia_genero', x='contenidos', orientation='h',
                 title="Volumen de Contenidos por Familia", color_discrete_sequence=[AZUL])
    fig.update_traces(texttemplate='<b>%{x:,}</b>', textposition='outside', cliponaxis=False,
                      marker=dict(opacity=0.85, line=dict(width=0.5, color=OSCURO)))
    fig.update_layout(template='plotly_white', margin=dict(l=40, r=60, t=50, b=40),
                      xaxis_title="Cantidad de Títulos Disponibles",
                      xaxis_range=[0, df['contenidos'].max() * 1.15], yaxis_title="")
    return fig


@app.callback(
    Output('scatter-rendimiento-familias', 'figure'),
    [Input('filter-tipo', 'value'), Input('filter-idioma', 'value')]
)
def update_scatter_familias(tipo, idioma):
    df_base = aplicar_filtros(cat_genero, tipo, idioma)
    rend = calcular_rendimiento_familias(df_base)
    if rend.empty:
        return figura_vacia("Desempeño Promedio de Géneros")
    fig = px.scatter(rend, x='popularidad_mediana', y='calificacion_promedio', size='contenidos',
                     color='familia_genero', text='familia_genero',
                     title="Desempeño Promedio de Géneros (Tamaño = Volumen)")
    fig.update_traces(textposition='top center', textfont=dict(size=11, color=OSCURO),
                      marker=dict(line=dict(width=1, color='white')))
    fig.add_vline(x=rend['popularidad_mediana'].median(), line_width=1.5, line_dash="dot", line_color=GRIS)
    fig.add_hline(y=rend['calificacion_promedio'].median(), line_width=1.5, line_dash="dot", line_color=GRIS)
    fig.update_layout(template='plotly_white', showlegend=False, margin=dict(l=40, r=40, t=50, b=40),
                      xaxis_title="Popularidad Mediana ➔ MAYOR ALCANCE",
                      yaxis_title="Calificación Promedio ➔ MAYOR CALIDAD")
    return fig


@app.callback(
    Output('bar-top-paises', 'figure'),
    [Input('filter-tipo', 'value'), Input('filter-idioma', 'value'), Input('filter-familia', 'value')]
)
def update_top_paises(tipo, idioma, familia):
    df = aplicar_filtros(cat_pais, tipo, idioma, familia)
    if df.empty:
        return figura_vacia("Top 10 Países de Producción")
    top_p = df['pais'].value_counts().head(10).reset_index()
    top_p.columns = ['pais', 'total']
    top_p['pais'] = top_p['pais'].map(PAISES_ES).fillna(top_p['pais'])
    top_p = top_p.sort_values(by='total', ascending=True)
    fig = px.bar(top_p, y='pais', x='total', orientation='h',
                 title="Top 10 Países de Producción", color_discrete_sequence=[VERDE])
    fig.update_traces(texttemplate='<b>%{x:,}</b>', textposition='outside', cliponaxis=False,
                      marker=dict(opacity=0.85, line=dict(width=0.5, color=OSCURO)))
    fig.update_layout(template='plotly_white', margin=dict(l=40, r=60, t=50, b=40),
                      xaxis_title="Relaciones contenido–país", xaxis_range=[0, top_p['total'].max() * 1.15], yaxis_title="")
    return fig


@app.callback(
    Output('bar-top-idiomas', 'figure'),
    [Input('filter-tipo', 'value'), Input('filter-familia', 'value')]
)
def update_top_idiomas(tipo, familia):
    df = aplicar_filtros(cat_score, tipo, None, familia)
    if df.empty:
        return figura_vacia("Top 10 Idiomas Originales")
    top_i = df['idioma'].value_counts().head(10).reset_index()
    top_i.columns = ['idioma', 'total']
    top_i = top_i.sort_values(by='total', ascending=True)
    fig = px.bar(top_i, y='idioma', x='total', orientation='h',
                 title="Top 10 Idiomas Originales", color_discrete_sequence=[ROJO])
    fig.update_traces(texttemplate='<b>%{x:,}</b>', textposition='outside', cliponaxis=False,
                      marker=dict(opacity=0.85, line=dict(width=0.5, color=OSCURO)))
    fig.update_layout(template='plotly_white', margin=dict(l=40, r=60, t=50, b=40),
                      xaxis_title="Títulos", xaxis_range=[0, top_i['total'].max() * 1.15], yaxis_title="")
    return fig


@app.callback(
    Output('line-evolucion-incorporacion', 'figure'),
    Input('filter-tipo', 'value')
)
def update_evolucion(tipo):
    fig = go.Figure()
    mostrar_peliculas = (not tipo or tipo == 'TODOS' or tipo == 'Película') and 'Películas' in evolucion.columns
    mostrar_series = (not tipo or tipo == 'TODOS' or tipo == 'Serie') and 'Series' in evolucion.columns
    if mostrar_peliculas:
        fig.add_trace(go.Scatter(
            x=evolucion['year_added'], y=evolucion['Películas'], mode='lines+markers+text', name='Películas',
            line=dict(color=ROJO, width=3), fill='tozeroy', fillcolor='rgba(229, 9, 20, 0.1)',
            text=evolucion['Películas'], texttemplate='%{y:,}', textposition='top center', textfont=dict(size=10, color=ROJO)))
    if mostrar_series:
        fig.add_trace(go.Scatter(
            x=evolucion['year_added'], y=evolucion['Series'], mode='lines+markers+text', name='Series',
            line=dict(color=AZUL, width=3, dash='dash'), fill='tozeroy', fillcolor='rgba(23, 105, 224, 0.1)',
            text=evolucion['Series'], texttemplate='%{y:,}', textposition='bottom center', textfont=dict(size=10, color=AZUL)))
    anio_min, anio_max = evolucion['year_added'].min(), evolucion['year_added'].max()
    fig.update_layout(
        title="Ritmo Anual de Incorporación de Catálogo", template='plotly_white',
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", title=""),
        xaxis_title="Año de Ingreso", yaxis_title="Cantidad de Títulos Añadidos",
        xaxis=dict(range=[anio_min - 0.6, anio_max + 0.6], dtick=1), yaxis=dict(range=[0, 1500], dtick=250),
        margin=dict(l=60, r=40, t=60, b=90)
    )
    fig.add_annotation(
        text="Nota: la fuente trae exactamente 1.000 películas y 1.000 series por año (991 series en 2025). "
             "Es un patrón de muestreo, no una tendencia real: no usar para proyectar crecimiento.",
        xref="paper", yref="paper", x=0, y=-0.32, showarrow=False, font=dict(size=10, color=GRIS), align="left")
    return fig


@app.callback(
    [Output('scatter-oportunidad-comercial', 'figure'),
     Output('comercial-points-count', 'children'),
     Output('tabla-top-roi', 'data')],
    [Input('search-title-comercial', 'value'),
     Input('filter-temporada', 'value')]
)
def update_oportunidad_comercial(search, temporada):
    df = oportunidades_tab.copy()

    if search:
        df = df[df['title'].astype(str).str.contains(search, case=False, na=False)]
    if temporada and temporada != 'TODAS':
        df = df[df['temporada'] == temporada]

    total = len(df)
    conteo_texto = f"Mostrando {total:,} de {len(oportunidades_tab):,} películas con datos financieros"

    fig = px.scatter(
        df,
        x='budget', y='revenue',
        color='oportunidad_comercial',
        size='popularity',
        hover_name='title',
        hover_data={'roi': ':.2f', 'vote_average': True, 'temporada': True,
                     'oportunidad_comercial': False, 'budget': ':,.0f', 'revenue': ':,.0f'},
        title="Presupuesto vs. Recaudación (tamaño = popularidad)",
        category_orders={'oportunidad_comercial': ORDEN_OPORTUNIDAD},
        color_discrete_map=COLOR_OPORTUNIDAD,
        log_x=True, log_y=True,
        opacity=0.7,
    )

    # Línea de "punto de equilibrio" (recaudación = presupuesto). Se calcula
    # sobre el universo completo (no el filtrado) para que la línea no salte
    # de escala cada vez que se busca o filtra por temporada.
    lim = max(oportunidades_tab['budget'].max(), oportunidades_tab['revenue'].max())
    fig.add_trace(go.Scatter(
        x=[1, lim], y=[1, lim], mode='lines',
        line=dict(color=OSCURO, width=1.5, dash='dash'),
        name='Punto de equilibrio', hoverinfo='skip', showlegend=True
    ))

    fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')), selector=dict(mode='markers'))

    fig.update_layout(
        template='plotly_white',
        legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center', title=""),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title="Presupuesto (USD, escala log)",
        yaxis_title="Recaudación (USD, escala log)",
    )

    tabla_data = df.nlargest(10, 'roi').to_dict('records') if total else []
    return fig, conteo_texto, tabla_data


if __name__ == '__main__':
    app.run(debug=True, port=8050)