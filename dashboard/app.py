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
Diseño Widescreen Horizontal con Barra Lateral de Navegación (Sin Scroll).
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
_tpl.font.size = 16
_tpl.font.color = '#1F2937'
_tpl.title.font.size = 18
_tpl.title.font.color = OSCURO
_tpl.title.x = 0.01
_tpl.xaxis.gridcolor = '#EEF1F5'
_tpl.yaxis.gridcolor = '#EEF1F5'
_tpl.xaxis.linecolor = '#D1D5DB'
_tpl.yaxis.linecolor = '#D1D5DB'
_tpl.hoverlabel.bgcolor = OSCURO
_tpl.hoverlabel.font.color = 'white smoke'
_tpl.hoverlabel.font.size = 14
_tpl.colorway = [AZUL, ROJO, VERDE, AMBAR, MORADO, '#0891B2', '#DB2777', '#65A30D', GRIS, '#EA580C']

BASE = Path(__file__).resolve().parent.parent / 'data' / 'processed'
_REQUERIDOS = [
    'top_candidatos_campania.csv', 'catalogo_marketing_score.csv',
    'catalogo_por_pais.csv', 'catalogo_por_genero.csv', 'oportunidades_comerciales.csv'
]
_faltan = [f for f in _REQUERIDOS if not (BASE / f).exists()]
if _faltan:
    raise FileNotFoundError(f"Faltan archivos en {BASE}: {_faltan}. Ejecute primero: python src/etl_marketing.py")

top_campanias = pd.read_csv(BASE / 'top_candidatos_campania.csv', low_memory=False)
cat_score = pd.read_csv(BASE / 'catalogo_marketing_score.csv', low_memory=False)
cat_pais = pd.read_csv(BASE / 'catalogo_por_pais.csv', low_memory=False)
oportunidades = pd.read_csv(BASE / 'oportunidades_comerciales.csv', low_memory=False)
cat_genero = pd.read_csv(BASE / 'catalogo_por_genero.csv', low_memory=False)

# UMBRAL_BUDGET_MINIMO_TAB: TMDB trae presupuestos placeholder ($1, $5, $84...)
# que no son datos financieros reales y, con un denominador tan chico, inflan el ROI
# a millones de veces. Se restringe SOLO dentro de la vista 'Oportunidades Comerciales'.
UMBRAL_BUDGET_MINIMO_TAB = 100_000
oportunidades_tab = oportunidades[oportunidades['budget'] >= UMBRAL_BUDGET_MINIMO_TAB].copy()

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
    Excluye 'Unknown' y 'Noticias'."""
    df = df_genero[~df_genero['familia_genero'].isin(['Unknown', 'Noticias'])]
    df = df.drop_duplicates(['uid', 'familia_genero'])
    return (
        df.groupby('familia_genero')
        .agg(
            contenidos=('uid', 'nunique'),
            popularidad_mediana=('popularity', 'median'),
            calificacion_promedio=('vote_average', 'mean'),
            votos_medianos=('vote_count', 'median'),
        )
        .reset_index()
    )


# --- Identificador único de título (show_id se repite entre películas y series) ---
cat_score['uid'] = cat_score['type'] + '_' + cat_score['show_id'].astype(str)
cat_pais['uid'] = cat_pais['type'] + '_' + cat_pais['show_id'].astype(str)
oportunidades['uid'] = 'Movie_' + oportunidades['show_id'].astype(str)
oportunidades_tab['uid'] = 'Movie_' + oportunidades_tab['show_id'].astype(str)
cat_score['cuadrante_corto'] = cat_score['cuadrante_campania'].str.split(':').str[0]

# Enriquecer cat_score con recaudación (revenue) y presupuesto (budget) para el filtro Top recaudación (%)
if 'revenue' not in cat_score.columns:
    _fin = oportunidades[['uid', 'budget', 'revenue']].drop_duplicates('uid')
    cat_score = cat_score.merge(_fin, on='uid', how='left')

FAMILIAS_NO_PROMOCIONABLES = ['Unknown', 'Noticias']
familias_disp = sorted(f for f in cat_genero['familia_genero'].dropna().unique() if f not in FAMILIAS_NO_PROMOCIONABLES)
UIDS_POR_FAMILIA = cat_genero.groupby('familia_genero')['uid'].agg(set).to_dict()
idiomas_disp = cat_score['idioma'].value_counts().head(15).index.tolist()
MIN_VOTOS_CANDIDATO = 100
UMBRAL_POP = cat_score['popularity'].median()
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


def figura_vacia(titulo, mensaje="Sin datos para la combinación de filtros seleccionada"):
    fig = go.Figure()
    fig.update_layout(
        template='plotly_white',
        title=titulo,
        annotations=[dict(
            text=mensaje,
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.5,
            y=0.5,
            font=dict(size=14, color=GRIS)
        )]
    )
    return fig


# -----------------------------------------------------------------------------
# 2. INICIALIZACIÓN E INTERFAZ (BARRA LATERAL + SENTIDO HORIZONTAL SIN SCROLL)
# -----------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME])
server = app.server

ESTILOS = r"""
<style>
* { box-sizing: border-box; }
html, body {
    margin: 0;
    padding: 0;
    height: 100vh;
    width: 100vw;
    overflow: hidden;
    font-size: 16px;
    background: #F5F5F5;
    font-family: Lato, 'Segoe UI', Arial, sans-serif;
}
.fs-7 { font-size: .88rem !important; }
.fs-8 { font-size: .80rem !important; }
small, .small { font-size: .80rem !important; }

/* Alto contraste para opciones, búsqueda y valor seleccionado en dcc.Dropdown (Dash 2/3/4) */
.dash-dropdown,
.dash-dropdown-content,
.dash-dropdown-options,
.dash-dropdown-option,
.dash-dropdown-option label,
.dash-dropdown-option span,
.dash-dropdown-value,
.dash-dropdown-value-item,
.dash-dropdown-search,
.dash-dropdown-search input,
[role="listbox"],
[role="option"],
[role="option"] label,
[role="option"] span,
.Select-control,
.Select-value-label,
.Select-option,
.Select-input > input,
.Select-menu-outer {
    color: #111827 !important;
    opacity: 1 !important;
}
.dash-dropdown-option,
.dash-dropdown-option label,
[role="option"],
[role="option"] label,
.Select-option {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
}
.dash-dropdown-option:hover,
[role="option"]:hover,
.Select-option.is-focused {
    background-color: #E5E7EB !important;
    color: #000000 !important;
}
.dash-dropdown-option[aria-selected="true"],
.dash-dropdown-option[aria-selected="true"] label,
.dash-dropdown-option[aria-selected="true"] span,
[role="option"][aria-selected="true"],
[role="option"][aria-selected="true"] label {
    color: #E50914 !important;
    font-weight: 700 !important;
    background-color: #FEF2F2 !important;
}
.dash-dropdown-search::placeholder,
.dash-dropdown-search input::placeholder {
    color: #4B5563 !important;
    opacity: 1 !important;
}
.Select-menu-outer,
.dash-dropdown-content {
    background-color: #FFFFFF !important;
}

/* Estructura general horizontal (Sidebar Izquierda + Panel Derecho) */
.dashboard-shell {
    display: flex;
    flex-direction: row;
    height: 100vh;
    width: 100vw;
    overflow: hidden;
}

/* Barra lateral izquierda */
.app-sidebar {
    width: 265px;
    min-width: 265px;
    height: 100vh;
    background: linear-gradient(180deg, #0B1220 0%, #151F36 65%, #231118 100%);
    border-right: 3px solid #E50914;
    color: #F9FAFB;
    display: flex;
    flex-direction: column;
    padding: 14px 14px 10px 14px;
    box-shadow: 4px 0 18px rgba(11, 18, 32, 0.18);
    z-index: 20;
    overflow-y: auto;
}
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding-bottom: 12px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}
.brand-icon {
    width: 42px;
    height: 42px;
    min-width: 42px;
    border-radius: 12px;
    background: rgba(229, 9, 20, 0.18);
    border: 1px solid rgba(229, 9, 20, 0.55);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FF4B55;
    font-size: 1.3rem;
}
.sidebar-section-label {
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9CA3AF;
    font-weight: 700;
    margin-bottom: 6px;
    padding-left: 4px;
}

/* Botones de navegación en la barra lateral */
.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 14px;
}
.sidebar-nav-btn {
    display: flex;
    align-items: center;
    gap: 11px;
    width: 100%;
    text-align: left;
    padding: 9px 11px;
    border-radius: 11px;
    border: 1px solid transparent;
    background: rgba(255, 255, 255, 0.04);
    color: #D1D5DB;
    font-weight: 600;
    font-size: 0.88rem;
    line-height: 1.2;
    cursor: pointer;
    transition: all 0.18s ease;
}
.sidebar-nav-btn:hover {
    background: rgba(255, 255, 255, 0.11);
    color: #FFFFFF;
    transform: translateX(2px);
}
.sidebar-nav-btn .nav-icon-box {
    width: 32px;
    height: 32px;
    min-width: 32px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.08);
    color: #9CA3AF;
    font-size: 0.95rem;
    transition: all 0.18s ease;
}
.sidebar-nav-btn.active {
    background: linear-gradient(90deg, rgba(229, 9, 20, 0.25) 0%, rgba(229, 9, 20, 0.10) 100%);
    border-color: rgba(229, 9, 20, 0.55);
    border-left: 4px solid #E50914;
    color: #FFFFFF;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.18);
}
.sidebar-nav-btn.active .nav-icon-box {
    background: #E50914;
    color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(229, 9, 20, 0.45);
}

/* Bloque de filtros globales en barra lateral */
.sidebar-filters {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 12px;
    padding: 11px;
    margin-bottom: auto;
}
/* Clase exclusiva para los 3 títulos de filtro (evita afectar a las opciones del dropdown) */
.sidebar-filters .sidebar-filter-label {
    color: #F3F4F6;
    font-size: 0.80rem;
    font-weight: 600;
    margin-bottom: 4px;
    display: block;
}
.sidebar-filters .Select-control {
    border-radius: 8px !important;
    min-height: 32px !important;
}
.sidebar-footer {
    font-size: 0.72rem;
    color: #9CA3AF;
    text-align: center;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    margin-top: 8px;
}

/* Área principal derecha (Sentido Horizontal) */
.app-main {
    flex: 1;
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 12px 18px;
    overflow: hidden;
    gap: 10px;
}

/* KPIs superiores compactos */
.card { border-radius: 12px !important; }
.kpi-card {
    border-left: 4px solid var(--c) !important;
    transition: transform .15s ease, box-shadow .15s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(17,24,39,.08) !important;
}
.kpi-icon {
    width: 40px;
    height: 40px;
    min-width: 40px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.20rem;
    color: var(--c);
    background: var(--bg);
}
.kpi-label {
    letter-spacing: .04em;
    text-transform: uppercase;
    font-size: .68rem !important;
    color: #6B7280;
}

/* Contenedor de las 5 vistas horizontales */
.views-container {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
}
.view-panel {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.view-panel.d-none {
    display: none !important;
}

/* Cabecera interna de cada vista */
.view-header-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 9px 15px;
    box-shadow: 0 2px 8px rgba(17, 24, 39, 0.05);
    flex-shrink: 0;
}
.section-title { color: #111827; font-size: 1.05rem; }
.section-title i { color: #E50914; }

/* Filas de contenido horizontal que ocupan el alto restante */
.horizontal-content-row {
    flex: 1;
    min-height: 0;
    --bs-gutter-y: 0 !important;
    margin-top: 0 !important;
}
.horizontal-content-row > * { margin-top: 0 !important; }
.panel-box {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 2px 10px rgba(17, 24, 39, 0.05);
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.panel-box-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 6px;
    margin-bottom: 6px;
    border-bottom: 1px solid #EEF1F5;
    flex-shrink: 0;
}
.panel-box-title {
    font-weight: 700;
    font-size: 0.92rem;
    color: #111827;
    margin: 0;
}
.panel-box-body {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Hallazgos y recomendaciones: cuadrículas que reparten el alto disponible por igual */
.cards-grid {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}
.cards-grid.rows-2 { grid-template-rows: repeat(2, minmax(0, 1fr)); }
.cards-grid.rows-3 { grid-template-rows: repeat(3, minmax(0, 1fr)); }

/* Tarjeta de hallazgo */
.hallazgo-card {
    position: relative;
    min-height: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 5px;
    padding: 14px 18px 12px 18px;
    background: linear-gradient(180deg, var(--bg) 0%, #FFFFFF 62%);
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(17, 24, 39, 0.05);
    transition: transform .15s ease, box-shadow .15s ease;
}
.hallazgo-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: var(--c);
}
.hallazgo-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(17, 24, 39, 0.09);
}
.hallazgo-top {
    display: flex;
    align-items: center;
    gap: 9px;
}
.hallazgo-icon {
    width: 38px;
    height: 38px;
    min-width: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    color: #FFFFFF;
    background: var(--c);
}
.hallazgo-etiqueta {
    font-size: clamp(0.78rem, 1.5vh, 0.95rem);
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.hallazgo-valor {
    font-weight: 800;
    line-height: 1.05;
    font-size: clamp(2.5rem, 6.4vh, 3.9rem);
    margin-top: 2px;
}
.hallazgo-valor-largo { font-size: clamp(1.7rem, 3.9vh, 2.5rem); line-height: 1.15; }
.hallazgo-titulo {
    font-weight: 700;
    font-size: clamp(1.1rem, 2.3vh, 1.4rem);
    color: #111827;
    line-height: 1.2;
}
.hallazgo-detalle {
    margin: 0;
    font-size: clamp(0.92rem, 1.95vh, 1.15rem);
    line-height: 1.35;
    color: #6B7280;
}

/* Tarjeta de recomendación */
.rec-item-card {
    min-height: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 5px;
    padding: 10px 14px 10px 18px;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(17, 24, 39, 0.05);
    position: relative;
    transition: transform .15s ease, box-shadow .15s ease;
}
.rec-item-card::before {
    content: '';
    position: absolute;
    top: 0; bottom: 0; left: 0;
    width: 4px;
    background: #E50914;
}
.rec-item-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(17, 24, 39, 0.09);
}
.rec-top {
    display: flex;
    align-items: center;
    gap: 9px;
}
.rec-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    min-width: 32px;
    border-radius: 9px;
    background: #E50914;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 1rem;
    box-shadow: 0 2px 6px rgba(229, 9, 20, 0.30);
}
.rec-titulo {
    font-weight: 700;
    font-size: clamp(1.05rem, 2.2vh, 1.35rem);
    color: #111827;
    line-height: 1.2;
}
.rec-icono {
    margin-left: auto;
    color: #D1D5DB;
    font-size: 1.15rem;
}
.rec-detalle {
    margin: 0;
    font-size: clamp(0.9rem, 1.9vh, 1.1rem);
    line-height: 1.35;
    color: #6B7280;
}
.nota-metodologica {
    flex-shrink: 0;
    margin-top: 10px;
    padding: 8px 12px;
    border-radius: 10px;
    background: #FFF8E6;
    border: 1px solid #F5E1A4;
    color: #6B5B1E;
    font-size: clamp(0.85rem, 1.7vh, 1rem);
    line-height: 1.35;
}

/* Encabezados de la vista de hallazgos */
#view-hallazgos .section-title { font-size: 1.4rem; }
#view-hallazgos .view-header-card small { font-size: 0.98rem !important; }
#view-hallazgos .panel-box-title { font-size: 1.12rem; }
#view-hallazgos .panel-box-header .badge { font-size: 0.82rem; }

/* Pantallas bajas */
@media (max-height: 820px) and (min-width: 1200px) {
    .hallazgo-card { padding: 10px 14px 8px 14px; gap: 3px; }
    .hallazgo-detalle, .rec-detalle { font-size: 0.86rem; line-height: 1.26; }
    .hallazgo-titulo, .rec-titulo { font-size: 1rem; }
    .hallazgo-valor { font-size: 2.3rem; }
    .hallazgo-valor-largo { font-size: 1.55rem; }
    .rec-badge { width: 28px; height: 28px; min-width: 28px; }
    .nota-metodologica { font-size: 0.8rem; margin-top: 8px; padding: 6px 10px; }
    #view-hallazgos .panel-box-title { font-size: 1rem; }
    #view-hallazgos .section-title { font-size: 1.25rem; }
    .hallazgo-icon { width: 32px; height: 32px; min-width: 32px; }
    .rec-item-card { padding: 7px 12px 7px 16px; gap: 3px; }
}
/* Ventanas angostas */
@media (max-width: 1199px) {
    .cards-grid.rows-2, .cards-grid.rows-3 { grid-template-rows: none; grid-auto-rows: minmax(150px, auto); }
}
@media (max-width: 767px) {
    .cards-grid { grid-template-columns: 1fr; }
}

/* Botones */
.btn-danger { background: #E50914; border-color: #E50914; }
.btn-danger:hover { background: #B8070F; border-color: #B8070F; }

/* Adaptabilidad */
@media (max-width: 1199px) {
    html, body, .dashboard-shell, .app-main {
        height: auto !important;
        overflow-y: auto !important;
    }
    .dashboard-shell { flex-direction: column; }
    .app-sidebar { width: 100%; height: auto; border-right: none; border-bottom: 3px solid #E50914; }
    .sidebar-nav { flex-direction: row; flex-wrap: wrap; }
    .sidebar-nav-btn { width: auto; flex: 1 1 180px; }
}
</style>
"""
app.index_string = app.index_string.replace('</head>', ESTILOS + '</head>')


def _tinte(hex_color, alpha=0.12):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def make_kpi_card(title, value, subtitle, icon, color=AZUL):
    return dbc.Card(dbc.CardBody(
        dbc.Row([
            dbc.Col([
                html.Div(title, className="kpi-label fw-bold"),
                html.Div([
                    html.Span(value, className="fw-bold me-2", style={'fontSize': '1.35rem', 'color': OSCURO, 'lineHeight': '1.1'}),
                ]),
                html.Small(subtitle, className="text-muted text-truncate d-block", style={'fontSize': '0.73rem'})
            ], width=9),
            dbc.Col(html.Div(html.I(className=f"fa-solid {icon}"), className="kpi-icon"),
                    width=3, className="d-flex align-items-center justify-content-end")
        ], className="g-0 align-items-center"),
        className="py-2 px-3"
    ), className="shadow-sm border-0 h-100 kpi-card", style={'--c': color, '--bg': _tinte(color)})


# Definición de los 5 botones de la barra lateral
NAV_ITEMS = [
    ('matriz', 'Qué promocionar', 'fa-bullseye'),
    ('hallazgos', 'Hallazgos y recomendaciones', 'fa-lightbulb'),
    ('familias', 'Géneros', 'fa-chart-column'),
    ('geo', 'Geografía', 'fa-earth-americas'),
    ('comercial', 'Oportunidades', 'fa-coins'),
]

sidebar = html.Aside([
    # Marca / Encabezado compacto
    html.Div([
        html.Div(html.I(className="fa-solid fa-circle-play"), className="brand-icon"),
        html.Div([
            html.Div("StreamView Analytics", className="fw-bold text-white", style={'fontSize': '1.02rem', 'lineHeight': '1.15'}),
            html.Div("Gerencia de Marketing", style={'fontSize': '0.74rem', 'color': '#AAB4C8'}),
        ])
    ], className="sidebar-brand"),

    # Menú vertical con los 5 botones solicitados
    html.Div("Secciones del Dashboard", className="sidebar-section-label"),
    html.Nav([
        html.Button([
            html.Span(html.I(className=f"fa-solid {icono}"), className="nav-icon-box"),
            html.Span(etiqueta)
        ], id=f"btn-nav-{clave}", n_clicks=0,
           className="sidebar-nav-btn active" if idx == 0 else "sidebar-nav-btn")
        for idx, (clave, etiqueta, icono) in enumerate(NAV_ITEMS)
    ], className="sidebar-nav"),

    # Filtros globales integrados en la barra lateral para liberar espacio vertical
    html.Div("Filtros Globales", className="sidebar-section-label"),
    html.Div([
        html.Div([
            html.Label("Tipo de contenido", className="sidebar-filter-label"),
            dcc.Dropdown(
                id='filter-tipo',
                options=[{'label': 'Todos', 'value': 'TODOS'}] +
                        [{'label': i, 'value': i} for i in sorted(cat_score['tipo_es'].dropna().unique())],
                value='TODOS', clearable=False, style={'color': OSCURO}
            )
        ], className="mb-2"),
        html.Div([
            html.Label("Idioma original", className="sidebar-filter-label"),
            dcc.Dropdown(
                id='filter-idioma',
                options=[{'label': 'Todos', 'value': 'TODOS'}] +
                        [{'label': i, 'value': i} for i in idiomas_disp],
                value='TODOS', clearable=False, style={'color': OSCURO}
            )
        ], className="mb-2"),
        html.Div([
            html.Label("Familia de género", className="sidebar-filter-label"),
            dcc.Dropdown(
                id='filter-familia',
                options=[{'label': 'Todas', 'value': 'TODOS'}] +
                        [{'label': i, 'value': i} for i in familias_disp],
                value='TODOS', clearable=False, style={'color': OSCURO}
            )
        ], className="mb-2"),
        dbc.Button(
            [html.I(className="fa-solid fa-rotate-left me-2"), "Restablecer filtros"],
            id='btn-reset', color='light', outline=True, size='sm', className="w-100 mt-1 fw-semibold"
        ),
        html.Small(
            "Actualiza KPIs, Qué promocionar, Géneros y Geografía.",
            className="d-block mt-2", style={'color': '#9CA3AF', 'fontSize': '0.70rem', 'lineHeight': '1.25'}
        )
    ], className="sidebar-filters"),

    html.Div("StreamView Analytics © 2026", className="sidebar-footer")
], className="app-sidebar")

# Fila superior de KPIs globales
kpis = dbc.Row(id='kpi-row', className="g-2 flex-shrink-0")

# -----------------------------------------------------------------------------
# VISTAS HORIZONTALES POR CADA BOTÓN DE LA BARRA LATERAL
# -----------------------------------------------------------------------------

# 1. VISTA: QUÉ PROMOCIONAR (Izquierda: Matriz Scatter | Derecha: Tabla Candidatos)
view_matriz = html.Div([
    html.Div(
        dbc.Row([
            dbc.Col([
                html.H5([html.I(className="fa-solid fa-bullseye me-2"), "¿Qué promocionar? Matriz de Priorización y Candidatos"],
                        className="fw-bold mb-0 section-title"),
                html.Small("Explora los cuadrantes a la izquierda y consulta o exporta la nómina priorizada a la derecha.",
                           className="text-muted")
            ], width=12, lg=4),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Label("Cuadrante:", className="fw-semibold fs-8 mb-0"),
                        dcc.Dropdown(
                            id='filter-cuadrante',
                            options=[{'label': 'Todos los cuadrantes', 'value': 'TODOS'}] +
                                    [{'label': i, 'value': i} for i in sorted(cat_score['cuadrante_campania'].dropna().unique())],
                            value='TODOS', clearable=False
                        )
                    ], width=12, md=5),
                    dbc.Col([
                        html.Label("Buscar título:", className="fw-semibold fs-8 mb-0"),
                        dbc.Input(id='search-title', placeholder="Ej: Stranger Things...", type="text", size="sm")
                    ], width=12, md=4),
                    dbc.Col([
                        html.Label("Top recaudación (%):", className="fw-semibold fs-8 mb-0"),
                        dcc.Slider(
                            id='sample-slider', min=10, max=50, step=None,
                            marks={10: '10%', 20: '20%', 30: '30%', 40: '40%', 50: '50%'},
                            value=50
                        )
                    ], width=12, md=3),
                ], className="g-2 align-items-center")
            ], width=12, lg=8)
        ], className="g-2 align-items-center"),
        className="view-header-card"
    ),

    dbc.Row([
        # Columna Izquierda: Gráfico de Matriz 2x2
        dbc.Col(html.Div([
            html.Div([
                html.Span([html.I(className="fa-solid fa-border-all me-2 text-danger"), "Matriz de Priorización Estratégica (2x2)"],
                          className="panel-box-title"),
                html.Span(id='matrix-points-count', className="badge bg-light text-primary border fw-semibold")
            ], className="panel-box-header"),
            html.Div(
                dcc.Graph(id='scatter-matriz-campania', config={'displaylogo': False, 'responsive': True},
                          style={'height': '100%', 'width': '100%', 'minHeight': '340px'}),
                className="panel-box-body"
            )
        ], className="panel-box"), width=12, lg=6, className="h-100"),

        # Columna Derecha: Tabla de Candidatos + Botón Exportar
        dbc.Col(html.Div([
            html.Div([
                html.Div([
                    html.Span([html.I(className="fa-solid fa-ranking-star me-2 text-danger"), "Candidatos Destacados para Campañas"],
                              className="panel-box-title d-block"),
                    html.Small(id='tabla-count', className="text-muted", style={'fontSize': '0.73rem'})
                ]),
                dbc.Button([html.I(className="fa-solid fa-file-arrow-down me-1"), "Exportar CSV"],
                           id='btn-export', color='danger', size='sm')
            ], className="panel-box-header"),
            html.Div(
                dash_table.DataTable(
                    id='tabla-top-campanias',
                    columns=[
                        {"name": "Título", "id": "title"},
                        {"name": "Tipo", "id": "tipo_es"},
                        {"name": "Géneros", "id": "genres"},
                        {"name": "Idioma", "id": "idioma"},
                        {"name": "Pop.", "id": "popularity", "type": "numeric",
                         "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
                        {"name": "Calif.", "id": "vote_average", "type": "numeric",
                         "format": dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed)},
                        {"name": "Votos", "id": "vote_count", "type": "numeric",
                         "format": dash_table.Format.Format(group=',')},
                        {"name": "Score", "id": "score_campania", "type": "numeric",
                         "format": dash_table.Format.Format(precision=3, scheme=dash_table.Format.Scheme.fixed)},
                        {"name": "Cuadrante", "id": "cuadrante_corto"},
                    ],
                    data=[],
                    page_size=8,
                    sort_action='native',
                    filter_action='native',
                    style_table={'overflowX': 'auto', 'overflowY': 'auto'},
                    style_header={'backgroundColor': OSCURO, 'color': 'white', 'fontWeight': 'bold', 'fontSize': '12px', 'padding': '7px'},
                    style_cell={'textAlign': 'left', 'padding': '6px 8px', 'fontSize': '12.5px',
                                'maxWidth': '170px', 'overflow': 'hidden', 'textOverflow': 'ellipsis'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': GRIS_CLARO}]
                ),
                className="panel-box-body justify-content-between"
            )
        ], className="panel-box"), width=12, lg=6, className="h-100"),
    ], className="g-3 horizontal-content-row")
], id="view-matriz", className="view-panel active")


# 2. VISTA: HALLAZGOS Y RECOMENDACIONES (Izquierda: Cuadrícula 2x2 Hallazgos | Derecha: Recomendaciones 2x3)
def hallazgo_card(etiqueta, numero, titulo, detalle, color, icono="fa-circle-info"):
    """Tarjeta de hallazgo: [icono + etiqueta] / cifra destacada / título / detalle."""
    valor_largo = len(str(numero)) > 8
    return html.Div([
        html.Div([
            html.Div(html.I(className=f"fa-solid {icono}"), className="hallazgo-icon"),
            html.Span(etiqueta, className="hallazgo-etiqueta", style={'color': color}),
        ], className="hallazgo-top"),
        html.Div(numero, className="hallazgo-valor hallazgo-valor-largo" if valor_largo else "hallazgo-valor",
                 style={'color': color}),
        html.Div(titulo, className="hallazgo-titulo"),
        html.P(detalle, className="hallazgo-detalle"),
    ], className="hallazgo-card", style={'--c': color, '--bg': _tinte(color, 0.10)})


def recomendacion_card(num, titulo, detalle, icono):
    return html.Div([
        html.Div([
            html.Span(num, className="rec-badge"),
            html.Span(titulo, className="rec-titulo"),
            html.I(className=f"fa-solid {icono} rec-icono"),
        ], className="rec-top"),
        html.P(detalle, className="rec-detalle"),
    ], className="rec-item-card")


_n_total = len(cat_score)
_n_prom = int((cat_score['cuadrante_campania'] == CUADRANTE_PROMOCIONAR).sum())
_top3 = ', '.join(top_campanias['title'].head(3))
_rf = calcular_rendimiento_familias(cat_genero)
_mejores = _rf[_rf['contenidos'] >= 1000].nlargest(2, 'calificacion_promedio')['familia_genero'].tolist()
_n_joyas = int((oportunidades_tab['oportunidad_comercial'] == JOYA_OCULTA).sum())
_top_joya = oportunidades_tab[oportunidades_tab['oportunidad_comercial'] == JOYA_OCULTA].nlargest(1, 'roi')
_n_series_top = int((top_campanias['tipo_es'] == 'Serie').sum())

RECOMENDACIONES_DATA = [
    ("1", "Campañas Principales", "Activar pauta prioritaria para el cuadrante Promocionar y el ranking top del territorio objetivo.", "fa-bullhorn"),
    ("2", "Colecciones Descubrir", "Curar carruseles de descubrimiento con títulos de alta valoración y baja visibilidad actual.", "fa-binoculars"),
    ("3", "Validación Awareness", "Testear mensajes y segmentación de títulos Awareness antes de comprometer alto presupuesto.", "fa-flask"),
    ("4", "Segmentación Creativa", "Adaptar piezas publicitarias según familia de género, idioma original y país de producción.", "fa-sliders"),
    ("5", "Impulso a Joyas Ocultas", "Potenciar películas de alto ROI con poca exposición y auditar costos en populares poco rentables.", "fa-gem"),
    ("6", "Integración de Conversión", "Incorporar métricas internas (reproducciones, CTR y conversión) para calibrar el score.", "fa-link"),
]

view_hallazgos = html.Div([
    html.Div([
        html.H5([html.I(className="fa-solid fa-lightbulb me-2"), "Hallazgos Clave y Recomendaciones Estratégicas"],
                className="fw-bold mb-0 section-title"),
        html.Small("Síntesis ejecutiva sobre el catálogo completo (sin filtros). Izquierda: diagnóstico cuantitativo · Derecha: plan de acción.",
                   className="text-muted")
    ], className="view-header-card"),

    dbc.Row([
        # Columna Izquierda: 4 Hallazgos Clave en cuadrícula 2x2
        dbc.Col(html.Div([
            html.Div([
                html.Span([html.I(className="fa-solid fa-magnifying-glass-chart me-2 text-danger"), "Diagnóstico: 4 Hallazgos Clave del Catálogo"],
                          className="panel-box-title"),
                html.Span("Base: Catálogo Completo", className="badge bg-light text-secondary border")
            ], className="panel-box-header"),
            html.Div(
                html.Div([
                    hallazgo_card("Cobertura", f"{_n_prom / _n_total:.0%}", "del catálogo es “Promocionar”",
                                  f"{_n_prom:,} de {_n_total:,} títulos combinan alto alcance y alta valoración: base de la campaña principal.", VERDE, "fa-bullseye"),
                    hallazgo_card("Formato", f"{_n_series_top}/{len(top_campanias)}", "candidatos del Top son series",
                                  f"Lideran {_top3}. Estrategia sugerida: series ancla sin excluir películas de alto score.", AZUL, "fa-tv"),
                    hallazgo_card("Calidad", ' y '.join(_mejores), "mejor valoración en familias grandes",
                                  "Volumen no equivale a potencial: Drama y Comedia dominan en cantidad, pero no en calidad percibida.", AMBAR, "fa-masks-theater"),
                    hallazgo_card("Rentabilidad", f"{_n_joyas}", "películas “joya oculta”",
                                  f"Alta rentabilidad y poca exposición (presupuesto ≥ USD 100.000). Mayor ROI: {_top_joya['title'].iloc[0] if len(_top_joya) else '—'}.", MORADO, "fa-gem"),
                ], className="cards-grid rows-2"),
                className="panel-box-body"
            )
        ], className="panel-box"), width=12, lg=6, className="h-100"),

        # Columna Derecha: 6 Recomendaciones accionables en cuadrícula 2x3 + Nota de Score
        dbc.Col(html.Div([
            html.Div([
                html.Span([html.I(className="fa-solid fa-compass-drafting me-2 text-danger"), "Plan de Acción: Recomendaciones para Marketing"],
                          className="panel-box-title"),
                html.Span("6 Líneas de Acción", className="badge bg-danger")
            ], className="panel-box-header"),
            html.Div([
                html.Div([recomendacion_card(num, tit, desc, ico) for num, tit, desc, ico in RECOMENDACIONES_DATA],
                         className="cards-grid rows-3"),
                html.Div([
                    html.I(className="fa-solid fa-scale-balanced me-2", style={'color': AMBAR}),
                    html.Strong("Nota metodológica: "),
                    "El score de campaña es descriptivo (45% popularidad, 30% valoración, 25% votos) y orienta la priorización inicial."
                ], className="nota-metodologica")
            ], className="panel-box-body")
        ], className="panel-box"), width=12, lg=6, className="h-100"),
    ], className="g-3 horizontal-content-row")
], id="view-hallazgos", className="view-panel d-none")


# 3. VISTA: GÉNEROS (Izquierda: Volumen por Familia | Derecha: Desempeño Alcance vs Calidad)
view_familias = html.Div([
    html.Div([
        html.H5([html.I(className="fa-solid fa-chart-column me-2"), "Rendimiento Estratégico por Familias de Géneros"],
                className="fw-bold mb-0 section-title"),
        html.Small("Comparativa horizontal: inventario disponible a la izquierda vs. desempeño real de audiencia (alcance y valoración) a la derecha.",
                   className="text-muted")
    ], className="view-header-card"),

    dbc.Row([
        dbc.Col(html.Div([
            html.Div([
                html.Span([html.I(className="fa-solid fa-layer-group me-2 text-primary"), "1. Disponibilidad de Inventario por Familia"],
                          className="panel-box-title"),
                html.Span("Volumen de Títulos", className="badge bg-light text-secondary border")
            ], className="panel-box-header"),
            html.Div(
                dcc.Graph(id='bar-volumen-familias', config={'displaylogo': False, 'responsive': True},
                          style={'height': '100%', 'width': '100%', 'minHeight': '360px'}),
                className="panel-box-body"
            )
        ], className="panel-box"), width=12, lg=6, className="h-100"),

        dbc.Col(html.Div([
            html.Div([
                html.Span([html.I(className="fa-solid fa-chart-line me-2 text-danger"), "2. Alcance vs. Calidad Percibida por Familia"],
                          className="panel-box-title"),
                html.Span("Tamaño = Volumen", className="badge bg-light text-secondary border")
            ], className="panel-box-header"),
            html.Div(
                dcc.Graph(id='scatter-rendimiento-familias', config={'displaylogo': False, 'responsive': True},
                          style={'height': '100%', 'width': '100%', 'minHeight': '360px'}),
                className="panel-box-body"
            )
        ], className="panel-box"), width=12, lg=6, className="h-100"),
    ], className="g-3 horizontal-content-row")
], id="view-familias", className="view-panel d-none")


# 4. VISTA: GEOGRAFÍA (Panel unificado con interruptor Países/Idiomas)
view_geografia = html.Div([
    html.Div([
        html.H5([html.I(className="fa-solid fa-earth-americas me-2"), "Geografía"],
                className="fw-bold mb-0 section-title"),
        html.Small("Segmentación de la producción por países de origen e idiomas originales.",
                   className="text-muted")
    ], className="view-header-card"),

    dbc.Row([
        dbc.Col(html.Div([
            html.Div([
                html.Span([html.I(className="fa-solid fa-map-location-dot me-2 text-success"), "Top Países / Top Idiomas"],
                          className="panel-box-title"),
                dbc.RadioItems(
                    id='selector-geo-idioma',
                    options=[
                        {'label': 'Países', 'value': 'PAIS'},
                        {'label': 'Idiomas', 'value': 'IDIOMA'},
                    ],
                    value='PAIS',
                    inline=True,
                    className="fw-bold"
                )
            ], className="panel-box-header"),
            html.Div([
                html.Small(id='sub-geo-idioma', className="text-muted d-block mb-2", style={'fontSize': '0.9rem'}),
                dcc.Graph(id='bar-geo-idioma', config={'displaylogo': False, 'responsive': True},
                          style={'height': '100%', 'width': '100%', 'minHeight': '360px'})
            ], className="panel-box-body")
        ], className="panel-box"), width=12, lg=8, className="mx-auto h-100"),
    ], className="g-3 horizontal-content-row")
], id="view-geo", className="view-panel d-none")


# 5. VISTA: OPORTUNIDADES COMERCIALES
view_comercial = html.Div([
    html.Div(
        dbc.Row([
            dbc.Col([
                html.H5([html.I(className="fa-solid fa-coins me-2"), "Oportunidades Comerciales (Películas con Datos Financieros)"],
                        className="fw-bold mb-0 section-title"),
                html.Small(f"Películas con presupuesto ≥ USD 100k ({len(oportunidades_tab):,} analizadas · ROI mediano {oportunidades_tab['roi'].median():.1f}x · {(oportunidades_tab['oportunidad_comercial'] == JOYA_OCULTA).sum():,} Joyas Ocultas).",
                           className="text-muted")
            ], width=12, lg=6),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Label("Buscar película:", className="fw-semibold fs-8 mb-0"),
                        dbc.Input(id='search-title-comercial', placeholder="Ej: Interstellar...", type="text", size="sm")
                    ], width=12, md=6),
                    dbc.Col([
                        html.Label("Temporada de incorporación:", className="fw-semibold fs-8 mb-0"),
                        dcc.Dropdown(
                            id='filter-temporada',
                            options=[{'label': 'Todas las temporadas', 'value': 'TODAS'}] +
                                    [{'label': t, 'value': t} for t in ['Verano', 'Otoño', 'Invierno', 'Primavera']],
                            value='TODAS', clearable=False
                        )
                    ], width=12, md=6),
                ], className="g-2 align-items-center")
            ], width=12, lg=6),
        ], className="g-2 align-items-center"),
        className="view-header-card"
    ),

    dbc.Row([
        # Columna Izquierda: Dispersión Presupuesto vs Recaudación
        dbc.Col(html.Div([
            html.Div([
                html.Span([html.I(className="fa-solid fa-sack-dollar me-2 text-success"), "Presupuesto vs. Recaudación (Tamaño = Popularidad)"],
                          className="panel-box-title"),
                html.Span(id='comercial-points-count', className="badge bg-light text-primary border fw-semibold")
            ], className="panel-box-header"),
            html.Div(
                dcc.Graph(id='scatter-oportunidad-comercial', config={'displaylogo': False, 'responsive': True},
                          style={'height': '100%', 'width': '100%', 'minHeight': '340px'}),
                className="panel-box-body"
            )
        ], className="panel-box"), width=12, lg=6, className="h-100"),

        # Columna Derecha: Tabla Top 10 Películas por ROI
        dbc.Col(html.Div([
            html.Div([
                html.Span([html.I(className="fa-solid fa-trophy me-2 text-warning"), "Top 10 Películas por Retorno sobre Presupuesto (ROI)"],
                          className="panel-box-title"),
                html.Span("Presupuesto ≥ USD 100k", className="badge bg-light text-secondary border")
            ], className="panel-box-header"),
            html.Div(
                dash_table.DataTable(
                    id='tabla-top-roi',
                    columns=[
                        {"name": "Título", "id": "title"},
                        {"name": "Temp.", "id": "temporada"},
                        {"name": "Presupuesto", "id": "budget", "type": "numeric",
                         "format": dash_table.Format.Format(group=',')},
                        {"name": "Recaudación", "id": "revenue", "type": "numeric",
                         "format": dash_table.Format.Format(group=',')},
                        {"name": "ROI (x)", "id": "roi", "type": "numeric",
                         "format": dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed)},
                        {"name": "Pop.", "id": "popularity", "type": "numeric",
                         "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
                        {"name": "Clasificación", "id": "oportunidad_comercial"},
                    ],
                    data=oportunidades_tab.nlargest(10, 'roi').to_dict('records'),
                    page_size=10,
                    sort_action='native',
                    style_table={'overflowX': 'auto', 'overflowY': 'auto'},
                    style_header={'backgroundColor': OSCURO, 'color': 'white', 'fontWeight': 'bold', 'fontSize': '12px', 'padding': '7px'},
                    style_cell={'textAlign': 'left', 'padding': '6px 8px', 'fontSize': '12.5px',
                                'maxWidth': '175px', 'overflow': 'hidden', 'textOverflow': 'ellipsis'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': GRIS_CLARO}]
                ),
                className="panel-box-body"
            )
        ], className="panel-box"), width=12, lg=6, className="h-100"),
    ], className="g-3 horizontal-content-row")
], id="view-comercial", className="view-panel d-none")


# Layout Principal (Shell Horizontal: Barra Lateral + Área Principal)
app.layout = html.Div([
    dcc.Store(id='active-view-store', data='matriz'),
    html.Div(id='resize-trigger', style={'display': 'none'}),
    dcc.Download(id='download-candidatos'),
    sidebar,
    html.Main([
        kpis,
        html.Div([
            view_matriz,
            view_hallazgos,
            view_familias,
            view_geografia,
            view_comercial,
        ], className="views-container")
    ], className="app-main")
], className="dashboard-shell")


# -----------------------------------------------------------------------------
# 3. CALLBACKS
# -----------------------------------------------------------------------------
@app.callback(
    [Output(f"view-{k}", "className") for k, _, _ in NAV_ITEMS] +
    [Output(f"btn-nav-{k}", "className") for k, _, _ in NAV_ITEMS] +
    [Output("active-view-store", "data")],
    [Input(f"btn-nav-{k}", "n_clicks") for k, _, _ in NAV_ITEMS],
    prevent_initial_call=True
)
def cambiar_vista_lateral(*_):
    ctx = dash.callback_context
    if not ctx.triggered:
        activa = 'matriz'
    else:
        boton_id = ctx.triggered[0]['prop_id'].split('.')[0]
        activa = boton_id.replace('btn-nav-', '')

    clases_vistas = ["view-panel active" if k == activa else "view-panel d-none" for k, _, _ in NAV_ITEMS]
    clases_botones = ["sidebar-nav-btn active" if k == activa else "sidebar-nav-btn" for k, _, _ in NAV_ITEMS]
    return (*clases_vistas, *clases_botones, activa)


# Clientside callback para reajustar los gráficos Plotly al cambiar de botón en la barra lateral
app.clientside_callback(
    """
    function(vistaActiva) {
        setTimeout(function() {
            window.dispatchEvent(new Event('resize'));
            if (window.Plotly) {
                document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {
                    if (gd.offsetParent !== null) {
                        window.Plotly.Plots.resize(gd);
                    }
                });
            }
        }, 40);
        setTimeout(function() {
            window.dispatchEvent(new Event('resize'));
        }, 150);
        return '';
    }
    """,
    Output('resize-trigger', 'children'),
    Input('active-view-store', 'data')
)


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
     Output('filter-cuadrante', 'value'), Output('search-title', 'value'), Output('sample-slider', 'value')],
    Input('btn-reset', 'n_clicks'), prevent_initial_call=True
)
def reset_filtros(_):
    return 'TODOS', 'TODOS', 'TODOS', 'TODOS', '', 50


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
        make_kpi_card("Catálogo Evaluado", f"{n:,}".replace(',', '.'), "Con votos y valoración válidos", "fa-clapperboard", AZUL),
        make_kpi_card("Candidatos Promoción", f"{n_prom:,}".replace(',', '.'), f"{n_prom / n:.0%} del catálogo filtrado" if n else "Alto alcance y alta valoración", "fa-bullhorn", ROJO),
        make_kpi_card("Score Promedio", f"{df['score_campania'].mean():.2f}" if n else nd, "Escala estandarizada (0-1)", "fa-ranking-star", AMBAR),
        make_kpi_card("Joyas Ocultas", f"{joyas:,}".replace(',', '.'), "Alta rentabilidad, poca exposición", "fa-gem", MORADO),
    ]
    return [dbc.Col(t, width=12, sm=6, lg=3) for t in tarjetas]


@app.callback(
    [Output('scatter-matriz-campania', 'figure'), Output('matrix-points-count', 'children')],
    [Input('filter-tipo', 'value'), Input('filter-idioma', 'value'), Input('filter-familia', 'value'),
     Input('filter-cuadrante', 'value'), Input('search-title', 'value'), Input('sample-slider', 'value')]
)
def update_matriz(tipo, idioma, familia, cuadrante, search, top_recaudacion_pct):
    df = aplicar_filtros(cat_score, tipo, idioma, familia)
    if cuadrante and cuadrante != 'TODOS':
        df = df[df['cuadrante_campania'] == cuadrante]
    if search:
        df = df[df['title'].astype(str).str.contains(search, case=False, na=False, regex=False)]

    # Filtrar solo títulos con recaudación registrada y seleccionar el Top X% (10% a 50%) más recaudador
    df_rec = df[df['revenue'].fillna(0) > 0]
    total_con_rec = len(df_rec)

    if total_con_rec == 0:
        msg = (
            "Sin datos de recaudación para Series (TMDB solo reporta taquilla en Películas)"
            if tipo == 'Serie'
            else "Sin títulos con recaudación registrada para los filtros seleccionados"
        )
        return figura_vacia("Matriz de Priorización Estratégica (2x2)", msg), "0 títulos con recaudación"

    pct = top_recaudacion_pct if top_recaudacion_pct else 50
    n_top = max(1, int(round(total_con_rec * (pct / 100))))
    df = df_rec.nlargest(n_top, 'revenue')

    conteo_str = f"{len(df):,}".replace(',', '.')
    total_str = f"{total_con_rec:,}".replace(',', '.')
    conteo_texto = f"Top {pct}% recaudación · {conteo_str} de {total_str} títulos"

    fig = px.scatter(
        df, x='popularity', y='vote_average', color='cuadrante_campania', hover_name='title',
        hover_data={
            'tipo_es': True,
            'idioma': True,
            'revenue': ':$,.0f',
            'score_campania': ':.3f',
            'vote_count': True,
            'cuadrante_campania': False,
        },
        labels={'revenue': 'Recaudación (USD)', 'tipo_es': 'Tipo', 'idioma': 'Idioma', 'score_campania': 'Score', 'vote_count': 'Votos'},
        color_discrete_map={
            'Promocionar: alto alcance y alta valoración': VERDE,
            'Awareness: alto alcance; validar mensaje y segmentación': AMBAR,
            'Descubrir: alta valoración; aumentar visibilidad': AZUL,
            'No priorizar: bajo alcance y valoración': GRIS
        },
        log_x=True, opacity=0.72
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')), selector=dict(type='scatter'))
    fig.add_vline(x=UMBRAL_POP, line_width=1.8, line_dash="dash", line_color=OSCURO)
    fig.add_hline(y=UMBRAL_VAL, line_width=1.8, line_dash="dash", line_color=OSCURO)
    for x, y, txt, col in [(0.98, 0.98, " PROMOCIONAR", VERDE), (0.02, 0.98, " DESCUBRIR", AZUL),
                           (0.98, 0.02, " AWARENESS", AMBAR), (0.02, 0.02, " NO PRIORIZAR", GRIS)]:
        fig.add_annotation(x=x, y=y, xref="paper", yref="paper", text=txt, showarrow=False,
                           font=dict(color=col, size=12), bgcolor="rgba(255,255,255,0.85)")
    fig.update_layout(
        template='plotly_white', showlegend=False, margin=dict(l=45, r=25, t=20, b=45),
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
    texto = f"{len(df):,} candidatos con ≥ {MIN_VOTOS_CANDIDATO} votos (top 200 en vista; CSV exporta todos).".replace(',', '.')
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
                 color_discrete_sequence=[AZUL])
    fig.update_traces(texttemplate='<b>%{x:,}</b>', textposition='outside', cliponaxis=False,
                      marker=dict(opacity=0.85, line=dict(width=0.5, color=OSCURO)))
    fig.update_layout(template='plotly_white', margin=dict(l=40, r=55, t=20, b=40),
                      xaxis_title="Cantidad de Títulos Disponibles",
                      xaxis_range=[0, df['contenidos'].max() * 1.18], yaxis_title="")
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
                     color='familia_genero', text='familia_genero')
    fig.update_traces(textposition='top center', textfont=dict(size=11, color=OSCURO),
                      marker=dict(line=dict(width=1, color='white')))
    fig.add_vline(x=rend['popularidad_mediana'].median(), line_width=1.5, line_dash="dot", line_color=GRIS)
    fig.add_hline(y=rend['calificacion_promedio'].median(), line_width=1.5, line_dash="dot", line_color=GRIS)
    fig.update_layout(template='plotly_white', showlegend=False, margin=dict(l=45, r=35, t=20, b=40),
                      xaxis_title="Popularidad Mediana ➔ MAYOR ALCANCE",
                      yaxis_title="Calificación Promedio ➔ MAYOR CALIDAD")
    return fig


# CALLBACK UNIFICADO DE GEOGRAFÍA (Países / Idiomas)
@app.callback(
    [Output('bar-geo-idioma', 'figure'), Output('sub-geo-idioma', 'children')],
    [Input('filter-tipo', 'value'), Input('filter-idioma', 'value'), Input('filter-familia', 'value'),
     Input('selector-geo-idioma', 'value')]
)
def update_geo_idioma(tipo, idioma, familia, modo):
    if modo == 'IDIOMA':
        df = aplicar_filtros(cat_score, tipo, idioma, familia)
        if df.empty:
            return figura_vacia("Sin idiomas para la combinación de filtros seleccionada"), ""

        conteo = df['idioma'].value_counts().head(10).reset_index()
        conteo.columns = ['etiqueta', 'total']
        lider_nom = conteo.iloc[0]['etiqueta']
        lider_val = f"{conteo.iloc[0]['total']:,}".replace(',', '.')
        conteo = conteo.sort_values('total', ascending=True)
        color_barra = ROJO
        titulo_x = "Catálogo por idioma original (títulos)"
        sub_decision = f"Focaliza localización creativa en {lider_nom} ({lider_val} títulos disponibles)."
    else:
        df = aplicar_filtros(cat_pais, tipo, idioma, familia)
        if df.empty:
            return figura_vacia("Sin países para la combinación de filtros seleccionada"), ""

        conteo = df['pais'].value_counts().head(10).reset_index()
        conteo.columns = ['etiqueta', 'total']
        conteo['etiqueta'] = conteo['etiqueta'].map(PAISES_ES).fillna(conteo['etiqueta'])
        lider_nom = conteo.iloc[0]['etiqueta']
        lider_val = f"{conteo.iloc[0]['total']:,}".replace(',', '.')
        conteo = conteo.sort_values('total', ascending=True)
        color_barra = VERDE
        titulo_x = "Producción por país de origen (títulos)"
        sub_decision = f"Orienta campañas por origen hacia {lider_nom} ({lider_val} títulos asociados)."

    fig = px.bar(conteo, y='etiqueta', x='total', orientation='h', color_discrete_sequence=[color_barra])
    fig.update_traces(texttemplate='<b>%{x:,}</b>', textposition='outside', cliponaxis=False,
                      marker=dict(opacity=0.85, line=dict(width=0.5, color=OSCURO)))
    fig.update_layout(template='plotly_white', margin=dict(l=35, r=55, t=10, b=40),
                      xaxis_title=titulo_x, xaxis_range=[0, conteo['total'].max() * 1.2], yaxis_title="")

    return fig, sub_decision


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
    total_str = f"{total:,}".replace(',', '.')
    len_str = f"{len(oportunidades_tab):,}".replace(',', '.')
    conteo_texto = f"{total_str} de {len_str} películas"

    if df.empty:
        return figura_vacia("Oportunidades Comerciales"), conteo_texto, []

    fig = px.scatter(
        df,
        x='budget', y='revenue',
        color='oportunidad_comercial',
        size='popularity',
        hover_name='title',
        hover_data={'roi': ':.2f', 'vote_average': True, 'temporada': True,
                    'oportunidad_comercial': False, 'budget': ':,.0f', 'revenue': ':,.0f'},
        category_orders={'oportunidad_comercial': ORDEN_OPORTUNIDAD},
        color_discrete_map=COLOR_OPORTUNIDAD,
        log_x=True, log_y=True,
        opacity=0.72,
    )

    lim = max(oportunidades_tab['budget'].max(), oportunidades_tab['revenue'].max())
    fig.add_trace(go.Scatter(
        x=[1, lim], y=[1, lim], mode='lines',
        line=dict(color=OSCURO, width=1.5, dash='dash'),
        name='Punto de equilibrio', hoverinfo='skip', showlegend=True
    ))

    fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')), selector=dict(mode='markers'))

    fig.update_layout(
        template='plotly_white',
        legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center', title="", font=dict(size=10)),
        margin=dict(l=45, r=25, t=20, b=80),
        xaxis_title="Presupuesto (USD, escala log)",
        yaxis_title="Recaudación (USD, escala log)",
    )

    tabla_data = df.nlargest(10, 'roi').to_dict('records') if total else []
    return fig, conteo_texto, tabla_data


if __name__ == '__main__':
    app.run(debug=False, port=8050)