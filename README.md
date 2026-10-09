# StreamView Analytics — Proyecto EP3 (Gerencia de Marketing)

**Asignatura:** ADY1104 – Visualización de Datos · **Docente:** Jorge Alberto Castro Silvestre
**Equipo:** Nicolas Fonseca · Cristobal Segovia · Bastian Rubio

## Descripción
Solución de visualización de datos para la Gerencia de Marketing de StreamView Analytics: priorización de contenidos para campañas, segmentación por cuadrantes (popularidad vs. valoración), análisis por género, país e idioma, y oportunidades comerciales (ROI). Incluye EDA, dashboard interactivo (Dash/Plotly), informe ejecutivo y resumen ejecutivo.

## Estructura
```
ep3_streamview/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 CSV originales (películas y series)
│   └── processed/           CSV generados por el EDA y el ETL
├── notebooks/
│   └── 02_eda_gerencia_marketing_Listo.ipynb
├── src/
│   ├── etl_marketing.py     ETL: genera las tablas derivadas del dashboard
│   ├── build_resumen_ejecutivo.py  Genera docs/Resumen_Ejecutivo.pdf e images/resumen/
│   └── build_figuras_informe.py    Regenera las figuras del informe (images/informe/)
├── dashboard/
│   ├── app.py               Dashboard (ejecución recomendada)
│   └── dashboard_Final.ipynb  Versión notebook (ETL + dashboard)
├── images/                  Figuras exportadas por el EDA (images/marketing/)
└── docs/                    Informe ejecutivo (PDF) y resumen ejecutivo
```

## Instalación
```bash
pip install -r requirements.txt
```

## Orden de ejecución
1. **EDA:** abrir `notebooks/02_eda_gerencia_marketing_Listo.ipynb` y ejecutar todo. Lee `data/raw/` y exporta a `data/processed/` y `images/marketing/`.
2. **ETL (obligatorio tras el EDA):** `python src/etl_marketing.py` genera las tablas del dashboard a partir de `data/processed/catalogo_marketing_limpio.csv`. Las tablas ya vienen generadas en el proyecto; el EDA guarda sus copias de referencia en `data/processed/eda/` y no las pisa.
3. **Dashboard:** `python dashboard/app.py` y abrir http://127.0.0.1:8050

Todas las rutas son relativas; el proyecto funciona sin modificaciones desde cualquier ubicación.

## Datos
| Archivo | Descripción |
|---|---|
| `raw/Netflix_Movies_Detailed_up_to_2025.csv` | Películas (fuente original) |
| `raw/Netflix_TV_Shows_Detailed_up_to_2025.csv` | Series (fuente original) |
| `processed/catalogo_marketing_limpio.csv` | Catálogo integrado y depurado |
| `processed/catalogo_marketing_score.csv` | Catálogo con score de campaña y cuadrantes |
| `processed/top_candidatos_campania.csv` | Top 15 candidatos a campaña |
| `processed/catalogo_por_genero.csv` | Catálogo expandido por género/familia |
| `processed/rendimiento_familias.csv` | Métricas por familia de género |
| `processed/catalogo_por_pais.csv` | Catálogo expandido por país de producción |
| `processed/evolucion_incorporacion.csv` | Incorporación anual de títulos |
| `processed/oportunidades_comerciales.csv` | Películas con presupuesto/recaudación y ROI |

## Metodología (resumen)
- **Score de campaña** = 0,45·percentil(popularidad) + 0,30·percentil(valoración) + 0,25·percentil(log(1+votos)). Descriptivo, no predictivo.
- **Cuadrantes:** Promocionar, Awareness, Descubrir, Baja prioridad (medianas de popularidad y valoración).
- **ROI:** (recaudación − presupuesto) / presupuesto; en el dashboard se restringe a presupuestos ≥ USD 100.000 para excluir valores simbólicos.

## Limitaciones
Los datos son de catálogo (sin reproducciones, clics, conversión ni retención). `evolucion_incorporacion.csv` muestra 1.000 títulos por año por tipo, patrón atribuible al muestreo de la fuente. Ver el informe ejecutivo para el detalle.
