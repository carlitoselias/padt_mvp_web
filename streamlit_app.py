# streamlit_app.py
import os

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium

# ============================
# 1) configuración
# ============================
st.set_page_config(page_title='VIALIS', layout='wide')
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'streamlit', 'df_all_plot.geojson')
CONTENT_DIR = os.path.join(BASE_DIR, 'content')

# ============================
# 1B) estilos globales (textos base más grandes)
# ============================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 18px !important;
    }
    h1 {
        font-size: 2.1rem !important;
    }
    h2 {
        font-size: 1.6rem !important;
    }
    h3 {
        font-size: 1.3rem !important;
    }
    .stMarkdown p {
        font-size: 1.0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# 1C) estilos solo para la columna de simulación
# ============================
st.markdown(
    """
    <style>
    /* Aumenta fuente solo dentro del div de simulación */
    #sim-col * {
        font-size: 19px !important;
    }

    #sim-col h2, #sim-col h3 {
        font-size: 22px !important;
        font-weight: 700 !important;
    }

    /* Etiquetas personalizadas de sliders */
    #sim-col .slider-label {
        font-size: 22px !important;
        font-weight: 800 !important;
        margin-top: 0.8rem;
        margin-bottom: 0.15rem;
    }

    #sim-col .stCaption, #sim-col caption {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# utilidad: leer markdown externos (sin caché)
# ============================
def load_markdown(filename: str) -> str:
    """lee un archivo markdown desde la carpeta content sin usar caché de streamlit"""
    path = os.path.join(CONTENT_DIR, filename)
    if not os.path.exists(path):
        return f'*no se encontró el archivo `{filename}`*'
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# ============================
# sidebar / navegación básica
# ============================
sidebar_section = st.sidebar.selectbox(
    'navegación',
    ['VIALIS', 'ABOUT US', 'EQUIPO', 'CONTACTO'],
    index=0
)

if sidebar_section == 'ABOUT US':
    st.sidebar.markdown(load_markdown('about_us.md'))

elif sidebar_section == 'EQUIPO':
    st.sidebar.markdown(load_markdown('equipo.md'))

elif sidebar_section == 'CONTACTO':
    st.sidebar.markdown(load_markdown('contacto.md'))

# ============================
# 2) carga de datos
# ============================
@st.cache_data
def load_data(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'no se encontró el archivo: {path}')

    gdf_raw = gpd.read_file(path)

    if gdf_raw.crs is None:
        gdf_raw = gdf_raw.set_crs(32719, allow_override=True)

    gdf_metric = gdf_raw.to_crs(32719)
    gdf_wgs84 = gdf_raw.to_crs(4326).copy()

    gdf_wgs84['centroid'] = gdf_wgs84.geometry.centroid
    gdf_wgs84['centroid_m'] = gdf_metric.geometry.centroid

    # asegurar columnas esenciales
    for col in ['IEV', 'D_v', 'A_tp', 'E_u', 'F_p', 'name', 'origen']:
        if col not in gdf_wgs84.columns:
            gdf_wgs84[col] = pd.NA

    return gdf_wgs84

gdf = load_data(DATA_PATH)

# ============================
# 3) buffers
# ============================
@st.cache_data
def build_buffers(_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    centroids_m = gpd.GeoSeries(_gdf['centroid_m'], crs=32719)
    buffers_m = centroids_m.buffer(400)
    buffers_wgs84 = gpd.GeoSeries(buffers_m, crs=32719).to_crs(4326)
    gdf_buf = _gdf.copy()
    gdf_buf['buffer_geom'] = buffers_wgs84
    return gdf_buf

def clip_0_10(x) -> float:
    try:
        if pd.isna(x):
            return None
        return max(0.0, min(10.0, float(x)))
    except Exception:
        return None

# ============================
# 4) pesos y función IEV
# ============================
W_FP, W_DV, W_ATP, W_EU = 0.30, 0.25, 0.20, 0.25

def calc_iev(D_v, A_tp, E_u, F_p=None) -> float:
    weights = {'D_v': W_DV, 'A_tp': W_ATP, 'E_u': W_EU}
    vals = {'D_v': D_v, 'A_tp': A_tp, 'E_u': E_u}
    if F_p is not None:
        weights['F_p'] = W_FP
        vals['F_p'] = F_p
    valid = {k: v for k, v in vals.items() if v is not None and pd.notna(v)}
    total_w = sum(weights[k] for k in valid)
    if total_w == 0:
        return float('nan')
    score = sum(weights[k] * float(valid[k]) for k in valid) / total_w
    return max(0.0, min(10.0, score))

st.title('VIALIS · visualizador de estrés vial urbano')

# -------------------------------------------------
# 5A) construcción del mapa
# -------------------------------------------------
colormap = cm.LinearColormap(
    colors=['#2ca25f', '#ffffbf', '#d73027'],
    vmin=0, vmax=10
)

m = folium.Map(location=[-33.47, -70.68], zoom_start=11, tiles='CartoDB positron')

gdf_buf = build_buffers(gdf)
gdf_buf_geo = gdf_buf.set_geometry('buffer_geom', drop=False)

buffers_fg = folium.FeatureGroup(name='áreas de influencia (800 m)', show=True)

def style_fn(feature):
    iev_raw = feature['properties'].get('IEV', None)
    iev = clip_0_10(iev_raw)
    fill_color = '#CFCFCF' if iev is None else colormap(iev)
    return {
        'fillColor': fill_color,
        'color': '#FFFFFF',
        'weight': 0.5,
        'opacity': 0.4,
        'fillOpacity': 0.25
    }

gj = folium.GeoJson(
    data=gdf_buf_geo[['name', 'origen', 'IEV', 'buffer_geom']].to_json(),
    name='IEV buffers',
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=['name', 'origen', 'IEV'],
        aliases=['nombre', 'origen', 'IEV'],
        sticky=True, localize=True, labels=True,
    ),
)
gj.add_to(buffers_fg)
buffers_fg.add_to(m)

for _, row in gdf.iterrows():
    try:
        iev_val = clip_0_10(row['IEV'])
        color = '#CFCFCF' if iev_val is None else colormap(iev_val)
        folium.CircleMarker(
            location=[row['centroid'].y, row['centroid'].x],
            radius=5.5,
            color=color,
            weight=0.8,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=f"{row.get('name','(sin nombre)')}<br>IEV: {0.0 if iev_val is None else iev_val:.2f}",
        ).add_to(m)
    except Exception:
        continue

colormap.caption = 'índice de estrés vial (0–10)  ← verde (bajo) | rojo (alto)'
colormap.add_to(m)

minx, miny, maxx, maxy = gpd.GeoSeries(gdf['centroid'], crs=4326).total_bounds
m.fit_bounds([[miny - 0.01, minx - 0.01], [maxy + 0.01, maxx + 0.01]])
folium.LayerControl(collapsed=True).add_to(m)

# -------------------------------------------------
# 5B) layout de tres columnas horizontales
# -------------------------------------------------
col_map, col_sliders, col_radar = st.columns([2, 1, 1], gap='large')

# ====== columna 1: mapa ======
with col_map:
    st.subheader('mapa IEV')
    map_data = st_folium(m, width=None, height=700)

# ====== columna 2: sliders ======
with col_sliders:
    # contenedor para aplicar estilos específicos
    st.markdown('<div id="sim-col">', unsafe_allow_html=True)

    st.subheader('simulación IEV')
    selected_row = None
    if map_data and map_data.get('last_object_clicked'):
        lat = map_data['last_object_clicked']['lat']
        lon = map_data['last_object_clicked']['lng']
        click_pt_m = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(32719).iloc[0]
        dists = gdf['centroid_m'].distance(click_pt_m)
        sel = dists.idxmin()
        selected_row = gdf.loc[sel]

    if selected_row is None:
        st.info('haz clic en un punto del mapa para simular cambios.')
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        row = selected_row
        iev_actual = float(row.get('IEV', float('nan')))
        st.caption(
            f"**{row.get('name','(sin nombre)')}** · origen: {row.get('origen','-')} "
            f"· IEV actual: {iev_actual:.2f}"
        )

        def descr(label: str):
            DESCS = {
                'D_v': 'diseño vial: ancho de calles, conectividad e intersecciones.',
                'A_tp': 'acceso transporte público: buses, metro e intermodalidad.',
                'E_u': 'entorno urbano: mezcla de usos y soporte peatonal.',
                'F_p': 'flujos de personas: presión de movilidad y permanencia.',
            }
            return DESCS[label]

        # ----- slider D_v -----
        st.markdown('<div class="slider-label">D_v · diseño vial</div>', unsafe_allow_html=True)
        D_v_val = st.slider(
            'D_v · diseño vial',
            0.0,
            10.0,
            float(row['D_v']) if pd.notna(row['D_v']) else 5.0,
            0.1,
            help=descr('D_v'),
            label_visibility='collapsed',
        )

        # ----- slider A_tp -----
        st.markdown('<div class="slider-label">A_tp · acceso transporte público</div>', unsafe_allow_html=True)
        A_tp_val = st.slider(
            'A_tp · acceso transporte público',
            0.0,
            10.0,
            float(row['A_tp']) if pd.notna(row['A_tp']) else 5.0,
            0.1,
            help=descr('A_tp'),
            label_visibility='collapsed',
        )

        # ----- slider E_u -----
        st.markdown('<div class="slider-label">E_u · entorno urbano</div>', unsafe_allow_html=True)
        E_u_val = st.slider(
            'E_u · entorno urbano',
            0.0,
            10.0,
            float(row['E_u']) if pd.notna(row['E_u']) else 5.0,
            0.1,
            help=descr('E_u'),
            label_visibility='collapsed',
        )

        # ----- slider F_p -----
        st.markdown('<div class="slider-label">F_p · flujos personas</div>', unsafe_allow_html=True)
        F_p_val = st.slider(
            'F_p · flujos personas',
            0.0,
            10.0,
            float(row['F_p']) if pd.notna(row['F_p']) else 5.0,
            0.1,
            help=descr('F_p'),
            label_visibility='collapsed',
        )

        new_iev = calc_iev(D_v_val, A_tp_val, E_u_val, F_p_val)
        delta = new_iev - iev_actual if pd.notna(iev_actual) else 0.0
        signo = '▲' if delta >= 0 else '▼'

        # ============================
        # bloque de resultado con contraste adaptativo
        # ============================
        iev_for_color = clip_0_10(new_iev)
        box_color = '#CFCFCF' if iev_for_color is None else colormap(iev_for_color)

        if iev_for_color is None:
            text_color = '#000000'
        elif iev_for_color <= 4:
            text_color = '#FFFFFF'  # verde → texto blanco
        elif iev_for_color <= 7:
            text_color = '#000000'  # amarillo → texto negro
        else:
            text_color = '#FFFFFF'  # rojo → texto blanco

        st.markdown(
            f"""
            <div style="background-color:{box_color};
                        padding:12px;
                        border-radius:8px;
                        text-align:center;">
                <span style="color:{text_color};
                             font-weight:700;
                             font-size:1.15rem;">
                    nuevo IEV simulado: {new_iev:.2f}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if pd.notna(iev_actual):
            st.write(f'{signo} variación vs. actual: {delta:+.2f} pts')

        st.markdown('</div>', unsafe_allow_html=True)  # cierra #sim-col

# ====== columna 3: radar chart ======
with col_radar:
    st.subheader('comparación de subíndices')
    if 'selected_row' in locals() and selected_row is not None:
        labels_radar = ['D_v', 'A_tp', 'E_u', 'F_p']
        point_vals = [D_v_val, A_tp_val, E_u_val, F_p_val]
        mean_vals = [
            float(gdf['D_v'].mean(skipna=True)),
            float(gdf['A_tp'].mean(skipna=True)),
            float(gdf['E_u'].mean(skipna=True)),
            float(gdf['F_p'].mean(skipna=True)),
        ]

        labels_ext = labels_radar + [labels_radar[0]]
        point_vals_ext = point_vals + [point_vals[0]]
        mean_vals_ext = mean_vals + [mean_vals[0]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=point_vals_ext,
                theta=labels_ext,
                fill='toself',
                name='simulación',
                line=dict(color='royalblue'),
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=mean_vals_ext,
                theta=labels_ext,
                fill='toself',
                name='promedio',
                line=dict(color='firebrick'),
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickfont=dict(size=14),
                ),
                angularaxis=dict(
                    tickfont=dict(size=14),
                    rotation=90,
                    direction='clockwise',
                ),
            ),
            legend=dict(font=dict(size=13)),
            font=dict(size=14),
            margin=dict(l=30, r=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('selecciona un punto en el mapa para ver el radar de subíndices.')
