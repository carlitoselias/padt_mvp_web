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
# 1) Configuración
# ============================
st.set_page_config(page_title="VIALIS", layout="wide")
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "streamlit", "df_all_plot.geojson")

# ============================
# Sidebar / navegación básica
# ============================
sidebar_section = st.sidebar.selectbox(
    "Navegación",
    ["VIALIS", "ABOUT US", "EQUIPO", "CONTACTO"],
    index=0
)

if sidebar_section == "ABOUT US":
    st.sidebar.markdown(
        """
        **VIALIS** nace como una herramienta digital interdisciplinaria para comprender, visualizar y anticipar los
        efectos del desarrollo urbano en la movilidad de las ciudades. Nuestro objetivo es transformar la forma
        en que se planifica el territorio, pasando de diagnósticos estáticos a una gestión urbana dinámica,
        basada en evidencia y datos reales.  

        El proyecto nace desde la convergencia entre arquitectura, planificación urbana, ciencia de datos y
        tecnología, con el propósito de construir un modelo territorial de Índice de Estrés Vial Urbano, capaz de
        medir y simular la presión que ejerce la movilidad sobre la red vial y los entornos urbanos. A través del
        uso de datos anonimizados de telecomunicaciones móviles (Telco) y fuentes abiertas como
        OpenStreetMap, el sistema combina información de flujos de personas, diseño vial, transporte público y
        entorno urbano para generar un indicador comprensible, escalable y útil para la toma de decisiones
        públicas.  

        Este enfoque busca apoyar a municipios, gobiernos regionales y ministerios en el diseño de políticas y
        proyectos con enfoque de Desarrollo Orientado al Transporte (TOD), alineado con la Estrategia Nacional
        de Movilidad Sostenible (ENMS) y los Objetivos de Desarrollo Sostenible (ODS 11).  

        A través de un dashboard interactivo, este MVP permite visualizar el nivel de estrés vial por zonas
        urbanas, comparar escenarios, y evaluar cómo distintas condiciones —como la apertura de una estación,
        el aumento de densidad o la modificación de la red vial— impactan en la movilidad cotidiana.  

        Planificar ciudades más justas, eficientes y sostenibles requiere herramientas que traduzcan la
        complejidad urbana en información accesible, visual y accionable- VIALIS avanza precisamente hacia
        eso: un ecosistema de análisis territorial proactivo, donde los datos se convierten en decisiones y las
        decisiones en ciudades más habitables.  
        """
    )

elif sidebar_section == "EQUIPO":
    st.sidebar.markdown(
        """
        ### **EQUIPO**  

        **JULIO NAZAR MIRANDA**  
        Arquitecto - Magíster en Proyecto Urbano P.U.C de Chile  
        Director de proyecto VIALIS  

        **RODRIGO TRONCOSO OLCHEVSKAIA**  
        Magíster en Economía con Mención en Economía Financiera P.U.C de Chile  
        Doctor en Economía de la PUC de Chile  
        Docente e Investigador Escuela Políticas Públicas UDD  

        **LORETO BRAVO CELEDÓN**  
        Ingeniera Civil mención Transporte  
        PhD Computer Science Carleton University (Canadá)  
        Directora Instituto Data Science UDD  

        **CARLOS ELÍAS PÉREZ PIZARRO**  
        Ingeniero Industrial UNAB - Master en Data Science UDD  
        Data Scientist Instituto Data Science UDD  
        Diseñador MVP Software  

        **XAVIERA GONZÁLEZ SEMERTZAKIS**  
        Arquitectura UDD (proceso titulación)  
        Secretaria ejecutiva y apoyo logístico  
        """
    )
    
elif sidebar_section == "CONTACTO":
    st.sidebar.markdown(
        """
        **JULIO NAZAR MIRANDA**  
        jnazar@udd.cl  
        """
    )

# ============================
# 2) Carga de datos
# ============================
@st.cache_data
def load_data(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    gdf_raw = gpd.read_file(path)

    if gdf_raw.crs is None:
        gdf_raw = gdf_raw.set_crs(32719, allow_override=True)

    gdf_metric = gdf_raw.to_crs(32719)
    gdf_wgs84 = gdf_raw.to_crs(4326).copy()

    gdf_wgs84["centroid"] = gdf_wgs84.geometry.centroid
    gdf_wgs84["centroid_m"] = gdf_metric.geometry.centroid

    # asegurar columnas esenciales
    for col in ["IEV", "D_v", "A_tp", "E_u", "F_p", "name", "origen"]:
        if col not in gdf_wgs84.columns:
            gdf_wgs84[col] = pd.NA

    return gdf_wgs84

gdf = load_data(DATA_PATH)

# ============================
# 3) Buffers
# ============================
@st.cache_data
def build_buffers(_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    centroids_m = gpd.GeoSeries(_gdf["centroid_m"], crs=32719)
    buffers_m = centroids_m.buffer(400)
    buffers_wgs84 = gpd.GeoSeries(buffers_m, crs=32719).to_crs(4326)
    gdf_buf = _gdf.copy()
    gdf_buf["buffer_geom"] = buffers_wgs84
    return gdf_buf

def clip_0_10(x) -> float:
    try:
        if pd.isna(x):
            return None
        return max(0.0, min(10.0, float(x)))
    except Exception:
        return None

# ============================
# 4) Pesos y función IEV
# ============================
W_FP, W_DV, W_ATP, W_EU = 0.30, 0.25, 0.20, 0.25

def calc_iev(D_v, A_tp, E_u, F_p=None) -> float:
    weights = {"D_v": W_DV, "A_tp": W_ATP, "E_u": W_EU}
    vals = {"D_v": D_v, "A_tp": A_tp, "E_u": E_u}
    if F_p is not None:
        weights["F_p"] = W_FP
        vals["F_p"] = F_p
    valid = {k: v for k, v in vals.items() if v is not None and pd.notna(v)}
    total_w = sum(weights[k] for k in valid)
    if total_w == 0:
        return float("nan")
    score = sum(weights[k] * float(valid[k]) for k in valid) / total_w
    return max(0.0, min(10.0, score))

st.title("VIALIS · Estrés Vial Urbano")

# -------------------------------------------------
# 5A) Construcción del mapa (sin cambios lógicos)
# -------------------------------------------------
colormap = cm.LinearColormap(
    colors=["#2ca25f", "#ffffbf", "#d73027"],
    vmin=0, vmax=10
)

m = folium.Map(location=[-33.47, -70.68], zoom_start=11, tiles="CartoDB positron")

gdf_buf = build_buffers(gdf)
gdf_buf_geo = gdf_buf.set_geometry("buffer_geom", drop=False)

buffers_fg = folium.FeatureGroup(name="Áreas de influencia (800 m)", show=True)

def style_fn(feature):
    iev_raw = feature["properties"].get("IEV", None)
    iev = clip_0_10(iev_raw)
    fill_color = "#CFCFCF" if iev is None else colormap(iev)
    return {
        "fillColor": fill_color,
        "color": "#FFFFFF",  # <- sacamos borde negro duro y lo dejamos blanco finito
        "weight": 0.5,
        "opacity": 0.4,
        "fillOpacity": 0.25  # más bajo para que el punto destaque
    }

gj = folium.GeoJson(
    data=gdf_buf_geo[["name", "origen", "IEV", "buffer_geom"]].to_json(),
    name="IEV buffers",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=["name", "origen", "IEV"],
        aliases=["Nombre", "Origen", "IEV"],
        sticky=True, localize=True, labels=True,
    ),
)
gj.add_to(buffers_fg)
buffers_fg.add_to(m)

# Marcadores (sacamos borde negro)
for _, row in gdf.iterrows():
    try:
        iev_val = clip_0_10(row["IEV"])
        color = "#CFCFCF" if iev_val is None else colormap(iev_val)
        folium.CircleMarker(
            location=[row["centroid"].y, row["centroid"].x],
            radius=5.5,
            color=color,          # antes "#FFFFFF"
            weight=0.8,           # más suave
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=f"{row.get('name','(sin nombre)')}<br>IEV: {0.0 if iev_val is None else iev_val:.2f}",
        ).add_to(m)
    except Exception:
        continue

colormap.caption = "Índice de Estrés Vial (0–10)  ← verde (bajo) | rojo (alto)"
colormap.add_to(m)

minx, miny, maxx, maxy = gpd.GeoSeries(gdf["centroid"], crs=4326).total_bounds
m.fit_bounds([[miny - 0.01, minx - 0.01], [maxy + 0.01, maxx + 0.01]])
folium.LayerControl(collapsed=True).add_to(m)

# -------------------------------------------------
# 5B) Layout de dos columnas
# -------------------------------------------------
col_map, col_side = st.columns([2, 1], gap="large")

with col_map:
    map_data = st_folium(m, width=None, height=600)

with col_side:
    st.subheader("Simulación IEV del punto seleccionado")

    # Detectar clic en el mapa
    selected_row = None
    if map_data and map_data.get("last_object_clicked"):
        lat = map_data["last_object_clicked"]["lat"]
        lon = map_data["last_object_clicked"]["lng"]
        click_pt_m = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(32719).iloc[0]
        dists = gdf["centroid_m"].distance(click_pt_m)
        sel = dists.idxmin()
        selected_row = gdf.loc[sel]

    if selected_row is None:
        st.info("Haz clic en un punto del mapa para simular cambios en ese nodo.")
    else:
        row = selected_row

        iev_actual = float(row.get("IEV", float("nan")))
        st.caption(
            f"**{row.get('name','(sin nombre)')}** · Origen: {row.get('origen','-')} "
            f"· IEV actual: {iev_actual:.2f}"
        )

        # ========= sliders con descripción corta =========
        def descr(label: str):
            DESCS = {
                "D_v": "Diseño vial: ancho de calles, conectividad y densidad de intersecciones.",
                "A_tp": "Acceso TP: acceso a transporte público e intermodalidad.",
                "E_u": "Entorno urbano: mezcla de usos, veredas, soporte peatonal.",
                "F_p": "Flujos de personas: presión real de movilidad y permanencia.",
            }
            return DESCS[label]

        D_v_val = st.slider(
            "D_v · Diseño vial",
            0.0, 10.0,
            float(row["D_v"]) if pd.notna(row["D_v"]) else 5.0,
            0.1,
            help=descr("D_v")
        )
        A_tp_val = st.slider(
            "A_tp · Acceso a transporte público",
            0.0, 10.0,
            float(row["A_tp"]) if pd.notna(row["A_tp"]) else 5.0,
            0.1,
            help=descr("A_tp")
        )
        E_u_val = st.slider(
            "E_u · Entorno urbano",
            0.0, 10.0,
            float(row["E_u"]) if pd.notna(row["E_u"]) else 5.0,
            0.1,
            help=descr("E_u")
        )
        F_p_val = st.slider(
            "F_p · Flujos de personas",
            0.0, 10.0,
            float(row["F_p"]) if pd.notna(row["F_p"]) else 5.0,
            0.1,
            help=descr("F_p")
        )

        new_iev = calc_iev(D_v_val, A_tp_val, E_u_val, F_p_val)
        delta = new_iev - iev_actual if pd.notna(iev_actual) else 0.0
        signo = "▲" if delta >= 0 else "▼"

        st.success(f"Nuevo IEV simulado: {new_iev:.2f}")
        if pd.notna(iev_actual):
            st.write(f"{signo} Variación vs. actual: {delta:+.2f} pts")

        # ========= radar chart al lado de los sliders (mismo panel) =========
        st.markdown("---")
        st.caption("Subíndices normalizados (0–10). Se actualizan con tus sliders.")

        labels_radar = ["Diseño vial", "Acceso TP", "Entorno urbano", "Flujos personas"]
        point_vals = [D_v_val, A_tp_val, E_u_val, F_p_val]
        mean_vals = [
            float(gdf["D_v"].mean(skipna=True)),
            float(gdf["A_tp"].mean(skipna=True)),
            float(gdf["E_u"].mean(skipna=True)),
            float(gdf["F_p"].mean(skipna=True)),
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=point_vals,
            theta=labels_radar,
            fill='toself',
            name="Escenario simulado",
            line=dict(color="royalblue")
        ))
        fig.add_trace(go.Scatterpolar(
            r=mean_vals,
            theta=labels_radar,
            fill='toself',
            name="Promedio general",
            line=dict(color="firebrick")
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)