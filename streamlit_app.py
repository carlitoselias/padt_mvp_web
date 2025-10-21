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
st.set_page_config(page_title="MVP IEV", layout="wide")
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "streamlit", "df_all_plot.geojson")

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

# ============================
# 5) Configuración UI
# ============================
st.title("MVP: Simulador de Índice de Estrés Vial con Datos de Movilidad")

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
    return {"fillColor": fill_color, "color": "#111111", "weight": 0.6,
            "opacity": 0.6, "fillOpacity": 0.35}

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

# Marcadores
for _, row in gdf.iterrows():
    try:
        iev_val = clip_0_10(row["IEV"])
        color = "#CFCFCF" if iev_val is None else colormap(iev_val)
        folium.CircleMarker(
            location=[row["centroid"].y, row["centroid"].x],
            radius=5.5,
            color="#FFFFFF",
            weight=1.2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            popup=f"{row.get('name','(sin nombre)')}<br>IEV: {0.0 if iev_val is None else iev_val:.2f}",
        ).add_to(m)
    except Exception:
        continue

colormap.caption = "Índice de Estrés Vial (0–10)  ← verde (bajo) | rojo (alto)"
colormap.add_to(m)

minx, miny, maxx, maxy = gpd.GeoSeries(gdf["centroid"], crs=4326).total_bounds
m.fit_bounds([[miny - 0.01, minx - 0.01], [maxy + 0.01, maxx + 0.01]])
folium.LayerControl(collapsed=True).add_to(m)

map_data = st_folium(m, width=1000, height=600)

# ============================
# 6) Interacción por clic
# ============================
if map_data and map_data.get("last_object_clicked"):
    lat = map_data["last_object_clicked"]["lat"]
    lon = map_data["last_object_clicked"]["lng"]
    click_pt_m = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(32719).iloc[0]
    dists = gdf["centroid_m"].distance(click_pt_m)
    sel = dists.idxmin()
    row = gdf.loc[sel]

    st.subheader(f"Punto seleccionado: {row.get('name','(sin nombre)')}")
    tab1, tab2 = st.tabs(["🔧 Simulación IEV", "📊 Subíndices normalizados"])

    # --- TAB 1: Simulación
    with tab1:
        iev_actual = float(row.get("IEV", float("nan")))
        st.caption(f"Origen: {row.get('origen','-')}  |  IEV actual: {iev_actual:.2f}")
        D_v = st.slider("Diseño vial (D_v)", 0.0, 10.0, float(row["D_v"]) if pd.notna(row["D_v"]) else 5.0, 0.1)
        A_tp = st.slider("Acceso TP (A_tp)", 0.0, 10.0, float(row["A_tp"]) if pd.notna(row["A_tp"]) else 5.0, 0.1)
        E_u = st.slider("Entorno urbano (E_u)", 0.0, 10.0, float(row["E_u"]) if pd.notna(row["E_u"]) else 5.0, 0.1)
        F_p = st.slider("Flujos de personas (F_p)", 0.0, 10.0, float(row["F_p"]) if pd.notna(row["F_p"]) else 5.0, 0.1)

        new_iev = calc_iev(D_v, A_tp, E_u, F_p)
        st.success(f"Nuevo IEV simulado: {new_iev:.2f}")
        if pd.notna(row["IEV"]):
            delta = float(new_iev) - float(row["IEV"])
            signo = "▲" if delta >= 0 else "▼"
            st.write(f"{signo} Variación vs. actual: {delta:+.2f} puntos")

    # --- TAB 2: Radar chart ---
    with tab2:
        st.caption("Comparación de subíndices normalizados (0–10).")

        cols_radar = ["D_v", "A_tp", "E_u", "F_p"]
        labels = ["Diseño vial", "Acceso TP", "Entorno urbano", "Flujos personas"]

        point_vals = [float(row[c]) if pd.notna(row[c]) else 0 for c in cols_radar]
        mean_vals = [float(gdf[c].mean(skipna=True)) if c in gdf.columns else 0 for c in cols_radar]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=point_vals,
            theta=labels,
            fill='toself',
            name="Punto seleccionado",
            line=dict(color="royalblue")
        ))
        fig.add_trace(go.Scatterpolar(
            r=mean_vals,
            theta=labels,
            fill='toself',
            name="Promedio general",
            line=dict(color="firebrick")
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Haz clic sobre un punto para editar sus subíndices y explorar sus métricas.")