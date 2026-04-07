import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import folium
from streamlit_folium import st_folium
import os
import kagglehub
from surprise import KNNBasic

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistema de Recomendación de Restaurantes Limeños",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS PERSONALIZADO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Header principal */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(229, 57, 53, 0.3);
    box-shadow: 0 8px 32px rgba(229, 57, 53, 0.15);
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.4rem;
    margin: 0;
    line-height: 1.2;
}
.main-header p {
    color: #90caf9;
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
}
.accent { color: #e53935; }

/* Tarjetas de métricas */
.metric-card {
    background: linear-gradient(135deg, #1e1e2e, #252540);
    border: 1px solid rgba(144, 202, 249, 0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e53935;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #90caf9;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

/* Tarjetas de restaurante */
.rest-card {
    background: linear-gradient(135deg, #1e1e2e, #252540);
    border: 1px solid rgba(229, 57, 53, 0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s;
}
.rest-card:hover { border-color: rgba(229, 57, 53, 0.6); }
.rest-card .rank {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #e53935;
    float: left;
    margin-right: 1rem;
    line-height: 1;
}
.rest-card .name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #ffffff;
}
.rest-card .meta {
    font-size: 0.8rem;
    color: #90caf9;
    margin-top: 0.2rem;
}
.rest-card .rating-bar {
    background: rgba(229,57,53,0.15);
    border-radius: 4px;
    height: 6px;
    margin-top: 0.5rem;
}
.rest-card .rating-fill {
    background: linear-gradient(90deg, #e53935, #ff6f61);
    border-radius: 4px;
    height: 6px;
}

/* Badges */
.badge {
    display: inline-block;
    background: rgba(229,57,53,0.15);
    color: #ff6f61;
    border: 1px solid rgba(229,57,53,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin-right: 4px;
    font-weight: 500;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #90caf9 !important;
    font-weight: 500;
}

/* Divider */
hr { border-color: rgba(144,202,249,0.15); }

/* Ocultar menú hamburguesa */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
#REVIEWS_PATH  = "dataset/Lima_Restaurants_2025_08_13.csv"
#METADATA_PATH = "dataset/restaurant_metadata.csv"
_KAGGLE_PATH  = kagglehub.dataset_download("bandrehc/lima-restaurant-review")
REVIEWS_PATH  = os.path.join(_KAGGLE_PATH, "Lima_Restaurants_2025_08_13.csv")
METADATA_PATH = os.path.join(_KAGGLE_PATH, "restaurant_metadata.csv")



DISTRITOS   = ["Todos", "Miraflores", "San_Isidro", "Barranco", "Lince",
               "Magdalena", "Surco", "Surquillo"]
CATEGORIAS  = ["Todas", "Restaurante peruano", "Restaurante chino", "Pizzería",
               "Marisquería", "Restaurante italiano", "Cafetería",
               "Hamburguesería", "Restaurante de sushi", "Restaurante de comida rápida"]

# ─────────────────────────────────────────────────────────────────────────────
#  CARGA Y ENTRENAMIENTO (con caché)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos():
    reviews  = pd.read_csv(REVIEWS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    # Filtrar coordenadas inválidas
    metadata = metadata[(metadata['lat'] < -10) & (metadata['lat'] > -14)]

    # Filtrado por densidad
    uc = reviews['username'].value_counts()
    pc = reviews['id_place'].value_counts()
    df = reviews[
        reviews['username'].isin(uc[uc >= 10].index) &
        reviews['id_place'].isin(pc[pc >= 20].index)
    ]
    df = df.drop_duplicates(subset=['username', 'id_place']).reset_index(drop=True)
    return df, metadata

@st.cache_resource(show_spinner=False)
def cargar_modelo():
    """Carga el modelo híbrido exportado desde el notebook."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "modelo_hibrido.pkl")
    if not os.path.exists(model_path):
        st.error("❌ No se encontró modelo_hibrido.pkl. "
                 "Ejecuta primero el notebook para exportar el modelo.")
        st.stop()
    with open(model_path, "rb") as f:
        m = pickle.load(f)
    metricas = {
        "rmse_ubcf" : m["rmse_ubcf"],  "mae_ubcf" : m["mae_ubcf"],
        "rmse_ibcf" : m["rmse_ibcf"],  "mae_ibcf" : m["mae_ibcf"],
        "rmse_hybrid": m["rmse_hybrid"], "mae_hybrid": m["mae_hybrid"],
        "w_ubcf"    : m["w_ubcf"],     "w_ibcf"   : m["w_ibcf"],
    }
    return m["ubcf"], m["ibcf"], m["w_ubcf"], m["w_ibcf"], metricas

# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN DE RECOMENDACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def recomendar(ubcf, ibcf, w_ubcf, w_ibcf, username, df, metadata, top_n=10,
               distrito="Todos", categoria="Todas", min_stars=1.0):
    ya_visto   = df[df['username'] == username]['id_place'].unique()
    candidatos = [p for p in df['id_place'].unique() if p not in ya_visto]

    # Filtrar candidatos por metadatos
    meta_filtrada = metadata.copy()
    if distrito != "Todos":
        meta_filtrada = meta_filtrada[meta_filtrada['district'] == distrito]
    if categoria != "Todas":
        meta_filtrada = meta_filtrada[meta_filtrada['category'] == categoria]
    meta_filtrada = meta_filtrada[meta_filtrada['stars'] >= min_stars]

    candidatos = [p for p in candidatos if p in meta_filtrada['id_place'].values]

    if not candidatos:
        return pd.DataFrame()

    preds = []
    for p in candidatos:
        est = w_ubcf * ubcf.predict(uid=username, iid=p).est \
            + w_ibcf * ibcf.predict(uid=username, iid=p).est
        preds.append((p, est))
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]

    result = pd.DataFrame(preds_sorted, columns=['id_place', 'rating_estimado'])
    result = result.merge(
        metadata[['id_place', 'title', 'category', 'district',
                  'address', 'stars', 'reviews', 'lat', 'long']],
        on='id_place', how='left'
    )
    return result

# ─────────────────────────────────────────────────────────────────────────────
#  RECOMENDACIÓN COLD-START (usuario nuevo vía IBCF manual)
# ─────────────────────────────────────────────────────────────────────────────
def recomendar_nuevo_usuario(ibcf, ratings_usuario, df_ratings, df_meta,
                              top_n=10, distrito="Todos", categoria="Todas"):
    """
    Recomienda restaurantes a un usuario nuevo usando IBCF manual (cold-start).

    El usuario no existe en el trainset, por lo que no se puede usar UBCF.
    En cambio, usamos la matriz de similitud de IBCF para encontrar restaurantes
    similares a los que el usuario valoró bien (rating >= 4).

    score(candidato) = sum(sim(candidato, r) * rating(r)) / sum(sim(candidato, r))
                       para cada r valorado con rating >= 3
    """
    # Restaurantes ya valorados por el usuario nuevo
    ya_visto   = set(ratings_usuario.keys())
    candidatos = [p for p in df_ratings['id_place'].unique() if p not in ya_visto]

    # Filtrar candidatos por metadatos si se especificó
    meta_filt = df_meta.copy()
    if distrito != "Todos":
        meta_filt = meta_filt[meta_filt['district'] == distrito]
    if categoria != "Todas":
        meta_filt = meta_filt[meta_filt['category'] == categoria]
    candidatos = [p for p in candidatos if p in meta_filt['id_place'].values]

    if not candidatos:
        return pd.DataFrame()

    # Construir scores usando similitud IBCF
    # ibcf.get_neighbors devuelve vecinos del ítem en el espacio interno
    scores = []
    for cand in candidatos:
        try:
            iid_inner = ibcf.trainset.to_inner_iid(cand)
        except ValueError:
            continue  # restaurante no visto en entrenamiento

        numerador   = 0.0
        denominador = 0.0

        for r_place, r_val in ratings_usuario.items():
            if r_val < 3:          # ignorar ratings bajos
                continue
            try:
                r_inner = ibcf.trainset.to_inner_iid(r_place)
            except ValueError:
                continue
            sim = ibcf.sim[iid_inner, r_inner]
            if sim > 0:
                numerador   += sim * r_val
                denominador += sim

        if denominador > 0:
            score = numerador / denominador
            scores.append((cand, score))

    if not scores:
        # Fallback: popularidad filtrada
        pop = (df_ratings[df_ratings['id_place'].isin(candidatos)]
               .groupby('id_place')['rating']
               .agg(['mean','count'])
               .reset_index())
        pop = pop[pop['count'] >= 5].sort_values('mean', ascending=False)
        scores = [(row['id_place'], row['mean'])
                  for _, row in pop.head(top_n).iterrows()]

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    result = pd.DataFrame(scores_sorted, columns=['id_place', 'rating_estimado'])
    result = result.merge(
        df_meta[['id_place','title','category','district',
                 'stars','reviews','address','lat','long']],
        on='id_place', how='left'
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  APP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>Sistema de Recomendación de<br><span class="accent">Restaurantes Limeños</span> 🍲</h1>
    <p>Modelo Híbrido UBCF + IBCF · Pesos optimizados por Grid Search · Datos Google Maps 2025</p>
</div>
""", unsafe_allow_html=True)

# ── Cargar datos ──────────────────────────────────────────────────────────────
with st.spinner("⏳ Cargando datos y modelo Híbrido UBCF+IBCF..."):
    df, metadata = cargar_datos()
    ubcf_m, ibcf_m, w_ubcf, w_ibcf, metricas = cargar_modelo()

# ── Métricas del modelo ───────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{df['username'].nunique():,}</div>
        <div class="label">Usuarios</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{df['id_place'].nunique():,}</div>
        <div class="label">Restaurantes</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{metricas['rmse_hybrid']:.4f}</div>
        <div class="label">RMSE Híbrido</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{metricas['mae_hybrid']:.4f}</div>
        <div class="label">MAE Híbrido</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")

    # Usuario
    st.markdown("### 👤 Usuario")
    usuarios_disponibles = sorted(df['username'].value_counts().head(200).index.tolist())
    modo = st.radio("Modo de entrada", ["Seleccionar de lista", "Escribir manualmente"],
                    label_visibility="collapsed")

    if modo == "Seleccionar de lista":
        username = st.selectbox("Usuario", usuarios_disponibles)
    else:
        username = st.text_input("Escribe el username",
                                 placeholder="Ej: Carlos Alberto")

    st.markdown("---")
    st.markdown("### 🔍 Filtros")

    distrito  = st.selectbox("📍 Distrito", DISTRITOS)
    categoria = st.selectbox("🍴 Categoría", CATEGORIAS)
    min_stars = st.slider("⭐ Rating mínimo en Google Maps", 1.0, 5.0, 3.5, 0.1)
    top_n     = st.slider("🏆 Número de recomendaciones", 5, 20, 10)

    st.markdown("---")
    modo_app = st.radio("🎯 Modo", ["👤 Usuario existente", "✨ Crear perfil nuevo"],
                        label_visibility="collapsed")
    st.markdown("---")
    recomendar_btn = st.button("🚀 Generar recomendaciones", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 📊 Sobre el modelo")
    st.markdown(f"""
    <small style='color:#90caf9'>
    <b>Híbrido UBCF + IBCF</b><br>
    Pesos optimizados por Grid Search.<br><br>
    <b>Pesos óptimos:</b><br>
    • UBCF: {w_ubcf:.1f} &nbsp;|&nbsp; IBCF: {w_ibcf:.1f}<br><br>
    <b>Métricas:</b><br>
    • RMSE UBCF:  {metricas['rmse_ubcf']:.4f}<br>
    • RMSE IBCF:  {metricas['rmse_ibcf']:.4f}<br>
    • RMSE Híbrido: <b>{metricas['rmse_hybrid']:.4f}</b><br><br>
    <b>Filtros:</b><br>
    • Usuarios ≥ 10 reseñas<br>
    • Restaurantes ≥ 20 reseñas<br>
    • Dispersidad: ~98%
    </small>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONTENIDO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

# Persistir resultados en session_state para que no desaparezcan al redibujar
if "recomendaciones" not in st.session_state:
    st.session_state.recomendaciones = None
    st.session_state.username_actual = None
    st.session_state.n_vistas        = 0

if recomendar_btn and username:
    if username not in df['username'].values:
        st.error(f"⚠️ Usuario **{username}** no encontrado en el dataset. "
                 "Prueba con otro nombre o selecciónalo de la lista.")
        st.session_state.recomendaciones = None
    else:
        result = recomendar(
            ubcf_m, ibcf_m, w_ubcf, w_ibcf,
            username, df, metadata,
            top_n=top_n, distrito=distrito,
            categoria=categoria, min_stars=min_stars
        )
        if result.empty:
            st.warning("No se encontraron restaurantes con esos filtros. "
                       "Intenta ampliar los criterios.")
            st.session_state.recomendaciones = None
        else:
            st.session_state.recomendaciones = result
            st.session_state.username_actual = username
            st.session_state.n_vistas        = len(df[df['username'] == username])

# Mostrar resultados desde session_state (persisten aunque el mapa redibuje)
if st.session_state.recomendaciones is not None:
    recomendaciones = st.session_state.recomendaciones
    username        = st.session_state.username_actual
    n_vistas        = st.session_state.n_vistas

    st.markdown(f"### 🎯 Recomendaciones para **{username}**")
    st.markdown(f"<small style='color:#90caf9'>Ha valorado {n_vistas} restaurantes · "
                f"Mostrando top {len(recomendaciones)} sugerencias</small>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🏆 Recomendaciones", "🗺️ Mapa", "📊 Análisis"])

    # ── TAB 1: Lista de recomendaciones ──────────────────────────────
    with tab1:
        col_lista, col_chart = st.columns([1.2, 1])

        with col_lista:
            for i, row in recomendaciones.iterrows():
                rank  = i + 1
                pct   = int((row['rating_estimado'] / 5) * 100)
                stars_google = row['stars'] if not pd.isna(row['stars']) else 0
                n_reviews    = int(row['reviews']) if not pd.isna(row['reviews']) else 0

                st.markdown(f"""
                <div class="rest-card">
                    <span class="rank">#{rank}</span>
                    <div style="overflow:hidden">
                        <div class="name">{row['title']}</div>
                        <div class="meta">
                            📍 {row['district']} &nbsp;|&nbsp;
                            🍴 {row['category']} &nbsp;|&nbsp;
                            ⭐ {stars_google:.1f} ({n_reviews:,} reseñas Google)
                        </div>
                        <div class="meta" style="margin-top:4px">
                            {row.get('address','')[:60]}
                        </div>
                        <div style="display:flex;align-items:center;gap:8px;margin-top:6px">
                            <div class="rating-bar" style="flex:1">
                                <div class="rating-fill" style="width:{pct}%"></div>
                            </div>
                            <span style="color:#ff6f61;font-weight:700;font-size:0.9rem">
                                {row['rating_estimado']:.2f} ★
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_chart:
            st.markdown("**Rating estimado por restaurante**")
            matplotlib.rcParams['figure.facecolor'] = '#1e1e2e'
            matplotlib.rcParams['axes.facecolor']   = '#1e1e2e'
            matplotlib.rcParams['text.color']       = 'white'
            matplotlib.rcParams['axes.labelcolor']  = '#90caf9'
            matplotlib.rcParams['xtick.color']      = '#90caf9'
            matplotlib.rcParams['ytick.color']      = 'white'

            fig, ax = plt.subplots(figsize=(5, len(recomendaciones) * 0.5 + 1))
            nombres = [t[:28]+'…' if len(t)>28 else t
                       for t in recomendaciones['title']]
            colors  = ['#e53935' if i == 0 else '#4a6fa5'
                       for i in range(len(recomendaciones))]

            ax.barh(nombres, recomendaciones['rating_estimado'],
                    color=colors, edgecolor='none', height=0.6)
            ax.invert_yaxis()
            ax.set_xlim(0, 5.5)
            ax.set_xlabel('Rating estimado', color='#90caf9')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#333355')
            ax.spines['bottom'].set_color('#333355')

            for j, v in enumerate(recomendaciones['rating_estimado']):
                ax.text(v + 0.05, j, f'{v:.2f}', va='center',
                        fontsize=8, color='#ff6f61')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── TAB 2: MAPA ──────────────────────────────────────────────────
    with tab2:
        map_data = recomendaciones.dropna(subset=['lat','long']).copy()
        map_data = map_data[(map_data['lat'] < -10) & (map_data['lat'] > -14)]
        map_data['rank'] = range(1, len(map_data) + 1)

        if map_data.empty:
            st.warning("No hay coordenadas disponibles para los restaurantes filtrados.")
        else:
            center_lat = map_data['lat'].mean()
            center_lon = map_data['long'].mean()

            # Crear mapa folium con tiles CartoDB (sin token Mapbox)
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=14,
                tiles="CartoDB dark_matter",
            )

            def hex_color(rank, total):
                if rank == 1:
                    return "#e53935"
                ratio = (rank - 1) / max(total - 1, 1)
                r = int(229 - ratio * (229 - 74))
                g = int(57  + ratio * (111 - 57))
                b = int(53  + ratio * (165 - 53))
                return f"#{r:02x}{g:02x}{b:02x}"

            total = len(map_data)

            for _, row in map_data.iterrows():
                rank      = int(row['rank'])
                color     = hex_color(rank, total)
                n_reviews = int(row['reviews']) if not pd.isna(row['reviews']) else 0
                stars_g   = row['stars'] if not pd.isna(row['stars']) else 0.0
                radius    = int(8 + (row['rating_estimado'] - 1) / 4 * 12)

                popup_html = f"""
                <div style="font-family:sans-serif;background:#1e1e2e;color:#fff;
                            padding:12px;border-radius:10px;min-width:220px;
                            border-left:4px solid {color}">
                    <div style="font-size:0.72rem;color:#90caf9;
                                text-transform:uppercase;letter-spacing:1px">
                        #{rank} recomendación
                    </div>
                    <div style="font-size:1rem;font-weight:700;margin:4px 0">
                        {row['title']}
                    </div>
                    <div style="font-size:0.8rem;color:#90caf9">
                        🍴 {row['category']}<br>
                        📍 {row['district']}<br>
                        🏠 {str(row.get('address',''))[:50]}
                    </div>
                    <hr style="border-color:#333;margin:8px 0">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="color:#ff6f61;font-weight:700;font-size:1.1rem">
                            {row['rating_estimado']:.2f} ★ Híbrido
                        </span>
                        <span style="color:#90caf9;font-size:0.8rem">
                            {stars_g:.1f}★ Google ({n_reviews:,})
                        </span>
                    </div>
                </div>
                """

                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.75,
                    weight=2,
                    popup=folium.Popup(popup_html, max_width=280),
                    tooltip=f"#{rank} {row['title']} — {row['rating_estimado']:.2f}★",
                ).add_to(m)

                folium.Marker(
                    location=[row['lat'], row['long']],
                    icon=folium.DivIcon(
                        html=f'''<div style="
                            font-weight:800;font-size:11px;color:white;
                            text-align:center;line-height:20px;
                            width:20px;margin-top:-10px;margin-left:-10px;
                            text-shadow:0 0 4px #000,0 0 4px #000;">{rank}</div>''',
                        icon_size=(20, 20),
                        icon_anchor=(0, 0),
                    )
                ).add_to(m)

            st_folium(m, width="100%", height=500)

            st.markdown("""
            <div style='display:flex;gap:20px;margin-top:8px;
                        font-size:0.8rem;color:#90caf9;flex-wrap:wrap'>
                <span>🔴 #1 Mejor recomendación</span>
                <span>🔵 Resto en degradado azul</span>
                <span>🔢 Número = ranking</span>
                <span>⭕ Tamaño = rating estimado</span>
                <span>🖱️ Click en círculo para detalles</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>**Resumen de restaurantes en el mapa:**",
                        unsafe_allow_html=True)
            st.dataframe(
                map_data[['rank','title','district','rating_estimado','stars','lat','long']]
                .rename(columns={
                    'rank':'#', 'title':'Restaurante', 'district':'Distrito',
                    'rating_estimado':'Rating modelo', 'stars':'★ Google',
                    'lat':'Latitud', 'long':'Longitud'
                }),
                hide_index=True,
                width='stretch',
            )
    # ── TAB 3: ANÁLISIS ───────────────────────────────────────────────
    with tab3:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Distribución por distrito**")
            dist_counts = recomendaciones['district'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(5, 3),
                                     facecolor='#1e1e2e')
            ax2.set_facecolor('#1e1e2e')
            wedges, texts, autotexts = ax2.pie(
                dist_counts.values,
                labels=dist_counts.index,
                autopct='%1.0f%%',
                colors=['#e53935','#4a6fa5','#55A868','#c44e52',
                        '#8172b2','#64b5cd','#ccb974'],
                startangle=90,
                textprops={'color': 'white', 'fontsize': 9}
            )
            for at in autotexts:
                at.set_color('white')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        with col_b:
            st.markdown("**Rating estimado vs Rating Google Maps**")
            fig3, ax3 = plt.subplots(figsize=(5, 3),
                                     facecolor='#1e1e2e')
            ax3.set_facecolor('#1e1e2e')
            ax3.scatter(
                recomendaciones['stars'],
                recomendaciones['rating_estimado'],
                c='#e53935', s=80, alpha=0.85,
                edgecolors='white', linewidths=0.5
            )
            ax3.set_xlabel('Rating Google Maps', color='#90caf9')
            ax3.set_ylabel('Rating estimado', color='#90caf9')
            ax3.set_xlim(1, 5.5)
            ax3.set_ylim(1, 5.5)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_color('#333355')
            ax3.spines['bottom'].set_color('#333355')
            ax3.tick_params(colors='#90caf9')

            for _, row in recomendaciones.iterrows():
                if not pd.isna(row['stars']):
                    ax3.annotate(
                        row['title'][:15],
                        (row['stars'], row['rating_estimado']),
                        fontsize=6, color='#90caf9',
                        xytext=(4, 4), textcoords='offset points'
                    )
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # Historial del usuario
        st.markdown(f"**Historial de valoraciones de {username}**")
        historial = df[df['username'] == username].merge(
            metadata[['id_place','title','category','district']],
            on='id_place', how='left'
        )[['title','category','district','rating']].sort_values('rating', ascending=False)

        st.dataframe(
            historial.rename(columns={
                'title':'Restaurante','category':'Categoría',
                'district':'Distrito','rating':'Rating dado'
            }),
            hide_index=True,
            width='stretch',
            height=250,
        )

# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN: NUEVO PERFIL (cold-start via IBCF)
# ─────────────────────────────────────────────────────────────────────────────
if modo_app == "✨ Crear perfil nuevo":

    # Session state para el perfil nuevo
    if "perfil_muestra" not in st.session_state:
        st.session_state.perfil_muestra   = None
        st.session_state.perfil_ratings   = {}
        st.session_state.perfil_resultado = None

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e1e2e,#252540);
                border:1px solid rgba(229,57,53,0.3);border-radius:16px;
                padding:1.5rem 2rem;margin-bottom:1.5rem'>
        <h3 style='font-family:Syne,sans-serif;color:white;margin:0 0 0.5rem'>
            ✨ Crea tu perfil
        </h3>
        <p style='color:#90caf9;margin:0;font-size:0.9rem'>
            Valora una muestra de restaurantes conocidos. Usaremos el componente
            <b>IBCF</b> del modelo híbrido para encontrar locales similares a los
            que te gustaron — sin necesidad de estar en el dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_conf, col_gen = st.columns([2, 1])
    with col_conf:
        n_muestra = st.slider("¿Cuántos restaurantes quieres valorar?", 5, 10, 7)
    with col_gen:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Generar muestra", use_container_width=True):
            # Elegir restaurantes populares y bien valorados para la muestra
            populares = (metadata[metadata['stars'] >= 4.0]
                         .dropna(subset=['title','stars','reviews'])
                         .sort_values('reviews', ascending=False)
                         .head(80)
                         .sample(n=n_muestra, random_state=None)
                         .reset_index(drop=True))
            st.session_state.perfil_muestra  = populares
            st.session_state.perfil_ratings  = {}
            st.session_state.perfil_resultado = None

    # ── Mostrar muestra para valorar ─────────────────────────────────────────
    if st.session_state.perfil_muestra is not None:
        muestra = st.session_state.perfil_muestra

        st.markdown("### 📋 Valora estos restaurantes")
        st.markdown("<small style='color:#90caf9'>Selecciona tu puntuación del 1 al 5 "
                    "(o déjalo en 0 si no lo conoces)</small>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        ratings_temp = {}
        cols_rating = st.columns(2)
        for idx, (_, row) in enumerate(muestra.iterrows()):
            col = cols_rating[idx % 2]
            with col:
                n_rev = int(row['reviews']) if not pd.isna(row['reviews']) else 0
                st.markdown(f"""
                <div class="rest-card" style="margin-bottom:0.4rem">
                    <div class="name">{row['title']}</div>
                    <div class="meta">
                        📍 {row.get('district','')} &nbsp;|&nbsp;
                        🍴 {row.get('category','')} &nbsp;|&nbsp;
                        {row['stars']:.1f} Google ({n_rev:,} reseñas)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                val = st.select_slider(
                    f"Tu rating — {row['title'][:25]}",
                    options=[0, 1, 2, 3, 4, 5],
                    value=0,
                    label_visibility="collapsed",
                    key=f"rating_{row['id_place']}"
                )
                if val > 0:
                    ratings_temp[row['id_place']] = val

        st.session_state.perfil_ratings = ratings_temp

        # ── Botón recomendar por perfil ───────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        n_valorados = sum(1 for v in ratings_temp.values() if v > 0)
        st.markdown(f"<small style='color:#90caf9'>Has valorado "
                    f"<b>{n_valorados}</b> de {len(muestra)} restaurantes</small>",
                    unsafe_allow_html=True)

        if n_valorados < 1:
            st.info("Valora al menos 1 restaurante para obtener recomendaciones.")
        else:
            if st.button("🍲 Recomendar por mi perfil", type="primary",
                         use_container_width=True):
                with st.spinner("Buscando restaurantes similares vía IBCF..."):
                    resultado_perfil = recomendar_nuevo_usuario(
                        ibcf_m,
                        ratings_temp,
                        df,
                        metadata,
                        top_n=top_n,
                        distrito=distrito,
                        categoria=categoria,
                    )
                st.session_state.perfil_resultado = resultado_perfil

    # ── Mostrar resultados del perfil ─────────────────────────────────────────
    if st.session_state.get("perfil_resultado") is not None:
        res = st.session_state.perfil_resultado

        if res.empty:
            st.warning("No hay suficientes coincidencias. Intenta ampliar los filtros.")
        else:
            st.markdown("---")
            st.markdown(f"### 🎯 Recomendaciones para tu perfil")
            st.markdown(f"<small style='color:#90caf9'>Basado en IBCF · "
                        f"{len(res)} restaurantes recomendados · "
                        f"Valoraste {len(st.session_state.perfil_ratings)} restaurantes</small>",
                        unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            tab_r1, tab_r2 = st.tabs(["🏆 Recomendaciones", "🗺️ Mapa"])

            with tab_r1:
                col_l, col_c = st.columns([1.2, 1])
                with col_l:
                    for i, row in res.reset_index(drop=True).iterrows():
                        rank       = i + 1
                        pct        = int((row['rating_estimado'] / 5) * 100)
                        stars_g    = row['stars']   if not pd.isna(row['stars'])   else 0
                        n_reviews  = int(row['reviews']) if not pd.isna(row['reviews']) else 0
                        st.markdown(f"""
                        <div class="rest-card">
                            <span class="rank">#{rank}</span>
                            <div style="overflow:hidden">
                                <div class="name">{row['title']}</div>
                                <div class="meta">
                                    📍 {row['district']} &nbsp;|&nbsp;
                                    🍴 {row['category']} &nbsp;|&nbsp;
                                    {stars_g:.1f} Google ({n_reviews:,})
                                </div>
                                <div style="display:flex;align-items:center;
                                            gap:8px;margin-top:6px">
                                    <div class="rating-bar" style="flex:1">
                                        <div class="rating-fill"
                                             style="width:{pct}%"></div>
                                    </div>
                                    <span style="color:#ff6f61;font-weight:700;
                                                 font-size:0.9rem">
                                        {row['rating_estimado']:.2f}
                                    </span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_c:
                    st.markdown("**Rating estimado (IBCF)**")
                    fig_p, ax_p = plt.subplots(
                        figsize=(5, len(res) * 0.5 + 1),
                        facecolor='#1e1e2e'
                    )
                    ax_p.set_facecolor('#1e1e2e')
                    nombres_p = [t[:28]+'…' if len(t)>28 else t
                                 for t in res['title']]
                    colors_p  = ['#e53935' if i==0 else '#4a6fa5'
                                 for i in range(len(res))]
                    ax_p.barh(nombres_p, res['rating_estimado'],
                              color=colors_p, edgecolor='none', height=0.6)
                    ax_p.invert_yaxis()
                    ax_p.set_xlim(0, 5.5)
                    ax_p.set_xlabel('Rating estimado IBCF', color='#90caf9')
                    ax_p.spines['top'].set_visible(False)
                    ax_p.spines['right'].set_visible(False)
                    ax_p.spines['left'].set_color('#333355')
                    ax_p.spines['bottom'].set_color('#333355')
                    ax_p.tick_params(colors='#90caf9')
                    for j, v in enumerate(res['rating_estimado']):
                        ax_p.text(v+0.05, j, f'{v:.2f}', va='center',
                                  fontsize=8, color='#ff6f61')
                    plt.tight_layout()
                    st.pyplot(fig_p)
                    plt.close()

            with tab_r2:
                map_p = res.dropna(subset=['lat','long']).copy()
                map_p = map_p[(map_p['lat'] < -10) & (map_p['lat'] > -14)]
                map_p['rank'] = range(1, len(map_p) + 1)

                if map_p.empty:
                    st.warning("No hay coordenadas disponibles.")
                else:
                    center_lat_p = map_p['lat'].mean()
                    center_lon_p = map_p['long'].mean()
                    m_p = folium.Map(
                        location=[center_lat_p, center_lon_p],
                        zoom_start=14,
                        tiles="CartoDB dark_matter",
                    )

                    def hex_color_p(rank, total):
                        if rank == 1: return "#e53935"
                        ratio = (rank-1) / max(total-1, 1)
                        r = int(229 - ratio*(229-74))
                        g = int(57  + ratio*(111-57))
                        b = int(53  + ratio*(165-53))
                        return f"#{r:02x}{g:02x}{b:02x}"

                    for _, row in map_p.iterrows():
                        rank_p  = int(row['rank'])
                        color_p = hex_color_p(rank_p, len(map_p))
                        n_rev_p = int(row['reviews']) if not pd.isna(row['reviews']) else 0
                        stars_p = row['stars'] if not pd.isna(row['stars']) else 0.0
                        radius_p = int(8 + (row['rating_estimado']-1)/4*12)

                        popup_p = f"""
                        <div style="font-family:sans-serif;background:#1e1e2e;
                                    color:#fff;padding:12px;border-radius:10px;
                                    min-width:200px;border-left:4px solid {color_p}">
                            <div style="font-size:0.72rem;color:#90caf9">#{rank_p}</div>
                            <div style="font-weight:700">{row['title']}</div>
                            <div style="font-size:0.8rem;color:#90caf9">
                                {row['category']} · {row['district']}<br>
                                {stars_p:.1f} Google ({n_rev_p:,} reseñas)
                            </div>
                            <div style="color:#ff6f61;font-weight:700;margin-top:6px">
                                {row['rating_estimado']:.2f} IBCF
                            </div>
                        </div>"""

                        folium.CircleMarker(
                            location=[row['lat'], row['long']],
                            radius=radius_p, color=color_p,
                            fill=True, fill_color=color_p,
                            fill_opacity=0.75, weight=2,
                            popup=folium.Popup(popup_p, max_width=260),
                            tooltip=f"#{rank_p} {row['title']} — {row['rating_estimado']:.2f}",
                        ).add_to(m_p)
                        folium.Marker(
                            location=[row['lat'], row['long']],
                            icon=folium.DivIcon(
                                html=f'''<div style="font-weight:800;font-size:11px;
                                    color:white;text-align:center;line-height:20px;
                                    width:20px;margin-top:-10px;margin-left:-10px;
                                    text-shadow:0 0 4px #000">{rank_p}</div>''',
                                icon_size=(20,20), icon_anchor=(0,0),
                            )
                        ).add_to(m_p)

                    st_folium(m_p, width="100%", height=450)

elif st.session_state.get("recomendaciones") is None and not recomendar_btn:
    # Estado inicial — pantalla de bienvenida
    st.markdown("""
    <div style='text-align:center; padding: 3rem 2rem;
                background: linear-gradient(135deg,#1e1e2e,#252540);
                border-radius:16px; border:1px solid rgba(144,202,249,0.15)'>
        <div style='font-size:4rem'>🍲</div>
        <h2 style='font-family:Syne,sans-serif;color:white;margin:1rem 0 0.5rem'>
            Sistema de Recomendación de Restaurantes Limeños
        </h2>
        <p style='color:#90caf9;max-width:560px;margin:0 auto'>
            Usa el panel izquierdo para buscar recomendaciones por usuario existente,
            o crea tu <b>perfil nuevo</b> valorando restaurantes conocidos.
        </p>
        <br>
        <div style='display:flex;justify-content:center;gap:2rem;margin-top:1rem;flex-wrap:wrap'>
            <div style='color:#90caf9;font-size:0.85rem'>📍 7 distritos</div>
            <div style='color:#90caf9;font-size:0.85rem'>🍴 15 categorías</div>
            <div style='color:#90caf9;font-size:0.85rem'>🤝 Modelo Híbrido</div>
            <div style='color:#90caf9;font-size:0.85rem'>🗺️ Mapa interactivo</div>
            <div style='color:#90caf9;font-size:0.85rem'>👤 Perfil nuevo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
