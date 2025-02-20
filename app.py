%%writefile app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Dashboard Harga Kamera Indonesia",
    page_icon="/api/placeholder/32/32",
    layout="wide"
)

# Function to format price
def format_price(price):
    return f"Rp{price:,.0f}".replace(',', '.')

# Load and prepare data
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1WwL2jnNXBEVNTDOX-r4aMOtxEGLjljtlrLEq6aV0Q-s/export?format=csv"
    df = pd.read_csv(url)
    df['Jumlah piksel'] = df['Jumlah piksel'].str.replace(' MP', '').astype(float)
    return df

@st.cache_resource
def train_models(df):
    # Calculate camera age
    current_year = datetime.now().year
    df['Umur Kamera'] = current_year - df['Tahun Rilis']

    # Add price per megapixel feature
    df['Harga per Megapixel'] = df['Harga'] / df['Jumlah piksel']

    # Add ISO range feature
    df['ISO Range'] = df['ISO max'] - df['ISO min']

    # Split data by condition
    df_new = df[df['Kondisi'] == 'Baru']
    df_used = df[df['Kondisi'] == 'Bekas']

    # Create label encoders
    le_brand = LabelEncoder()
    le_category = LabelEncoder()
    le_format = LabelEncoder()

    features = [
        'Brand_encoded', 'Category_encoded',
        'Jumlah piksel', 'ISO min', 'ISO max', 'ISO Range',
        'fps', 'Format_encoded', 'Tahun Rilis', 'Umur Kamera',
        'Harga per Megapixel'
    ]

    # Process new cameras
    df_new_encoded = df_new.copy()
    df_new_encoded['Brand_encoded'] = le_brand.fit_transform(df_new['Merek'])
    df_new_encoded['Category_encoded'] = le_category.fit_transform(df_new['Kategori'])
    df_new_encoded['Format_encoded'] = le_format.fit_transform(df_new['Format'])

    X_new = df_new_encoded[features]
    y_new = df_new_encoded['Harga']

    # Process used cameras
    df_used_encoded = df_used.copy()
    df_used_encoded['Brand_encoded'] = le_brand.fit_transform(df_used['Merek'])
    df_used_encoded['Category_encoded'] = le_category.fit_transform(df_used['Kategori'])
    df_used_encoded['Format_encoded'] = le_format.fit_transform(df_used['Format'])

    X_used = df_used_encoded[features]
    y_used = df_used_encoded['Harga']

    # Scale features
    scaler_new = StandardScaler()
    scaler_used = StandardScaler()
    X_new_scaled = scaler_new.fit_transform(X_new)
    X_used_scaled = scaler_used.fit_transform(X_used)

    # Train models
    model_new = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_used = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model_new.fit(X_new_scaled, y_new)
    model_used.fit(X_used_scaled, y_used)

    return {
        'model_new': model_new,
        'model_used': model_used,
        'scaler_new': scaler_new,
        'scaler_used': scaler_used,
        'le_brand': le_brand,
        'le_category': le_category,
        'le_format': le_format,
        'features': features
    }

df = load_data()
models_data = train_models(df)

# Sidebar navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard", "Prediksi Harga"])

if page == "Dashboard":
    st.image("/api/placeholder/64/64", width=64)
    st.title("Dashboard Analisis Data Kamera di Indonesia")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Total Model Kamera", df['Model'].nunique())
    with col2:
        brands = ", ".join(sorted(df['Merek'].unique()))
        st.write("**Merek Tersedia:**", brands)

    st.markdown("---")

    # Second row: Price metrics
    col1, col2, col3 = st.columns(3)

    # Average prices
    with col1:
        avg_new = df[df['Kondisi'] == 'Baru']['Harga'].mean()
        avg_used = df[df['Kondisi'] == 'Bekas']['Harga'].mean()
        st.metric("Rata-rata Harga Kamera Baru", f"Rp {avg_new:,.0f}")
        st.metric("Rata-rata Harga Kamera Bekas", f"Rp {avg_used:,.0f}")

    # Highest prices
    with col2:
        max_new = df[df['Kondisi'] == 'Baru']['Harga'].max()
        max_used = df[df['Kondisi'] == 'Bekas']['Harga'].max()
        st.metric("Harga Tertinggi Kamera Baru", f"Rp {max_new:,.0f}")
        st.metric("Harga Tertinggi Kamera Bekas", f"Rp {max_used:,.0f}")

    # Lowest prices
    with col3:
        min_new = df[df['Kondisi'] == 'Baru']['Harga'].min()
        min_used = df[df['Kondisi'] == 'Bekas']['Harga'].min()
        st.metric("Harga Terendah Kamera Baru", f"Rp {min_new:,.0f}")
        st.metric("Harga Terendah Kamera Bekas", f"Rp {min_used:,.0f}")

    # Distribution plots row
    st.subheader("Distribusi dan Perbandingan")
    tab1, tab2, tab3 = st.tabs(["Distribusi Harga", "Perbandingan Merek", "Analisis Teknis"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Price distribution by brand
            fig_price_brand = px.box(df, x='Merek', y='Harga',
                                   color='Kondisi',
                                   title='Distribusi Harga Berdasarkan Merek')
            st.plotly_chart(fig_price_brand, use_container_width=True)

        with col2:
            # Price by category
            fig_category = px.violin(df, x='Kategori', y='Harga',
                                   color='Kondisi', box=True,
                                   title='Distribusi Harga Berdasarkan Kategori')
            st.plotly_chart(fig_category, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Brand market share
            brand_share = df['Merek'].value_counts()
            fig_brand_share = px.pie(values=brand_share.values,
                                   names=brand_share.index,
                                   title='Market Share Merek Kamera')
            st.plotly_chart(fig_brand_share, use_container_width=True)

        with col2:
            # Average price by brand and condition
            avg_price = df.groupby(['Merek', 'Kondisi'])['Harga'].mean().reset_index()
            fig_avg_price = px.bar(avg_price, x='Merek', y='Harga',
                                 color='Kondisi',
                                 title='Rata-rata Harga Berdasarkan Merek dan Kondisi')
            st.plotly_chart(fig_avg_price, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            # Megapixels vs Price
            fig_mp = px.scatter(df, x='Jumlah piksel', y='Harga',
                              color='Merek', size='fps',
                              hover_data=['Model'],
                              title='Hubungan Megapixel dengan Harga')
            st.plotly_chart(fig_mp, use_container_width=True)

        with col2:
            # ISO Range analysis
            fig_iso = px.scatter(df, x='ISO Range', y='Harga',
                               color='Format', size='Jumlah piksel',
                               hover_data=['Model'],
                               title='Hubungan ISO Range dengan Harga')
            st.plotly_chart(fig_iso, use_container_width=True)

    # Timeline analysis
    st.subheader("Analisis Timeline")
    fig_timeline = px.line(df.groupby('Tahun Rilis')['Harga'].mean().reset_index(),
                          x='Tahun Rilis', y='Harga',
                          title='Trend Harga Rata-rata Berdasarkan Tahun Rilis')
    st.plotly_chart(fig_timeline, use_container_width=True)

else:
    st.image("/api/placeholder/64/64", width=64)
    st.title("Prediksi Harga Kamera")
    st.markdown("---")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Merek", df['Merek'].unique())
        category = st.selectbox("Kategori", df['Kategori'].unique())
        condition = st.selectbox("Kondisi", df['Kondisi'].unique())
        format_type = st.selectbox("Format", df['Format'].unique())

    with col2:
        megapixels = st.number_input("Jumlah Piksel (MP)",
                                    min_value=float(df['Jumlah piksel'].min()),
                                    max_value=float(df['Jumlah piksel'].max()))
        iso_min = st.number_input("ISO Minimum",
                                 min_value=int(df['ISO min'].min()),
                                 max_value=int(df['ISO min'].max()))
        iso_max = st.number_input("ISO Maximum",
                                 min_value=int(df['ISO max'].min()),
                                 max_value=int(df['ISO max'].max()))
        fps = st.number_input("FPS",
                             min_value=int(df['fps'].min()),
                             max_value=int(df['fps'].max()))
        year = st.number_input("Tahun Rilis",
                              min_value=int(df['Tahun Rilis'].min()),
                              max_value=int(df['Tahun Rilis'].max()))

    if st.button("Prediksi Harga"):
        # Calculate derived features
        current_year = datetime.now().year
        camera_age = current_year - year
        iso_range = iso_max - iso_min

        # Create input data
        input_data = pd.DataFrame({
            'Brand_encoded': [models_data['le_brand'].transform([brand])[0]],
            'Category_encoded': [models_data['le_category'].transform([category])[0]],
            'Jumlah piksel': [megapixels],
            'ISO min': [iso_min],
            'ISO max': [iso_max],
            'ISO Range': [iso_range],
            'fps': [fps],
            'Format_encoded': [models_data['le_format'].transform([format_type])[0]],
            'Tahun Rilis': [year],
            'Umur Kamera': [camera_age],
            'Harga per Megapixel': [0]  #Updated after prediction
        })

        # Select appropriate model and scaler based on condition
        if condition == 'Baru':
            model = models_data['model_new']
            scaler = models_data['scaler_new']
        else:
            model = models_data['model_used']
            scaler = models_data['scaler_used']

        # Scale input data
        input_scaled = scaler.transform(input_data[models_data['features']])

        # Make prediction
        predicted_price = model.predict(input_scaled)[0]

        # Display prediction
        st.success(f"Prediksi Harga: {format_price(predicted_price)}")

        # Show similar cameras
        st.subheader("Kamera dengan harag serupa:")
        similar_cameras = df[
            (df['Merek'] == brand) &
            (df['Kategori'] == category) &
            (df['Kondisi'] == condition)
        ].head()

        if not similar_cameras.empty:
            st.dataframe(similar_cameras[['Model', 'Harga', 'Jumlah piksel', 'Tahun Rilis']])
        else:
            st.info("Tidak ditemukan kamera dengan harga serupa dalam database")

# CSS
st.markdown("""
    
    """, unsafe_allow_html=True)
