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
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

# Set page config
st.set_page_config(
    page_title="Dashboard Harga Kamera Indonesia",
    page_icon="https://i.pinimg.com/1200x/2c/64/d2/2c64d2b0c32c1d17bf8f87863b34d367.jpg",
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
    le_model = LabelEncoder()  # Added Model encoder
    le_category = LabelEncoder()
    le_format = LabelEncoder()

    # Updated features list to match ML-only version
    features = [
        'Brand_encoded', 'Model_encoded', 'Category_encoded',  # Added Model_encoded
        'Jumlah piksel', 'ISO min', 'ISO max', 'ISO Range',
        'fps', 'Format_encoded', 'Tahun Rilis', 'Umur Kamera',
        'Harga per Megapixel'
    ]

    # Process new cameras data
    df_new_encoded = df_new.copy()
    df_new_encoded['Brand_encoded'] = le_brand.fit_transform(df_new['Merek'])
    df_new_encoded['Model_encoded'] = le_model.fit_transform(df_new['Model'])  # Added Model encoding
    df_new_encoded['Category_encoded'] = le_category.fit_transform(df_new['Kategori'])
    df_new_encoded['Format_encoded'] = le_format.fit_transform(df_new['Format'])

    X_new = df_new_encoded[features]
    y_new = df_new_encoded['Harga']

    # Process used cameras data
    df_used_encoded = df_used.copy()
    df_used_encoded['Brand_encoded'] = le_brand.fit_transform(df_used['Merek'])
    df_used_encoded['Model_encoded'] = le_model.fit_transform(df_used['Model'])  # Added Model encoding
    df_used_encoded['Category_encoded'] = le_category.fit_transform(df_used['Kategori'])
    df_used_encoded['Format_encoded'] = le_format.fit_transform(df_used['Format'])

    X_used = df_used_encoded[features]
    y_used = df_used_encoded['Harga']

    # Scale features before splitting (changed to match ML-only version)
    scaler_new = StandardScaler()
    scaler_used = StandardScaler()
    X_new_scaled = scaler_new.fit_transform(X_new)
    X_used_scaled = scaler_used.fit_transform(X_used)

    # Split scaled data
    X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new_scaled, y_new, test_size=0.2, random_state=42)
    X_used_train, X_used_test, y_used_train, y_used_test = train_test_split(X_used_scaled, y_used, test_size=0.2, random_state=42)

    # Train models with same parameters as ML-only version
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

    # Fit models
    model_new.fit(X_new_train, y_new_train)
    model_used.fit(X_used_train, y_used_train)

    # Calculate metrics
    y_new_pred = model_new.predict(X_new_test)
    new_metrics = {
        'MAPE': mean_absolute_percentage_error(y_new_test, y_new_pred),
        'MAE': mean_absolute_error(y_new_test, y_new_pred),
        'RMSE': np.sqrt(mean_squared_error(y_new_test, y_new_pred)),
        'R2': r2_score(y_new_test, y_new_pred)
    }

    y_used_pred = model_used.predict(X_used_test)
    used_metrics = {
        'MAPE': mean_absolute_percentage_error(y_used_test, y_used_pred),
        'MAE': mean_absolute_error(y_used_test, y_used_pred),
        'RMSE': np.sqrt(mean_squared_error(y_used_test, y_used_pred)),
        'R2': r2_score(y_used_test, y_used_pred)
    }

    return {
        'model_new': model_new,
        'model_used': model_used,
        'scaler_new': scaler_new,
        'scaler_used': scaler_used,
        'le_brand': le_brand,
        'le_model': le_model,  # Added Model encoder to return dict
        'le_category': le_category,
        'le_format': le_format,
        'features': features,
        'new_metrics': new_metrics,
        'used_metrics': used_metrics
    }
    
df = load_data()
models_data = train_models(df)

# Sidebar navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard", "Prediksi Harga"])

if page == "Dashboard":
    st.image("https://i.pinimg.com/1200x/2c/64/d2/2c64d2b0c32c1d17bf8f87863b34d367.jpg", width=64)
    st.title("Dashboard Analisis Data Kamera di Indonesia")
    st.write("**Tahun Rilis 2013-2024**")
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
        st.metric("Rata-rata Harga Kamera Baru", f"Rp{avg_new:,.0f}".replace(',', '.'))
        st.metric("Rata-rata Harga Kamera Bekas", f"Rp{avg_used:,.0f}".replace(',', '.'))

    # Highest prices
    with col2:
        max_new = df[df['Kondisi'] == 'Baru']['Harga'].max()
        max_used = df[df['Kondisi'] == 'Bekas']['Harga'].max()
        st.metric("Harga Tertinggi Kamera Baru", f"Rp{max_new:,.0f}".replace(',', '.'))
        st.metric("Harga Tertinggi Kamera Bekas", f"Rp{max_used:,.0f}".replace(',', '.'))

    # Lowest prices
    with col3:
        min_new = df[df['Kondisi'] == 'Baru']['Harga'].min()
        min_used = df[df['Kondisi'] == 'Bekas']['Harga'].min()
        st.metric("Harga Terendah Kamera Baru", f"Rp{min_new:,.0f}".replace(',', '.'))
        st.metric("Harga Terendah Kamera Bekas", f"Rp{min_used:,.0f}".replace(',', '.'))

    # Distribution plots with tabs
st.subheader("Analisis Data Kamera")
tabs = st.tabs(["Analisis Harga Kamera", "Popularitas dan Tren Kamera", 
                "Analisis Merek dan Kategori", "Analisis Kamera Bekas vs Baru"])

# Tab 1: Analisis Harga Kamera
with tabs[0]:
    st.subheader("Distribusi Harga tiap Merek")
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution for new cameras
        fig_new = px.box(df[df['Kondisi'] == 'Baru'], 
                        x='Merek', y='Harga',
                        title='Distribusi Harga Kamera Baru per Merek')
        st.plotly_chart(fig_new, use_container_width=True)
    
    with col2:
        # Distribution for used cameras
        fig_used = px.box(df[df['Kondisi'] == 'Bekas'], 
                         x='Merek', y='Harga',
                         title='Distribusi Harga Kamera Bekas per Merek')
        st.plotly_chart(fig_used, use_container_width=True)

    st.subheader("Distribusi Harga berdasarkan Tahun Rilis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_year_new = px.box(df[df['Kondisi'] == 'Baru'], 
                             x='Tahun Rilis', y='Harga',
                             title='Distribusi Harga Kamera Baru per Tahun')
        st.plotly_chart(fig_year_new, use_container_width=True)
    
    with col2:
        fig_year_used = px.box(df[df['Kondisi'] == 'Bekas'], 
                              x='Tahun Rilis', y='Harga',
                              title='Distribusi Harga Kamera Bekas per Tahun')
        st.plotly_chart(fig_year_used, use_container_width=True)

    st.subheader("Distribusi Harga berdasarkan Format Kamera")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_format_new = px.box(df[df['Kondisi'] == 'Baru'], 
                               x='Format', y='Harga',
                               title='Distribusi Harga Kamera Baru per Format')
        st.plotly_chart(fig_format_new, use_container_width=True)
    
    with col2:
        fig_format_used = px.box(df[df['Kondisi'] == 'Bekas'], 
                                x='Format', y='Harga',
                                title='Distribusi Harga Kamera Bekas per Format')
        st.plotly_chart(fig_format_used, use_container_width=True)

    st.subheader("Top 10 Kamera Termahal")
    col1, col2 = st.columns(2)
    
    with col1:
        top_new = df[df['Kondisi'] == 'Baru'].nlargest(10, 'Harga')
        fig_top_new = px.bar(top_new, x='Model', y='Harga',
                            title='10 Kamera Baru Termahal',
                            text=top_new['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))
        fig_top_new.update_traces(textposition='outside')
        st.plotly_chart(fig_top_new, use_container_width=True)
    
    with col2:
        top_used = df[df['Kondisi'] == 'Bekas'].nlargest(10, 'Harga')
        fig_top_used = px.bar(top_used, x='Model', y='Harga',
                             title='10 Kamera Bekas Termahal',
                             text=top_used['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))
        fig_top_used.update_traces(textposition='outside')
        st.plotly_chart(fig_top_used, use_container_width=True)

# Tab 2: Popularitas dan Tren Kamera
with tabs[1]:
    st.subheader("Top 10 Model Kamera Terbanyak menurut Tahun Rilis")
    model_counts = df.groupby(['Tahun Rilis', 'Model']).size().reset_index(name='count')
    top_models = model_counts.nlargest(10, 'count')
    fig_top_models = px.bar(top_models, x='Model', y='count',
                           color='Tahun Rilis',
                           title='10 Model Kamera Terbanyak')
    st.plotly_chart(fig_top_models, use_container_width=True)

    st.subheader("Distribusi Kamera Berdasarkan Tahun Rilis")
    year_dist = df['Tahun Rilis'].value_counts().reset_index()
    year_dist.columns = ['Tahun Rilis', 'Jumlah']
    fig_year_dist = px.line(year_dist.sort_values('Tahun Rilis'),
                           x='Tahun Rilis', y='Jumlah',
                           title='Distribusi Kamera per Tahun Rilis',
                           markers=True)
    st.plotly_chart(fig_year_dist, use_container_width=True)

# Tab 3: Analisis Merek dan Kategori
with tabs[2]:
    st.subheader("Distribusi Kamera Berdasarkan Merek")
    col1, col2 = st.columns(2)
    
    with col1:
        brand_new = df[df['Kondisi'] == 'Baru']['Merek'].value_counts()
        fig_brand_new = px.pie(values=brand_new.values,
                              names=brand_new.index,
                              title='Distribusi Merek Kamera Baru')
        st.plotly_chart(fig_brand_new, use_container_width=True)
    
    with col2:
        brand_used = df[df['Kondisi'] == 'Bekas']['Merek'].value_counts()
        fig_brand_used = px.pie(values=brand_used.values,
                               names=brand_used.index,
                               title='Distribusi Merek Kamera Bekas')
        st.plotly_chart(fig_brand_used, use_container_width=True)

    st.subheader("Proporsi Kategori Kamera")
    col1, col2 = st.columns(2)
    
    with col1:
        cat_new = df[df['Kondisi'] == 'Baru']['Kategori'].value_counts()
        fig_cat_new = px.pie(values=cat_new.values,
                            names=cat_new.index,
                            title='Proporsi Kategori Kamera Baru')
        st.plotly_chart(fig_cat_new, use_container_width=True)
    
    with col2:
        cat_used = df[df['Kondisi'] == 'Bekas']['Kategori'].value_counts()
        fig_cat_used = px.pie(values=cat_used.values,
                             names=cat_used.index,
                             title='Proporsi Kategori Kamera Bekas')
        st.plotly_chart(fig_cat_used, use_container_width=True)

    st.subheader("Rata-rata Harga Kamera Per Merek")
    col1, col2 = st.columns(2)
    
    with col1:
        avg_price_new = df[df['Kondisi'] == 'Baru'].groupby('Merek')['Harga'].mean().reset_index()
        fig_avg_new = px.bar(avg_price_new, x='Merek', y='Harga',
                            title='Rata-rata Harga Kamera Baru per Merek',
                            text=avg_price_new['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))
        fig_avg_new.update_traces(textposition='outside')
        st.plotly_chart(fig_avg_new, use_container_width=True)
    
    with col2:
        avg_price_used = df[df['Kondisi'] == 'Bekas'].groupby('Merek')['Harga'].mean().reset_index()
        fig_avg_used = px.bar(avg_price_used, x='Merek', y='Harga',
                             title='Rata-rata Harga Kamera Bekas per Merek',
                             text=avg_price_used['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))
        fig_avg_used.update_traces(textposition='outside')
        st.plotly_chart(fig_avg_used, use_container_width=True)

# Tab 4: Analisis Kamera Bekas vs Baru
with tabs[3]:
    st.subheader("Perbandingan Kamera Bekas vs Baru")
    
    # Comparison metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Statistik Kamera Baru:**")
        new_stats = df[df['Kondisi'] == 'Baru']['Harga'].describe()
        st.write(f"- Rata-rata Harga: Rp{new_stats['mean']:,.0f}".replace(',', '.'))
        st.write(f"- Harga Tertinggi: Rp{new_stats['max']:,.0f}".replace(',', '.'))
        st.write(f"- Harga Terendah: Rp{new_stats['min']:,.0f}".replace(',', '.'))
        st.write(f"- Jumlah Model: {len(df[df['Kondisi'] == 'Baru'])}")
    
    with col2:
        st.write("**Statistik Kamera Bekas:**")
        used_stats = df[df['Kondisi'] == 'Bekas']['Harga'].describe()
        st.write(f"- Rata-rata Harga: Rp{used_stats['mean']:,.0f}".replace(',', '.'))
        st.write(f"- Harga Tertinggi: Rp{used_stats['max']:,.0f}".replace(',', '.'))
        st.write(f"- Harga Terendah: Rp{used_stats['min']:,.0f}".replace(',', '.'))
        st.write(f"- Jumlah Model: {len(df[df['Kondisi'] == 'Bekas'])}")

    # Price comparison by year
    st.subheader("Perbandingan Harga Berdasarkan Tahun")
    year_price = df.groupby(['Tahun Rilis', 'Kondisi'])['Harga'].mean().reset_index()
    fig_year_compare = px.line(year_price, x='Tahun Rilis', y='Harga',
                              color='Kondisi', title='Trend Harga Rata-rata per Tahun',
                              markers=True)
    st.plotly_chart(fig_year_compare, use_container_width=True)

    # Brand distribution comparison
    st.subheader("Perbandingan Distribusi Merek")
    brand_dist = pd.DataFrame({
        'Baru': df[df['Kondisi'] == 'Baru']['Merek'].value_counts(),
        'Bekas': df[df['Kondisi'] == 'Bekas']['Merek'].value_counts()
    }).fillna(0)
    fig_brand_compare = px.bar(brand_dist, barmode='group',
                              title='Perbandingan Jumlah Model per Merek')
    st.plotly_chart(fig_brand_compare, use_container_width=True)

else:
    st.image("https://i.pinimg.com/1200x/2c/64/d2/2c64d2b0c32c1d17bf8f87863b34d367.jpg", width=64)
    st.title("Prediksi Harga Kamera")
    st.markdown("---")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Merek", df['Merek'].unique())  # Removed sorting to match ML-only version
        model = st.selectbox("Model", df[df['Merek'] == brand]['Model'].unique())  # Added Model selection
        category = st.selectbox("Kategori", df['Kategori'].unique())
        condition = st.selectbox("Kondisi", df['Kondisi'].unique())
        format_type = st.selectbox("Format", df['Format'].unique())

    with col2:
        megapixels = st.selectbox("Jumlah Piksel (MP)", 
                                 df['Jumlah piksel'].unique())
        iso_min = st.selectbox("ISO Minimum",
                              df['ISO min'].unique())
        iso_max = st.selectbox("ISO Maximum",
                              df['ISO max'].unique())
        fps = st.selectbox("FPS",
                          df['fps'].unique())
        year = st.selectbox("Tahun Rilis",
                           df['Tahun Rilis'].unique())

    if st.button("Prediksi Harga"):
        # Calculate derived features
        current_year = datetime.now().year
        camera_age = current_year - year
        iso_range = iso_max - iso_min
        price_per_mp = 0  # Will be updated after prediction

        # Create input data with Model encoding
        input_data = pd.DataFrame({
            'Brand_encoded': [models_data['le_brand'].transform([brand])[0]],
            'Model_encoded': [models_data['le_model'].transform([model])[0]],  # Added Model encoding
            'Category_encoded': [models_data['le_category'].transform([category])[0]],
            'Jumlah piksel': [megapixels],
            'ISO min': [iso_min],
            'ISO max': [iso_max],
            'ISO Range': [iso_range],
            'fps': [fps],
            'Format_encoded': [models_data['le_format'].transform([format_type])[0]],
            'Tahun Rilis': [year],
            'Umur Kamera': [camera_age],
            'Harga per Megapixel': [price_per_mp]
        })

        # Select appropriate model and scaler
        if condition == 'Baru':
            model = models_data['model_new']
            scaler = models_data['scaler_new']
            metrics = models_data['new_metrics']
        else:
            model = models_data['model_used']
            scaler = models_data['scaler_used']
            metrics = models_data['used_metrics']

        # Scale input data
        input_scaled = scaler.transform(input_data[models_data['features']])

        # Make prediction and ensure non-negative
        predicted_price = max(0, model.predict(input_scaled)[0])  # Added max(0, ...) to prevent negative prices

        # Display prediction
        st.success(f"Prediksi Harga: {format_price(predicted_price)}")

        # Display model metrics
        st.subheader("Metrik Performa Model")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAPE", f"{metrics['MAPE']*100:.2f}%")
            st.caption("Mean Absolute Percentage Error")
        with col2:
            st.metric("MAE", f"Rp{metrics['MAE']:,.0f}".replace(',', '.'))
            st.caption("Mean Absolute Error")
        with col3:
            rmse_millions = metrics['RMSE'] / 1_000_000
            st.metric("RMSE", f"Rp{rmse_millions:.1f}M")
            st.caption("Root Mean Squared Error")
        with col4:
            st.metric("R²", f"{metrics['R2']:.3f}")
            st.caption("Coefficient of Determination")

        # Show similar cameras
        st.subheader("Kamera Similar")
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
