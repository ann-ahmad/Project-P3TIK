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
    # Split data by condition
    df_new = df[df['Kondisi'] == 'Baru']
    df_used = df[df['Kondisi'] == 'Bekas']
    
    # Create encoders
    le_model = LabelEncoder()
    
    # Process new cameras data
    df_new_encoded = df_new.copy()
    df_new_encoded['Model_encoded'] = le_model.fit_transform(df_new['Model'])
    
    X_new = df_new_encoded[['Model_encoded']]
    y_new = df_new_encoded['Harga']
    
    # Process used cameras data
    df_used_encoded = df_used.copy()
    df_used_encoded['Model_encoded'] = le_model.fit_transform(df_used['Model'])
    
    X_used = df_used_encoded[['Model_encoded']]
    y_used = df_used_encoded['Harga']
    
    # Split data untuk training dan testing
    X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
        X_new, y_new, test_size=0.2, random_state=42
    )
    X_used_train, X_used_test, y_used_train, y_used_test = train_test_split(
        X_used, y_used, test_size=0.2, random_state=42
    )
    
    # Train models
    model_new = xgb.XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model_used = xgb.XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # Fit models
    model_new.fit(X_new_train, y_new_train)
    model_used.fit(X_used_train, y_used_train)
    
    # Calculate metrics for both models
    def calculate_metrics(model, X_test, y_test):
        y_pred = model.predict(X_test)
        return {
            'MAPE': mean_absolute_percentage_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
    
    metrics_new = calculate_metrics(model_new, X_new_test, y_new_test)
    metrics_used = calculate_metrics(model_used, X_used_test, y_used_test)
    
    return {
        'model_new': model_new,
        'model_used': model_used,
        'le_model': le_model,
        'metrics_new': metrics_new,
        'metrics_used': metrics_used,
        # Tambahkan data test untuk validasi prediksi
        'test_data': {
            'new': (X_new_test, y_new_test),
            'used': (X_used_test, y_used_test)
        }
    }
    
df = load_data()
models_data = train_models(df)

# Sidebar navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard", "Prediksi Harga"])

if page == "Dashboard":
    st.image("https://i.pinimg.com/1200x/2c/64/d2/2c64d2b0c32c1d17bf8f87863b34d367.jpg", width=64)
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

    # Fungsi untuk membuat dan menampilkan chart dengan ID unik
    def create_and_show_chart(fig, container, idx):
        fig.update_layout(height=400)  # Menyamakan tinggi semua chart
        container.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")
   
    # Analysis Tabs
    st.subheader("Analisis Data Kamera")
    tabs = st.tabs([
        "Analisis Harga Kamera",
        "Popularitas dan Tren Kamera",
        "Analisis Merek dan Kategori",
        "Analisis Kamera Bekas vs Baru"
    ])

    with tabs[0]:
        st.subheader("Distribusi Harga tiap Merek")
        col1, col2 = st.columns(2)

        with col1:
        # Distribution for new cameras
            avg_price_new = df[df['Kondisi'] == 'Baru'].groupby('Merek')['Harga'].mean().reset_index()
            fig_new = px.bar(avg_price_new, x='Merek', y='Harga',
                             title='Rata-rata Harga Kamera Baru per Merek',
                             text=avg_price_new['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))
            fig_new.update_traces(textposition='outside')
            create_and_show_chart(fig_new, col1, "new_price_brand")   

        with col2:
            # Distribution for used cameras
            avg_price_used = df[df['Kondisi'] == 'Bekas'].groupby('Merek')['Harga'].mean().reset_index()
            fig_used = px.bar(avg_price_used, x='Merek', y='Harga',
                              title='Rata-rata Harga Kamera Bekas per Merek',
                              text=avg_price_used['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))
            fig_used.update_traces(textposition='outside')
            create_and_show_chart(fig_used, col2, "used_price_brand")
            
        st.subheader("Distribusi Harga berdasarkan Tahun Rilis")
        
        # Calculate average price by year for both conditions
        year_price_new = df[df['Kondisi'] == 'Baru'].groupby('Tahun Rilis')['Harga'].mean().reset_index()
        year_price_used = df[df['Kondisi'] == 'Bekas'].groupby('Tahun Rilis')['Harga'].mean().reset_index()
            
        col1, col2 = st.columns(2)
            
        with col1:
            fig_year_new = px.line(year_price_new, x='Tahun Rilis', y='Harga',
                                   title='Trend Harga Rata-rata Kamera Baru per Tahun',
                                   markers=True)
            create_and_show_chart(fig_year_new, col1, "new_price_year")
                
        with col2:
            fig_year_used = px.line(year_price_used, x='Tahun Rilis', y='Harga',
                                    title='Trend Harga Rata-rata Kamera Bekas per Tahun',
                                    markers=True)
            create_and_show_chart(fig_year_used, col2, "used_price_year")
            
        st.subheader("Distribusi Harga berdasarkan Format Kamera")
        col1, col2 = st.columns(2)

        with col1:
            format_price_new = df[df['Kondisi'] == 'Baru'].groupby('Format')['Harga'].mean().reset_index()
            fig_format_new = px.bar(format_price_new, x='Format', y='Harga',
                                    title='Rata-rata Harga Kamera Baru per Format',
                                    text=format_price_new['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))
            fig_format_new.update_traces(textposition='outside')
            create_and_show_chart(fig_format_new, col1, "new_price_format")
                
        with col2:
            format_price_used = df[df['Kondisi'] == 'Bekas'].groupby('Format')['Harga'].mean().reset_index()
            fig_format_used = px.bar(format_price_used, x='Format', y='Harga',
                                     title='Rata-rata Harga Kamera Bekas per Format',
                                     text=format_price_used['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))
            fig_format_used.update_traces(textposition='outside')
            create_and_show_chart(fig_format_used, col2, "used_price_format")
            
        st.subheader("Top 10 Kamera Termahal")
        col1, col2 = st.columns(2)
            
        with col1:
            top_new = df[df['Kondisi'] == 'Baru'].nlargest(10, 'Harga')
            fig_top_new = px.bar(top_new, x='Model', y='Harga',
                                 title='10 Kamera Baru Termahal',
                                 text=top_new['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')),
                                 hover_data=['Merek', 'Tahun Rilis'])
            fig_top_new.update_traces(textposition='outside')
            create_and_show_chart(fig_top_new, col1, "top_10_new")
                
        with col2:
            top_used = df[df['Kondisi'] == 'Bekas'].nlargest(10, 'Harga')
            fig_top_used = px.bar(top_used, x='Model', y='Harga',
                                  title='10 Kamera Bekas Termahal',
                                  text=top_used['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')), 
                                  hover_data=['Merek', 'Tahun Rilis'])
            fig_top_used.update_traces(textposition='outside')
            create_and_show_chart(fig_top_used, col2, "top_10_used")

    # Tab 2: Popularitas dan Tren Kamera
    with tabs[1]:
        st.subheader("Top 10 Merek dengan Model Terbanyak per Tahun")
        
        # Calculate number of models released by each brand per year
        brand_models_per_year = df.groupby(['Tahun Rilis', 'Merek'])['Model'].nunique().reset_index()
        top_brands = brand_models_per_year.nlargest(10, 'Model')
        
        fig_top_brands = px.bar(top_brands, x='Merek', y='Model',
                                color='Tahun Rilis',
                                title='10 Merek dengan Model Terbanyak per Tahun')
        create_and_show_chart(fig_top_brands, st, "top_brands_by_year")
        
        st.subheader("Distribusi Model Kamera Berdasarkan Tahun Rilis")
        
        # Count unique models per year
        models_per_year = df.groupby('Tahun Rilis')['Model'].nunique().reset_index()
        models_per_year.columns = ['Tahun Rilis', 'Jumlah Model']
        
        fig_year_dist = px.line(models_per_year.sort_values('Tahun Rilis'),
                                x='Tahun Rilis', y='Jumlah Model',
                                title='Jumlah Model Kamera yang Dirilis per Tahun',
                                markers=True)
        create_and_show_chart(fig_year_dist, st, "models_by_year")
        

    # Tab 3: Analisis Merek dan Kategori
    with tabs[2]:
        st.subheader("Distribusi Kamera Berdasarkan Merek")
        col1, col2 = st.columns(2)

        with col1:
            brand_new = df[df['Kondisi'] == 'Baru']['Merek'].value_counts()
            fig_brand_new = px.pie(values=brand_new.values,
                                   names=brand_new.index,
                                   title='Distribusi Merek Kamera Baru')
            create_and_show_chart(fig_brand_new, col1, "brand_dist_new")

        with col2:
            brand_used = df[df['Kondisi'] == 'Bekas']['Merek'].value_counts()
            fig_brand_used = px.pie(values=brand_used.values,
                                    names=brand_used.index,
                                    title='Distribusi Merek Kamera Bekas')
            create_and_show_chart(fig_brand_used, col2, "brand_dist_used")
        
        st.subheader("Proporsi Kategori Kamera")
        col1, col2 = st.columns(2)
        
        with col1:
            cat_new = df[df['Kondisi'] == 'Baru']['Kategori'].value_counts()
            fig_cat_new = px.pie(values=cat_new.values,
                                 names=cat_new.index,
                                 title='Proporsi Kategori Kamera Baru')
            create_and_show_chart(fig_cat_new, col1, "cat_dist_new")
            
        with col2:
            cat_used = df[df['Kondisi'] == 'Bekas']['Kategori'].value_counts()
            fig_cat_used = px.pie(values=cat_used.values,
                                  names=cat_used.index,
                                  title='Proporsi Kategori Kamera Bekas')
            create_and_show_chart(fig_cat_used, col2, "cat_dist_used")
            
        st.subheader("Rata-rata Harga Kamera Per Merek")
        avg_price = df.groupby(['Merek', 'Kondisi'])['Harga'].mean().reset_index()
            
        fig_avg = px.bar(avg_price, x='Merek', y='Harga',
                         color='Kondisi',
                         barmode='group',
                         title='Perbandingan Rata-rata Harga Kamera per Merek',
                         text=avg_price['Harga'].apply(lambda x: f"Rp{x:,.0f}".replace(',', '.')))
        fig_avg.update_traces(textposition='outside')
        create_and_show_chart(fig_avg, st, "avg_price_comparison")

    # Tab 4: Analisis Kamera Bekas vs Baru
    with tabs[3]:
        st.subheader("Perbandingan Kamera Bekas vs Baru")
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
        create_and_show_chart(fig_year_compare, st, "year_price_comparison")
        
        # Brand distribution comparison
        st.subheader("Perbandingan Distribusi Merek")
        brand_dist = pd.DataFrame({
            'Baru': df[df['Kondisi'] == 'Baru']['Merek'].value_counts(),
            'Bekas': df[df['Kondisi'] == 'Bekas']['Merek'].value_counts()
        }).fillna(0)
        fig_brand_compare = px.bar(brand_dist, barmode='group',
                                   title='Perbandingan Jumlah Model per Merek')
        create_and_show_chart(fig_brand_compare, st, "brand_model_comparison")

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
        if model:
            model_encoded = models_data['le_model'].transform([model])
            
            if condition == 'Baru':
                predictor = models_data['model_new']
                metrics = models_data['metrics_new']
                X_test, y_test = models_data['test_data']['new']
            else:
                predictor = models_data['model_used']
                metrics = models_data['metrics_used']
                X_test, y_test = models_data['test_data']['used']
                
            predicted_price = predictor.predict([[model_encoded[0]]])[0]
            
            st.success(f"Prediksi Harga: {format_price(predicted_price)}")
            
            st.subheader("Metrik Evaluasi Model")
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
                
            similar_models = df[
                (df['Kondisi'] == condition) &
                (df['Merek'] == model_data['Merek']) &
                (df['Kategori'] == model_data['Kategori']) &
                (abs(df['Tahun Rilis'] - model_data['Tahun Rilis']) <= 2)
            ]
            
            if len(similar_models) > 0:
                similar_encoded = models_data['le_model'].transform(similar_models['Model'])
                similar_predictions = predictor.predict(similar_encoded.reshape(-1, 1))
                
                local_metrics = {
                    'MAPE': mean_absolute_percentage_error(similar_models['Harga'], similar_predictions),
                    'MAE': mean_absolute_error(similar_models['Harga'], similar_predictions),
                    'RMSE': np.sqrt(mean_squared_error(similar_models['Harga'], similar_predictions)),
                    'R2': r2_score(similar_models['Harga'], similar_predictions)
                }
                
                st.subheader("Metrik untuk Kamera Similar")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAPE Lokal", f"{local_metrics['MAPE']*100:.2f}%")
                with col2:
                    st.metric("MAE Lokal", f"Rp{local_metrics['MAE']:,.0f}".replace(',', '.'))
                with col3:
                    rmse_millions = local_metrics['RMSE'] / 1_000_000
                    st.metric("RMSE Lokal", f"Rp{rmse_millions:.1f}M")
                with col4:
                    st.metric("R² Lokal", f"{local_metrics['R2']:.3f}")
        
            # Display similar cameras
            st.subheader("Kamera Similar")
            similar_cameras = similar_models[similar_models['Model'] != model].sort_values('Harga')
            
            if not similar_cameras.empty:
                st.dataframe(
                    similar_cameras[['Model', 'Kategori', 'Tahun Rilis', 'Harga']]
                    .head()
                )
            else:
                st.info("Tidak ditemukan kamera dengan spesifikasi serupa dalam database")

# CSS
st.markdown("""
    
    """, unsafe_allow_html=True)
