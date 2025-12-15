import streamlit as st
import pandas as pd
import altair as alt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Analisis Pelanggan Mall",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Analisis Pelanggan Mall")
st.write("""
Aplikasi ini melakukan **analisis gabungan**:
- **Clustering** untuk segmentasi pelanggan
- **Regresi** untuk memprediksi Spending Score  
berdasarkan **satu input pengguna**.
""")

# ===============================
# UPLOAD DATA
# ===============================
uploaded = st.file_uploader(
    "📂 Unggah dataset Mall_Customers.csv",
    type=["csv"]
)

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📌 Pratinjau Dataset")
    st.dataframe(df, use_container_width=True)

    # ===============================
    # ====== CLUSTERING MODEL ======
    # ===============================
    X_cluster = df[['Age', 'Annual Income (k$)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # ===============================
    # ====== REGRESSION MODEL ======
    # ===============================
    Xr = df[['Age', 'Annual Income (k$)']]
    yr = df['Spending Score (1-100)']

    X_train, X_test, y_train, y_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # ===============================
    # VISUALISASI CLUSTER
    # ===============================
    st.subheader("🎯 Visualisasi Segmentasi Pelanggan")

    chart = alt.Chart(df).mark_circle(size=90).encode(
        x='Annual Income (k$)',
        y='Age',
        color='Cluster:N',
        tooltip=[
            'Age',
            'Annual Income (k$)',
            'Spending Score (1-100)',
            'Cluster'
        ]
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    # ===============================
    # ===== INPUT GABUNGAN =====
    # ===============================
    st.subheader("🧠 Analisis Pelanggan Baru (Clustering + Regresi)")

    age = st.number_input("Umur Pelanggan", 1, 100, 30)
    income = st.number_input("Pendapatan Tahunan (k$)", 1, 200, 50)

    if st.button("🔍 Analisis Pelanggan"):
        # ---- CLUSTER PREDICTION ----
        input_df = pd.DataFrame({
            'Age': [age],
            'Annual Income (k$)': [income]
        })

        input_scaled = scaler.transform(input_df)
        cluster_result = kmeans.predict(input_scaled)[0]

        # ---- REGRESSION PREDICTION ----
        spending_pred = rf.predict([[age, income]])[0]

        # ===============================
        # OUTPUT HASIL
        # ===============================
        st.success("✅ Hasil Analisis Pelanggan")

        st.write(f"🧩 **Masuk Cluster** : **Cluster {cluster_result}**")
        st.write(f"💰 **Prediksi Spending Score** : **{spending_pred:.2f}**")

        # ===============================
        # INTERPRETASI GABUNGAN
        # ===============================
        if spending_pred < 40:
            level = "rendah"
        elif spending_pred < 70:
            level = "sedang"
        else:
            level = "tinggi"

        if cluster_result == 0:
            profil = "pelanggan usia muda dengan pendapatan menengah"
        elif cluster_result == 1:
            profil = "pelanggan berpendapatan tinggi"
        elif cluster_result == 2:
            profil = "pelanggan usia lebih tua dengan pendapatan menengah ke bawah"
        else:
            profil = "pelanggan usia menengah dengan pendapatan tinggi"

        st.info(
            f"📌 **Kesimpulan:** Pelanggan ini termasuk {profil} "
            f"dengan tingkat pembelanjaan **{level}**."
        )

    # ===============================
    # METRIK REGRESI (OPSIONAL)
    # ===============================
    st.subheader("📊 Evaluasi Model Regresi")

    y_pred = rf.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    st.write(f"📉 RMSE : {rmse:.2f}")
    st.write(f"📈 R² Score : {r2:.3f}")
