import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Analisis Pelanggan Mall", page_icon="🛍️", layout="wide")

st.title("🛍️ Analisis Pelanggan Mall")
st.write("""
Aplikasi ini menampilkan **segmentasi pelanggan (clustering)** dan  
**prediksi Spending Score (regresi)** sebagai dua analisis yang terpisah.
""")

uploaded = st.file_uploader("📂 Unggah dataset Mall_Customers.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📌 Pratinjau Dataset")
    st.dataframe(df)

    st.subheader("🎯 Segmentasi Pelanggan (K-Means Clustering)")

    X_cluster = df[['Age','Annual Income (k$)']]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_cluster)

    km = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = km.fit_predict(Xs)

    chart = alt.Chart(df).mark_circle(size=80).encode(
        x='Annual Income (k$)',
        y='Age',
        color='Cluster:N',
        tooltip=['Age','Annual Income (k$)','Spending Score (1-100)','Cluster']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    st.subheader("📊 Profil Rata-Rata Tiap Cluster")
    cluster_stats = (
        df.groupby('Cluster')[['Age','Annual Income (k$)']]
        .mean()
        .round(2)
        .reset_index()
    )

    st.dataframe(cluster_stats)

    st.info("""
    🔍 **Catatan:**  
    Clustering dilakukan hanya menggunakan **Age** dan **Annual Income**  
    untuk menghindari kebocoran data (data leakage).
    """)
    
    st.subheader("🤖 Prediksi Spending Score (Random Forest Regression)")

    Xr = df[['Age','Annual Income (k$)']]
    yr = df['Spending Score (1-100)']

    X_train, X_test, y_train, y_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    st.write(f"📉 **RMSE**: {rmse:.2f}")
    st.write(f"📈 **R² Score**: {r2:.3f}")

    st.subheader("🧮 Prediksi Berdasarkan Input Pengguna")

    age = st.number_input("Umur Pelanggan", 1, 100, 30)
    income = st.number_input("Pendapatan Tahunan (k$)", 1, 200, 50)

    if st.button("🔮 Prediksi Spending Score"):
        hasil = rf.predict([[age, income]])
        score = hasil[0]

        if score < 40:
            komentar = "Pelanggan memiliki tingkat pembelanjaan rendah."
        elif score < 70:
            komentar = "Pelanggan memiliki tingkat pembelanjaan sedang."
        else:
            komentar = "Pelanggan dengan tingkat pembelanjaan tinggi (loyal/premium)."

        st.success(f"✨ Perkiraan Spending Score: {score:.2f}")
        st.info(f"📌 Interpretasi: {komentar}")
