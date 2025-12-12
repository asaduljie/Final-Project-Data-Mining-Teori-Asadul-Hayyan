
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
st.write("Aplikasi clustering & prediksi Spending Score menggunakan Random Forest.")

uploaded = st.file_uploader("📂 Unggah dataset Mall_Customers.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📌 Pratinjau Dataset")
    st.dataframe(df)

    # Clustering
    st.subheader("🎯 Clustering Pelanggan (K-Means)")
    X = df[['Age','Annual Income (k$)','Spending Score (1-100)']]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = km.fit_predict(Xs)

    chart = alt.Chart(df).mark_circle(size=80).encode(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        color='Cluster:N',
        tooltip=['Age','Annual Income (k$)','Spending Score (1-100)','Cluster']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    st.subheader("📊 Statistik Tiap Cluster")
    st.write(df.groupby('Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean())

    st.subheader("🤖 Prediksi Spending Score (Random Forest)")
    Xr = df[['Age','Annual Income (k$)']]
    yr = df['Spending Score (1-100)']

    X_train, X_test, y_train, y_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    st.write("📉 RMSE:", mean_squared_error(y_test, pred, squared=False))
    st.write("📈 R²:", r2_score(y_test, pred))

    st.subheader("🧮 Prediksi Berdasarkan Input Pengguna")
    age = st.number_input("Umur Pelanggan", 1, 100, 30)
    income = st.number_input("Pendapatan Tahunan (k$)", 1, 200, 50)

    if st.button("🔮 Prediksi Spending Score"):
        hasil = rf.predict([[age, income]])
        st.success(f"✨ Perkiraan Spending Score: {hasil[0]:.2f}")
