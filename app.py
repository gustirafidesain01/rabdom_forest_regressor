import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Regression.csv')
    return data

# Preprocessing function
def preprocess_data(data):
    # Convert categorical variables to numerical
    data['sex'] = data['sex'].map({'male': 0, 'female': 1})
    data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
    data = pd.get_dummies(data, columns=['region'], drop_first=True)
    return data

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Prediksi Biaya Asuransi", layout="wide", initial_sidebar_state="expanded")

    # Sidebar for navigation
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Prediksi", "Penjelasan Algoritma"])

    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)

    if page == "Prediksi":
        # Add a title and description
        st.title("Prediksi Biaya Asuransi")
        st.write("Aplikasi ini memprediksi biaya asuransi berdasarkan berbagai faktor. Silakan masukkan detail di bawah ini untuk mendapatkan prediksi.")
        
        # Display dataset
        st.subheader("Dataset")
        st.dataframe(data)  # Display the entire dataset

        # Display statistics of the dataset
        st.subheader("Statistik Dataset")
        st.write(data.describe())  # Display descriptive statistics

        # Plotting the distribution of the target variable 'charges'
        st.subheader("Distribusi Biaya Asuransi")
        plt.figure(figsize=(10, 6))
        sns.histplot(data['charges'], bins=30, kde=True)
        plt.title('Distribusi Biaya Asuransi')
        plt. xlabel('Biaya Asuransi')
        plt.ylabel('Frekuensi')
        st.pyplot(plt)

        # Features and target variable
        X = data.drop('charges', axis=1)
        y = data['charges']

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Evaluasi Model")
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")
        st.write(f"**Akurasi Model:** {r2 * 100:.2f}%")  # Display accuracy as a percentage

        # Feature importance
        feature_importances = model.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        st.subheader("Pentingnya Fitur")
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title('Pentingnya Fitur dalam Prediksi')
        st.pyplot(plt)

        # User input for prediction
        st.subheader("Buat Prediksi")
        user_input = {}
        for feature in X.columns:
            if feature == 'age':
                user_input[feature] = st.number_input(feature, min_value=18, max_value=64, value=30, step=1)  # Set age range
            elif feature in ['bmi', 'children']:
                user_input[feature] = st.number_input(feature, min_value=0.0, value=30.0, step=1.0)
            elif feature == 'sex':
                user_input[feature] = st.selectbox(
                    feature,
                    options=[0, 1],
                    format_func=lambda x: '0' if x == 0 else '1'
                )
            elif feature == 'smoker':
                user_input[feature] = st.selectbox(
                    feature,
                    options=[0, 1],
                    format_func=lambda x: '0' if x == 0 else '1'
                )
            else:
                user_input[feature] = st.selectbox(feature, options=data[feature].unique())

        # Convert user input to DataFrame
        user_input_df = pd.DataFrame(user_input, index=[0])

        # Predict charges
        if st.button("Prediksi Biaya", key="predict"):
            prediction = model.predict(user_input_df)
            st.success(f"Prediksi Biaya Asuransi: **{prediction[0]:.2f}**")

        # Plotting actual vs predicted charges
        st.subheader("Perbandingan Biaya Asuransi Aktual dan Prediksi")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Biaya Asuransi Aktual')
        plt.ylabel('Biaya Asuransi Prediksi')
        plt.title('Perbandingan Biaya Asuransi Aktual dan Prediksi')
        st.pyplot(plt)

    elif page == "Penjelasan Algoritma":
        st.title("Penjelasan Algoritma")
        st.write("""
            **Random Forest Regressor** adalah metode pembelajaran ensemble yang bekerja dengan membangun beberapa pohon keputusan selama pelatihan dan mengeluarkan rata-rata prediksi dari pohon-pohon individu. 
            
            Berikut adalah beberapa alasan mengapa Random Forest adalah pilihan yang tepat untuk masalah ini:
            - **Kekuatan**: Lebih tahan terhadap overfitting dibandingkan dengan satu pohon keputusan.
            - **Menangani Non-linearitas**: Dapat menangkap hubungan kompleks dalam data.
            - **Pentingnya Fitur**: Memberikan wawasan tentang fitur mana yang paling penting untuk prediksi.
            - **Versatilitas**: Dapat menangani data numerik dan kategorikal dengan efektif.
        """)
        
        st.write("""
            **Cara Kerja Random Forest:**
            1. **Bootstrap Sampling**: Mengambil sampel acak dari dataset dengan pengembalian untuk membangun beberapa pohon keputusan.
            2 . **Pembentukan Pohon**: Setiap pohon dibangun dengan memilih subset acak dari fitur untuk menentukan split pada setiap node.
            3. **Voting**: Setelah semua pohon dibangun, prediksi dilakukan dengan cara voting (untuk klasifikasi) atau rata-rata (untuk regresi) dari semua pohon.
            
            **Keuntungan Menggunakan Random Forest:**
            - **Robustness**: Mampu menangani data yang hilang dan outlier.
            - **Interpretabilitas**: Memberikan informasi tentang fitur mana yang paling berpengaruh terhadap prediksi.
            - **Fleksibilitas**: Dapat digunakan untuk berbagai jenis masalah, baik klasifikasi maupun regresi.
            - **Pengurangan Varians**: Mengurangi varians model dengan menggabungkan hasil dari banyak pohon.

            **Visualisasi Random Forest:**
            Untuk memberikan pemahaman yang lebih baik, berikut adalah diagram yang menunjukkan bagaimana Random Forest bekerja:
        """)
        
        

        st.write("""
            **Contoh Aplikasi Random Forest:**
            - **Prediksi Harga**: Digunakan dalam industri real estate untuk memprediksi harga rumah.
            - **Deteksi Penipuan**: Mendeteksi transaksi yang mencurigakan dalam sistem perbankan.
            - **Analisis Risiko**: Memprediksi risiko kesehatan berdasarkan data demografis dan medis.

            Dengan menggunakan Random Forest dalam aplikasi ini, kami dapat memberikan prediksi biaya asuransi yang lebih akurat dan dapat diandalkan berdasarkan berbagai faktor yang mempengaruhi biaya tersebut.
        """)

        # Unique and attractive design for the explanation page
        st.markdown("""
            <style>
            .stTitle {
                color: #4CAF50;
                font-size: 2.5em;
                text-align: center;
            }
            .stSubheader {
                color: #2196F3;
                font-size: 1.5em;
                text-align: left;
            }
            .stMarkdown {
                font-size: 1.2em;
                line-height: 1.5;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<h1 class='stTitle'>Penjelasan Algoritma</h1>", unsafe_allow_html=True)
        st.markdown("<h2 class='stSubheader'>Mengapa Memilih Random Forest?</h2>", unsafe_allow_html=True)
        st.markdown("<div class='stMarkdown'>Random Forest adalah metode yang kuat dan fleksibel untuk memprediksi biaya asuransi. Dengan kemampuannya untuk menangani data yang kompleks dan memberikan wawasan tentang fitur penting, algoritma ini menjadi pilihan utama dalam analisis data.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()