import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Keputusan Nelayan Berlayar",
    layout="centered"
)

st.title("🚤 Penentuan Keputusan Nelayan Berlayar")
st.write(
    "Pilih kondisi cuaca dan teknis kapal untuk mendapatkan "
    "rekomendasi keputusan berlayar."
)

# ===============================
# LOAD DATASET
# ===============================
file_path = "Weather-for-Boating-Activities.csv"
data = pd.read_csv(file_path)

# ===============================
# PREPROCESSING
# ===============================
label_encoders = {}
encoded_data = data.copy()

for col in data.columns:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = encoded_data.iloc[:, :-1]
y = encoded_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)
clf.fit(X_train, y_train)

accuracy = accuracy_score(y_test, clf.predict(X_test))

# ===============================
# MAPPING INDONESIA → DATASET
# ===============================
map_wind_wave = {
    "Rendah": "Low",
    "Sedang": "Medium",
    "Tinggi": "High"
}

map_weather = {
    "Cerah": "Sunny",
    "Berawan": "Cloudy",
    "Hujan": "Rainy"
}

map_day = {
    "Hari Kerja": "Weekday",
    "Akhir Pekan": "Weekend"
}

map_boat = {
    "Buruk": "Poor",
    "Sedang": "Average",   
    "Baik": "Good"
}

decision_id = {
    "Yes": "Ya",
    "No": "Tidak"
}

# ===============================
# FORM INPUT (BAHASA INDONESIA)
# ===============================
with st.form("form_prediksi"):
    wind_id = st.selectbox(
        "Kecepatan Angin",
        ["Rendah", "Sedang", "Tinggi"]
    )

    wave_id = st.selectbox(
        "Ketinggian Gelombang",
        ["Rendah", "Sedang", "Tinggi"]
    )

    weather_id = st.selectbox(
        "Kondisi Cuaca",
        ["Cerah", "Berawan", "Hujan"]
    )

    day_id = st.selectbox(
        "Hari",
        ["Hari Kerja", "Akhir Pekan"]
    )

    boat_id = st.selectbox(
        "Kondisi Teknis Kapal",
        ["Buruk", "Sedang", "Baik"]
    )

    submit = st.form_submit_button("Prediksi Keputusan")

# ===============================
# PROSES PREDIKSI
# ===============================
if submit:
    raw_input = {
        "Wind Speed": map_wind_wave[wind_id],
        "Wave Height": map_wind_wave[wave_id],
        "Weather": map_weather[weather_id],
        "Day of the Week": map_day[day_id],
        "Boat Technical Condition": map_boat[boat_id]
    }

    input_values = []
    for col in X.columns:
        encoded_value = label_encoders[col].transform(
            [raw_input[col]]
        )[0]
        input_values.append(encoded_value)

    prediction = clf.predict([input_values])
    decision_raw = label_encoders[y.name].inverse_transform(prediction)[0]
    decision = decision_id[decision_raw]

    probabilities = clf.predict_proba([input_values])[0]
    class_labels = label_encoders[y.name].inverse_transform(clf.classes_)

    prob_dict = {
        decision_id[class_labels[i]]: round(probabilities[i] * 100, 2)
        for i in range(len(class_labels))
    }

    # ===============================
    # OUTPUT
    # ===============================
    st.divider()
    st.subheader("Data Kondisi")

    st.write(f"**Kecepatan Angin:** {wind_id}")
    st.write(f"**Ketinggian Gelombang:** {wave_id}")
    st.write(f"**Kondisi Cuaca:** {weather_id}")
    st.write(f"**Hari:** {day_id}")
    st.write(f"**Kondisi Teknis Kapal:** {boat_id}")

    st.divider()

    if decision == "Ya":
        st.success("Keputusan yang disarankan: **Berlayar**")
    else:
        st.error("Keputusan yang disarankan: **Tidak Berlayar**")

    st.subheader("Probabilitas Keputusan")
    for label, prob in prob_dict.items():
        st.write(f"**{label}:** {prob}%")

