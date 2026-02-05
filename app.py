import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, _tree
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

# ===============================
# CUSTOM BACKGROUND & STYLE
# ===============================
st.markdown(
    """
    <style>
    /* Background halaman */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    /* Container utama */
    section.main > div {
        max-width: 650px;
        margin: auto;
    }

    /* Card form */
    div[data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.10);
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        cursor: pointer !important;
    }

    /* Judul */
    h1, h2, h3 {
        color: #ffffff;
        text-align: center;
    }

    /* Teks & label */
    label, p {
        color: #f1f1f1 !important;
    }

    /* Input */
    .stSelectbox div {
        border-radius: 10px;
    }

    div[data-testid="stFormSubmitButton"] button {
        background-color: #2f4752 !important;
        border-radius: 12px !important;
        height: 48px !important;
        width: 220px !important;
        border: white 1px solid !important;
    }

    div[data-testid="stFormSubmitButton"] button span {
        color: white !important;
        opacity: 1 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.title("🚤 Penentuan Keputusan Nelayan Berlayar")
st.write(
    "Pilih kondisi cuaca dan teknis kapal untuk mendapatkan "
    "rekomendasi keputusan berlayar."
)

# ===============================
# LOAD DATASET
# ===============================
data = pd.read_csv("Weather-for-Boating-Activities.csv")

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
# ANALISIS FAKTOR PENTING
# ===============================
importance_df = pd.DataFrame({
    "Faktor": X.columns,
    "Nilai Kepentingan": clf.feature_importances_
}).sort_values(by="Nilai Kepentingan", ascending=False)

feature_translation = {
    "Wind Speed": "Kecepatan Angin",
    "Wave Height": "Ketinggian Gelombang",
    "Weather": "Kondisi Cuaca",
    "Day of the Week": "Hari",
    "Boat Technical Condition": "Kondisi Teknis Kapal"
}

importance_df["Faktor"] = importance_df["Faktor"].map(feature_translation)
importance_df.index = range(1, len(importance_df) + 1)

# ===============================
# MAPPING INPUT
# ===============================
map_wind_wave = {"Rendah": "Low", "Sedang": "Medium", "Tinggi": "High"}
map_weather = {"Cerah": "Sunny", "Berawan": "Cloudy", "Hujan": "Rainy"}
map_day = {"Hari Kerja": "Weekday", "Akhir Pekan": "Weekend"}
map_boat = {"Buruk": "Poor", "Sedang": "Average", "Baik": "Good"}
decision_id = {"Yes": "Ya", "No": "Tidak"}

# ===============================
# FORM INPUT
# ===============================
with st.form("form_prediksi"):
    wind_id = st.selectbox("Kecepatan Angin", ["Rendah", "Sedang", "Tinggi"])
    wave_id = st.selectbox("Ketinggian Gelombang", ["Rendah", "Sedang", "Tinggi"])
    weather_id = st.selectbox("Kondisi Cuaca", ["Cerah", "Berawan", "Hujan"])
    day_id = st.selectbox("Hari", ["Hari Kerja", "Akhir Pekan"])
    boat_id = st.selectbox("Kondisi Teknis Kapal", ["Buruk", "Sedang", "Baik"])
    submit = st.form_submit_button("🔍 Prediksi Keputusan")

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

    input_values = [
        label_encoders[col].transform([raw_input[col]])[0]
        for col in X.columns
    ]

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
    # OUTPUT PREDIKSI
    # ===============================
    st.divider()
    st.subheader("📋 Data Kondisi")
    st.write(f"Kecepatan Angin: {wind_id}")
    st.write(f"Ketinggian Gelombang: {wave_id}")
    st.write(f"Kondisi Cuaca: {weather_id}")
    st.write(f"Hari: {day_id}")
    st.write(f"Kondisi Teknis Kapal: {boat_id}")

    st.divider()
    st.subheader("📊 Probabilitas Keputusan")

    labels = list(prob_dict.keys())
    sizes = list(prob_dict.values())

    fig, ax = plt.subplots(figsize=(1, 1), facecolor="none")
    ax.set_facecolor("none")
    wedges, texts, autotexts = ax.pie(
    sizes,
    autopct="%1.1f%%",
    startangle=90,
    radius=0.6,
    pctdistance=0.7,
    textprops={"fontsize": 4, "color": "white"}
    )

    ax.axis("equal")

    ax.legend(
        wedges,
        labels,
        title="Keputusan",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=4,
        title_fontsize=4,
    )

    st.pyplot(fig, transparent=True)

    # ===============================
    # INTERPRETASI DECISION TREE
    # ===============================
    node_indicator = clf.decision_path([input_values])
    leaf_id = clf.apply([input_values])

    factors_used = []
    for node_id in node_indicator.indices:
        if node_id == leaf_id[0]:
            continue
        feature_index = clf.tree_.feature[node_id]
        if feature_index != _tree.TREE_UNDEFINED:
            factors_used.append(feature_translation[X.columns[feature_index]])

    factors_used = list(dict.fromkeys(factors_used))
    prob_main = prob_dict[decision]

    # ===============================
    # FAKTOR PENTING GLOBAL
    # ===============================
    st.divider()
    st.subheader("📌 Faktor Penting")
    st.markdown("""
    <style>
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 15px;
        color: white;
        background: transparent;
    }

    .custom-table th {
        background-color: rgba(255,255,255,0.08);
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
    }

    .custom-table td {
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.15);
        text-align: center;
        background: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

    html_table = importance_df.to_html(classes="custom-table", index=True, escape=False)
    st.markdown(html_table, unsafe_allow_html=True)

    # ===============================
    # SARAN
    # ===============================
    st.divider()
    st.subheader("📝 Saran")

    top_factors = importance_df.head(3)["Faktor"].tolist()

    saran_text = (
        f"Berdasarkan hasil prediksi, pada kondisi ini "
        f"machine memberikan probabilitas keputusan **{decision} untuk belayar sebesar {prob_main}%**. "
        f"Keputusan tersebut terutama dipengaruhi oleh faktor "
        f"**{', '.join(factors_used)}**.\n\n"
        f"Secara umum, faktor yang paling dominan dalam sistem ini adalah "
        f"**{top_factors[0]}**, diikuti oleh **{top_factors[1]}** dan "
        f"**{top_factors[2]}**.\n\n"
        f"Oleh karena itu, disarankan agar nelayan selalu memperhatikan "
        f"faktor-faktor utama tersebut sebelum berlayar."
    )

    st.write(saran_text)
