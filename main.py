import streamlit as st
import numpy as np
from PIL import Image
import sqlite3
import pandas as pd
from datetime import datetime
import os
import requests

# ⚠️ OpenCV léger
import cv2

# IA
from transformers import pipeline

# ===================== CONFIG =====================
st.set_page_config(page_title="Gestion Incidents Classe", layout="wide")
st.title("📸 Gestion intelligente des incidents")

# ===================== DOSSIER =====================
if not os.path.exists("incidents_photos"):
    os.makedirs("incidents_photos")

# ===================== DATABASE =====================
def init_db():
    conn = sqlite3.connect('incidents.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    classe TEXT,
                    description TEXT,
                    photo_path TEXT,
                    analyse_ia TEXT,
                    objets TEXT,
                    meteo TEXT,
                    gravite TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

def ajouter_incident(date, classe, desc, photo_path, analyse, objets, meteo, gravite):
    conn = sqlite3.connect('incidents.db')
    c = conn.cursor()
    c.execute("INSERT INTO incidents (date, classe, description, photo_path, analyse_ia, objets, meteo, gravite) VALUES (?,?,?,?,?,?,?,?)",
              (date, classe, desc, photo_path, analyse, objets, meteo, gravite))
    conn.commit()
    conn.close()

# ===================== IA =====================
@st.cache_resource
def load_model():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

detector = load_model()

def analyser_image(image):
    labels = [
        "classe propre",
        "classe sale",
        "désordre",
        "chaises renversées",
        "tables en désordre",
        "déchets au sol",
        "classe organisée",
        "dégradation",
        "incivilité"
    ]

    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    result = detector(image, candidate_labels=labels)
    top = result[0]

    return top['label'], top['score']

# 🔍 simulation objets (léger cloud-friendly)
def detecter_objets_simple(analyse_label):
    mapping = {
        "chaises renversées": ["chaise"],
        "tables en désordre": ["table"],
        "déchets au sol": ["déchets"],
        "désordre": ["chaise", "table"],
    }
    return mapping.get(analyse_label, ["inconnu"])

# ===================== METEO =====================
API_KEY = "TA_CLE_API"  # ⚠️ mets ta clé ici

def get_weather(city="Paris"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric&lang=fr"
        data = requests.get(url).json()
        return f"{data['weather'][0]['description']} - {data['main']['temp']}°C"
    except:
        return "Indisponible"

# ===================== UI =====================
tab1, tab2, tab3 = st.tabs(["📸 Nouvel Incident", "📋 Historique", "📊 Dashboard"])

# ===================== TAB 1 =====================
with tab1:
    st.subheader("Détection intelligente")

    classe = st.selectbox("Classe", ["6A","6B","5A","5B","4A","4B","3A","3B"])
    description = st.text_area("Description")

    ville = st.text_input("Ville (météo)", "Paris")

    photo_file = st.camera_input("📷 Photo")

    if photo_file:
        img_array = np.frombuffer(photo_file.getvalue(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        st.image(img, channels="BGR")

        if st.button("🔍 Analyse complète"):
            with st.spinner("Analyse IA..."):

                label, score = analyser_image(img)
                objets = detecter_objets_simple(label)
                meteo = get_weather(ville)

                st.success(f"Analyse : {label} ({score:.1%})")
                st.write("Objets détectés :", ", ".join(objets))
                st.write("Météo :", meteo)

                # gravité intelligente
                if score > 0.7 and label in ["désordre","classe sale","dégradation"]:
                    gravite = "⚠️ Élevée"
                elif score > 0.5:
                    gravite = "⚡ Moyenne"
                else:
                    gravite = "🟢 Faible"

                st.info(f"Gravité : {gravite}")

                if st.button("💾 Enregistrer"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"incidents_photos/{timestamp}.jpg"
                    cv2.imwrite(path, img)

                    ajouter_incident(
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        classe,
                        description,
                        path,
                        label,
                        ", ".join(objets),
                        meteo,
                        gravite
                    )

                    st.success("Enregistré")

# ===================== TAB 2 =====================
with tab2:
    conn = sqlite3.connect('incidents.db')
    df = pd.read_sql_query("SELECT * FROM incidents ORDER BY date DESC", conn)
    conn.close()

    if not df.empty:
        st.dataframe(df)
    else:
        st.info("Aucun incident")

# ===================== TAB 3 =====================
with tab3:
    if 'df' in locals() and not df.empty:

        st.metric("Total", len(df))
        st.metric("Graves", len(df[df['gravite'].str.contains("Élevée")]))
        st.metric("Classes", df['classe'].nunique())

        st.bar_chart(df['classe'].value_counts())

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("Infos")
    st.write("App IA + météo")

    if st.button("🗑️ Reset"):
        conn = sqlite3.connect('incidents.db')
        conn.execute("DELETE FROM incidents")
        conn.commit()
        conn.close()
        st.success("Reset OK")
        st.rerun()