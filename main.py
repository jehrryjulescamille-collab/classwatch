import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import pandas as pd
from datetime import datetime
import os

# CLIP pour l'analyse IA
from transformers import pipeline

# ===================== CONFIGURATION =====================
st.set_page_config(page_title="Gestion Incidents Classe", layout="wide")
st.title("📸 Gestion des Incidents en Classe")

# Dossier pour sauvegarder les photos
if not os.path.exists("incidents_photos"):
    os.makedirs("incidents_photos")

# ===================== BASE DE DONNÉES =====================
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
                    gravite TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

def ajouter_incident(date, classe, desc, photo_path, analyse, gravite):
    conn = sqlite3.connect('incidents.db')
    c = conn.cursor()
    c.execute("INSERT INTO incidents (date, classe, description, photo_path, analyse_ia, gravite) VALUES (?,?,?,?,?,?)",
              (date, classe, desc, photo_path, analyse, gravite))
    conn.commit()
    conn.close()

# ===================== ANALYSE IA (CLIP) =====================
@st.cache_resource
def load_clip():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

detector = load_clip()

def analyser_image(image):
    # Labels en français pour meilleure précision
    labels = [
        "classe bien ordonnée et propre",
        "classe désordonnée",
        "classe sale avec poussière partout",
        "chaises renversées",
        "tables en désordre",
        "déchets par terre",
        "classe propre et organisée",
        "problème de propreté",
        "incivilité ou dégradation"
    ]
    
    # Convertir en PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    result = detector(image, candidate_labels=labels)
    
    # Meilleur résultat + score
    top = result[0]
    return f"{top['label'].capitalize()} ({top['score']:.1%})", top['score']

# ===================== INTERFACE =====================
tab1, tab2, tab3 = st.tabs(["📸 Nouvel Incident", "📋 Historique", "📊 Tableau de Bord"])

with tab1:
    st.subheader("Prendre une photo de l'incident")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        classe = st.selectbox("Classe / Salle", 
                            ["6A", "6B", "5A", "5B", "4A", "4B", "3A", "3B", "Autre"])
        description = st.text_area("Description de l'incident", 
                                 placeholder="Ex: Chaises renversées après la récré...")
        
        # Prise de photo
        photo_file = st.camera_input("📷 Prendre une photo")
        
        if photo_file is not None:
            # Sauvegarder l'image
            img_array = np.frombuffer(photo_file.getvalue(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            photo_path = f"incidents_photos/incident_{timestamp}.jpg"
            cv2.imwrite(photo_path, img)
            
            st.image(img, channels="BGR", caption="Photo capturée")
            
            # Analyse IA
            if st.button("🔍 Analyser avec l'IA", type="primary"):
                with st.spinner("Analyse en cours..."):
                    analyse_ia, score = analyser_image(img)
                    st.success(f"**Analyse IA :** {analyse_ia}")
                    
                    # Détermination gravité
                    if score > 0.7 and ("désordonnée" in analyse_ia or "sale" in analyse_ia or "renversées" in analyse_ia):
                        gravite = "⚠️ Élevée"
                    elif score > 0.5:
                        gravite = "⚡ Moyenne"
                    else:
                        gravite = "🟢 Faible"
                    
                    st.info(f"**Gravité estimée :** {gravite}")
                    
                    # Sauvegarde
                    if st.button("💾 Enregistrer l'incident"):
                        ajouter_incident(
                            datetime.now().strftime("%Y-%m-%d %H:%M"),
                            classe,
                            description,
                            photo_path,
                            analyse_ia,
                            gravite
                        )
                        st.success("Incident enregistré avec succès !")

with tab2:
    st.subheader("Historique des Incidents")
    
    conn = sqlite3.connect('incidents.db')
    df = pd.read_sql_query("SELECT * FROM incidents ORDER BY date DESC", conn)
    conn.close()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        # Filtre par classe
        classe_filter = st.selectbox("Filtrer par classe", ["Toutes"] + sorted(df['classe'].unique()))
        if classe_filter != "Toutes":
            df = df[df['classe'] == classe_filter]
            st.dataframe(df)
    else:
        st.info("Aucun incident enregistré pour le moment.")

with tab3:
    st.subheader("Tableau de Bord")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Incidents", len(df) if 'df' in locals() else 0)
    with col2:
        if 'df' in locals() and not df.empty:
            st.metric("Incidents Élevés", len(df[df['gravite'].str.contains('Élevée')]))
    with col3:
        st.metric("Classes Concernées", df['classe'].nunique() if 'df' in locals() and not df.empty else 0)
    
    if 'df' in locals() and not df.empty:
        st.bar_chart(df['classe'].value_counts())

# Sidebar
with st.sidebar:
    st.header("À propos")
    st.write("Application de gestion des incidents en classe avec détection IA.")
    st.caption("Utilise OpenAI CLIP pour l'analyse zéro-shot.")
    
    if st.button("🗑️ Supprimer tous les incidents (attention)"):
        if st.checkbox("Confirmer la suppression"):
            conn = sqlite3.connect('incidents.db')
            conn.execute("DELETE FROM incidents")
            conn.commit()
            conn.close()
            st.success("Base de données vidée.")
            st.rerun()