import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import sqlite3
import os
from datetime import datetime, timedelta
import uuid

# --- Pfade ---
MODEL_PATH = "model/model.h5"
UPLOAD_FOLDER = "uploads"
DB_PATH = "fundbuero.db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- DB Setup ---
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    filename TEXT,
    category TEXT,
    confidence REAL,
    timestamp TEXT,
    session_id TEXT
)
""")
conn.commit()

# --- Laden des Modells ---
model = load_model(MODEL_PATH)
CATEGORIES = ["Schere", "Flasche", "Federtasche"]

# --- Hilfsfunktionen ---
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def save_entry(filename, category, confidence, session_id):
    timestamp = datetime.now().isoformat()
    entry_id = str(uuid.uuid4())
    c.execute("INSERT INTO items VALUES (?, ?, ?, ?, ?, ?)",
              (entry_id, filename, category, confidence, timestamp, session_id))
    conn.commit()

def delete_old_entries():
    cutoff = datetime.now() - timedelta(weeks=2)
    c.execute("DELETE FROM items WHERE timestamp <= ?", (cutoff.isoformat(),))
    conn.commit()

# --- Streamlit Session ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

delete_old_entries()

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Gehe zu:", ["Gegenstand melden", "Gegenstand suchen"])

if choice == "Gegenstand melden":
    st.title("Gegenstand melden")
    upload_type = st.radio("Bildquelle wählen:", ["Hochladen", "Kamera"])

    image_file = None
    if upload_type == "Hochladen":
        image_file = st.file_uploader("Wähle ein Bild", type=["jpg", "jpeg", "png"])
    else:
        image_file = st.camera_input("Foto aufnehmen")

    if image_file:
        img = Image.open(image_file)
        img_array = preprocess_image(img)
        preds = model.predict(img_array)
        confidence = float(np.max(preds))
        category = CATEGORIES[int(np.argmax(preds))]

        if confidence >= 0.5:
            filename = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.png")
            img.save(filename)
            save_entry(filename, category, confidence, st.session_state.session_id)
            st.success(f"Gegenstand gespeichert: {category} ({confidence*100:.1f}%)")
        else:
            st.warning("Vorhersage zu unsicher. Bild nicht gespeichert.")

    # Eigene Einträge löschen
    st.subheader("Meine Einträge löschen")
    c.execute("SELECT id, filename, category FROM items WHERE session_id=?", (st.session_state.session_id,))
    user_items = c.fetchall()
    for item_id, filename, category in user_items:
        if st.button(f"Lösche {category}", key=item_id):
            c.execute("DELETE FROM items WHERE id=?", (item_id,))
            conn.commit()
            st.success(f"{category} gelöscht")

elif choice == "Gegenstand suchen":
    st.title("Gegenstand suchen")
    category_search = st.selectbox("Kategorie auswählen", CATEGORIES)
    c.execute("SELECT filename FROM items WHERE category=?", (category_search,))
    results = c.fetchall()
    if results:
        for (filename,) in results:
            st.image(filename, use_column_width=True)
    else:
        st.info("Keine Einträge gefunden.")
