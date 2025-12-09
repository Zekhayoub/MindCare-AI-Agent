import pandas as pd
import joblib
import numpy as np
import os
from dotenv import load_dotenv

# Imports pour le RAG Vectoriel (Expert)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_mistralai import MistralAIEmbeddings
except ImportError:
    print(" Modules RAG manquants (pip install faiss-cpu langchain-mistralai)")

# --- CONFIGURATION ---
MODEL_PATH = 'models/LogisticRegression.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
ADVICE_DB_PATH = 'conseils_emotions.csv'
VECTORSTORE_PATH = 'vectorstore_psychology' # Dossier créé par build_rag.py

# Seuil pour considérer une émotion comme "secondaire"
SECONDARY_THRESHOLD = 0.10

LABEL_MAP = {
    0: 'Sadness', 1: 'Joy', 2: 'Love',
    3: 'Anger', 4: 'Fear', 5: 'Surprise'
}

# --- BASE DE DONNÉES GÉOGRAPHIQUE (RAG STRUCTURÉ) ---
LOCATIONS = {
    "sadness": {"name": "Parc de Bruxelles", "desc": "une promenade apaisante au grand air", "lat": 50.8454, "lon": 4.3642},
    "anger":   {"name": "Basic-Fit Gare Centrale", "desc": "une séance de sport pour évacuer la tension", "lat": 50.8452, "lon": 4.3594},
    "fear":    {"name": "Bibliothèque Royale", "desc": "un environnement calme et sécurisant", "lat": 50.8432, "lon": 4.3571},
    "joy":     {"name": "Grand-Place", "desc": "un lieu social pour célébrer ce moment", "lat": 50.8468, "lon": 4.3524},
    "love":    {"name": "Grand-Place (Soirée)", "desc": "une ambiance romantique et chaleureuse", "lat": 50.8468, "lon": 4.3524},
    "surprise":{"name": "Musée des Sciences", "desc": "de quoi nourrir votre curiosité", "lat": 50.8367, "lon": 4.3766}
}

class MindCareTools:
    def __init__(self):
        print(" Chargement des outils MindCare...")
        load_dotenv() # Pour charger la clé API si besoin ici
        
        # 1. Modèles ML
        try:
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.advice_df = pd.read_csv(ADVICE_DB_PATH)
            self.advice_df['emotion'] = self.advice_df['emotion'].str.strip().str.lower()
            print(" Modèles ML et CSV chargés.")
        except FileNotFoundError as e:
            print(f" ERREUR CRITIQUE : {e}")
            self.model = None
            self.advice_df = pd.DataFrame()

        # 2. RAG Vectoriel (Mémoire Longue)
        self.vector_db = None
        api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_KEY_1")
        
        if os.path.exists(VECTORSTORE_PATH) and api_key:
            try:
                print(" Chargement de la Base Vectorielle (Manuel Psy)...")
                embeddings = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
                # allow_dangerous_deserialization=True est requis en local pour FAISS
                self.vector_db = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
                print(f" Base Vectorielle chargée.")
            except Exception as e:
                print(f" Erreur chargement RAG : {e}")
        else:
            print(f" RAG non chargé (Dossier '{VECTORSTORE_PATH}' manquant ou pas de clé API).")

    def classify_emotion(self, text):
        """TOOL A: Analyse l'émotion (Principale + Secondaires)."""
        if self.model is None: return {"error": "Modèle non chargé"}

        vec_text = self.vectorizer.transform([text])
        probas = self.model.predict_proba(vec_text)[0]
        
        max_proba = np.max(probas)
        pred_index = np.argmax(probas)
        primary_emotion = LABEL_MAP.get(pred_index, "unknown")
        
        # Secondaires
        secondary_emotions = {}
        all_scores = {}
        for index, score in enumerate(probas):
            label = LABEL_MAP.get(index, "unknown")
            all_scores[label] = float(score)
            if label != primary_emotion and score >= SECONDARY_THRESHOLD:
                secondary_emotions[label] = float(score)

        # Incertitude
        threshold_main = 0.35
        final_emotion = primary_emotion
        is_ambiguous = False
        
        if max_proba < threshold_main:
            final_emotion = "unknown"
            is_ambiguous = True
        
        return {
            "emotion": final_emotion,
            "confidence": round(max_proba, 2),
            "is_ambiguous": is_ambiguous,
            "secondary_emotions": secondary_emotions,
            "all_scores": all_scores
        }

    def get_advice(self, emotion):
        """TOOL B: Conseil CSV."""
        if emotion == "unknown": return "Demandez des précisions.", "Clarification"
        if self.advice_df.empty: return "Erreur base de données.", "Error"

        row = self.advice_df[self.advice_df['emotion'] == emotion.lower()]
        if not row.empty:
            return row.iloc[0]['advice'], row.iloc[0]['notes']
        return f"Soutien général pour {emotion}.", "General support"

    def get_activity(self, emotion):
        """TOOL C: Activité Locale."""
        emotion_key = emotion.lower()
        if emotion_key in LOCATIONS:
            place = LOCATIONS[emotion_key]
            return f"Suggestion d'activité : {place['desc']} à {place['name']}."
        return "Aucune activité spécifique."

    def query_knowledge_base(self, query):
        """
        TOOL D (NOUVEAU - RAG EXPERT): Recherche sémantique dans le manuel de psychologie.
        Utile pour des questions complexes (ex: "Comment calmer une crise ?").
        """
        if self.vector_db is None:
            return "Base de connaissances indisponible."
        
        try:
            # Recherche des 2 passages les plus pertinents
            results = self.vector_db.similarity_search(query, k=2)
            knowledge = "\n\n".join([doc.page_content for doc in results])
            return f"INFO DU MANUEL CLINIQUE :\n{knowledge}"
        except Exception as e:
            return f"Erreur de recherche : {e}"

# --- TEST RAPIDE ---
if __name__ == "__main__":
    tools = MindCareTools()
    print("\n---  Test RAG Vectoriel ---")
    q = "technique respiration"
    print(f"Question : {q}")
    res = tools.query_knowledge_base(q)
    print(f"Réponse RAG : {res[:100]}...")