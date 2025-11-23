import pandas as pd
import joblib
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/LogisticRegression.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
ADVICE_DB_PATH = 'conseils_emotions.csv'

# Mapping inverse (Chiffre -> Mot)
LABEL_MAP = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

class MindCareTools:
    def __init__(self):
        print(" Chargement des outils MindCare...")
        try:
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.advice_df = pd.read_csv(ADVICE_DB_PATH)
            self.advice_df['emotion'] = self.advice_df['emotion'].str.strip().str.lower()
            print("✅ Modèle et Base de conseils chargés avec succès.")
        except FileNotFoundError as e:
            print(f" ERREUR CRITIQUE : Fichier manquant. {e}")
            self.model = None
            self.advice_df = pd.DataFrame()

    def classify_emotion(self, text):
        """
        TOOL A: Analyse l'émotion et retourne TOUS les scores.
        """
        if self.model is None:
            return {"error": "Modèle non chargé"}

        # 1. Vectorisation
        vec_text = self.vectorizer.transform([text])
        
        # 2. Prédiction des probabilités
        probas = self.model.predict_proba(vec_text)[0]
        
        # --- NOUVEAU : Récupérer tous les scores ---
        all_scores = {}
        for index, score in enumerate(probas):
            label = LABEL_MAP.get(index, "unknown")
            all_scores[label] = float(score) # On garde le score brut
            
        # 3. Récupération du gagnant
        max_proba = np.max(probas)
        pred_index = np.argmax(probas)
        emotion_label = LABEL_MAP.get(pred_index, "unknown")
        
        # 4. Logique d'Incertitude
        threshold = 0.20
        is_ambiguous = False
        
        if max_proba < threshold:
            emotion_label = "unknown"
            is_ambiguous = True
        
        return {
            "emotion": emotion_label,
            "confidence": round(max_proba, 2),
            "is_ambiguous": is_ambiguous,
            "all_scores": all_scores  # On ajoute le détail ici
        }

    def get_advice(self, emotion):
        """ TOOL B: Cherche un conseil. """
        if emotion == "unknown":
            return "Je ne suis pas sûr de bien comprendre votre émotion. Pouvez-vous m'en dire plus ?", "Ask for clarification"
        
        if self.advice_df.empty:
            return "Base de conseils non disponible.", "Error"

        row = self.advice_df[self.advice_df['emotion'] == emotion.lower()]
        
        if not row.empty:
            advice = row.iloc[0]['advice']
            note = row.iloc[0]['notes']
            return advice, note
        else:
            return f"Je comprends que vous ressentez : {emotion}.", "General support"

# --- TEST INTERACTIF COMPLET ---
if __name__ == "__main__":
    tools = MindCareTools()
    
    print("\n---  TEST INTERACTIF AVEC DÉTAILS ---")
    print("Tapez 'exit' pour quitter.")
    print("-------------------------------------------")
    
    while True:
        try:
            user_input = input("\nVous: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            # 1. Test Tool A
            result = tools.classify_emotion(user_input)
            emotion = result.get('emotion')
            conf = result.get('confidence')
            all_scores = result.get('all_scores', {})
            
            print(f" [Tool A] Emotion Dominante: {emotion.upper()} (Confiance: {conf:.0%})")
            
            # --- AFFICHAGE DES POURCENTAGES ---
            print(" Détail des probabilités :")
            # On trie du plus grand au plus petit pour la lisibilité
            sorted_scores = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)
            
            for label, score in sorted_scores:
                barre = "█" * int(score * 20) # Petite barre visuelle
                print(f"   - {label.ljust(10)} : {score:.1%}  {barre}")
            # ----------------------------------

            # 2. Test Tool B
            if emotion != 'unknown':
                conseil, note = tools.get_advice(emotion)
                print(f"\n [Tool B] Conseil : {conseil}")
            else:
                print("\n [Tool B] Pas de conseil (Score trop faible)")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erreur : {e}")