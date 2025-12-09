import pandas as pd
import json
import os

# --- CONFIGURATION ---
INPUT_CSV = "conseils_emotions.csv"
OUTPUT_JSONL = "fine_tuning_data.jsonl"

print(" PRÉPARATION DES DONNÉES POUR FINE-TUNING (MISTRAL/GPT)...")

# 1. Chargement des données brutes
try:
    df = pd.read_csv(INPUT_CSV)
    print(f" Fichier CSV chargé : {len(df)} exemples trouvés.")
except FileNotFoundError:
    print(" Erreur : 'conseils_emotions.csv' introuvable.")
    exit()

# 2. Conversion au format Chat (Instruction Tuning)
# On crée des paires (User -> Assistant) pour apprendre au modèle comment réagir.
training_data = []

# On ajoute quelques variations pour enrichir le dataset (Data Augmentation simple)
templates = [
    "I feel {emotion}",
    "I am feeling very {emotion} today",
    "Why do I feel {emotion}?",
]

print(" Conversion et Augmentation des données...")

for index, row in df.iterrows():
    emotion = row['emotion'].lower()
    advice = row['advice']
    note = row['notes']
    
    # Construction de la réponse idéale de l'assistant
    # On combine le conseil et la note psy pour une réponse complète
    assistant_response = f"{advice} {note}"

    # Pour chaque émotion, on crée 3 exemples d'entraînement différents
    for template in templates:
        user_message = template.format(emotion=emotion)
        
        # Structure JSONL standard (OpenAI/Mistral format)
        entry = {
            "messages": [
                {"role": "system", "content": "You are MindCare, an empathetic mental health assistant."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response}
            ]
        }
        training_data.append(entry)

# 3. Sauvegarde en JSONL
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for entry in training_data:
        json.dump(entry, f)
        f.write('\n')

print(f" Fichier d'entraînement généré : {OUTPUT_JSONL}")
print(f" Nombre total de séquences d'entraînement : {len(training_data)}")
print("\n--- EXEMPLE D'UNE LIGNE GÉNÉRÉE ---")
print(json.dumps(training_data[0], indent=2))
print("-" * 30)
print(" Ce fichier est prêt à être envoyé sur la plateforme Mistral pour un Fine-Tuning.")