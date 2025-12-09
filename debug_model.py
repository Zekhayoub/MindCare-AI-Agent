import joblib
import pandas as pd

print("🕵️ EXAMEN DU CERVEAU DE L'IA...")

# 1. Chargement
try:
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model = joblib.load('models/LogisticRegression.pkl')
    print(" Fichiers chargés.")
except:
    print(" Erreur : Fichiers introuvables.")
    exit()

# 2. Vérification du vocabulaire (Est-ce qu'il connait "not" ?)
vocab = vectorizer.vocabulary_
print(f"\n Taille du vocabulaire : {len(vocab)} mots")

mots_cles = ["not", "no", "never", "happy"]
print("\n Vérification des mots-clés :")
for mot in mots_cles:
    if mot in vocab:
        print(f"    '{mot}' est présent (ID: {vocab[mot]})")
    else:
        print(f"    '{mot}' a été SUPPRIMÉ ! (Le problème est ici)")

# 3. Test de prédiction mathématique brute
phrase = "I feel not happy"
print(f"\n Test avec la phrase : '{phrase}'")

# On regarde les N-grams (groupes de mots)
vec = vectorizer.transform([phrase])
print(f"   -> L'IA voit {vec.nnz} éléments (mots ou groupes de mots).")

# On affiche ce qu'elle voit vraiment
feature_names = vectorizer.get_feature_names_out()
print("   -> Détails de ce que l'IA détecte :")
for col_index in vec.indices:
    print(f"      - '{feature_names[col_index]}'")

# Prédiction
proba = model.predict_proba(vec)[0]
classes = model.classes_
print("\n Scores calculés :")
for i, emotion in enumerate(classes):
    print(f"   - {emotion}: {proba[i]:.4f}")

gagnant = classes[proba.argmax()]
print(f"\n Résultat final : {gagnant}")