import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

# 1. Charger les clés
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_KEY_1")

print("---  DIAGNOSTIC MISTRAL ---")

if not api_key:
    print(" ERREUR : Aucune clé API trouvée dans le fichier .env")
    exit()

print(f" Clé détectée : {api_key[:5]}... (Masquée)")

# 2. Tentative de connexion simple
print(" Connexion aux serveurs Mistral...")
try:
    llm = ChatMistralAI(
        api_key=api_key,
        model="mistral-large-latest",
        temperature=0
    )
    
    # 3. La question test
    print(" Envoi du message : 'Are you Mistral?'")
    response = llm.invoke("Are you Mistral?")
    
    print("\n SUCCÈS ! Réponse reçue :")
    print("-" * 30)
    print(response.content)
    print("-" * 30)
    print(" Tout est opérationnel pour la Phase 4.")

except Exception as e:
    print(f"\n ÉCHEC DE CONNEXION : {e}")
    print("Vérifiez votre connexion internet ou votre crédit sur console.mistral.ai")