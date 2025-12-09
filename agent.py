import os
import sys
import getpass
import traceback
from dotenv import load_dotenv

# --- CHARGEMENT ENVIRONNEMENT ---
load_dotenv()

print(" Chargement des modules...")
try:
    from langchain_mistralai import ChatMistralAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import tool
    from langchain_core.prompts import PromptTemplate
    from mindcare_tools import MindCareTools
    print(" Modules chargés.")
except ImportError as e:
    print(f" Erreur Import : {e}")
    sys.exit(1)

# --- 1. SYSTÈME DE MULTI-CLÉS (FAILOVER) ---
print("\n Test des clés API Mistral...")

# Liste des clés à tester (depuis le .env)
potential_keys = [
    os.getenv("MISTRAL_KEY_1"),
    os.getenv("MISTRAL_KEY_2"),
    os.getenv("MISTRAL_API_KEY") # Au cas où une clé unique traîne
]

# On nettoie la liste (enlève les vides)
valid_keys = [k for k in potential_keys if k and len(k) > 10]

# Si aucune clé trouvée dans le .env, on demande manuellement
if not valid_keys:
    print(" Aucune clé trouvée dans le fichier .env")
    manual_key = getpass.getpass(" Entrez une clé manuellement : ").strip()
    valid_keys.append(manual_key)

active_llm = None

# BOUCLE DE TEST DES CLÉS
for index, key in enumerate(valid_keys):
    print(f"    Tentative avec la Clé #{index + 1}...", end=" ")
    try:
        # On tente une connexion simple
        test_llm = ChatMistralAI(api_key=key, model="mistral-large-latest", temperature=0.2)
        # On envoie un "ping" (un message vide ou très court) pour vérifier que la clé marche
        test_llm.invoke("Ping")
        
        # Si ça passe sans erreur, on garde cette configuration
        print(" SUCCÈS !")
        active_llm = test_llm
        # On définit la variable globale pour que les autres outils LangChain soient contents
        os.environ["MISTRAL_API_KEY"] = key 
        break # On sort de la boucle, on a trouvé une clé qui marche
    except Exception as e:
        print(f" ÉCHEC.")
        # print(f"      Raison : {e}") # Décommenter pour voir l'erreur technique

if active_llm is None:
    print("\n ERREUR FATALE : Aucune clé API ne fonctionne.")
    print("   -> Vérifiez votre fichier .env ou vos crédits Mistral.")
    sys.exit(1)

# --- 2. OUTILS ---
try:
    print("🔌 Connexion aux outils...")
    MINDCARE_TOOLS = MindCareTools()
except Exception as e:
    print(f" Erreur Outils : {e}")
    sys.exit(1)

@tool
def emotion_classifier(text: str) -> str:
    """
    Useful to identify the user's emotion.
    Returns a text description of the emotion and confidence.
    """
    try:
        return str(MINDCARE_TOOLS.classify_emotion(text))
    except Exception as e:
        return f"Error: {e}"

@tool
def advice_lookup(emotion: str) -> str:
    """
    Useful to get advice based on an emotion (e.g., 'sadness', 'joy').
    Input must be a single emotion word.
    """
    try:
        if "unknown" in emotion.lower():
            return "No advice available for unknown emotion."
        return str(MINDCARE_TOOLS.get_advice(emotion))
    except Exception as e:
        return f"Error: {e}"

tools = [emotion_classifier, advice_lookup]

# --- 3. PROMPT ReAct ---
template = """
You are MINDCARE, an empathetic mental health assistant.

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:"""

