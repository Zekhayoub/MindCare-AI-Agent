import os
import sys
from dotenv import load_dotenv

# --- IMPORTS RAG ---
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_mistralai import MistralAIEmbeddings
    from langchain_community.vectorstores import FAISS
    print(" Modules RAG chargés.")
except ImportError as e:
    print(f" ERREUR : Il manque des modules. {e}")
    print("Faites : pip install faiss-cpu langchain-community")
    sys.exit(1)

# 1. Configuration
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_KEY_1")

if not api_key:
    print(" Clé API manquante. Vérifiez votre fichier .env")
    sys.exit(1)

print(" Démarrage de l'indexation RAG...")

# 2. Chargement du "Livre"
try:
    loader = TextLoader("psychology_guide.txt", encoding="utf-8")
    documents = loader.load()
    print(f" Document chargé : {len(documents[0].page_content)} caractères.")
except FileNotFoundError:
    print(" ERREUR : Le fichier 'psychology_guide.txt' est introuvable !")
    sys.exit(1)

# 3. Découpage (Pour que l'IA lise paragraphe par paragraphe)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print(f" Découpage effectué : {len(docs)} passages extraits.")

# 4. Vectorisation & Stockage
print(" Calcul des vecteurs (Embeddings)... Patientez...")
try:
    # On transforme le texte en mathématiques
    embeddings = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
    db = FAISS.from_documents(docs, embeddings)
    
    # 5. Sauvegarde sur le disque
    db.save_local("vectorstore_psychology")
    print("\n SUCCÈS ! La mémoire a été sauvegardée dans le dossier 'vectorstore_psychology'.")
    print(" Vous pouvez passer à l'intégration dans l'agent.")

except Exception as e:
    print(f" Erreur lors de la vectorisation : {e}")
