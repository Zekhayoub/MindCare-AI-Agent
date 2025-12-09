import os
import getpass

# --- 1. IMPORTS ROBUSTES (Correction des erreurs "ImportError") ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool

# Bloc de secours pour trouver les fonctions LangChain où qu'elles soient
try:
    # Chemin standard
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    print(" Import standard échoué, utilisation des chemins alternatifs...")
    try:
        # Chemins alternatifs pour certaines versions
        from langchain.agents.agent import AgentExecutor
        from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
    except ImportError:
        # Dernier recours (souvent nécessaire pour les versions très récentes/anciennes)
        from langchain.agents import AgentExecutor
        from langchain.agents import create_tool_calling_agent

# Importation de VOTRE boîte à outils (Phase 2)
from mindcare_tools import MindCareTools

# --- 2. CONFIGURATION API (Sécurité) ---
# Si la clé n'est pas trouvée, on la demande dans le terminal
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass(" Entrez votre Clé API Mistral (elle sera masquée) : ")

# --- 3. INITIALISATION DES OUTILS ---
print(" Initialisation des outils MindCare...")
MINDCARE_TOOLS = MindCareTools()

@tool
def emotion_classifier(text: str) -> dict:
    """
    Identifies the primary emotion (Sadness, Joy, Fear, Anger, Love, Surprise) 
    and returns its confidence score. If confidence is low, returns 'unknown'.
    """
    return MINDCARE_TOOLS.classify_emotion(text)

@tool
def advice_lookup(emotion: str) -> str:
    """
    Retrieves psychological advice based on a specific emotion (e.g., 'sadness').
    """
    if emotion.lower() == "unknown":
        return "No advice available for unknown emotion."
    return MINDCARE_TOOLS.get_advice(emotion)

tools = [emotion_classifier, advice_lookup]

# --- 4. LE CERVEAU (MISTRAL CLOUD) ---
print(" Connexion au Cloud Mistral AI (Large)...")

# On utilise le modèle le plus puissant disponible avec votre budget
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.5
)

# --- 5. LE PROMPT (Personnalité de l'Agent) ---
SYSTEM_PROMPT = """
You are MINDCARE, an empathetic mental health assistant.
Your goal is to help the user identify their emotions and provide supportive advice.

### PROCESS:
1. ALWAYS use the tool 'emotion_classifier' on the user's input first.
2. Analyze the result:
   - If 'unknown' (low confidence): DO NOT use advice_lookup. Ask a gentle clarification question.
   - If emotion detected: Use 'advice_lookup' to get the advice.
3. **FINAL ANSWER:** - Synthesize the advice into a warm, human-like response. 
   - Do not copy-paste; rephrase naturally in the user's language (French or English).
   - Be supportive and professional.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# --- 6. CRÉATION DE L'AGENT ---
try:
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
except NameError as e:
    print(f" Erreur fatale : Les fonctions de l'agent n'ont pas pu être importées. {e}")
    exit()

# --- 7. INTERFACE DE TEST INTERACTIVE ---
if __name__ == "__main__":
    print("\n --- MINDCARE (Mistral Edition) EST PRÊT --- ")
    print("Tapez 'quit' pour sortir.")
    
    chat_history = []
    
    while True:
        try:
            user_input = input("\nVous: ")
            if user_input.lower() in ["quit", "exit"]:
                print("MindCare: Prenez soin de vous. Au revoir !")
                break
            
            # Lancement de l'agent avec mémoire
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            ai_response = response["output"]
            print(f"\nMindCare: {ai_response}")
            
            # Mise à jour de la mémoire
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=ai_response))
            
        except Exception as e:
            print(f" Erreur : {e}")