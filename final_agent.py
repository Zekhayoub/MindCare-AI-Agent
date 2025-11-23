# --- DÉBUT DU FICHIER final_agent.py ---
import os
import sys
import getpass
from dotenv import load_dotenv

# --- 1. CHARGEMENT ET IMPORTS ---
print(" Chargement des modules...")
load_dotenv()

try:
    from langchain_mistralai import ChatMistralAI
    # On utilise create_react_agent qui est plus robuste pour ce cas
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import tool
    from langchain_core.prompts import PromptTemplate
    from mindcare_tools import MindCareTools
    print("✅ Modules chargés.")
except ImportError as e:
    print(f" ERREUR IMPORT : {e}")
    sys.exit(1)

# --- 2. TEST DES CLÉS API (FAILOVER) ---
print(" Vérification des clés API...")
potential_keys = [os.getenv("MISTRAL_KEY_1"), os.getenv("MISTRAL_KEY_2"), os.getenv("MISTRAL_API_KEY")]
valid_keys = [k for k in potential_keys if k and len(k) > 10]

if not valid_keys:
    print(" Aucune clé dans .env")
    manual_key = getpass.getpass(" Entrez une clé maintenant : ").strip()
    valid_keys.append(manual_key)

active_llm = None
for i, key in enumerate(valid_keys):
    try:
        # On utilise une temperature basse pour que le modèle suive bien le format ReAct
        test_llm = ChatMistralAI(api_key=key, model="mistral-large-latest", temperature=0.1)
        test_llm.invoke("Hi")
        print(f"✅ Clé #{i+1} valide.")
        active_llm = test_llm
        os.environ["MISTRAL_API_KEY"] = key
        break
    except Exception:
        print(f" Clé #{i+1} invalide.")

if not active_llm:
    print(" AUCUNE CLÉ VALIDE. Arrêt.")
    sys.exit(1)

# --- 3. OUTILS ---
print(" Chargement des outils...")
try:
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
        return str(MINDCARE_TOOLS.get_advice(emotion))
    except Exception as e:
        return f"Error: {e}"

tools = [emotion_classifier, advice_lookup]

# --- 4. PROMPT & AGENT (ReAct) ---
print(" Assemblage de l'Agent ReAct...")

# Ce prompt est CRUCIAL pour que ReAct fonctionne. Ne pas modifier.
template = """
You are MINDCARE, a supportive mental health assistant.

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes Action: the action to take, should be one of [{tool_names}] 

Action Input: the input to the action Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No Final Answer: [your response here]


Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

try:
    agent = create_react_agent(active_llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True, # Corrige automatiquement les erreurs de format
        max_iterations=5
    )
    print("✅ Agent assemblé avec succès.")
except Exception as e:
    print(f" Erreur Assemblage : {e}")
    sys.exit(1)

# --- 5. BOUCLE PRINCIPALE ---
if __name__ == "__main__":
    print("\n" + "="*40)
    print("✨ MINDCARE EST EN LIGNE (Mode ReAct) ✨")
    print("Tapez 'quit' pour sortir.")
    print("="*40 + "\n")
    
    # Mémoire simple sous forme de texte pour ReAct
    chat_history_str = ""
    
    while True:
        try:
            user_input = input("Vous: ")
            if user_input.lower() in ["quit", "exit"]:
                print("MindCare: Au revoir !")
                break
            
            if not user_input.strip(): continue
            
            print("   (MindCare réfléchit...)")
            
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history_str
            })
            
            output = response['output']
            print(f"\nMindCare: {output}\n")
            
            # Mise à jour de la mémoire texte
            chat_history_str += f"\nHuman: {user_input}\nAI: {output}"
            
        except Exception as e:
            print(f" Erreur conversation : {e}")

# --- FIN DU FICHIER final_agent.py ---