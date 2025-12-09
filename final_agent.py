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
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import tool
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    from mindcare_tools import MindCareTools
    print(" Modules chargés.")
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
        # Temperature 0.2 : Créativité faible pour respecter les consignes strictes
        test_llm = ChatMistralAI(api_key=key, model="mistral-large-latest", temperature=0.2)
        test_llm.invoke("Hi")
        print(f" Clé #{i+1} valide.")
        active_llm = test_llm
        os.environ["MISTRAL_API_KEY"] = key
        break
    except Exception:
        print(f" Clé #{i+1} invalide.")

if not active_llm:
    print(" AUCUNE CLÉ VALIDE. Arrêt.")
    sys.exit(1)

# --- 3. DÉFINITION DES OUTILS (LES 4 PILIERS) ---
print(" Connexion aux outils...")
try:
    MINDCARE_TOOLS = MindCareTools()
except Exception as e:
    print(f" Erreur Outils : {e}")
    sys.exit(1)

@tool
def emotion_classifier(text: str) -> str:
    """Useful to identify the user's emotion. Returns emotion name and confidence."""
    try:
        return str(MINDCARE_TOOLS.classify_emotion(text))
    except Exception as e:
        return f"Error: {e}"

@tool
def advice_lookup(emotion: str) -> str:
    """Useful to get a quick supportive tip based on an emotion (e.g., 'sadness', 'joy')."""
    try:
        return str(MINDCARE_TOOLS.get_advice(emotion))
    except Exception as e:
        return f"Error: {e}"

@tool
def activity_recommendation(emotion: str) -> str:
    """Useful to suggest a specific real-world place in Brussels (Park, Gym...) based on emotion."""
    try:
        return str(MINDCARE_TOOLS.get_activity(emotion))
    except Exception as e:
        return f"Error: {e}"

@tool
def knowledge_retriever(query: str) -> str:
    """
    Useful for deep psychological questions or "How-to" questions (e.g. "How to breathe?", "Why am I angry?").
    It searches in a Clinical Psychology Manual using Vector RAG.
    """
    try:
        return str(MINDCARE_TOOLS.query_knowledge_base(query))
    except Exception as e:
        return f"Error: {e}"

# Liste complète des 4 outils
tools = [emotion_classifier, advice_lookup, activity_recommendation, knowledge_retriever]

# --- 4. PROMPT & AGENT (ReAct Expert) ---
print(" Assemblage de l'Agent Expert...")

template = """
You are MINDCARE, an advanced mental health assistant.
Your tone must be warm, professional, and deeply empathetic.

TOOLS AVAILABLE:
----------------
{tools}

FORMAT INSTRUCTIONS (ReAct):
----------------------------
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final response to the human

FEW-SHOT EXAMPLES (How you should behave):
------------------------------------------
Question: "I feel completely lost and alone."
Thought: The user expresses deep sadness. I must first identify the emotion.
Action: emotion_classifier
Action Input: "I feel completely lost and alone"
Observation: {{'emotion': 'Sadness', 'confidence': 0.85}}
Thought: Emotion is Sadness. I will get advice and suggest an activity.
Action: advice_lookup
Action Input: "sadness"
Observation: "Reach out to a friend..."
Thought: I will also check for a local place.
Action: activity_recommendation
Action Input: "sadness"
Observation: "Suggestion: Parc de Bruxelles..."
Final Answer: I hear how heavy things feel right now. You are not alone. It is important to reach out... [Advice]. If you feel up to it, a walk in the Parc de Bruxelles might offer a moment of peace.

Question: "I am having a panic attack, how do I breathe?"
Thought: The user is in acute distress and asks for a specific technique. I need deep clinical knowledge.
Action: knowledge_retriever
Action Input: "how to breathe during a panic attack"
Observation: "Technique de la respiration carrée (Box Breathing): Inspirer 4s..."
Thought: I have the specific technique. I will guide the user through it step-by-step.
Final Answer: I am here with you. Let's try the Box Breathing technique together. 1. Inhale for 4 seconds... [Instructions].

Question: "I ate a sandwich."
Thought: This statement is neutral. I will check for hidden emotions just in case.
Action: emotion_classifier
Action Input: "I ate a sandwich"
Observation: {{'emotion': 'unknown', 'confidence': 0.15}}
Thought: Confidence is too low. No advice needed. I should ask for clarification.
Final Answer: That sounds like a nice lunch! How are you feeling otherwise today?

------------------------------------------
Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

try:
    agent = create_react_agent(active_llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=6
    )
    print(" Agent assemblé avec succès.")
except Exception as e:
    print(f" Erreur Assemblage : {e}")
    sys.exit(1)

# --- 5. BOUCLE PRINCIPALE ---
if __name__ == "__main__":
    print("\n" + "="*40)
    print(" MINDCARE EST EN LIGNE (Full RAG + Expert) ")
    print("Tapez 'quit' pour sortir.")
    print("="*40 + "\n")
    
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
            
            chat_history_str += f"\nHuman: {user_input}\nAI: {output}"
            
        except Exception as e:
            print(f" Erreur conversation : {e}")
# --- FIN DU FICHIER final_agent.py ---