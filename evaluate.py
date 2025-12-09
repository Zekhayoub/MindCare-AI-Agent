import os
import sys
import pandas as pd
import time
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Import de votre agent MindCare
try:
    from final_agent import agent_executor as mindcare_agent
except ImportError:
    print(" Erreur : Impossible d'importer 'final_agent.py'.")
    sys.exit(1)

# --- CONFIGURATION ---
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_KEY_1")

if not api_key:
    print(" Clé API manquante.")
    sys.exit(1)

print(" INITIALISATION DU TEST A/B (SCÉNARIOS COMPLEXES)...")

# 1. CHALLENGER (IA Standard)
baseline_llm = ChatMistralAI(api_key=api_key, model="mistral-large-latest", temperature=0.5)

# 2. JUGE (Evaluateur)
judge_llm = ChatMistralAI(api_key=api_key, model="mistral-large-latest", temperature=0)

# --- DATASET DE TEST "TUEUR DE BASELINE" ---
test_cases = [
    {
        "input": "I feel lazy and sad. What is the '5 minute rule' for depression?",
        "type": "RAG Savoir (Précis)",
        "expected": "Doit citer le manuel : 'Dites-vous que vous ne ferez l'activité que 5 minutes'. La Baseline va inventer une autre règle."
    },
    {
        "input": "I am so angry I need to punch something. Give me a specific gym name in Brussels Center.",
        "type": "RAG Lieu (Local)",
        "expected": "Doit nommer 'Basic-Fit Gare Centrale'. La Baseline ne connait pas votre base de données locale."
    },
    {
        "input": "I am panic. Tell me exactly how many seconds to inhale according to the 'Box Breathing' technique.",
        "type": "RAG Technique",
        "expected": "Doit répondre '4 secondes' (info du fichier texte). La Baseline pourrait donner un autre chiffre."
    },
    {
        "input": "I feel not happy at all today.",
        "type": "Négation Complexe",
        "expected": "Doit détecter la Tristesse (Sadness) malgré le mot 'happy'."
    },
    {
        "input": "I want to stimulate my curiosity. Where can I go in Brussels?",
        "type": "RAG Lieu (Culture)",
        "expected": "Doit proposer le 'Musée des Sciences' (lié à l'émotion Surprise/Curiosité)."
    }
]

# --- FONCTION DE NOTATION ---
def run_judge(user_input, ai_response, expected):
    judge_template = """
    Role: Expert AI Auditor.
    Task: Compare the AI response against the Expected Behavior.
    
    Context:
    - User Question: "{input}"
    - Expected Behavior (Gold Standard): "{expected}"
    - AI Response: "{response}"
    
    Grading Criteria:
    1/5: Fails completely (Wrong info, dangerous, or misses the specific data required).
    2/5: Vague or generic answer (e.g. "Go to a park" instead of a specific name).
    3/5: Correct but lacks empathy or detail.
    4/5: Good answer with specific info.
    5/5: Perfect answer. Cites the specific rule/location/time AND is empathetic.
    
    Output: Just the integer number (1-5).
    """
    prompt = ChatPromptTemplate.from_template(judge_template)
    try:
        res = (prompt | judge_llm).invoke({"input": user_input, "response": ai_response, "expected": expected})
        import re
        match = re.search(r'\d', res.content)
        return int(match.group()) if match else 3
    except:
        return 3

# --- BOUCLE ---
results = []
print(f"\n Démarrage du duel sur {len(test_cases)} rounds...\n")

for i, case in enumerate(test_cases):
    u_in = case["input"]
    print(f" Round {i+1}: {u_in[:40]}...")
    
    # Baseline
    print("   🔹 Baseline...", end="\r")
    try:
        base_resp = baseline_llm.invoke(f"User: {u_in}").content
    except: base_resp = "Error"
    s_base = run_judge(u_in, base_resp, case["expected"])
    print(f"   🔹 Baseline: {s_base}/5")

    # MindCare
    print("    MindCare...", end="\r")
    try:
        # On reset l'historique
        res = mindcare_agent.invoke({"input": u_in, "chat_history": ""})
        mind_resp = res['output']
    except Exception as e:
        mind_resp = str(e)
    s_mind = run_judge(u_in, mind_resp, case["expected"])
    print(f"    MindCare: {s_mind}/5")
    print("-" * 30)

    results.append({
        "Type": case["type"],
        "Input": u_in,
        "Baseline Score": s_base,
        "MindCare Score": s_mind,
        "Gain": s_mind - s_base
    })

# --- ANALYSE ---
df = pd.DataFrame(results)
avg_base = df["Baseline Score"].mean()
avg_mind = df["MindCare Score"].mean()
improvement = ((avg_mind - avg_base) / avg_base) * 100 if avg_base > 0 else 0

print("\n" + "="*40)
print(" RÉSULTATS DU DUEL")
print("="*40)
print(df[["Type", "Baseline Score", "MindCare Score", "Gain"]])

print(f"\n Moyenne Baseline : {avg_base:.2f}/5")
print(f" Moyenne MindCare : {avg_mind:.2f}/5")
print(f" AMÉLIORATION : +{improvement:.1f}%")

df.to_csv("evaluation_results.csv", index=False)