import streamlit as st
import time
import pandas as pd
import altair as alt
import tiktoken 
from langchain_core.messages import HumanMessage, AIMessage

# --- IMPORTS BACKEND ---
try:
    from final_agent import agent_executor
    from mindcare_tools import MindCareTools, LOCATIONS 
except ImportError:
    st.error(" Fichiers manquants. Assurez-vous d'être dans le bon dossier.")
    st.stop()

tools_instance = MindCareTools()

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="MINDCARE AI",
    page_icon="🧠",
    layout="wide"
)

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .stButton button {width: 100%; border-radius: 8px;}
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- COULEURS ---
EMOTION_COLORS = {
    "Joy": "#FFD700", "Love": "#FF69B4", "Surprise": "#FFA500",
    "Unknown": "#808080", "Fear": "#9370DB", "Sadness": "#1E90FF", "Anger": "#FF4500"
}

# --- INITIALISATION MÉMOIRE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = {k: 0 for k in EMOTION_COLORS.keys()}
if "emotion_timeline" not in st.session_state:
    st.session_state.emotion_timeline = []
if "show_kpi" not in st.session_state:
    st.session_state.show_kpi = False
if "total_co2" not in st.session_state:
    st.session_state.total_co2 = 0.0


def get_emotion_score(emotion_name):
    mapping = {
        "joy": 1.0, "love": 1.0, "surprise": 0.5, "unknown": 0.0,
        "fear": -0.5, "sadness": -1.0, "anger": -1.0
    }
    return mapping.get(emotion_name.lower(), 0.0)

def calculate_co2(text_input, text_output):
    """Calcule l'empreinte carbone EXACTE."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = len(encoding.encode(text_input + text_output))
        return round(tokens * 0.0002, 5)
    except:
        return 0.0

# ==================================================
#  ZONE PRINCIPALE - CHAT
# ==================================================

st.title("🧠 MindCare AI")
st.markdown("##### *Votre compagnon de soutien émotionnel intelligent*")
st.info("💡 **Info:** Je suis connecté à Mistral Large et j'utilise un modèle de Régression Logistique pour analyser vos émotions.")

# Affichage Historique
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

# Saisie
user_input = st.chat_input("Exprimez ce que vous ressentez...")

if user_input:
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Analyse Technique
    try:
        raw_analysis = tools_instance.classify_emotion(user_input)
        detected_emotion = raw_analysis.get('emotion', 'unknown').lower()
        confidence = raw_analysis.get('confidence', 0)
        # On récupère bien les secondaires ici
        secondary = raw_analysis.get('secondary_emotions', {})
        
        cap_emotion = detected_emotion.capitalize()
        if cap_emotion in st.session_state.emotion_log:
            st.session_state.emotion_log[cap_emotion] += 1
        elif "Unknown" in st.session_state.emotion_log:
             st.session_state.emotion_log["Unknown"] += 1
            
        score = get_emotion_score(detected_emotion)
        st.session_state.emotion_timeline.append(
            {"Step": len(st.session_state.emotion_timeline) + 1, "Score": score, "Emotion": cap_emotion}
        )
    except Exception:
        detected_emotion = "unknown"
        confidence = 0
        secondary = {}

    # Réponse IA
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        with st.spinner("Analyse en cours..."):
            try:
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                ai_response = response["output"]
                
                # Simulation écriture
                full_response = ""
                for chunk in ai_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                st.session_state.chat_history.append(AIMessage(content=ai_response))
                
                # --- CALCUL GREEN AI ---
                cost = calculate_co2(user_input, ai_response)
                st.session_state.total_co2 += cost
                
                # --- ZONE DEBUG AMÉLIORÉE (CORRECTION ICI) ---
                with st.expander("🔍 Analyse Technique & Impact"):
                    
                    # Ligne 1 : Les indicateurs principaux
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Émotion ML", f"{detected_emotion.upper()}", f"{confidence:.0%}")
                    with c2:
                        # Score de positivité réintégré !
                        st.metric("Positivité", f"{get_emotion_score(detected_emotion)}")
                    with c3:
                        st.metric("Coût Carbone", f"{cost} g")
                    
                    st.divider()
                    
                    # Ligne 2 : Les émotions secondaires avec barres de progression
                    st.markdown("**Nuances émotionnelles (Secondaires > 10%) :**")
                    if secondary:
                        for emo, score_val in secondary.items():
                            col_txt, col_bar = st.columns([1, 3])
                            with col_txt:
                                st.write(f"**{emo}**")
                            with col_bar:
                                # Barre de progression visuelle
                                st.progress(float(score_val), text=f"{score_val:.1%}")
                    else:
                        st.caption("Aucune émotion secondaire significative détectée.")

            except Exception as e:
                st.error(f"Erreur de connexion : {e}")
                if st.button("Recharger"):
                    st.rerun()

# ==================================================
#  BARRE LATÉRALE (DASHBOARD & KPI)
# ==================================================
with st.sidebar:
    st.title("Tableau de Bord")
    
    if st.button("🗑️ Nouvelle Session", type="primary"):
        st.session_state.chat_history = []
        st.session_state.emotion_log = {k: 0 for k in EMOTION_COLORS.keys()}
        st.session_state.emotion_timeline = []
        st.session_state.total_co2 = 0.0
        st.session_state.show_kpi = False
        st.rerun()
        
    st.divider()
    
    # 1. ANALYSE COMPARATIVE
    st.subheader("1️⃣ Point de départ")
    initial_mood = st.slider("Comment vous sentez-vous (0-10) ?", 0, 10, 5, key="initial_mood")
    
    st.divider()

    # 2. GRAPHIQUES
    if sum(st.session_state.emotion_log.values()) > 0:
        dom_emotion = max(st.session_state.emotion_log, key=st.session_state.emotion_log.get)
    else:
        dom_emotion = "-"
        
    st.subheader("📊 Analyse temps réel")
    col1, col2 = st.columns(2)
    col1.metric("Messages", len(st.session_state.emotion_timeline))
    col2.metric("Dominante", dom_emotion)
    
    st.metric("🌿 Impact Carbone Total", f"{st.session_state.total_co2:.4f} gCO2")

    if len(st.session_state.emotion_timeline) > 0:
        df_time = pd.DataFrame(st.session_state.emotion_timeline)
        line = alt.Chart(df_time).mark_line(interpolate='monotone', color='gray').encode(
            x=alt.X('Step', axis=alt.Axis(tickMinStep=1)),
            y=alt.Y('Score', scale=alt.Scale(domain=[-1.5, 1.5]))
        )
        points = alt.Chart(df_time).mark_circle(size=100).encode(
            x='Step',
            y='Score',
            color=alt.Color('Emotion', scale=alt.Scale(domain=list(EMOTION_COLORS.keys()), range=list(EMOTION_COLORS.values()))),
            tooltip=['Step', 'Emotion', 'Score']
        )
        rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y')
        st.altair_chart(line + points + rule, use_container_width=True)
    else:
        st.info("En attente de données...")

    st.divider()

    # 3. ANALYSE COMPARATIVE (BILAN)
    st.subheader("2️⃣ Bilan & Action")
    
    if st.button("🏁 Clôturer la session"):
        st.session_state.show_kpi = True
    
    if st.session_state.show_kpi:
        final_mood = st.slider("Comment vous sentez-vous MAINTENANT ?", 0, 10, initial_mood, key="final_mood")
        gain = final_mood - initial_mood
        
        st.markdown("### Résultat MindCare :")
        st.metric(label="Amélioration du Moral", value=f"{gain} pts", delta=gain)
        
        if gain > 0:
            st.balloons()
        
        if dom_emotion != "-":
            location_data = LOCATIONS.get(dom_emotion.lower())
            
            if location_data:
                st.markdown("---")
                st.success(f"📍 **Recommandation pour vous :**")
                st.write(f"Basé sur votre état dominant (**{dom_emotion}**), nous vous suggérons :")
                st.write(f"👉 **{location_data['name']}**")
                
                map_df = pd.DataFrame({
                    'lat': [location_data['lat']],
                    'lon': [location_data['lon']]
                })
                st.map(map_df, zoom=14)