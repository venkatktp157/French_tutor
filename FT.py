#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
import warnings
warnings.filterwarnings('ignore')

# ✅ Load API key from secrets.toml
groq_api_key = st.secrets["GROQ_API_KEY"]

# ❗ Safety check
if not groq_api_key:
    st.error("❌ GROQ_API_KEY missing in secrets. Add it in Streamlit Settings.")
    st.stop()

# 🚀 Initialize Groq model
chat = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# 🧠 Setup memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

st.title("👩‍🏫 French Grammar Tutor - Dual Agent App")

terminate = st.button("🛑 End Conversation")

if not terminate:
    student_input = st.text_input("👨‍🎓 Student: Posez votre question sur la grammaire française")

    if student_input:
        st.session_state.memory.chat_memory.add_user_message(student_input)

        teacher_prompt = f"""
        You are a native French grammar expert tutoring a student.

        For the student's question, respond with:
        1. A clear answer in French.
        2. A brief grammar explanation in French.
        3. An English translation of your full response.
        4. A list of 2–3 synonyms if vocabulary is involved.
        5. Alternative phrasing that conveys the same meaning.
        6. Common learner mistakes or misconceptions to watch out for.
        7. Whether the sentence/phrasing is formal or informal.

        Question: {student_input}
        """
        try:
            response = chat.invoke([HumanMessage(content=teacher_prompt)])
            st.markdown("### 👩‍🏫 Teacher's Response")
            st.write(response.content)

            follow_up = chat.invoke([HumanMessage(content="Ask the student for the next question.")])
            st.markdown("### 👨‍🎓 Student's Turn")
            st.write(follow_up.content)
        except Exception as e:
            st.error(f"❌ Groq model error: {e}")
else:
    st.success("Conversation ended. Merci beaucoup!")
