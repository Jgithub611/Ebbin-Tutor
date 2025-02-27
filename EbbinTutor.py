import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import os
import streamlit as st
import asyncio
import time
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() #Cargar varialbles de entorno
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-pro')

st.set_page_config(
    page_title="EbbinTutor - Tutor Académico IA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

#CSS
st.markdown("""
<style>
    
    body {
        background-color: rgba(20, 20, 20, 0.95); 
        color: #E0E0E0;
        font-family: 'Arial', sans-serif;
        margin: 0;
        padding: 0;
    }

    /* Contenedor principal */
    .stApp {
        max-width: 1200px;
        margin: auto;
        background: rgba(255, 255, 255, 0.1); 
        backdrop-filter: blur(15px); 
        border-radius: 20px; 
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
    }

    /* Estilo de mensajes de chat */
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        background: rgba(255, 255, 255, 0.15); 
        backdrop-filter: blur(10px); 
        transition: transform 0.2s; 
    }

    .chat-message:hover {
        transform: scale(1.02); 

    .chat-message.user {
        background-color: rgba(58, 74, 90, 0.85); 
        border-left: 5px solid #4682B4; 
    }

    .chat-message.bot {
        background-color: rgba(47, 62, 70, 0.85); 
        border-left: 5px solid #2E8B57; 
    }

    .chat-message .avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.3); 
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5); 
    }

    .chat-message .message {
        flex-grow: 1;
        color: #D3D3D3; 
        font-size: 1rem;
    }

    .timestamp {
        font-size: 0.75rem;
        color: #A9A9A9; 
        margin-top: 0.5rem;
        text-align: right; 
    }

    /* Estilo de botones */
    .stButton > button {
        background-color: rgba(46, 139, 87, 0.9); 
        color: #FFFFFF; 
        border: none;
        border-radius: 25px; 
        padding: 12px 25px;
        font-size: 1rem; 
        transition: background-color 0.3s ease, transform 0.2s; 
        cursor: pointer; 
    }

    .stButton > button:hover {
        background-color: rgba(39, 122, 71, 0.9);
        transform: translateY(-2px); 
    }

    /* Estilo de entradas de texto */
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 12px 15px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        background-color: rgba(51, 51, 51, 0.8); 
        color: #E0E0E0; 
        transition: border-color 0.3s;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4682B4; 
        outline: none; 
    }

    /* Estilo de chips de tema */
    .topic-chip {
        display: inline-block;
        background-color: rgba(53, 79, 82, 0.8); 
        border-radius: 20px;
        padding: 8px 15px;
        margin: 5px;
        font-size: 0.9rem;
        color: #B0C4DE; 
        border: 1px solid rgba(82, 121, 111, 0.8); 
        transition: background-color 0.3s;
    }

    .topic-chip:hover {
        background-color: rgba(82, 121, 111, 0.9); 
    }

    /* Estilo del pie de página */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #A9A9A9; 
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: rgba(30, 30, 30, 0.9);
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.5); 
    }
</style>
""", unsafe_allow_html=True)

class EbbinTutorChatbot: 
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('spanish'))
        self.intents = {}
        self.words = []
        self.classes = []
        self.documents = []
        self.model = None

        self.model_path = 'chatbot_ebbin_tutor_model.keras'
        self.words_path = 'words.pkl'
        self.classes_path = 'classes.pkl'

        self.specialized_topics = {
            'ingles': ['ingles', 'inglés'],
            'matematicas': ['matematicas', 'matemáticas', 'mat'],
            'programacion': ['programacion', 'programación', 'codificacion', 'codigo'],
            'ciencias': ['ciencias', 'biologia', 'quimica'],
            'fisica': ['fisica', 'física', 'mecanica', 'termodinamica']
        }
        
    def preprocess_text(self, text: str) -> list: #Preprosamiento del texto
        tokens = word_tokenize(text.lower()) #Tokeniza y Convierte a minus
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and token.isalnum()] #Filtrado Stopwords
        return tokens

    def load_data(self, json_file: str):
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            for intent in data['intents']:
                self.classes.append(intent['tag'])
                
                for pattern in intent['patterns']:
                    words = self.preprocess_text(pattern)
                    self.words.extend(words)
                    self.documents.append((words, intent['tag'])) 
                    
                self.intents[intent['tag']] = intent['responses'] #Guardado de respuestas por intención
            
            self.words = sorted(list(set(self.words)))
            self.classes = sorted(list(set(self.classes)))
            
        except Exception as e:
            st.error(f"Error al cargar los datos de entrenamiento: {e}")
            raise

    def create_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)

        for doc in self.documents:
            bag = [1 if word in doc[0] else 0 for word in self.words] #Bag of words
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        train_x = np.array([x for x, y in training]) #Separación entradas
        train_y = np.array([y for x, y in training]) #Separación salidas
        
        return train_x, train_y 

    def build_model(self, input_shape: int, output_shape: int):
        model = Sequential([
            Dense(256, input_shape=(input_shape,), activation='relu'), 
            Dropout(0.6),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(output_shape, activation='softmax') #Capa de salida Softmax 
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, training_file: str, epochs: int = 500, batch_size: int = 32):
        self.load_data(training_file)
        train_x, train_y = self.create_training_data()

        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self.words = pickle.load(open(self.words_path, 'rb'))
            self.classes = pickle.load(open(self.classes_path, 'rb'))
        else:
            self.model = self.build_model(len(self.words), len(self.classes))
            history = self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0)
            self.model.save(self.model_path)
            pickle.dump(self.words, open(self.words_path, 'wb'))
            pickle.dump(self.classes, open(self.classes_path, 'wb'))
            return history

    def is_specialized_topic(self, sentence: str) -> bool: #Temas especializados
        words = self.preprocess_text(sentence)
        return any(word in topic_variants for topic_variants in self.specialized_topics.values() for word in words)

    def get_topic(self, sentence: str) -> str:
        words = self.preprocess_text(sentence)
        topic = next((key for key, variants in self.specialized_topics.items() 
                     if any(word in variants for word in words)), None)
        return topic

    async def get_gemini_response(self, question: str):
        try:
            words = self.preprocess_text(question)
            topic = self.get_topic(question)

            if topic:
                prompt = f"Actúa como un experto en {topic} y responde: {question}"
                try:
                    response = model.generate_content(prompt)
                    if response and hasattr(response, 'text'):
                        return response.text
                    else:
                        return None
                except Exception as e:
                    st.warning(f"Error 01: {str(e)}") #Error en la llamada a Gemini API
                    return None

            return None

        except Exception as e:
            st.warning(f"Se ha producido un error al obtener respuesta de Gemini: {str(e)}")
            return None

    async def predict(self, sentence: str): #Predice la respuesta
        if not sentence.strip():
            return "Por favor, escribe algo para que pueda ayudarte.", None

        if self.is_specialized_topic(sentence):
            gemini_response = await self.get_gemini_response(sentence)
            if gemini_response:
                return gemini_response, self.get_topic(sentence)

        sentence_words = self.preprocess_text(sentence)
        if not sentence_words:
            return "No entiendo tu entrada. ¿Podrías reformularla?", None

        bag = [1 if word in sentence_words else 0 for word in self.words]
        results = self.model.predict(np.array([bag]), verbose=0)[0]

        max_prob_idx = np.argmax(results)
        predicted_class = self.classes[max_prob_idx]
        probability = results[max_prob_idx]

        if probability > 0.7:
            return random.choice(self.intents[predicted_class]), self.get_topic(sentence)
        return "Lo siento, no entiendo tu pregunta. ¿Podrías reformularla?", None

@st.cache_resource
def load_chatbot():
    return EbbinTutorChatbot()

def display_message(message, is_user=False, topic=None):
    avatar_url = "Usser.png" if is_user else "Ebbin.png"  
    message_class = "user" if is_user else "bot"
    
    topic_html = f'<span class="topic-chip">{topic.capitalize()}</span>' if topic else ''
    
    html = f"""
    <div class="chat-message {message_class}">
        <div class="avatar">
            <img src="{avatar_url}">
        </div>
        <div class="message">
            <div class="timestamp">{datetime.now().strftime('%H:%M:%S')} {topic_html}</div>
            <p>{message}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

async def process_message(chatbot, user_input):
    display_message(user_input, True)
    
    with st.chat_message("assistant", avatar="Ebbin.png"):  
        message_placeholder = st.empty()
        message_placeholder.markdown("_Pensando..._")
        await asyncio.sleep(1)  
        
        response, topic = await chatbot.predict(user_input)
        message_placeholder.empty()
        display_message(response, False, topic)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.session_state.messages.append({
        "content": user_input,
        "is_user": True,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    st.session_state.messages.append({
        "content": response,
        "is_user": False,
        "topic": topic,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

def save_conversation():
    if "messages" in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversacion_ebbin_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            for msg in st.session_state.messages:
                sender = "Tú" if msg["is_user"] else "EbbinTutor"
                f.write(f"[{msg['timestamp']}] {sender}: {msg['content']}\n")
                
        st.success(f"La conversación guardada como {filename}")

def display_chat_history():
    for msg in st.session_state.messages:
        display_message(
            msg["content"],
            msg["is_user"],
            msg.get("topic") if not msg["is_user"] else None
        )

def main():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("📚 EbbinTutor")

    st.sidebar.title("⚙️ Configuración")
    
    chatbot = load_chatbot()
    
    data_path = "training_data.json"

    if not st.session_state.initialized:
        if st.sidebar.button("Inicializar EbbinTutor"):
            with st.spinner("Inicializando EbbinTutor..."):
                chatbot.train(data_path)
                st.session_state.chatbot = chatbot
                st.session_state.initialized = True
                st.success("¡Chatbot inicializado!")
                
                welcome = "¡Hola! Soy EbbinTutor, tu asistente educativo. Puedo ayudarte con inglés, matemáticas, programación, ciencias y física. ¿En qué puedo ayudarte hoy?"
                st.session_state.messages.append({
                    "content": welcome,
                    "is_user": False,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })

    st.sidebar.subheader("Temas especializados")
    topics_html = """
    <div style="font-size: 0.9rem;">
        <span class="topic-chip">Inglés</span>
        <span class="topic-chip">Matemáticas</span>
        <span class="topic-chip">Programación</span>
        <span class="topic-chip">Ciencias</span>
        <span class="topic-chip">Física</span>
    </div>
    """
    st.sidebar.markdown(topics_html, unsafe_allow_html=True)
    
    if st.session_state.messages and st.sidebar.button("Guardar conversación"):
        save_conversation()

    chat_container = st.container()
    with chat_container:
        display_chat_history()

    if st.session_state.initialized:
        user_input = st.chat_input("Escribe tu pregunta aquí...")
        if user_input:
            asyncio.run(process_message(st.session_state.chatbot, user_input))
    else:
        st.info("Por favor, inicializa EbbinTutor usando el botón en la barra lateral.")

    st.markdown('<div class="footer">Created by NeuroBits</div>', unsafe_allow_html=True)

main()