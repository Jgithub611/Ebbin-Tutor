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
import sympy as sp
from sympy import solve, Eq, symbols, simplify, expand, factor
from typing import List, Tuple

class EbbinTutorChat:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('spanish'))
        self.intents = {}
        self.words = []
        self.classes = []
        self.documents = []
        self.model = None
        self.x, self.y, self.z = symbols('x y z')
        self.common_symbols = {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }
        
    def preprocess_text(self, text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.isalnum()]
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
                    
                self.intents[intent['tag']] = intent['responses']
            
            self.words = sorted(list(set(self.words)))
            self.classes = sorted(list(set(self.classes)))
            
        except FileNotFoundError:
            print(f"Error al cargar el Json :C")
            raise
        except json.JSONDecodeError:
            raise
        except Exception as e:
            raise

    def create_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        #one-hot encoding
        training = []
        output_empty = [0] * len(self.classes)

        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            for word in self.words:
                bag.append(1 if word in pattern_words else 0)
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        train_x = np.array([x for x, y in training])
        train_y = np.array([y for x, y in training])
        
        return train_x, train_y

    def build_model(self, input_shape: int, output_shape: int):
        #red neuronal TensorFlow
        model = Sequential([
            Dense(128, input_shape=(input_shape,), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(output_shape, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, training_file: str, epochs: int = 200, batch_size: int = 32):
        self.load_data(training_file)
        train_x, train_y = self.create_training_data()
        self.model = self.build_model(len(self.words), len(self.classes))
        print(f"Entrenando modelo por {epochs} épocas...")
        history = self.model.fit(
            train_x, train_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        self.model.save('chatbot_model.keras')
        
        with open('words.pkl', 'wb') as f:
            pickle.dump(self.words, f)
        with open('classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)
            
        return history

    def solve_equation(self, equation_str: str) -> str:
        try:
            equation_str = equation_str.replace("^", "**")
            if "=" in equation_str:
                left, right = equation_str.split("=")
                left_expr = sp.sympify(left)
                right_expr = sp.sympify(right)
                equation = Eq(left_expr, right_expr)
                solution = solve(equation)
                return f"La solución es: {solution}"
            else:
                expr = sp.sympify(equation_str)
                simplified = simplify(expr)
                factored = factor(expr)
                expanded = expand(expr)
                
                result = f"Expresión original: {expr}\n"
                result += f"Simplificada: {simplified}\n"
                result += f"Factorizada: {factored}\n"
                result += f"Expandida: {expanded}"
                return result
                
        except Exception as e:
            return f"Error al procesar la ecuación: {str(e)}"

    def process_math_query(self, query: str) -> str:
        """Procesar consultas matemáticas"""
        keywords = {
            'resolver': 'solve',
            'simplificar': 'simplify',
            'factorizar': 'factor',
            'expandir': 'expand',
            'calcular': 'solve'
        }
        
        try:
            operation = next((op for key, op in keywords.items() if key in query.lower()), 'solve')
            import re
            math_expr = re.search(r'"([^"]*)"', query)
            if math_expr:
                math_expr = math_expr.group(1)
            else:
                math_expr = query.split()[-1]
            
            if operation == 'solve':
                return self.solve_equation(math_expr)
            elif operation == 'simplify':
                expr = sp.sympify(math_expr)
                return f"Expresión simplificada: {simplify(expr)}"
            elif operation == 'factor':
                expr = sp.sympify(math_expr)
                return f"Expresión factorizada: {factor(expr)}"
            elif operation == 'expand':
                expr = sp.sympify(math_expr)
                return f"Expresión expandida: {expand(expr)}"
                
        except Exception as e:
            return f"Error al procesar la consulta matemática: {str(e)}"

    def predict(self, sentence: str) -> Tuple[str, float]:
        """Predecir intención y procesar consulta"""
        # Verificar si es una consulta matemática
        math_keywords = ['resolver', 'ecuación', 'simplificar', 'factorizar', 'expandir']
        is_math_query = any(keyword in sentence.lower() for keyword in math_keywords)
        
        if is_math_query:
            response = self.process_math_query(sentence)
            return response, 1.0
        sentence_words = self.preprocess_text(sentence)
        bag = [1 if word in sentence_words else 0 for word in self.words]
        results = self.model.predict(np.array([bag]))[0]
        
        max_prob_idx = np.argmax(results)
        predicted_class = self.classes[max_prob_idx]
        probability = results[max_prob_idx]
        
        if probability > 0.25:
            response = random.choice(self.intents[predicted_class])
            return response, probability
        return "Lo siento, no entiendo tu pregunta.", 0.0

def main():
    quit_words = ["salir","quit","exit","adiós"]
    chatbot = EbbinTutorChat()
    training_file = 'training_data.json'
    history = chatbot.train(training_file)
    final_accuracy = history.history['accuracy'][-1]
    print(f"\nEntrenamiento completado con precisión final: {final_accuracy:.2%}")
    print("\n=== Tutor de estudios Ebbin tutor===")
    print("Puedes hacer preguntas matemáticas o conversar normalmente")
    print("Ejemplos de consultas matemáticas:")
    print("- 'resolver x^2 + 2x + 1 = 0'")
    print("- 'simplificar (x+1)(x-1)'")
    print("- 'factorizar x^2 - 4'")
    print("Escribe 'salir' para terminar")
    while True:
        user_input = input("\nTú: ")
        if user_input.lower() in quit_words:
            break
        response, confidence = chatbot.predict(user_input)
        if isinstance(response, str) and confidence < 1.0:
            print(f"Bot ({confidence:.2%} confianza): {response}")
        else:
            print(f"\nResultado:\n{response}")
if __name__ == "__main__":
    main()