import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from autocorrect import Speller
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import spacy

nltk.download()
df = pd.read_csv('train.csv')
df = pd.DataFrame(df)
df = df.drop_duplicates()

cleaned_text = df['full_text'].astype(str)

stop_words = set(stopwords.words('english'))
words = word_tokenize(" ".join(cleaned_text))
filtered_text = [word for word in words if word.lower() not in stop_words]

spell = Speller(lang='en')
corrected_text = ' '.join([spell(word) for word in filtered_text])

nlp = spacy.load("en_core_web_sm")
doc = nlp(corrected_text)
tokens = [token.text for token in doc]
lemmatized_tokens = [token.lemma_ for token in doc]
sentences = sent_tokenize(corrected_text)

df['level'] = df['cohesion']  
encoder = LabelEncoder()
df['level'] = encoder.fit_transform(df['level'])

train, test = train_test_split(df, test_size=0.2, random_state=42)

df.to_csv('cleaned_dataset.csv', index=False)