import streamlit as st
import numpy as np
import nltk
from nltk.corpus import stopwords
from string import punctuation
import spacy
from langdetect import detect
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import tensorflow as tf
import tensorflow_addons as tfa

nltk.download('stopwords')

nlp_eng = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
nlp_rus = spacy.load('ru_core_news_sm', disable=['ner', 'parser'])

stop_words_rus = stopwords.words('russian')
stop_words_eng = stopwords.words('english')
stop_words = stop_words_rus+stop_words_eng+['шт', 'каждый день', 'каждый', 'день', 'красная цена', 'красная', 'цена', 'верный', 'дикси', 'моя', 'моя цена', 'окей','то, что надо!', 'smart','spar', 'ашан']

tf.config.run_functions_eagerly(True)

with open('LSTM_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

loaded_model = load_model("LSTM_model.h5", custom_objects={'Addons>F1Score': tfa.metrics.F1Score})


def preprocess_sentences(sentences, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences


def tokenize_text(value):

   all_sentence = []

   try:
       lang = detect(value)

       if lang == 'en':
           doc = nlp_eng(value)
           lemmas = [token.lemma_ for token in doc if token.is_alpha and token.text not in punctuation and token.text.lower() not in stop_words]
           cleaned_sentence = " ".join(lemmas)
           all_sentence.append(cleaned_sentence)
       else:
           doc = nlp_rus(value)
           lemmas = [token.lemma_ for token in doc if token.is_alpha and token.text not in punctuation and token.text.lower() not in stop_words]
           cleaned_sentence = " ".join(lemmas)
           all_sentence.append(cleaned_sentence)
   except Exception as e:
       print(f"Error processing value: {value}")
       print(f"Error message: {str(e)}")

   padded_sequences = preprocess_sentences(all_sentence, tokenizer, 29)

   return padded_sequences


st.title("Анализатор категорий продуктов")

product = st.text_input("Введите название покупки")

if st.button("Анализировать"):
    if product:
        tokens = tokenize_text(product)
        prediction = np.argmax(loaded_model.predict(tokens), axis=1)

        dictionary = { "topic": ['автозапчасти', 'видеоигры', 'напитки', 'продукты питания', 'закуски и приправы', 'аквариум', 'одежда', 'уборка', 'электроника', 'образование'], "label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] }

        label = prediction.item()
        topic = dictionary['topic'][dictionary['label'].index(label)]

        st.write(f"Категория продукта: {topic}")
