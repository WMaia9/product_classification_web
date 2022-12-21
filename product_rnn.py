import streamlit as st
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from pre_treatment_product import pre_process_text
import pickle
from PIL import Image

pre_process = pre_process_text(stopwords_language='portuguese')
# ler as categorias
with open('product.json', 'r') as myfile:
    data=myfile.read()
produtos = json.loads(data)

label_segmento = np.array(produtos['segmento'])
label_categoria = np.array(produtos['categoria'])
label_subcategoria = np.array(produtos['subcategoria'])
label_produto = np.array(produtos['nm_product'])

# Abrindo o Tokenizador
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# carregando o modelo
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("full_MultiModel2.h5")
    return model
    
model = load_model()

# Número máximo que sequência que a rede neural irá utilizar
MAX_SEQUENCE_LENGTH = 15

st.title('Short Text Product Classification')

img = Image.open('mercado.jpg')
st.image(img)

st.text(" ")
st.text(" ")

# Texto do item
st.sidebar.header('Entrada do Texto')
text = st.sidebar.text_input("NOME DO ITEM", 'Biscoito de Chocolate')

# Função para limpar o dataset
def remove_stopwords(sentence):
  # Converting to Lowercase
  sentence = sentence.lower()

  return sentence

btn_predict = st.sidebar.button("REALIZAR CASSIFICAÇÃO")

if btn_predict:
  new_complaint = remove_stopwords(text)
  seq = tokenizer.texts_to_sequences([new_complaint])
  padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
  pred = model.predict(padded)
  st.header(f'Segmento:')
  st.text(label_segmento[np.argmax(pred[0])])
  st.header('Categoria:')
  st.text(label_categoria[np.argsort(pred[1].flatten())[::-1]][:3])
  st.header('Subcategoria:')
  st.text(label_subcategoria[np.argsort(pred[2].flatten())[::-1]][:3])
  st.header('Produto:')
  st.text(label_produto[np.argsort(pred[3].flatten())[::-1]][:5])