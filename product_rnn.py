import streamlit as st
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_addons as tfa
from pre_treatment_product import pre_process_text
import pickle
from PIL import Image

st.set_page_config(layout='wide')

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

st.title('Catalogador de Produtos')

img = Image.open('IMG_0058.jpeg')
st.image(img)

st.text(" ")
st.text(" ")

# Texto do item
st.sidebar.header('UNIVERSIDADE DE SÃO PAULO')
usp = Image.open('IMG_0059.png')
st.sidebar.image(usp)
#text = st.sidebar.text_input("NOME DO ITEM", 'Biscoito de Chocolate')
text = st.text_input("NOME DO ITEM", 'Biscoito de Chocolate')

btn_predict = st.button("REALIZAR CATALOGAÇÃO")

if btn_predict:
    new_complaint = pre_process.transform(text)
    seq = tokenizer.texts_to_sequences([new_complaint])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)

    segmento = label_segmento[np.argsort(pred[0].flatten())[::-1]][:5]
    categoria = label_categoria[np.argsort(pred[1].flatten())[::-1]][:5]
    subcategoria = label_subcategoria[np.argsort(pred[2].flatten())[::-1]][:5]
    produto = label_produto[np.argsort(pred[3].flatten())[::-1]][:5]

    # Criando o Data Frame
    index_labels = ['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5']
    labels = {
        'segmento': segmento,
        'categoria': categoria,
        'subcategoria': subcategoria,
        'produto': produto
    }
    df_product = pd.DataFrame(labels, index=index_labels)

    st.table(df_product)
