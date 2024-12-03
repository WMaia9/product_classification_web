import streamlit as st
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from pre_treatment_product import pre_process_text
from PIL import Image

#st.set_page_config(layout='wide')

pre_process = pre_process_text(stopwords_language='portuguese')
# ler as categorias
with open('product.json', 'r') as myfile:
    data = myfile.read()
produtos = json.loads(data)

label_segment = np.array(produtos['segmento'])
label_category = np.array(produtos['categoria'])
label_subcategory = np.array(produtos['subcategoria'])
label_product = np.array(produtos['nm_product'])

# Load the tokenizer
with open('tokenizer.json', 'r') as json_file:
    tokenizer_json = json_file.read()  # Read the JSON string
tokenizer = tokenizer_from_json(json.loads(tokenizer_json)) 

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("LSTModel.keras")
    return model

model = load_model()

# Maximum sequence length the neural network will use
MAX_SEQUENCE_LENGTH = 15

st.title('Product Cataloger')
st.text(" ")
st.text(" ")
img = Image.open('IMG_0058.jpeg')
st.image(img)
st.text(" ")
st.text(" ")

# Sidebar content
st.sidebar.header('UNIVERSITY OF SÃO PAULO')
usp = Image.open('IMG_0059.png')
st.sidebar.image(usp)
st.sidebar.header('Developed by MECAI students')

btnChoose = st.selectbox('TOOL', list(['Text', 'CSV']))

if btnChoose == 'Text':

    text = st.text_input("ITEM NAME", 'Biscoito de chocolate 200g')

    btn_predict = st.button("PERFORM CATALOGING")

    if btn_predict:
        new_complaint = pre_process.transform(text)
        seq = tokenizer.texts_to_sequences([new_complaint])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)

        segment = label_segment[np.argsort(pred[0].flatten())[::-1]][:5]
        category = label_category[np.argsort(pred[1].flatten())[::-1]][:5]
        subcategory = label_subcategory[np.argsort(pred[2].flatten())[::-1]][:5]
        product = label_product[np.argsort(pred[3].flatten())[::-1]][:5]

        # Criando o Data Frame
        index_labels = ['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5']
        labels = {
            'segment': segment,
            'category': category,
            'subcategory': subcategory,
            'product': product
        }
        df_product = pd.DataFrame(labels, index=index_labels)

        st.table(df_product)
        
else:
    uploaded_file = st.file_uploader("Upload the CSV")
    if uploaded_file is not None:
        dt = pd.read_csv(uploaded_file, names=['item_name'], header=None)
        btn_predict = st.button("PERFORM CATALOGING")
        if btn_predict:
            def classification(dt):
                df = pd.DataFrame(
                    [],
                    columns=[
                        'item_name',
                        'segment',
                        'category',
                        'subcategory',
                        'product_name',
                    ],
                )
                my_bar = st.progress(0)
                for index, text in dt.iterrows():
                    text_processed = pre_process.transform(text['item_name'])
                    seq = tokenizer.texts_to_sequences([text_processed])
                    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
                    pred = model.predict(padded)
                    df = df.append(
                        {
                            'item_name': text['nm_item'],
                            'segment': label_segment[np.argmax(pred[0])],
                            'category': label_category[np.argmax(pred[1])],
                            'subcategory': label_subcategory[np.argmax(pred[2])],
                            'product_name': label_product[np.argmax(pred[3])],
                        },
                        ignore_index=True,
                    )
                    my_bar.progress(index + 1)
                return df

            df = classification(dt)
            downdload = df.to_csv(index=False)
            st.download_button(
                label="Download dos items catalogados",
                data=downdload,
                file_name='large_df.csv',
                mime='text/csv',
            )

            st.table(df.head(50))