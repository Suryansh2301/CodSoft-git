import streamlit as st
import pickle

import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
# from streamlit import header

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf= pickle.load(open("Vectorizer.pkl", "rb"))
model = pickle.load(open("ExtraTreesClassifier.pkl", "rb"))
model_1 = pickle.load(open("MultinomialNB.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the messages")



if st.button("predict"):

    # 1. preprocess
    transform_sms =transform_text(input_sms)
    #2. vectorize
    vector_input  = tfidf.transform([transform_sms])
    #3. predict
    result = model.predict(vector_input)[0]
    # result_1 = model.predict(vector_input)[0]
    #4. Display
    if result == 1:
      st.header("Spam")
    else:
        st.header("Not Spam")

    # if result_1 == 1:
    #   st.header("Spam")
    # else:
    #     st.header("Not Spam")