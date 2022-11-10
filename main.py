import streamlit as st
import streamlit.components.v1 as components
import pickle
import string
import re
import pandas as pd
import nltk, nltk.classify.util, nltk.metrics
nltk.download('punkt')

from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
from nltk.classify import MaxentClassifier
from sklearn.metrics import confusion_matrix as cm, classification_report as cr
from nltk.corpus import stopwords
nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Import RF Model
model = pickle.load(open('MaxEntropy.pkl', 'rb'))

vect = pickle.load(open('vect_tfidf.pkl', 'rb'))

def remove_punct(text):
    #Remove Karakter ASCII, angka, punctuation
    text = text.encode('ascii', 'replace').decode('ascii')
    text = re.sub('x(\d+[a-zA-Z]+|[a-zA-Z]+\d+|\d+)',"",text)
    text = re.sub('[0-9]+', '', text)
    text = re.sub(r'[\W\s_]',' ',text)

    #Remove spasi diawal dan akhir, url, hastag, tagar, kata akhiran berlebihan, baris baru, tab
    text = re.sub("^\s+|\s+$", "", text)
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text) #url
    text = re.sub("@[A-Za-z0-9]+","", text)
    text = re.sub("#[A-Za-z0-9]+","", text)
    pola = re.compile(r'(.)\1{2,}', re.DOTALL)
    text = pola.sub(r'\1', text)
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ") 
    return text

normalized_word = pd.read_excel("Kamus_Normalisasi.xlsx")

normalized_word_dict = {}

for index, row in normalized_word.iterrows():
  if row[0] not in normalized_word_dict:
    normalized_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return ' '.join([normalized_word_dict[term] if term in normalized_word_dict else term for term in document.split()])

def tokenization(text):
    text = nltk.tokenize.word_tokenize(text)
    return text

list_stop = set(stopwords.words('indonesian'))
hapus = ["tidak","baik","kurang","rusak","lemot","lemah"]
for i in hapus:
    if i in list_stop:
        list_stop.remove(i)
        
def stopwords_removal(words):
    return [word for word in words if word not in list_stop]

factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming(text):
    text = [stemmer.stem(word) for word in text]
    return text

def getlabel(xlsx):
    book = pd.read_excel(xlsx)
    label=list(book["LABEL"])
    print("Jumlah label :",len(label))
    return  label

def getdata(csv):
    data=[]
    docid=[]
    book=pd.read_csv(csv)
    print("Shape :",book.shape) #formatnya (jumlah data, jumlah fitur)
    for i in range(book.shape[0]):
        row = book.iloc[i][1:].to_dict()
        id = book.iloc[i][0]
        data.append(row)
        docid.append(id)
    return data, docid

def bagidata(data, label, id, train=0.8):
    rasio=int(train*len(data))
    fitur_train, label_train = data[:rasio], label[:rasio]
    fitur_test, test_label = data[rasio:], label[rasio:]
    id_train, id_test= id[:rasio], id[rasio:]
    print(len(fitur_train),len(label_train))
    print(len(fitur_test),len(test_label))
    return fitur_train, label_train, fitur_test, test_label, id_train, id_test    

def joindata(data,label):
    dataset=[]
    for i in range(len(data)):
        row=[data[i],label[i]]
        dataset.append(row)
    return dataset
    
def main():
    data, id= getdata(r"datasett.csv")
    label = getlabel(r"DATASET.xlsx") 

        # data, id= getdata(r"/content/drive/MyDrive/smt 8/data scraping shopee/hasil_tfidf/datasett.csv")
        # label = getlabel(r"/content/drive/MyDrive/smt 8/data scraping shopee/scrapinggg/DATASET.xlsx") 

    newlabel=["negatif" if i == "N" else "positif" for i in label]
    newlabel=newlabel[:len(data)] #limit beda jumlah data disini

    print("\nJumlah data Negatif : ",len([ i for i in label if i == "N"]))
    print("Jumlah data Positif : ",len([ i for i in label if i == "P"]))

    fitur_train, label_train, fitur_test, label_test, id_train, id_test = bagidata(data, newlabel, id, train=0.8) 

    data_train=joindata(fitur_train, label_train)
    st.title('Aplikasi Website Analisis Sentimen thd Shopee')

    age = st.slider('Pilih Index Ddokumen Data Test', 0, 211)
    dokumen= age #max index 211 di data test
    out = str(id_test[dokumen])
    st.subheader(f"ID Dokumen [{out}] :")
    model.explain(fitur_test[dokumen])
    if model.classify(fitur_test[dokumen]) == "positif":
        st.success("Positif")
    else:
        st.warning("negatif")

    hasil = {}
    for i in range(211):
        hasil[i] = model.classify(fitur_test[dokumen])
    hasilt = pd.DataFrame(hasil.items(), columns=['Dokumen', 'Prediksi'])
    st.dataframe(hasilt, use_container_width=True)

    commen = st.text_area("Masukkan Komentar")   
    if st.button("Proses"):
        #Data Uji
        st.subheader("Data Uji")
        df = pd.DataFrame([commen], columns=(['data uji']))
        st.dataframe(df, use_container_width=True)

        #Case Folding
        st.subheader("Case Folding")
        df['CaseFolding'] = df['data uji'].str.lower()
        st.dataframe(df[['CaseFolding']], use_container_width=True)

        #Cleaning
        st.subheader("Cleaning")
        df['Cleaning'] = df['CaseFolding'].apply(lambda x: remove_punct(x))
        st.dataframe(df[['Cleaning']], use_container_width=True)

        #Normalisasi
        st.subheader("Normalisasi")
        df['Normalisasi'] = df['Cleaning'].apply(lambda x: normalized_term(x))
        st.dataframe(df[['Normalisasi']], use_container_width=True)

        #Token
        st.subheader("Tokenize")
        df['Tokenize'] = df['Normalisasi'].apply(lambda x: tokenization(x))
        st.dataframe(df[['Tokenize']], use_container_width=True)

        #Stopword
        st.subheader("Stopword")
        df['Stopword'] = df['Tokenize'].apply(lambda x: stopwords_removal(x))
        st.dataframe(df[['Stopword']], use_container_width=True)

        #Stemming
        st.subheader("Stemming")
        df['Stemming'] = df['Stopword'].apply(lambda x: stemming(x))
        st.dataframe(df[['Stemming']], use_container_width=True)
        
        dokumen=0 #max index 211 di data test
        out = str(id_test[dokumen])
        st.subheader(f"ID Dokumen [{out}] :")
        model.explain(fitur_test[dokumen])

        vect_text = vect.transform([commen])
        model.explain(vect_text)

        if model.classify(fitur_test[dokumen]) == "positif":
            st.success("Positif")
        else:
            st.warning("negatif")     

if __name__ == '__main__':
    main()