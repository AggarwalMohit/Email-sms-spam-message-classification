from django.http import HttpResponse,HttpRequest, request
from django.shortcuts import render

import pandas as pd
import joblib
import string
import nltk as nltk
from nltk.stem.porter import PorterStemmer

input_text=""

def home(request):
    return render(request,"home.html")
    

def result(request):
    def transformation(text):
        text=text.lower()
        text=nltk.word_tokenize(text)
    
        y=[]
        for i in text:
            if i.isalnum():
                y.append(i)
            
        text=y[:]
        y.clear()
    
        for i in text:
            if i not in nltk.corpus.stopwords.words('English') and i not in string.punctuation:
                y.append(i)
    
        text=y[:]
        y.clear()
    
        for i in text:
            y.append(PorterStemmer().stem(i))
            
        return " ".join(y)

    clf=joblib.load('final_model.sav')
    vectorizer=joblib.load('vectorizer.sav')
    input_text=request.POST['message']
    input_text_df=pd.DataFrame({"input_text":[input_text]})
    input_text_df['num_character']=input_text_df['input_text'].apply(len)
    input_text_df['num_words']=input_text_df['input_text'].apply(lambda x:len(nltk.word_tokenize(x)))
    input_text_df['num_sentence']=input_text_df['input_text'].apply(lambda x:len(nltk.sent_tokenize(x)))
    input_text_df["input_text_transformed"]=input_text_df["input_text"].apply(transformation)
    
    ##transformed_text=transformation(input_text)
    input_vectorized_tf=vectorizer.transform(input_text_df["input_text_transformed"])
    ##vectorized_text=vectorizer.fit_transform([transformed_text])
    result=clf.predict(input_vectorized_tf)
    final_verdict=""
    if result==0:
        final_verdict="Not a spam"
    else:
        final_verdict="Spam"

    
    return render(request,"result.html",{'answer':final_verdict})






