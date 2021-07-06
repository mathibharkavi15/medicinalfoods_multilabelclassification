import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
import re
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import numpy as np
import math
import pickle

app= Flask(__name__)

@app.route("/")

def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def effectivenessPrediction():
    inputQuery1 = request.form['query1']
    model = keras.models.load_model(r'C:\Users\mathi\Desktop\cdappdev\model_bigru_final1.h5')
    #model = keras.models.load_model(r'C:\Users\mathi\Desktop\cdappdev\RNN_Glove_model1.h5')
    data = [inputQuery1]
    print(data)
    
    labels=["Rvw_Prod","Rvw_Service","Rvw_Both","Prod_Pos","Prod_Neg","Prod_Neutral","Eff_Gen_Y","Eff_Gen_N","Eff_Gen_NotSay","Eff_Stress_Y","Eff_Stress_N","Eff_Stress_NotSay",
"Eff_Sleep_Y","Eff_Sleep_N","Eff_Sleep_NotSay","Eff_Anxty_Y","Eff_Anxty_N","Eff_Anxty_NotSay","SideEff_Y","SideEff_N"]
    
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=250)
    text_ds = tf.data.Dataset.from_tensor_slices(data).batch(1)
    vectorizer.adapt(text_ds)
    test_data = vectorizer(data).numpy()
    print(test_data)

    single = model.predict(test_data)

    result=[]
    for y_pred in single:
            class_labels=[labels[i] for i,prob in enumerate(y_pred) if prob > 0.5]
            result.append(class_labels)

    print(result)

    output = result
    
    return render_template('home.html', output1=result,query1 = request.form['query1'])
    
app.run()
