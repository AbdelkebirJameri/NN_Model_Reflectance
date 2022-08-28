import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

  

def load_model1():
    with open('finalized_model_ouarzazate.pkl', 'rb') as file:
        data1 = pickle.load(file)
    return data1

data1 = load_model1()
error1=data1["error"]
score1=data1["score"]

def explore_page_ouarzazate():
    st.title("MODEL ERRORS (OUARZAZATE) ")
    st.subheader(f' - Mean squar error:  {error1:.5f} ')
    st.subheader(f" - Accuracy assessment (R2) :    {score1:.5f}  ")

    st.title("Linear Graph")
    df1=pd.read_excel("DATA-OUARZAZATE.xlsx")
    df1=df1[["T_Air_C_Avg", "RH_Avg","V_Vent_m_s_Avg", "Ray_Tot_KWh_m2_Tot cumulé", "Perte_réflectance"]]
    for i in list(df1.columns):
         fig1=sns.lmplot(y=i, x='Perte_réflectance', data=df1)
        
         st.pyplot(fig1)
         
def load_model2():
    with open('finalized_model_temara.pkl', 'rb') as file2:
        data2 = pickle.load(file2)
    return data2

data2 = load_model2()
error2=data2["error"]
score2=data2["score"]

def explore_page_temara():
    st.title("MODEL ERRORS (TÉMARA) : ")
    st.subheader(f' - Mean squar error:  {error2:.5f} ')
    st.subheader(f" - Accuracy assessment (R2) :    {score2:.5f}  ")

    st.title("Linear Graph")
    df2=pd.read_excel("DATA-TEMARA.xlsx")
    df2=df2[["T_Air_C_Avg", "RH_Avg\t","V_Vent_m_s_Avg", "Ray_Tot_KWh_m2_Tot cumulé", "Perte_réflectance"]]
    for i in list(df2.columns):
         fig2=sns.lmplot(y=i, x='Perte_réflectance', data=df2)
        
         st.pyplot(fig2)
def load_model3():
    with open('finalized_model_ouarzazate.pkl', 'rb') as file3:
        data3 = pickle.load(file3)
    return data3

data3 = load_model3()
modul3=data3["model"]
    

def predicted_page_ouarzazate():
    st.title("Site de OUARZAZATE")
    st.write("Entrer les données:")
    
    input11= st.number_input("T_Air_C_Avg", 0, 60, 15, 1)
    input22 = st.number_input("RH_Avg", 0, 40, 20, 1)
    input33 = st.number_input("V_Vent_m_s_Avg", 0, 30, 2, 1)
    input44= st.number_input("Ray_Tot_KWh_m2_Tot cumulé", 0, 10000, 368, 1)
    
    submit1 = st.button("Perte_réflectance")
    if submit1:
        
        x1=np.array([[float(input11),float(input22),float(input33),float(input44)]])
        x1 = x1.astype(float)

        perte1= modul3.predict(x1)
        st.subheader(f"la prediction de la perte de réflectance est :    {perte1[0]:.5f}  %")

def load_model4():
    with open('finalized_model_temara.pkl', 'rb') as file4:
        data4 = pickle.load(file4)
    return data4

data4 = load_model4()
modul4=data4["model"]
    

def predicted_page_temara():
    st.title("Site de Témara")
    st.write("Entrer les données:")
    
    input15= st.number_input("T_Air_C_Avg", 0, 60, 15, 1)
    input25= st.number_input("RH_Avg", 0, 40, 20, 1)
    input35= st.number_input("V_Vent_m_s_Avg", 0, 30, 2, 1)
    input45= st.number_input("Ray_Tot_KWh_m2_Tot cumulé", 0, 10000, 368, 1)
    
    submit2 = st.button("Perte_réflectance")
    if submit2:
        
        x2=np.array([[float(input15),float(input25),float(input35),float(input45)]])
        x2 = x2.astype(float)

        perte2= modul4.predict(x2)
        st.subheader(f"la prediction de la perte de réflectance est :    {perte2[0]:.5f}  %")
    

st.sidebar.image('logo.png', caption='',channels="RGB")
switching1=st.sidebar.radio("SITES:", ("OUARZAZATE", "TÉMARA"))
st.markdown("***")

switching2=st.sidebar.radio("PRÉDIRE  OU EXPLORER DATA:", ("PRÉDIRE", "EXPLORER"))
if switching1=="OUARZAZATE" and switching2=="PRÉDIRE" :
    predicted_page_ouarzazate()

elif switching1=="OUARZAZATE" and switching2=="EXPLORER" :
    
    explore_page_ouarzazate()
elif switching1=="TÉMARA" and switching2=="PRÉDIRE" :
    predicted_page_temara()

elif switching1=="TÉMARA" and switching2=="EXPLORER" :
    explore_page_temara()