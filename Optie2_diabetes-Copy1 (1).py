#!/usr/bin/env python
# coding: utf-8

# In[25]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import streamlit as st


# In[26]:


# Titel en subtitel van de app

st.title('Diabetes detectie')
st.header('Bepaal de kans op diabetes middels deze app')

# Teskst behorende bij de app

st.text('Diabetes is een chronische stofwisselingsziekte. Bij diabetes zit er te veel suiker in het bloed. Het lichaam kan de bloedsuiker niet op peil houden. Met behulp van deze machine learning web app wordt het mogelijk om aan de hand van ingevoerde parameters een diagnose te maken over de mogelijkheid dat iemand diabetes heeft.De app maakt gebruik van historische data om de kans op diabetes te calculeren. Dit kan mensen helpen om betere en snellere diagnoses te maken of mensen helpen die geen tijd of geld hebben om een doctor te bezoeken.')

#Voeg afbeelding toe > werkt alleen als de persoon die het upload naar streamlit zelf de afbeelding opslaat en filepath noteerd naar de afbeelding
#Image = Image.open("C:\Users\joshua.bierenbrood\Documents\Data Science\Intro to datascience\Werkcollege week 3\diabetes.jpg")
#st.image(image, caption = 'ML', use_column_width = True)


# In[27]:


df = pd.read_csv('diabetes.csv')


# In[28]:


st.title("Diabetes Database")


# In[29]:


df.head()


# In[30]:


df.info()


# In[31]:


df.eq(0).sum()
#veel 0 waarden, vervangen door NaN en dan opvullen met gemiddelden


# In[32]:


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI',
    'DiabetesPedigreeFunction','Age']]= df[['Glucose','BloodPressure',
    'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)


# In[33]:


df.fillna(df.mean(), inplace=True)


# In[34]:


df.eq(0).sum() #checken


# In[35]:


df.describe()


# In[36]:


st.subheader('Informatie over de data:')
st.dataframe(df)


# In[37]:


st.subheader('Aantal diabetesgevallen in dataset:')
def countPlot():
    fig = plt.figure()
    sns.countplot(df["Outcome"])
    st.pyplot(fig)


# In[38]:


def histogram_pairplot():
    st.header("Histogram & Pair Plot")
    sd = st.selectbox(
        "Selecteer een optie om data te visualiseren", 
        [
            "Histogram", 
            "Pair Plot"   
        ]
    )

    fig = plt.figure()

    if sd == "Histogram":
        df.hist(bins = 50, figsize = (20,15))
    
    elif sd == "Pair Plot":
        sns.pairplot(df, hue="Outcome")

    st.pyplot(fig)


# In[39]:


cor = df.corr()
sns.heatmap(cor, annot = True)


# In[40]:


related = cor['Outcome'].sort_values(ascending=False)
related


# In[41]:


#Machine learning
x_data= df.drop('Outcome',axis=1)
y_data= df['Outcome']


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.3, random_state=42)


# In[43]:


def get_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17,0)
    glucose = st.sidebar.slider('glucose', 44, 199, 44)
    blood_pressure = st.sidebar.slider('blood_pressure',24, 122,24)
    skin_thickness = st.sidebar.slider('skin_thickness', 7, 99, 7)
    insulin = st.sidebar.slider('insulin', 14.0, 846.0, 14.0)
    BMI = st.sidebar.slider('BMI', 18.2, 67.1, 18.2)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.078)
    age = st.sidebar.slider('age', 21, 81, 21)
    
    user_data = {'zwangerschappen': pregnancies,
                 'glucose': glucose,
                 'bloeddruk': blood_pressure,
                 'huiddikte': skin_thickness,
                 'insuline': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'leeftijd': age
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features

gekozen_input = get_input()

st.subheader('Uw gekozen waarden:')
st.write('selecteer de voor u geldende parameters in de slider boxen om de app de kans op diabetes te laten berekenen')
                        


# In[44]:


RandomForest=RandomForestClassifier()
RandomForest.fit(x_train, y_train)
Predict= RandomForest.predict(x_test)
print(Predict)


# In[45]:


accuracyRFC = accuracy_score(y_test, Predict)
print("Accuracy with Random Forrest Classification:", accuracyRFC)


# In[46]:


st.subheader('Accuratiescore bij het model')
st.write(str(accuracy_score(y_test, Predict) * 100) + '%')


# In[47]:


diabetes_ja_nee = RandomForest.predict(gekozen_input)


# In[48]:


st.subheader('Wel (1) of geen (0) diabetes?')
st.write(diabetes_ja_nee)

