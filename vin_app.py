#importer les librairies 

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier



# Chargement des données
dt = pd.read_csv("winequality-red.csv", sep=";")


bins = (2, 5, 8)
group_names = ['bad', 'good']
dt['quality'] = pd.cut(dt['quality'], bins = bins, labels = group_names)


#Encoder les valeurs 
label_quality = LabelEncoder()
dt['quality'] = label_quality.fit_transform(dt['quality'])


# Préparation des données

X = dt.iloc[:, :-1]
y = dt.iloc[:, -1]


#TDivision des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#normalisation des données 
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



# Entraînement du modèle 
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# Création de l'application Web
st.title("Prédiction de la qualité du vin rouge")
st.write("Entrez les caractéristiques chimiques du vin rouge pour obtenir une prédiction de sa qualité.")
fixed_acidity = st.slider("Acidité fixe", 4.6, 16.0, 8.0, 0.1)
volatile_acidity = st.slider("Acidité volatile", 0.12, 1.58, 0.6, 0.01)
citric_acid = st.slider("Acide citrique", 0.0, 1.0, 0.3, 0.01)
residual_sugar = st.slider("Sucre résiduel", 0.9, 15.5, 8.1, 0.1)
chlorides = st.slider("Chlorures", 0.01, 0.61, 0.08, 0.01)
free_sulfur_dioxide = st.slider("Dioxyde de soufre libre", 1.0, 72.0, 36.0, 1.0)
total_sulfur_dioxide = st.slider("Dioxyde de soufre total", 6.0, 289.0, 144.0, 1.0)
density = st.slider("Densité", 0.990, 1.003, 0.996, 0.001)
pH = st.slider("pH", 2.74, 4.01, 3.31, 0.01)
sulphates = st.slider("Sulfates", 0.33, 2.00, 0.68, 0.01)
alcohol = st.slider("Alcool", 8.4, 14.9, 10.4, 0.1)


# Préparation des données de prédiction
prediction_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
                             total_sulfur_dioxide, density, pH, sulphates, alcohol]])
prediction = rfc.predict(prediction_data)

# Affichage de la prédiction
st.subheader("Résultat de la prédiction")

# Affichage de la prédiction
if prediction == 0:
    st.write("La qualité de ce vin rouge est mauvaise.")
else:
    st.write("La qualité de ce vin rouge est bonne!")

