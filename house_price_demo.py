from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import numpy as np
import joblib

df_original = pd.read_csv('melb_data.csv')

df_original = df_original.rename(
    columns= {
        'Lattitude' : 'Latitude',
        'Longtitude': 'Longitude'
    }
)

df_original.loc[df_original['Type'] == 'h', 'BuildingArea'] = df_original.loc[df_original['Type'] == 'h', 'BuildingArea'].fillna(144)
df_original.loc[df_original['Type'] == 't', 'BuildingArea'] = df_original.loc[df_original['Type'] == 't', 'BuildingArea'].fillna(130)
df_original.loc[df_original['Type'] == 'u', 'BuildingArea'] = df_original.loc[df_original['Type'] == 'u', 'BuildingArea'].fillna(75)

df_original.groupby('Type')['BuildingArea'].median()

df_original.loc[df_original['YearBuilt'].isnull(),'YearBuilt'] = df_original['YearBuilt'].fillna(1970)

df_original.loc[df_original['Car'].isnull(),'Car'] = df_original.loc[df_original['Car'].isnull(),'Car'].fillna(2.0)

df_for_council = df_original[['Postcode','Latitude','Longitude','Regionname','CouncilArea']] 
df_for_council = df_for_council.dropna() 

df_council_features = df_for_council[['Postcode','Latitude','Longitude','Regionname']] 

A = pd.get_dummies(df_council_features, columns=['Regionname'], dtype=int) 
b = df_for_council['CouncilArea'] 

A_train, A_test, b_train, b_test = train_test_split(A,b, test_size= 0.20, random_state= 20) 

rf_council = RandomForestClassifier(n_estimators=500, random_state=20, 
max_depth=30 , min_samples_leaf=6) 

rf_council.fit(A_train,b_train) 

missing_council = df_original.loc[df_original['CouncilArea'].isnull(),] 

A_missing = missing_council[['Postcode','Latitude','Longitude','Regionname']] 
A_missing_encoded = pd.get_dummies(A_missing, columns=['Regionname'], dtype=int) 

A_missing_encoded = A_missing_encoded.reindex(columns = A_train.columns, 
fill_value = 0)

predict_council = rf_council.predict(A_missing_encoded) 

df_original.loc[df_original['CouncilArea'].isnull(),'CouncilArea'] = predict_council


class DummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols = None):
        self.categorical_cols = categorical_cols
        self.columns_ = None
    
    def fit(self, X , y=None):
        X_dummies = pd.get_dummies(X, columns=self.categorical_cols, 
                                   drop_first = True)
        self.columns_ = X_dummies.columns
        return self
    
    def transform(self, X):
        X_dummies = pd.get_dummies(X, columns=self.categorical_cols,
                                   drop_first=True)
        for col in self.columns_:
            if col not in X_dummies:
                X_dummies[col] = 0
        X_dummies = X_dummies[self.columns_]
        return X_dummies



pipe = joblib.load('house_price_model.pkl')
st.title ("House Price Prediction (Gradient Boosting Model)")

Regionname = st.selectbox('Region:', df_original['Regionname'].unique())
CouncilArea = st.selectbox('Council Area:', df_original['CouncilArea'].unique())
Type = st.selectbox('Type:', df_original['Type'].unique())
Method = st.selectbox('Method:', df_original['Method'].unique())
SellerG = st.selectbox('Seller:', df_original['SellerG'].unique())

Rooms = st.slider('Rooms:',1,10,2)
Distance = st.slider('Distance(km):',0.0, 50.0, 5.0)
Landsize = st.number_input('Land Size:',0.0,45000.0,500.0)
BuildingArea = st.number_input('Building Area:', 0.0,45000.0,500.0)
Latitude = st.slider('Latitude:',-38.5, -37.8, 0.3)
Longitude = st.slider('Longitude:',144.40, 145.50, 0.5)

if st.button('Predict Price'):
    new_data = pd.DataFrame([{
        'Rooms' : Rooms,
        'Distance' : Distance, 
        'Landsize' : Landsize,
        'BuildingArea': BuildingArea,
        'Latitude': Latitude,
        'Longitude' : Longitude,
        'Regionname': Regionname,
        'CouncilArea' : CouncilArea,
        'Type' : Type,
        'Method' : Method,
        'SellerG' : SellerG
    }])

    prediction = pipe.predict(new_data)[0]
    st.success(f"Estimated House Price: AUD{prediction:,.0f}")
