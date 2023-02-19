import pandas as pd
import joblib
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.title("water potability")
left,right = st.columns(2)
def save_model(model):
    joblib.dump(model, './project/water_drop.joblib')
def load_model():
    return joblib.load('./project/water_drop.joblib')
def generate_data():
    dm_data=pd.read_csv("./project/water_drop.csv")
    dm_color=dm_data['color']
    dm_cut=dm_data["cut"]
    # print(diamonds_cut.value_counts())
    # print(diamonds_color.value_counts())
    x=dm_data.drop(["price"],axis=1)
    y=dm_data["price"]
    st.markdown(x)
    st.markdown(y)
    # st.markdown(dm_cut)
    # st.markdown(dm_color)

ss =st.button("click")
train=st.button("train")
if ss :
    generate_data()

if train :
    dm_data = pd.read_csv("./project/water_drop.csv")
    x=dm_data.drop(["Potability"],axis=1).astype("Float64")
    y=dm_data["Potability"].astype("Float64")
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    model=LogisticRegression()
    model.fit(x_train,y_train)
    save_model(model)

ph=st.slider("Input PH",min_value=0.2274,max_value=14.0000,step=.0001)
Hardness=st.slider("Hardness",min_value=73.4922,max_value=317.3381,step=.0001)
Solids=st.slider("Solids",min_value=56488.6724,max_value=320.9426,step=.0001)
Chloramines=st.slider("Chloramines",min_value=13.1270,max_value=1.39087,step=.0001)
Organic_carbon=st.slider("Organic_carbon",min_value=27.0067,max_value=2.2000,step=.0001)
Turbidity=st.slider("Turbidity",min_value=6.4947,max_value=1.4500,step=.0001)
# Potability=

predictb = left.button('คาดคะเน')
if predictb:
    model=load_model()
    data_key=(ph, Hardness,Solids,Chloramines,Organic_carbon,Turbidity)
    model_data=model.predictb()

