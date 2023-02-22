import water as wt
import ussi_project as up
import streamlit as st


st.sidebar.title("Menu")
select_box=st.sidebar.selectbox("Where are you going :",('"-"',"water potability","Prediction Prediction"))
st.title(select_box)
if select_box == '"-"':
    # st.title(select_box)
    st.title("UUUU")
if select_box == "water potability":
    # st.title(select_box)
    wt.water()
if select_box == "Prediction Prediction" :
    # st.title(select_box)
    up.population()
# up.page()