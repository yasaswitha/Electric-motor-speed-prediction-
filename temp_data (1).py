# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 22:37:05 2023

@author: Dibyanshu
"""

import pandas as pd
import streamlit as st
from pickle import load




st.title('Model Deployment: Ada Boost')




st.sidebar.header('User Input')




def user_input_features():
    ambient = st.sidebar.number_input("Insert ambient")
    u_d = st.sidebar.number_input("Insert u_d")
    u_q = st.sidebar.number_input("Insert u_q")
    i_d = st.sidebar.number_input("Insert i_d")
    pm = st.sidebar.number_input("Insert pm")
    data  = { 'ambient':ambient,
              'u_d':u_d,
              'u_q':u_q,
              'i_d':i_d,
              'pm':pm}
    features=pd.DataFrame(data,index=[0])
    return features
    


df = user_input_features()
st.subheader('User Input')
st.write(df)


loaded_model=load(open('ada_boost.sav' ,'rb'))


prediction=loaded_model.predict(df)
st.write("Motor speed is :",prediction)

