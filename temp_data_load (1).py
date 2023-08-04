# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 22:09:23 2023

@author: Dibyanshu
"""
import pandas as pd
from pickle import dump
from pickle import load
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

temperature = pd.read_csv("temperature_data.csv")

x=temperature[['ambient','u_d','u_q','i_d','pm']]
y=temperature[['motor_speed']]

model=AdaBoostRegressor()
model.fit(x,y)

dump(model,open('ada_boost.sav', 'wb'))

loaded_model=load(open('ada_boost.sav' ,'rb'))
result = loaded_model.score(x,y)
print(result)
