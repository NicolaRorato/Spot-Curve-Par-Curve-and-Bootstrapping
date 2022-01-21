# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 09:12:10 2022

@author: Nicola
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def MultipleRegression(x, y, intercept=True):
    '''Multiple Regression
        Inputs:
            x
            y
            intercept (Optional) = Boolean to regress with or without the intercept
            '''
    mlr = LinearRegression(fit_intercept=intercept)
    mlr.fit(x, y)
    return mlr.intercept_, mlr.coef_

strips = pd.read_csv('Homework 2 Data Strips.csv')
strips.rename({'Maturity':'T'}, axis=1, inplace=True)

strips['SpotRate'] = 2*((100/strips['Price'])**(1/(strips['T']*2)) - 1)
strips['D(T)'] = 1/(1+strips['SpotRate']/2)**(strips['T']*2)
strips['3-mForwardRate'] = (strips['D(T)']/strips['D(T)'].shift(-1) - 1)/0.25

## 1
plt.figure()
plt.plot(strips['T'], strips['SpotRate'])
plt.xlabel('Years')
plt.ylabel('Spot Rates')
plt.title('Spot Curve')

plt.figure()
plt.plot(strips['T'][:-2], strips['3-mForwardRate'][:-2])
plt.xlabel('Years')
plt.ylabel('Forward Rates')
plt.title('Forward Curve')

## 2
X = (pd.DataFrame([(strips['T']), (strips['T'])**2, (strips['T'])**3, (strips['T'])**4, (strips['T'])**5])).T
X.columns = ['T', 'T^2', 'T^3', 'T^4', 'T^5']
y = np.log(strips['D(T)'])
alpha, betas = MultipleRegression(X, y, False)
print("Intercept: ", alpha)
print("Coefficients:")
print(list(zip(X.columns, betas)))

## 3-4-5
strips_estimated = pd.DataFrame(columns = ['T'])
strips_estimated['T'] = np.arange(0.5, 25.5, 0.5)
strips_estimated['D(T)'] = np.exp(alpha +\
                                  betas[0]*strips_estimated['T'] +\
                                  betas[1]*strips_estimated['T']**2 +\
                                  betas[2]*strips_estimated['T']**3 +\
                                  betas[3]*strips_estimated['T']**4 +\
                                  betas[4]*strips_estimated['T']**5)

strips_estimated['SpotRate'] = 2*((1/strips_estimated['D(T)'])**(1/(strips_estimated['T']*2)) - 1)
strips_estimated['6-mForwardRate'] = (strips_estimated['D(T)']/strips_estimated['D(T)'].shift(-1) - 1)/0.5
strips_estimated['D(T)CumSum'] = strips_estimated['D(T)'].cumsum()
strips_estimated['ParRate'] = 2*((1-1*strips_estimated['D(T)'])/strips_estimated['D(T)CumSum'])

plt.figure()
plt.plot(strips_estimated['T'], strips_estimated['SpotRate'], label='Spot Curve')
plt.plot(strips_estimated['T'], strips_estimated['ParRate'], label='Par Curve')
plt.plot(strips_estimated['T'], strips_estimated['6-mForwardRate'], label='Forward Curve')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Rates')
plt.title('Strips Curves - Semiannual')

## 6-7
bonds = pd.read_csv('Homework 2 Data T-Note.csv', skipfooter=1)
bonds.rename({'Maturity':'T'},axis=1, inplace=True)

X = (pd.DataFrame([(bonds['T']), (bonds['T'])**2, (bonds['T'])**3, (bonds['T'])**4, (bonds['T'])**5])).T
X.columns = ['T', 'T^2', 'T^3', 'T^4', 'T^5']
y = bonds['Yield']
alpha, betas = MultipleRegression(X, y)
bonds_estimated = pd.DataFrame(columns = ['T'])
bonds_estimated['T'] = np.arange(0.5, 25.5, 0.5)
bonds_estimated['ParRate'] = (alpha +\
                                  betas[0]*bonds_estimated['T'] +\
                                  betas[1]*bonds_estimated['T']**2 +\
                                  betas[2]*bonds_estimated['T']**3 +\
                                  betas[3]*bonds_estimated['T']**4 +\
                                  betas[4]*bonds_estimated['T']**5)

arr = np.zeros(len(bonds_estimated['ParRate']))
arr[0] = 100/(bonds_estimated.iloc[0]['ParRate']/2 + 100)
bonds_estimated['D(T)'] = arr
for i in range(1, len(bonds_estimated['T'])):
    bonds_estimated['D(T)'][i] = (100 - bonds_estimated['ParRate'][i]/2*bonds_estimated['D(T)'].cumsum()[i])/(100 + bonds_estimated['ParRate'][i]/2)

bonds_estimated['SpotRate'] = 2*((1/bonds_estimated['D(T)'])**(1/(bonds_estimated['T']*2)) - 1)
bonds_estimated['6-mForwardRate'] = (bonds_estimated['D(T)']/bonds_estimated['D(T)'].shift(-1) - 1)/0.5

plt.figure()
plt.plot(bonds_estimated['T'], bonds_estimated['SpotRate'], label='Spot Curve')
plt.plot(bonds_estimated['T'], bonds_estimated['ParRate']/100, label='Par Curve')
plt.plot(bonds_estimated['T'], bonds_estimated['6-mForwardRate'], label='Forward Curve')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Rates')
plt.title('Bonds Curves - Semiannual')
