#Part Basic

import numpy as np
#Create your first array with the elements [1,22.4,5,35,4,6.7,3,8,40]
# and print it. Experiment what the following functions do: ndim, shape, size and dtype.
first_array = np.array([1, 22.4, 5, 35, 4, 6.7, 3, 8, 40])
print(first_array.ndim)
print(first_array.shape)
print(first_array.size)
print(first_array.dtype)

#Create your first matrix with the elements [['a', 'b'],['c', 'd'],[3, 3]] and print it
#Experiment what the following functions do: ndim, shape, size and dtype
initial_matrix = np.array([['a', 'b'],['c', 'd'],[3, 3]])
print(initial_matrix)
print(initial_matrix.ndim)
print(initial_matrix.shape)
print(initial_matrix.size)
print(initial_matrix.dtype)

#Create numpy 1 dimension array using each of the functions arange and rand
arange_array = np.arange(10)
random_array = np.random.rand(10)
print(arange_array)
print(random_array)

#Create numpy 2 dimensions matrix using each of the functions zeros and rand
zero_matrix = np.zeros((2,2))
random_matrix = np.random.rand(3,3)
print(zero_matrix)
print(random_matrix)

#Create an array containing 20 times the value 7. Reshape it to a 4 x 5 Matrix
matrix7 = np.array([7]*20)
print(matrix7)
matrix_reshaped = matrix7.reshape(4,5)
print(matrix_reshaped)

#Create a 6 x 6 matrix with all numbers up to 36, then print:
#   only the first element on it
#   only the last 2 rows for it
#   only the two mid columns and 2 mid rows for it
#   the sum of values for each column
initial_array = np.arange(1,37)
matrix66 = initial_array.reshape(6,6)
print(matrix66)
print("first element =", matrix66[0][0])
print('last 2 columns = ', matrix66[-2: , : ])
print('2 mid columns & 2 mid rows = ', matrix66[2:4,2:4])
print("Sum of values for each column =", np.sum(matrix66,axis=0))

#Part Advanced

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandas.tests.test_downstream import df

insurance = pd.read_csv("/Users/nevinmurad/Desktop/pandas-homework/insurance.csv")
#Load the insurance.csv in a DataFrame using pandas. Explore the dataset using functions like to_string(), columns,
# index, dtypes, shape, info() and describe(). Use this DataFrame for the following exercises

print("Insurance columns", insurance.columns)
print("Insurance to_string", insurance.to_string())
print("Insurance info", insurance.info())
print("Insurance index", insurance.index)
print("Insurance shape", insurance.shape)
print("Insurance dtypes", insurance.dtypes)
print("Insurance describe", insurance.describe)




#Print only the column age

print("Insurance age only",(insurance['age']))

#Print only the columns age,children and charges

print("Insurance age, children, charge only",(insurance[['age','children','charges']]))

#Print only the first 5 lines and only the columns age,children and charges

print("Insurance 5 first lines only",insurance.loc[[1,2,3,4,5],['age','children','charges']])

#What is the average, minimum and maximum charges ?

print("Insurance meadian is:",insurance.loc[:,'charges'].median())
print("Insurance means is:",insurance.loc[:,'charges'].mean())
print("Insurance min is:",insurance.loc[:,'charges'].min())

#What is the age and sex of the person that paid 10797.3362. Was he/she a smoker?

#Answ: 52years, Female, No Smoker

print("Person that paid 10797.3362", insurance.loc[insurance["charges"] == 10797.3362])

#What is the age of the person who paid the maximum charge?

#Answ: 54 years old

print("Person that paid max charge", insurance.loc[insurance["charges"] == insurance["charges"].max()])

#How many insured people do we have for each region?
#southwest    325
#northwest    325
#northeast    324

print("Print count per region", insurance["region"].value_counts())


#How many insured people are children?
#ANSWER: there is no people insured below 18 y/o so no children

print("Nb children", insurance.groupby("age").size())

#What do you expect to be the correlation between charges and age, bmi and children?
# I would expect that the correlation between charges and age to be strong because you will likely pay
# more as you age. The correlation between bmi and children should be low since there is no link
# between how many children you have and your BMI.
# My assumptions were correct since the correlation between charges and age is 0.299 and only 0.0128
# between bmi and children


print("Extracting correlation matrix", insurance.corr().to_string())


