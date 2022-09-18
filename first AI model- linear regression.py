#linear regression
# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Salary_Data.csv') #this step to read the data from the desired file
x= dataset.iloc[:,[0]].values # i should divide the dataset into x and y train iloc to divide the data, it takes coulmns and rows
# [0] means the first coulmn years of experince and it entered it as a matrix because i work with matrices
y= dataset.iloc[:,1].values # i want the y data vector not matrix so i didn't write [1]
# .values to extract the headers and deals only with numerical data
# split the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.30) # the output of this function give these varibles respectively
from sklearn.linear_model import LinearRegression # import leaner regression to learn the algorithim, i import a class
regressor = LinearRegression() #making object from the class
regressor.fit(x_train,y_train) #creating a model
# testing phase
model_output= regressor.predict(x_test)
print(model_output)
# mean square error
from sklearn.metrics import mean_squared_error # importing a function calculate the error
error = mean_squared_error(y_test, model_output) ; print(error)
# visualization phase
plt.scatter(x_train,y_train, color= 'blue') # this function dot points, so i should pass the coordinates
plt.plot(x_train, regressor.predict(x_train) ,  color= 'green') # this to plot the line
plt.title('experince Vs salary (trainging set)')
plt.xlabel('years of experince')
plt.ylabel('salary')
plt.show() # to show the graph
sal = regressor.predict([[7.5]]) ;  print(sal) # to predict the salary of specific number of years 

