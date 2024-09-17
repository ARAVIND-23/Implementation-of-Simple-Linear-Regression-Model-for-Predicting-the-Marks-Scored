# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ARAVIND G
RegisterNumber: 212223240011 
*/
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print("dataset.head()")
print(dataset.head())
print("dataset.tail()")
print(dataset.tail())
dataset.info()
#assigning hours to X & scores to Y
print("X & Y values")
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
print("Prediction values of X & Y")
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="pink")
plt.plot(X_train,reg.predict(X_train),color="brown")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

## Output:

![Screenshot 2024-09-17 193834](https://github.com/user-attachments/assets/966863cb-dfc2-4538-8379-bf94a1e69274)
![Screenshot 2024-09-17 193916](https://github.com/user-attachments/assets/c7e3fc8e-bd20-4b98-9622-4dbf5e545fbd)
![Screenshot 2024-09-17 193949](https://github.com/user-attachments/assets/cacb3c04-a5fb-494b-880a-223cd1624e3a)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
