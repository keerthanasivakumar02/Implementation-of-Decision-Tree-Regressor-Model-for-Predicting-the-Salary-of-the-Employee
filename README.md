

# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Keerthana S
RegisterNumber:  212223040092
*/
```
```

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("/content/Salary (2).csv")
data.head()

```


## Output:
![image](https://github.com/user-attachments/assets/5c47d33d-8dcf-4083-9062-16db3983f414)

```
data.info()
data.isnull().sum()
```
## Output:
![image](https://github.com/user-attachments/assets/1979288e-43e7-4937-967c-b7574d56f77d)
```

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
```
## Output:

## ![image](https://github.com/user-attachments/assets/dfaf13a4-49f7-4ec4-88df-ee0f7e9b30fb)
```
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
```
## Output:
![image](https://github.com/user-attachments/assets/1fa9ec2b-9360-4528-97aa-cc65bf81edf2)
```
r2=metrics.r2_score(y_test,y_pred)
r2
```
## Output:
![image](https://github.com/user-attachments/assets/ee9f6fa9-b19d-4557-b62e-bd27c221898f)
```
dt.predict([[5,6]])
```
## Output:
![image](https://github.com/user-attachments/assets/65ec2134-3958-4a64-8025-0eb2ae2b1ee7)

Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
