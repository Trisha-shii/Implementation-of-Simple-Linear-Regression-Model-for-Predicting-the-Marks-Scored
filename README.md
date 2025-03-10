# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Import essential libraries such as pandas, numpy, and matplotlib.
2.Load the Dataset: Read the dataset containing the input feature (e.g., study hours) and the target value (e.g., marks scored).
3.Data Preprocessing: Extract the independent variable (X) and dependent variable (Y) from the dataset.
4.Split the Data: Divide the dataset into training and testing sets using train_test_split() from sklearn.
5.Train the Model: Import LinearRegression from sklearn.
6.Create a model instance and fit it using the training data.
7.Predict Values: Use the .predict() method on the test data to predict marks.
8.Display Results: Print the actual and predicted marks.
9.Plot the regression line along with the training and testing data points for visual representation.
10.Evaluate the Model: Calculate the model’s performance using metrics like:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
 
 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Trisha Priyadarshni Parida
RegisterNumber: 212224230293
*/

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from Libraries import modules for mse,mae,rmse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error  #i.e mae, mse


#importing "student_scores.csv" file given to work with
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('/content/student_scores.csv')

df

df.head(10)

#Input Data    CAUTION : .values - method not a func()
# Segregating data to variables
X = df['Hours'].values
Y = df['Scores'].values

#splitting the entire data into 80:20 ratio for train:test, 0.2 = 20% test size
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2,  random_state = 0)

#import linear regression model and fit the model with the data

lr_model = LinearRegression()
lr_model.fit(X_train.reshape(-1,1),Y_train)

Y_pred = lr_model.predict(X_test.reshape(-1,1))
print("Predicted values: ",Y_pred)      #expected Values Based on Trained LR Model

print("Actual Values: ",Y_test)         #Actually Observed Values

#graph plot for training data -> X_train, Y_train
#for every term that calls for x variable, convert into 2d array by X_train/test.reshape(-1,1) func

plt.scatter(X_train.reshape(-1,1),Y_train,color='green',label='Data Points')

# Reshape X_train before passing it to predict
plt.plot(X_train.reshape(-1,1), lr_model.predict(X_train.reshape(-1,1)),color="purple",label="Best Fit Line")


plt.title("Training Data Graph Plot")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.show()

#graph plot for test data

plt.scatter(X_test.reshape(-1,1),Y_test,color='orange',label="Data Values")
plt.plot(X_test.reshape(-1,1), Y_pred, color = "blue",label="Best Fit Line")

plt.title("Testing Data Graph Plot")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.show()


#find mae,mse,rmse

mae = mean_absolute_error(Y_test,Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ",rmse)
print("\nRound/Approximated to 2 decimal digits:-\n")
print(f"Mean Absolute Error:{mae:.2f} ")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error:{rmse:.2f} ")
```

## Output:

![Screenshot 2025-03-10 083813](https://github.com/user-attachments/assets/075f0950-4727-4142-86df-126931f9d6fc)
![Screenshot 2025-03-10 083821](https://github.com/user-attachments/assets/69fd2fb9-c4fe-42f1-81b1-49cc46e98a4d)
![Screenshot 2025-03-10 083827](https://github.com/user-attachments/assets/e838b381-2961-47d0-a7d4-2f1675d6a0e1)
![Screenshot 2025-03-10 083833](https://github.com/user-attachments/assets/cb6fab6b-afd8-4c99-b09e-d531cc74f8ef)
![Screenshot 2025-03-10 083842](https://github.com/user-attachments/assets/4f87e1a3-1dcf-414d-9308-107a8dc58c54)
![Screenshot 2025-03-10 083858](https://github.com/user-attachments/assets/0aa5cf7f-ae15-4e3a-880b-f534f1650890)
![Screenshot 2025-03-10 083906](https://github.com/user-attachments/assets/ae38d235-01a5-48c0-a6ed-26a5c1a1c3a3)
![Screenshot 2025-03-10 083912](https://github.com/user-attachments/assets/b5413e3f-3eb4-4533-bc4a-6b7767b6f6b9)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
