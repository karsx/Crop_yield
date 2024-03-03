import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Given data
X = np.array([[1230, 1230, 1230, 1230, 1230, 1230],
    [480, 60, 36, 70, 20, 18],
    [1250,	75,	29,	78,	22,	19],	
    [450,	65,	35,	70,	19,	18],	
[1200,	80,	27,	79,	22,	19],	
[500,	70,	34,	74,	22,	16],	
[1275,	71,	28,	77,	21,	20],	
[425,	65,	37,	67,	18,	15],	
[1200,	77,	27,	78,	23,	20],	
[400,	50,	39,	60,	18,	15],	
[1280,	80,	26,	80,	24,	20],	
[415,	55,	38,	65,	19,	17],	
[1225,	79,	29,	79,	23,	20],	
[425,	50,	37,	65,	18,	19],	
[1250,	70,	24,	70,	22,	18],	
[400,	60,	39,	60,	18,	15],	
[1300,	80,	28,	80,	24,	20],	
[410,	55,	36,	65,	21,	16],
[1150,	77,	28,	76,	23,	20],	
[1200,	78,	27,	78,	23,	19],	
[410,	50,	37,	59,	19,	15],
[1280,	76,	26,	75,	24,	19],	
[425,	55,	38,	65,	19,	17],	
[1225,	73,	29,	73,	23,	20],	
[450,	50,	37,	65,	18,	19],	
[1250,	70,	24,	70,	22,	18],
[400,	60,	39,	60,	18,	15],
[1250,	80,	28,	80,	24,	20],	
[405,	55,	36,	60,	21,	16],	
[1200,	72,	29,	73,	21,	19],	
[1150,	80,	26,	75,	21,	20],	
[475,	55,	39,	61,	18,	16],	
[1275,	76,	26,	75,	24,	19],	
[450,	55,	38,	65,	19,	17],	
[1200,	73,	29,	73,	23,	20],	
[500,	50,	37,	65,	18,	19],	
[1300,	70,	24,	70,	22,	18],	
[425,	60,	39,	60,	18,	15],	
[1250,	75,	26,	75,	22,	19],
[440,	58,	37,	66,	22,	18],	
[1275,	78,	28,	77,	23,	21],
[405,	55,	36,	60,	21,	16],	
[1175,	78,	28,	75,	23,	22],
[410,	60,	39,	65,	18,	15],	
[1250,	80,	26,	78,	23,	19],
[460,	55,	38,	61,	20,	18],	
[1275,	76,	26,	75,	24,	19],	
[445,	60,	38,	68,	22,	18],	
[1220,	77,	29,	75,	22,	19],	
[450,	50,	37,	60,	18,	15],	
[1225,	79,	24,	79,	22,	19],	
[430,	65,	39,	65,	19,	16],	
[1275,	77,	27,	78,	21,	20],	
[400,	52,	38,	64,	19,	15],	
[1245,	78,	27,	78,	22,	19],	
[455,	58,	37,	61,	21,	18],	
[1280,	73,	28,	75,	24,	19],	
[475,	62,	37,	68,	22,	18],	
[1300,	79,	28,	77,	23,	19],	
[425,	55,	36,	65,	19,	15],	
[1175,	77,	25,	75,	22,	19],	
[425,	70,	39,	70,	20,	16],	
[1200,	75,	27,	79,	21,	20],	
[450,	56,	40,	67,	18,	15],
[1200,	75,	27,	76,	21,	19],	
[410,	55,	38,	68,	20,	16],
[1250,	75,	27,	75,	23,	20],	
[475,	60,	37,	63,	20,	18],	
[1225,	75,	29,	77,	23,	19],	
[455,	60,	38,	65,	20,	16],	
[1245,	77,	27,	75,	22,	20],	
[450,	59,	40,	67,	18,	16],	
[1200,	79,	27,	77,	23,	20],	
[475,	72,	36,	71,	21,	17],	
[1275,	77,	28,	76,	22,	19],	
[475,	58,	39,	68,	19,	16],	
[1300,	80,	28,	80,	24,	20],
[400,	50,	40,	60,	18,	15],	
[1175,	70,	28,	70,	22,	19],
[445,	65,	39,	65,	21,	19],	
[1200,	77,	29,	76,	22,	19],	
[450,	65,	38,	60,	20,	16],	
[1225,	75,	28,	79,	21,	19],	
[450,	65,	39,	70,	20,	19],
[1300,	76,	28,	77,	22,	20],	
[450,	70,	36,	72,	25,	18],
[1250,	77,	28,	76,	22,	19],	
[475,	60,	39,	70,	20,	17],	
[1200,	75,	28,	77,	23,	19],	
[410,	52,	40,	62,	19,	16],	
[1225,	75,	28,	75,	23,	20],	
[460,	60,	39,	60,	20,	16],	
[1150,	78,	29,	77,	21,	18],	
[475,	65,	38,	60,	20,	16],	
[1250,	77,	28,	78,	23,	20],	
[425,	60,	39,	65,	19,	17],	
[1220,	79,	28,	77,	23,	21],	
[480,	65,	36,	68,	21,	16],	
[1230,	80,	28,	80,	24,	20]
    
    # Add more rows with feature values
])

y = np.array([1230, 8, 11, 9, 11, 10, 11, 7, 12, 6, 12, 8, 11, 9, 11, 5.5, 12, 7, 11, 12, 6, 11, 7, 10, 9, 10, 6, 12, 7, 10, 11, 6, 11, 7, 10, 9, 10, 6, 12, 8, 11, 7, 11, 6, 11, 7, 11, 8, 10, 6, 11, 6, 11, 6.5, 10, 7.5, 11, 9, 10, 7, 11, 6.5, 11, 7, 11, 7, 11, 7, 10, 8.5, 9.5, 7.5, 10.5, 7, 10, 7.5, 12, 6, 10, 7.5, 9.5, 8, 9.5, 8.5, 9, 8, 10, 8, 11, 6.5, 11, 7, 9, 8, 9, 6.5, 10.5, 7, 12])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVR model
svr = SVR(kernel='linear')  # You can also try other kernels like 'rbf'

# Train the SVR model
svr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svr.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Display the results
print("R-squared (R²):", r2)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
