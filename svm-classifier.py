import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Reading data and discrimination
df = pd.read_excel("./dataset/iris.xls")
y, label = pd.factorize(df["iris"])
x = df.drop(["iris"], axis=1)

# Scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Generating the model iteration times and collect the accuracy scores
accuracies = []
iteration = 200

for i in range(iteration):
    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # Creating and fitting the model
    model = SVC(kernel="rbf")
    model.fit(x_train, y_train)

    # Predicting
    y_pred = model.predict(x_test)
    conf = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)

# Visualization
accuracies = np.resize(np.array(accuracies), new_shape=(iteration-1,1))
print(accuracies)
print("Average accuracy: %.3f" %(np.average(accuracies)))
print("Success")