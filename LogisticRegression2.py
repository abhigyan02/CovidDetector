import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

covid_data = pd.read_csv('covid_data.csv')

symptoms = covid_data[['Fever', 'Body Ache', 'Age', 'Runny Nose', 'Breathing Problems', 'Sore Throat', 'Dry Cough', 'Loss Of Taste/Smell', 'Red Eyes/Eye Irritation', 'Headache', 'Speech and Mobility Issue', 'Diarrhoea', 'Asthma', 'Heart Disease', 'Diabetic', 'Hypertension', 'Fatigue', 'Vaccinated']]
target = covid_data['Covid-19 affected']

X = np.array(symptoms).reshape(-1, 18)
Y = np.array(target)

symptoms_train, symptoms_test, target_train, target_test = train_test_split(symptoms, target, train_size=0.7)

model = LogisticRegression()
model.fit = model.fit(symptoms_train, target_train)

prediction = model.fit.predict(symptoms_test)

confusionMatrix = confusion_matrix(target_test, prediction, labels=model.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=model.classes_)
print("Confusion matrix: ")
display.plot()
plt.show()

print('Accuracy percentage', accuracy_score(target_test, prediction)*100, "%")
