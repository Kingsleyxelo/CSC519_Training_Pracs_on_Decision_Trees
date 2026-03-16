from sklearn.tree import DecisionTreeClassifier
import numpy as np

x = np.array([[2, 50],[3,60],[4,70],[5,80],[6,90],[7,95]])
y = np.array([0,0,0,1,1,1])

model = DecisionTreeClassifier(random_state=32)
model.fit(x,y)

new_stud = [[5,85]]
prediction = model.predict(new_stud)
print(f"Prediction for the new student is: {'Pass' if prediction[0] == 1 else 'Fail'}")