from sklearn.tree import DecisionTreeClassifier
import numpy as np

customer_data = np.array([
    [25, 30000, 1],
    [30, 45000, 3],
    [35, 60000, 2],
    [40, 75000, 5],
    [45, 90000, 4],
    [50, 55000, 0],
    [55, 120000, 6],
    [60, 80000, 2],
    [28, 40000, 1],
    [52, 65000, 3]
])
buy = np.array([0,0,1,1,1,0,1,1,0,1])

model = DecisionTreeClassifier(random_state=42)
model.fit(customer_data, buy)

new_customers = [[38, 70000, 4], [29, 35000, 1]]
prediction = model.predict(new_customers)
print(f"Customer A: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Customer B: {'Yes' if prediction[1] == 1 else 'No'}")