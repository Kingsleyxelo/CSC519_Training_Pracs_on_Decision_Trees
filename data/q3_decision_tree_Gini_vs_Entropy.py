from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

gini = DecisionTreeClassifier(criterion='gini', random_state=42)
entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)

gini.fit(X_train, y_train)
entropy.fit(X_train, y_train)

print("Gini Accuracy:", accuracy_score(y_test, gini.predict(X_test)))
print("Entropy Accuracy:", accuracy_score(y_test, entropy.predict(X_test)))