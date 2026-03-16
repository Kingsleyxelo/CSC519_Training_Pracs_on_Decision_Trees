from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(iris.data, iris.target)

plt.figure(figsize=(14,10))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree")
plt.show()