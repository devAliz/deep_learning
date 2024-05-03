import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# بارگیری و آماده‌سازی داده‌ها
data = pd.read_csv('iris.csv')
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# آموزش مدل
model = LogisticRegression()
model.fit(X_train, y_train)

# ارزیابی مدل
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy', accuracy)