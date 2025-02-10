# Импорт необходимых библиотек
import numpy as np # для линейной алгебры
import pandas as pd # для обработки данных и работы с CSV файлами

# Вывод списка всех файлов в директории входных данных
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Загрузка обучающих данных
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()  # Вывод первых 5 строк обучающих данных

# Загрузка тестовых данных
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()  # Вывод первых 5 строк тестовых данных

# Расчет процента выживших женщин
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

# Расчет процента выживших мужчин
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)

# Импорт алгоритма Random Forest
from sklearn.ensemble import RandomForestClassifier

# Подготовка данных для обучения модели
y = train_data["Survived"]  # Целевая переменная

features = ["Pclass", "Sex", "SibSp", "Parch"]  # Выбор признаков для обучения
X = pd.get_dummies(train_data[features])  # One-hot кодирование признаков для обучающих данных
X_test = pd.get_dummies(test_data[features])  # One-hot кодирование признаков для тестовых данных

# Создание и обучение модели Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Предсказание на тестовых данных
predictions = model.predict(X_test)

# Создание файла с предсказаниями для отправки на Kaggle
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")