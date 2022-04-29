import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Z space normalization
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from random import *
from csv import reader
from math import  *

# Csv dosyasını yükleme kodu
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Dize sütununu noktalı sayıya dönüştürme kodu
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Dize sütununu tam sayıya dönüştürme kodu
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Her sütun için minimum ve maksimum değerleri bulundu.
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Veri kümesi sütunlarını 0-1 aralığına normalize edildi.
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
# Bir veri kümesini k. kata bölme
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Doğruluk yüzdesi hesaplandı.
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# İki vektör arasındaki Öklid Mesafesi hesaplatıldı.
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)
# Benzer komşuların konumu tespit edildi.
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
# Komşuluk tahmini yapıldı.
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# KNN Algoritması
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return (predictions)

seed(1)
filename = 'iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(filename, names=names)
datasets = load_csv(filename)
# Değerler noktalı sayılara dönüştürüldü.
for i in range(len(datasets[0])-1):
	str_column_to_float(datasets, i)
# Değerler tam sayılara dönüştürüldü.
str_column_to_int(datasets, len(datasets[0])-1)
# print(dataset)
kul_giris=int(input("1-Max - min Hesaplama ; 2- Z-Space hesaplama: \t "))
if(kul_giris==1):
    n_folds = 5
    num_neighbors = 5
    scores = evaluate_algorithm(datasets, k_nearest_neighbors, n_folds, num_neighbors)
    print('Max-Min göre hesaplanmış değerler: %s' % scores)
    print('Doğruluk Oranı: %.3f%%' % (sum(scores) / float(len(scores))))
elif(kul_giris==2):
    #Z space Normalization
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def dist_func(item, data_point):
  sum = 0.0
  for i in range(2):
    diff = item[i] - data_point[i+1]
    sum += diff * diff
  return np.sqrt(sum)

def make_weights(k, distances):
  result = np.zeros(k, dtype=np.float32)
  sum = 0.0
  for i in range(k):
    result[i] += 1.0 / distances[i]
    sum += result[i]
  result /= sum
  return result

print("Ortalamalar....")

def show(v):
  print("idx = %3d (%3.2f %3.2f) Sınıf = %2d  " \
    % (v[0], v[1], v[2], v[3]), end="")

data=np.array(pd.read_csv("iris.csv"))
item = np.array([0.62, 0.35], dtype=np.float32)
N = len(data)
k = 6
c = 3
distances = np.zeros(N)
for i in range(N):
    distances[i] = dist_func(item, data[i])

ordering = distances.argsort()

k_near_dists = np.zeros(k, dtype=np.float32)
for i in range(k):
    idx = ordering[i]
    show(data[idx])
    print("Mesafe = %0.4f" % distances[idx])
    k_near_dists[i] = distances[idx]
votes = np.zeros(c, dtype=np.float32)
wts = make_weights(k, k_near_dists)

print("\n Ağırlıklar (ters mesafe tekniğiyle hesaplanmıştır.): ")

for i in range(len(wts)):
    print("%7.4f" % wts[i], end="")

print("\n\n Tahmin edilen ağırlıklı sınıf ortalamaları: ")

for i in range(k):
    idx = ordering[i]
    pred_class = np.int(data[idx][3])
    votes[pred_class] += wts[i] * 1.0
for i in range(c):
    print("[%d]  %0.4f" % (i, votes[i]))




