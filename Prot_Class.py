import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def one_hot_encoding(aminoacido):
    vec = np.zeros(len(aminoacidi))
    vec[aminoacid_index[aminoacido]] = 1
    return vec

def sequence_to_one_hot(seq):
    return np.concatenate([one_hot_encoding(aminoacido) for aminoacido in seq])

aminoacidi = 'ACDEFGHIKLMNOPQRSTUVWY'
aminoacid_index = {aminoacido: i for i, aminoacido in enumerate(aminoacidi)}

files = ["transport.xml", "trasducer.xml"]

train_data = []
validation_data = []
test_data = []

for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    namespace = {'ns': root.tag.split('}')[0].strip('{')}
    sequences = [entry.find('ns:sequence', namespace).text for entry in root.findall(".//ns:entry", namespace)]

    label = 0 if 'transport' in file else 1
    train_data.extend([(seq, label) for seq in sequences[:300]])
    validation_data.extend([(seq, label) for seq in sequences[300:400]])
    test_data.extend([(seq, label) for seq in sequences[400:500]])

train_sequences, train_labels = zip(*train_data)
validation_sequences, validation_labels = zip(*validation_data)
test_sequences, test_labels = zip(*test_data)

train_sequences = [sequence_to_one_hot(seq) for seq in train_sequences if 'X' not in seq]
validation_sequences = [sequence_to_one_hot(seq) for seq in validation_sequences if 'X' not in seq]
test_sequences = [sequence_to_one_hot(seq) for seq in test_sequences if 'X' not in seq]

max_length = max(len(seq) for seq in train_sequences + validation_sequences + test_sequences)

train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post', dtype='float32')
validation_sequences = pad_sequences(validation_sequences, maxlen=max_length, padding='post', dtype='float32')
test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post', dtype='float32')

train_df = pd.DataFrame(train_sequences)
validation_df = pd.DataFrame(validation_sequences)
test_df = pd.DataFrame(test_sequences)


print("Train DataFrame:")
print(train_df)
print("\nValidation DataFrame:")
print(validation_df)
print("\nTest DataFrame:")
print(test_df)

scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
validation_df = scaler.transform(validation_df)
test_df = scaler.transform(test_df)

pca = PCA(n_components=150)
train_pca = pca.fit_transform(train_df)
validation_pca = pca.transform(validation_df)
test_pca = pca.transform(test_df)

train_pca_df = pd.DataFrame(train_pca)
validation_pca_df = pd.DataFrame(validation_pca)
test_pca_df = pd.DataFrame(test_pca)


print("Train DataFrame dopo PCA:")
print(train_pca_df)
print("\nValidation DataFrame dopo PCA:")
print(validation_pca_df)
print("\nTest DataFrame dopo PCA:")
print(test_pca_df)

explained_variance = pca.explained_variance_ratio_
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

rf = RandomForestClassifier()

CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rf.fit(train_pca, np.array(train_labels))

print(CV_rf.best_params_)

validation_score_rf = CV_rf.score(validation_pca, np.array(validation_labels))
print("Random Forest Validation score: ", validation_score_rf)
y_pred_rf = CV_rf.predict(validation_pca)
cm_rf = confusion_matrix(validation_labels, y_pred_rf)
rf_validrep = classification_report(validation_labels, y_pred_rf)
print(rf_validrep)

sns.heatmap(cm_rf, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix')
plt.show()

svm = SVC()

svm.fit(train_pca, np.array(train_labels))

validation_score_svm = svm.score(validation_pca, np.array(validation_labels))
print("SVM Validation score: ", validation_score_svm)

y_pred_svm = svm.predict(validation_pca)
cm_svm = confusion_matrix(validation_labels, y_pred_svm)
report_Svm = classification_report(validation_labels, y_pred_svm)
print(report_Svm)

sns.heatmap(cm_svm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('SVM Confusion Matrix')
plt.show()

knn = KNeighborsClassifier()

knn.fit(train_pca, np.array(train_labels))

validation_score_knn = knn.score(validation_pca, np.array(validation_labels))
print("KNN Validation score: ", validation_score_knn)

y_pred_knn = knn.predict(validation_pca)
cm_knn = confusion_matrix(validation_labels, y_pred_knn)
report_Knn = classification_report(validation_labels, y_pred_knn)
print(report_Knn)

sns.heatmap(cm_knn, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('KNN Confusion Matrix')
plt.show()

y_pred_rf_test = CV_rf.predict(test_pca)
cm_rf_test = confusion_matrix(test_labels, y_pred_rf_test)
report_rf = classification_report(test_labels, y_pred_rf_test)
print(report_rf)

sns.heatmap(cm_rf_test, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix - Test Data')
plt.show()

validation_score_svm_test = svm.score(test_pca, np.array(test_labels))
print("SVM Test score: ", validation_score_svm_test)

y_pred_svm_test = svm.predict(test_pca)
cm_svm_test = confusion_matrix(test_labels, y_pred_svm_test)

sns.heatmap(cm_svm_test, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('SVM Confusion Matrix - Test Data')
plt.show()