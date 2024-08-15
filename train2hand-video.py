import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Memuat data yang telah diproses untuk dua tangan
data_dict = pickle.load(open('./data_video.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Memisahkan data menjadi data latih dan data uji
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Tampilkan bentuk data untuk memastikan dimensi benar
print(x_train.shape, x_test.shape)

# Melatih model dengan RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Memprediksi hasil dengan data uji
y_predict = model.predict(x_test)

# Mengukur akurasi prediksi
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Menyimpan model ke file untuk penggunaan di masa mendatang
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
