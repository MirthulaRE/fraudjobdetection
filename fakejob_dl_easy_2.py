# -*- coding: utf-8 -*-
"""fakejob_dl_easy_2.py"""

# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import nltk
from nltk.corpus import stopwords
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv("fake_job_postings.csv")  # Ensure this file is present
df.fillna(" ", inplace=True)  # Fill missing values

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    df[col] = df[col].apply(preprocess_text)
df['text'] = df[text_columns].apply(lambda x: ' '.join(x), axis=1)

# TF-IDF Vectorization (IMPORTANT FIX)
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['text']).toarray()  # Fit and transform text data

# Save the fitted TF-IDF Vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("✅ TF-IDF Vectorizer saved successfully as tfidf_vectorizer.pkl")

# One-hot encode labels
y = to_categorical(df['fraudulent'], num_classes=2)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Deep Neural Network (DNN)
dnn_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile DNN model
dnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train DNN model
dnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

# Save the trained model (IMPORTANT FIX)
dnn_model.save("dnn_model.keras")
print("✅ DNN Model saved successfully as dnn_model.h5")

# Evaluate the model
dnn_loss, dnn_accuracy = dnn_model.evaluate(X_test, y_test)
print(f"DNN Test Accuracy: {dnn_accuracy:.2%}")

# Classification Report
y_pred_dnn = np.argmax(dnn_model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
print("\nDNN Classification Report:")
print(classification_report(y_test_labels, y_pred_dnn))
