import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt_tab')
nltk.download('stopwords')

def load_data(data_path):
    data =[]

    for filename in os.listdir(data_path):

        if (filename.startswith('spm')):
            with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                data.append((content, 1))
        else:
            with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                data.append((content, 0)) 

    return pd.DataFrame(data, columns=['text', 'label'])

def train_data():
    training_data = load_data('train-mails')
    test_data = load_data('test-mails')

    X_train = training_data['text']
    y_train = training_data['label']
    X_test = test_data['text']
    y_test = test_data['label']

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    lgClassifier = LogisticRegression()
    lgClassifier.fit(X_train_tfidf, y_train)
    y_pred1 = lgClassifier.predict(X_test_tfidf)
    score1 = accuracy_score(y_test, y_pred1)
    cm1 = confusion_matrix(y_test, y_pred1)
    # accuracy = 0.9769230769230769

    nbClassifier = GaussianNB()
    nbClassifier.fit(X_train_tfidf.toarray(), y_train.to_numpy())
    y_pred2 = nbClassifier.predict(X_test_tfidf.toarray())
    score2 = accuracy_score(y_test, y_pred2)
    cm2 = confusion_matrix(y_test, y_pred2)
    # accuracy = 0.9461538461538461

    rfClassifier = RandomForestClassifier()
    rfClassifier.fit(X_train_tfidf, y_train)
    y_pred3 = rfClassifier.predict(X_test_tfidf)
    score3 = accuracy_score(y_test, y_pred3)
    cm3 = confusion_matrix(y_test, y_pred3)
    # accuracy = 0.9807692307692307

    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train_tfidf, y_train)
    y_pred4 = knn.predict(X_test_tfidf)
    score4 = accuracy_score(y_test, y_pred4)
    cm4 = confusion_matrix(y_test, y_pred4)
    # accuracy = 0.9730769230769231

    svc = SVC(kernel='linear')
    svc.fit(X_train_tfidf, y_train)
    y_pred5 = svc.predict(X_test_tfidf)
    score5 = accuracy_score(y_test, y_pred5)
    cm5 = confusion_matrix(y_test, y_pred5)
    # accuracy = 0.9846153846153847

    return svc, tfidf_vectorizer

def training(email, svc, tfidf_vectorizer):
    email_converted = tfidf_vectorizer.transform([email])
    result = svc.predict(email_converted)
    if(result[0]==1):
        return "spam"
    else:
        return "ham"
    

def preprocess_email_content(email_content):

    sentences = sent_tokenize(email_content)

    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        processed_sentences.append(words)

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    stemmed_sentences = []
    for sentence in processed_sentences:
        sentence = [word.lower() for word in sentence if word.isalpha()]
        sentence = [word for word in sentence if word not in stop_words]
        stemmed_sentence = [stemmer.stem(word) for word in sentence]
        
        stemmed_sentences.append(' '.join(stemmed_sentence))
    
    return ' , '.join(stemmed_sentences)


email = """
Dear User,

Thank you for using our service. Please let us know if you have any questions or concerns.

Best regards,
Customer Support
"""

processed_email = preprocess_email_content(email)
print("Processed Email:", processed_email)