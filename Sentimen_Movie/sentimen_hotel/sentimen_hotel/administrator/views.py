from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.utils.decorators import decorator_from_middleware
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib import messages
import random
from operator import itemgetter
import zipfile
import os
from shutil import copyfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from time import time

import nltk
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics



import shutil

import time
import numpy as np
import csv




# Create your views here.

def get_stemmed_text(corpus):
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

def remove_stop_words(corpus, english_stop_words):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
        ' '.join([word for word in review.split()
        if word not in english_stop_words])
        )
    return removed_stop_words







@login_required(login_url=settings.LOGIN_URL)
def index(request):
    return render(request, 'administrator/dashboard.html')

@login_required(login_url=settings.LOGIN_URL)
def tentang(request):
    return render(request, 'administrator/tentang.html')

@login_required(login_url=settings.LOGIN_URL)
def SVM(request):
    return render(request, 'administrator/SVM.html')

@login_required(login_url=settings.LOGIN_URL)
def SVMRBF(request):
    return render(request, 'administrator/SVMRBF.html')

@login_required(login_url=settings.LOGIN_URL)
def SVMRBFIG(request):
    return render(request, 'administrator/SVMRBFIG.html')

@login_required(login_url=settings.LOGIN_URL)
def sentimen(request):
    if request.method == 'POST':
        file = request.FILES['data']
        if default_storage.exists('dataset.csv'):
            default_storage.delete('dataset.csv')
        file_name = default_storage.save('dataset.csv', file)

        dataset = []
        data = pd.read_csv(default_storage.path('dataset.csv'))
        for x in range(len(data['Message'])):
            temp = []
            temp.append(data['Message'][x])
            temp.append(data['kelas'][x])
            # print(data['sentiment'][x])
            dataset.append(temp)
        # path = default_storage.save('dataset.csv', ContentFile(file.read()))
        messages.success(request,'Dataset berhasil diupload!')

        return render(request, 'administrator/sentimen.html',{'dataset': dataset})
    else:
        if default_storage.exists('dataset.csv'):
            dataset = []
            data = pd.read_csv(default_storage.path('dataset.csv'))
            print(data['kelas'])
            for x in range(len(data['Message'])):
                temp = []
                temp.append(data['Message'][x])
                temp.append(data['kelas'][x])
                # print(data['sentiment'][x])
                dataset.append(temp)

            # with open(default_storage.path('dataset.csv'), 'r') as data:
            #     reader = csv.reader(data)
            #     dataset = []
            #     for row in reader:
            #         dataset.append(row)
            # print(dataset)

        else:
            dataset = []
        # nama=[]
        # jumlah=[]
        # dataset=[]
        # if default_storage.exists('dataset'):
        #     for name in os.listdir(os.path.join(settings.BASE_DIR, 'media/dataset')):
        #         dataset.append([str(name),str(len(os.listdir(os.path.join(settings.BASE_DIR, 'media/dataset/'+name))))])
        # # print(dataset)
        return render(request, 'administrator/sentimen.html',{'dataset': dataset})



@login_required(login_url=settings.LOGIN_URL)
def hasil(request):
    if default_storage.exists('dataset.csv'):
        data = pd.read_csv(default_storage.path('dataset.csv'))
        tok = WordPunctTokenizer()
        pat1 = r'@[A-Za-z0-9]+'
        pat2 = r'https?://[A-Za-z0-9./]+'
        combined_pat = r'|'.join((pat1, pat2))
        def tweet_cleaner(Message):
            soup = BeautifulSoup(Message, 'lxml')
            souped = soup.get_text()
            stripped = re.sub(combined_pat, '', souped)
            try:
                clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
            except:
                clean = stripped
            letters_only = re.sub("[^a-zA-Z]", " ", clean)
            lower_case = letters_only.lower()
            # During the letters_only process two lines above, it has created unnecessay white spaces,
            # I will tokenize and join together to remove unneccessary white spaces
            words = tok.tokenize(lower_case)
            return (" ".join(words)).strip()
        testing = data.Message[:2000]
        test_result = []
        for t in testing:
            test_result.append(tweet_cleaner(t))
        # print(test_result)

        english_stop_words = stopwords.words('english')
        data_remove = remove_stop_words(test_result, english_stop_words)
        # data_remove[:2])

        stem_data = get_stemmed_text(data_remove)
        # stem_data[:2]
        # print(stem_data[:2])

        clean_df = pd.DataFrame(stem_data,columns=['Message'])
        clean_df['target'] = data.kelas
        if default_storage.exists('clean_review.csv'):
            default_storage.delete('clean_review.csv')
# clean_df.head()
        clean_df.to_csv(default_storage.path('clean_review.csv'), encoding='utf-8')

        return render(request, 'administrator/hasil.html',{'tokenize':test_result,'stopwords_removal':data_remove,'stemming':stem_data})
    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/sentimen_analisis/')
        pass

@login_required(login_url=settings.LOGIN_URL)
def fitur(request):
    if default_storage.exists('clean_review.csv'):
        csv = default_storage.path('clean_review.csv')
        my_df = pd.read_csv(csv, index_col=0)
        my_df.head(2000)
        x = my_df.Message
        y = my_df.target
        seed = 2000
        # Using CountVectorizer to convert text into tokens/features
        Encoder = LabelEncoder()
        Trans_Y = Encoder.fit_transform(y)
        Tfidf_vect = TfidfVectorizer(max_features=5000, min_df=4, max_df=0.8)
        Tfidf_vect.fit(x)
        X_Tfidf = Tfidf_vect.transform(x)
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X_Tfidf, Trans_Y, test_size=0.2, random_state=seed)
        ig = SelectKBest(score_func=mutual_info_classif, k=2100)
        X_new = ig.fit_transform(Train_X, Train_Y)
        feature_scores = [(item, score) for item, score in zip(Tfidf_vect.get_feature_names(), ig.scores_)]
        topk = sorted(feature_scores, key=lambda x: -x[1])[:-1]

        return render(request, 'administrator/fitur.html',{ 'topk': topk})
    else:
        messages.error(request, 'Dataset belum diinputkan!')
        return redirect('/administrator/sentimen_analisis/')
        pass

@login_required(login_url=settings.LOGIN_URL)
def klasifikasi(request):
    if request.method == 'GET':
        a = float(request.GET['num1'])
        b = float(request.GET['num2'])
        c = int(request.GET['num3'])
    if default_storage.exists('clean_review.csv'):
        csv = default_storage.path('clean_review.csv')
        my_df = pd.read_csv(csv, index_col=0)
        my_df.head(2000)
        x = my_df.Message
        y = my_df.target
        seed = 2000
        # Using CountVectorizer to convert text into tokens/features
        Encoder = LabelEncoder()
        Trans_Y = Encoder.fit_transform(y)
        Tfidf_vect = TfidfVectorizer(max_features=5000, min_df=4, max_df=0.8)
        Tfidf_vect.fit(x)
        X_Tfidf = Tfidf_vect.transform(x)
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X_Tfidf, Trans_Y, test_size=0.2, random_state=seed)
        ig = SelectKBest(score_func=mutual_info_classif, k=c)
        X_new = ig.fit_transform(Train_X, Train_Y)
        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        Train_x, Test_x, Train_y, Test_y = model_selection.train_test_split(X_new, Train_Y, random_state=seed)
        SVM_RBF_ig = SVC(C=a, kernel='rbf', gamma=b)
        SVM_RBF_ig.fit(Train_x, Train_y)
        # predict the labels on validation dataset
        predictions_SVM_RBF_ig = SVM_RBF_ig.predict(Test_x)
        # Use accuracy_score function to get the accuracy
        hasil_ig = accuracy_score(predictions_SVM_RBF_ig, Test_y) * 100
        # Making the Confusion Matrix
        accuracy_ig = {}
        cm = metrics.confusion_matrix(Test_y, predictions_SVM_RBF_ig)
        accuracy_ig[SVM_RBF_ig] = [SVM_RBF_ig,
                             round(metrics.accuracy_score(Test_y, predictions_SVM_RBF_ig), 2),
                             cm[0][0],  # true_negative
                             cm[0][1],  # false_positive
                             cm[1][0],  # false_negative
                             cm[1][1],  # true_positive
                             ]

        return render(request, 'administrator/klasifikasi.html',{'akurasi_ig':hasil_ig, 'matrix_ig':accuracy_ig})

    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/sentimen_analisis/')
        pass
@login_required(login_url=settings.LOGIN_URL)
def hasilsvm(request):
    if default_storage.exists('clean_review.csv'):
        csv = default_storage.path('clean_review.csv')
        my_df = pd.read_csv(csv, index_col=0)
        my_df.head(2000)
        x = my_df.Message
        y = my_df.target
        seed = 2000
        # Using CountVectorizer to convert text into tokens/features
        Encoder = LabelEncoder()
        Trans_Y = Encoder.fit_transform(y)
        Tfidf_vect = TfidfVectorizer(max_features=5000, min_df=4, max_df=0.8)
        Tfidf_vect.fit(x)
        X_Tfidf = Tfidf_vect.transform(x)
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X_Tfidf, Trans_Y, test_size=0.2, random_state=seed)
        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        SVM = SVC()
        SVM.fit(Train_X, Train_Y)
        # predict the labels on validation dataset
        predictions_SVM = SVM.predict(Test_X)
        # Use accuracy_score function to get the accuracy
        hasil_svm = accuracy_score(predictions_SVM, Test_Y) * 100
        # Making the Confusion Matrix
        accuracy_svm = {}
        cm = metrics.confusion_matrix(Test_Y, predictions_SVM)
        accuracy_svm[SVM] = [SVM,
                             round(metrics.accuracy_score(Test_Y, predictions_SVM), 2),
                             cm[0][0],  # true_negative
                             cm[0][1],  # false_positive
                             cm[1][0],  # false_negative
                             cm[1][1],  # true_positive
                             ]

        return render(request, 'administrator/hasilsvm.html',{'akurasi_svm':hasil_svm, 'matrix_svm':accuracy_svm})

    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/sentimen_analisis/')
        pass

@login_required(login_url=settings.LOGIN_URL)
def hasilsvmrbf(request):
    if request.method == 'GET':
        a = float(request.GET['num1'])
        b = float(request.GET['num2'])
        if default_storage.exists('clean_review.csv'):
            csv = default_storage.path('clean_review.csv')
            my_df = pd.read_csv(csv, index_col=0)
            my_df.head(2000)
            x = my_df.Message
            y = my_df.target
            seed = 2000
            # Using CountVectorizer to convert text into tokens/features
            Encoder = LabelEncoder()
            Trans_Y = Encoder.fit_transform(y)
            Tfidf_vect = TfidfVectorizer(max_features=5000, min_df=4, max_df=0.8)
            Tfidf_vect.fit(x)
            X_Tfidf = Tfidf_vect.transform(x)
            Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X_Tfidf, Trans_Y, test_size=0.2, random_state=seed)
            # Classifier - Algorithm - SVM
            # fit the training dataset on the classifier

            SVM_RBF = SVC(C=a, kernel='rbf', gamma=b)
            SVM_RBF.fit(Train_X, Train_Y)
            # predict the labels on validation dataset
            predictions_SVM_RBF = SVM_RBF.predict(Test_X)
            # Use accuracy_score function to get the accuracy
            hasil_rbf = accuracy_score(predictions_SVM_RBF, Test_Y) * 100
            # Making the Confusion Matrix
            accuracy_rbf = {}
            cm = metrics.confusion_matrix(Test_Y, predictions_SVM_RBF)
            accuracy_rbf[SVM_RBF] = [SVM_RBF,
                                 round(metrics.accuracy_score(Test_Y, predictions_SVM_RBF), 2),
                                 cm[0][0],  # true_negative
                                 cm[0][1],  # false_positive
                                 cm[1][0],  # false_negative
                                 cm[1][1],  # true_positive
                                 ]

            return render(request, 'administrator/hasilsvmrbf.html',{'akurasi_rbf':hasil_rbf, 'matrix_rbf':accuracy_rbf})

    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/sentimen_analisis/')
        pass
