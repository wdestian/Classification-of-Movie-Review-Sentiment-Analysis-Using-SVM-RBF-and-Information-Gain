from django.urls import path, include
from . import views

app_name = 'administrator'

urlpatterns = [
    path('', views.index, name='index'),
    path('sentimen_analisis/', views.sentimen, name='sentimen'),
    path('tentang/', views.tentang, name='tentang'),
    path('hasil/', views.hasil, name='hasil'),
    path('fitur/', views.fitur, name='fitur'),
    path('klasifikasi/', views.klasifikasi, name='klasifikasi'),
    path('SVM/', views.SVM, name='SVM'),
    path('hasilsvm/', views.hasilsvm, name='hasilsvm'),
    path('SVMRBF/', views.SVMRBF, name='SVMRBF'),
    path('hasilsvmrbf/', views.hasilsvmrbf, name='hasilsvmrbf'),
    path('SVMRBFIG/', views.SVMRBFIG, name='SVMRBFIG'),
    # path('latih/', views.latih, name='latih'),
    # path('save/', views.savekonfigurasi, name='save_konfigurasi'),
    # path('recognize/', views.face_recognition, name='face_recognition'),

]
