#import self as self
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer
from django.contrib import messages
from imblearn.over_sampling import SMOTE
from django.http import HttpResponse
from django.shortcuts import render
# Create your views here.
from django.shortcuts import render
#from . forms import MyForm
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .forms import ApprovalForm
from . models import approvals
from . serializers import approvalsSerializers
import pickle
import json
import numpy as np
from sklearn import preprocessing
import pandas as pd
import csv
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense

class ApprovalsView(viewsets.ModelViewSet):
	queryset = approvals.objects.all()
	serializer_class = approvalsSerializers


def ohevalue(df):
    ohe_col = pd.read_csv(r'D:/machine learning tutorial/machine learning project/bankloan.csv')
    ohe_col = pd.DataFrame(ohe_col)
    ohe_col = ohe_col.dropna()  # remove Nan or null values row
    ohe_col_x = ohe_col.drop('Loan_ID', axis=1)
    ohe_col_x['LoanAmount'] = (ohe_col_x['LoanAmount'] * 1000).astype(int)
    ohe_col_x1= ohe_col_x.drop('Loan_Status', axis=1)
    ohe_col_x1 = pd.get_dummies(ohe_col_x1)
    cat_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    #ai term gulo on hot encoded hobe karon form a option(male,female) silo
    #but jeta select korbo(jemon male select korlam) seita bade(female) ki hobe ta ...
    #ml model janbe na tai newdict[i]=0 bosabo
    df_processed = pd.get_dummies(df, columns=cat_columns)
    newdict = {}
    for i in ohe_col_x1:
        if i in df_processed.columns:
            newdict[i] = df_processed[i].values
            #print("if : ",newdict[i])
        else:
            newdict[i] = 0
            #print("else : ",newdict[i])
    newdf = pd.DataFrame(newdict)
    return newdf

#@api_view(["POST"])#it converts the page with written (keras.models.load_model is ok) to a new page
def approvereject(unit):
    try:
        mdl = keras.models.load_model("D:/machine learning tutorial/machine learning project/bank_loan.h5")
        #print("helloooooo")
        #print(mdl.summary())
        #https://datatofish.com/import-csv-file-python-using-pandas/
        df=pd.read_csv(r'D:/machine learning tutorial/machine learning project/bankloan.csv')
        df = df.dropna()  # remove Nan or null values row
        df = df.drop('Loan_ID', axis=1)
        df['LoanAmount'] = (df['LoanAmount'] * 1000).astype(int)
        pre_y = df['Loan_Status']
        pre_x = df.drop('Loan_Status', axis=1)
        dm_x = pd.get_dummies(pre_x)
        dm_y = pre_y.map(dict(Y=1, N=0))

        smote = SMOTE(sampling_strategy='minority')
        x1, y = smote.fit_sample(dm_x, dm_y)
        # number of row will be increase because minority more new data will be added
        # to equal to the number of majority data (using average calculation & other calculation)
        sc = MinMaxScaler()
        x = sc.fit_transform(x1)
        #balance_loan_status = Counter(y)  # 1 for yes, 0 for no

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
        y_pred = mdl.predict(x_test)
        #sc = MinMaxScaler()
        #x = sc.fit_transform(unit)
        x=sc.transform(unit)
        #x=unit
        y_pred=mdl.predict(x)
        #print("unit : ", unit)
        #print("x : ",x)
        y_pred = (y_pred > 0.5)
        approve_reject = pd.DataFrame(y_pred)  ####
        approve_reject = approve_reject.replace({True: 'Approved', False: 'Rejected'})
        #print("approve_reject : ",approve_reject)
        #print(df.head(5))
        #series=pd.Series([10],['A'])
        #print(series)
        #return JsonResponse('Your Status is {}'.format(approve_reject), safe=False)
        return approve_reject.values[0][0]

        #return HttpResponse ('<h1>keras.models.load_model is ok</h1>')
    except ValueError as e:
        print("failed...........................")
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def cxcontact(request):
    if(request.method=='POST'):
        form=ApprovalForm(request.POST)
        if(form.is_valid()):
            Firstname = form.cleaned_data['Firstname']
            Lastname = form.cleaned_data['Lastname']
            Dependents = form.cleaned_data['Dependents']
            ApplicantIncome = form.cleaned_data['ApplicantIncome']
            CoapplicantIncome  = form.cleaned_data['CoapplicantIncome']
            LoanAmount = form.cleaned_data['LoanAmount']
            Loan_Amount_Term = form.cleaned_data['Loan_Amount_Term']
            Credit_History = form.cleaned_data['Credit_History']
            Gender = form.cleaned_data['Gender']
            Married = form.cleaned_data['Married']
            Education = form.cleaned_data['Education']
            Self_Employed = form.cleaned_data['Self_Employed']
            Property_Area = form.cleaned_data['Property_Area']
            #print(Firstname, Lastname, Dependents, Married, Property_Area)

            myDict = (request.POST).dict()
            df = pd.DataFrame(myDict, index=[0])
            #print("df :", df)
            ohevalue_call=ohevalue(df)

            #print("ohevalue : ", ohevalue_call)
            #print("ohevalue : ", ohevalue_call.columns)
            #print("column name : ", ohevalue(df).columns)
            answer=approvereject(ohevalue_call)
            #print("answer :", answer)
            messages.success(request, 'Application Status: {}'.format(answer))
            #return JsonResponse('Application Status: {}'.format(answer), safe=False)
            #answer = serializers.serialize('json', self.get_queryset)
            #return HttpResponse(answer, content_type="application/json")
            #print(ohevalue(df))
            #return HttpResponse(json.dumps(answer), content_type="application/json")

    form = ApprovalForm()
    return render(request, 'myform/csform.html', {'form': form})

