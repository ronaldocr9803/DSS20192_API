# try:
    # now it can reach class A of file_a.py in folder a 
    # by relative import
# except (ModuleNotFoundError, ImportError) as e:
# 	print("{} fileure".format(type(e)))
# else:
# 	print("Import succeeded")
from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json 
import _pickle as pickle
#from .predictRF import predictIris
# from ..iris.predictRF import predictIris
import os
import sys
lib_path = os.path.abspath(os.path.join('../iris'))# Create your views here.
print("before: ")
print(sys.path)
print(lib_path)
sys.path.append(lib_path)
print(sys.path)
from predictRF import predictIris
import pandas as pd
import numpy as np
import datetime 
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time


@api_view(["POST"])
def testThang(request):
	try: 
		data = json.loads(request.body)
		#weight = str(height*10)
		# with open('iris_rfc.pkl', 'rb') as f:
		# 	my_random_forest = pickle.load(f)

		#my_random_forest = pickle.load(open("iris_rfc.pkl","rb"))
		#result = my_random_forest.predict([[5.84,3.0,3.75,1.1]])
		#result = my_random_forest.predict([[request['sl'],request['sw'],request['pl'],request['pw']]])[0]
		ir = predictIris(data['sl'],data['sw'],data['pl'],data['pw'])


		#return JsonResponse("Hahaaah response"+weight+"kg",safe=False)
		return Response({'result':ir.get_predict()})
	except ValueError as e:
		# return JsonResponse(e.args[0],status.HTTP_400_BAD_REQUEST)
		return Response(status=status.HTTP_400_BAD_REQUEST)

def time_unix(a: datetime):
    return int(time.mktime(a.timetuple())*1000)

@api_view(["GET"])
def testTime(request):
	try: 
		df=pd.read_csv('MyApp/data/IPN31152N.csv',index_col=0)
		df.index = pd.date_range(start='1972-01-01', end='2020-01-01', freq='M')
		train_df=df[df.index <= '2017-12-31']
		test_df=df[df.index > '2017-12-31']
		model1 = SARIMAX(train_df['IPN31152N'],order=(3,1,1),seasonal_order=(0,1,1,12)).fit()
		pred = model1.predict(start=len(train_df),end=len(train_df)+len(test_df)-1,type='levels')
		df_pred=pd.DataFrame(pred)
		df_pred.columns=['IPN31152N']
		results = {'test': [[time_unix(test_df.index[i]),test_df.iloc[i]['IPN31152N']] for i in range(0,len(test_df)-1)], 'predict': [[time_unix(df_pred.index[i]),df_pred.iloc[i]['IPN31152N']] for i in range(0,len(df_pred)-1)]}
		#context = {"data":json.dumps(results), "aa":"hihi","haha":"hahahaha"}
		return Response({'result':results})
	except ValueError as e:
		# return JsonResponse(e.args[0],status.HTTP_400_BAD_REQUEST)
		return Response(status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'POST'])
def overView(request):
	if request.method == 'GET':
		try: 
			df=pd.read_csv('MyApp/data/IPN31152N.csv',index_col=0)
			df.index = pd.date_range(start='1972-01-01', end='2020-01-01', freq='M')
			df_overview = df[df.index.year == 2019]
			results = {'overview': [[time_unix(df_overview.index[i]),df_overview.iloc[i]['IPN31152N']] for i in range(0,len(df_overview))]}
			#context = {"data":json.dumps(results), "aa":"hihi","haha":"hahahaha"}
			return Response({'result':results})
		except ValueError as e:
			# return JsonResponse(e.args[0],status.HTTP_400_BAD_REQUEST)
			return Response(status=status.HTTP_400_BAD_REQUEST)
	elif request.method == 'POST':
		try:
			data = json.loads(request.body)
			df=pd.read_csv('MyApp/data/IPN31152N.csv',index_col=0)
			year = int(data['year'])
			df.index = pd.date_range(start='1972-01-01', end='2020-01-01', freq='M')
			df_overview = df[df.index.year == year]
			results = {'overview': [[time_unix(df_overview.index[i]),df_overview.iloc[i]['IPN31152N']] for i in range(0,len(df_overview))]}
			#context = {"data":json.dumps(results), "aa":"hihi","haha":"hahahaha"}
			return Response({'result':results})
		except ValueError as e:
			# return JsonResponse(e.args[0],status.HTTP_400_BAD_REQUEST)
			return Response(status=status.HTTP_400_BAD_REQUEST)

@api_view(["GET"])
def averageByYear(request):
	try:
		df=pd.read_csv('MyApp/data/IPN31152N.csv',index_col=0)
		df.index = pd.date_range(start='1972-01-01', end='2020-01-01', freq='M')
		df_year = df.groupby(df.index.year).mean()
		results = {'overview_year': [[df_year.index[i],df_year.iloc[i]['IPN31152N']] for i in range(0,len(df_year))]}
		#context = {"data":json.dumps(results), "aa":"hihi","haha":"hahahaha"}
		return Response({'result':results})
	except ValueError as e:
		# return JsonResponse(e.args[0],status.HTTP_400_BAD_REQUEST)
		return Response(status=status.HTTP_400_BAD_REQUEST)
	


