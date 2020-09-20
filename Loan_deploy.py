# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:17:14 2020

@author: chiag
"""
import os as os
import sys
import pandas as pd
import xlrd
import numpy as np
from IPython.display import display
import math
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

#Classifiers
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pickle


#Clustering
from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import silhouette_visualizer
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as shc


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from apyori import apriori
import streamlit as st



import warnings
warnings.filterwarnings('ignore')

st.title(' Intelligent Decision-Making for Loan Application')
image = Image.open('cover_header.jpg')
st.image(image,  use_column_width=True)

@st.cache
def load_data(title):
    original_dataset = pd.read_csv(title+'.csv')
    return original_dataset

data = load_data('Bank_CS')
st.write("## Raw Data")
if st.checkbox('Show Raw Data',value=False):
        st.write(data)
        st.write("Number of rows:" , np.shape(data)[0])
        st.write("Number of columns:" , np.shape(data)[1])
        
cleaning_df = load_data('cleaning_df1')

if st.checkbox('Show Cleaned Data',value=False):
        st.write(cleaning_df)
        st.write("Number of rows:" , np.shape(cleaning_df)[0])
        st.write("Number of columns:" , np.shape(cleaning_df)[1])

balanced_df = load_data('balanced_df')

encoded_df = load_data('encoded_df')






################# TEMP START HERE #################
st.title('Exploratory Data Analysis(EDA)')
st.write("## Question1")
st.write("### Which group of people recorded the highest amount of loan?")
df_eda = cleaning_df.copy()

q1 = df_eda[["EMPLOYMENT_TYPE","LOAN_AMOUNT"]]

employer_q1 = q1["EMPLOYMENT_TYPE"]=='Employer'
employer_q1 = q1[employer_q1]
employer_q1 = employer_q1["LOAN_AMOUNT"].sum()
employer_q1 = int(employer_q1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
selfemp_q1  = q1["EMPLOYMENT_TYPE"]=='Self-Employed'
selfemp_q1  = q1[selfemp_q1]
selfemp_q1  = selfemp_q1["LOAN_AMOUNT"].sum()
selfemp_q1  = int(selfemp_q1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
govt_q1     = q1["EMPLOYMENT_TYPE"]=='Government'
govt_q1     = q1[govt_q1]
govt_q1     = govt_q1["LOAN_AMOUNT"].sum()
govt_q1     = int(govt_q1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
employee_q1 = q1["EMPLOYMENT_TYPE"]=='Employee'
employee_q1 = q1[employee_q1]
employee_q1 = employee_q1["LOAN_AMOUNT"].sum()
employee_q1 = int(employee_q1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
fresh_q1    = q1["EMPLOYMENT_TYPE"]=='Fresh Graduate'
fresh_q1    = q1[fresh_q1]
fresh_q1    = fresh_q1["LOAN_AMOUNT"].sum()
fresh_q1    = int(fresh_q1)

names       = 'Employer', 'Self-Employed', 'Government', 'Employee', 'Fresh Graduate'
size        = [employer_q1, selfemp_q1, govt_q1, employee_q1, fresh_q1]


q1_img = Image.open('Question1.jpg')
st.image(q1_img,  use_column_width=True)

question1     = {'EMPLOYMENT TYPE'      : ['Employer', 'Self-Employed', 'Government', 'Employee', 'Fresh Graduate'],
            'OVERALL LOAN AMOUNT'  : [employer_q1, selfemp_q1, govt_q1, employee_q1, fresh_q1]}
olm      = pd.DataFrame (question1, columns = ['EMPLOYMENT TYPE','OVERALL LOAN AMOUNT'])

olm.sort_values(by=['OVERALL LOAN AMOUNT'], inplace=True)
olm['OVERALL LOAN AMOUNT'] = olm.apply(lambda x: "{:,}".format(x['OVERALL LOAN AMOUNT']), axis=1)
olm.reset_index(drop=True,inplace=True)
olm

########### Question 2 ###########

st.write("## Question2")
st.write("###  Which group of people recorded the highest amount of loan?")

df_temp=df_eda.copy()
df_temp=df_temp[~df_temp.PROPERTY_TYPE.str.contains("Not Specified")]
question2 = df_temp.groupby(['PROPERTY_TYPE','EMPLOYMENT_TYPE'])['PROPERTY_TYPE'].count()

q2_img = Image.open('Question2.jpg')
st.image(q2_img,  use_column_width=True)
    
question2

########### Question 3 ###########

st.write("## Question3")
st.write("###  What is the frequency for property type in each state?")

df_temp=df_eda.copy()
df_temp=df_temp[~df_temp.PROPERTY_TYPE.str.contains("Not Specified")]
question3=df_temp.groupby('STATE')['PROPERTY_TYPE'].value_counts()

q3_img = Image.open('Question3.jpg')
st.image(q3_img,  use_column_width=True)
#statistic info
question3

########### Question 4 ###########

st.write("## Question4")
st.write("###  Which credit card type holder that gets the loan accepted the most?")

q4     = df_eda[["CREDIT_CARD_TYPES", "DECISION"]]

plat   = ((q4["CREDIT_CARD_TYPES"] == 'Platinum') & (q4["DECISION"] =='Accept'))
plat   = q4[plat]
plat   = len(plat.index)

gold   = ((q4["CREDIT_CARD_TYPES"] == 'Gold') & (q4["DECISION"] =='Accept'))
gold   = q4[gold]
gold   = len(gold.index)

norm   = ((q4["CREDIT_CARD_TYPES"] == 'Normal') & (q4["DECISION"] =='Accept'))
norm   = q4[norm]
norm   = len(norm.index)

nots   = ((q4["CREDIT_CARD_TYPES"] == 'Not Specified') & (q4["DECISION"] =='Accept'))
nots   = q4[nots]
nots   = len(nots.index)

names       = 'Platinum', 'Gold', 'Normal', 'Not Specified'
size        = [plat, gold, norm, nots]

fig4, ax = plt.subplots(figsize=(15,9))
my_circle   = plt.Circle((0,0), 0.7, color='white')


q4_img = Image.open('Question4.jpg')
st.image(q4_img,  use_column_width=True)
########### Question 5 ###########

st.write("## Question5")
st.write("###  What is the average income for all types of employment?")

df_temp=df_temp[~df_temp.CREDIT_CARD_TYPES.str.contains("Not Specified")]
question5 = df_temp.groupby(['CREDIT_CARD_TYPES','EMPLOYMENT_TYPE'])['MONTHLY_SALARY'].agg('mean')

#PLOT 

q5_img = Image.open('Question5.jpg')
st.image(q5_img,  use_column_width=True)

question5

########### Question 6 ###########


st.write("## Question6")
st.write("###  What is the loan amount distribution among all types of properties?")


q6_img = Image.open('Question6.jpg')
st.image(q6_img,  use_column_width=True)   


################# TEMP END HERE #################


########### Feature Selection ###########
st.title('Feature Selection')
dataset_selection = st.selectbox("Select a dataset to perform data mining",('','Raw Dataset (Imbalanced)','SMOTE Dataset (Balance)'))

if not dataset_selection:
    st.warning('Please select a dataset.')
    st.stop()
    st.success('Thank you for selecting a dataset.')

feature_selection_types = st.selectbox("Select a feature selection algorithm",('','Boruta','RFE'))

if not feature_selection_types:
    st.warning('Please select a feature selection algorithm.')
    st.stop()
    st.success('Thank you for selecting a feature selection.')
    
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


if(dataset_selection == 'Raw Dataset (Imbalanced)'):
    X = encoded_df.drop("DECISION",1)
    y = encoded_df["DECISION"]
    colnames = X.columns
    features =encoded_df.copy()
    

if(dataset_selection == 'SMOTE Dataset (Balance)'):
    X = balanced_df.drop("DECISION",1)
    y = balanced_df["DECISION"]
    colnames = X.columns
    features =balanced_df.copy()
    
features_selected = None
if (feature_selection_types == 'Boruta'):
    st.write('Boruta feature selection selected')
###########Boruta###########
    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 11)
    feat_selector = BorutaPy(rf, n_estimators="auto", random_state = 5)
    feat_selector.fit(X.values, y.values.ravel())
    
    boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
    boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
    
    boruta_score = boruta_score.sort_values("Score", ascending = False)
    
    print('---------Boruta Score on Balanced dataset----------')
    boruta_score
    features_selected = boruta_score.copy()


###########RFE###########
if (feature_selection_types == 'RFE'):
    st.write('RFE feature selection selected')
    rf = RandomForestClassifier(n_jobs = 1, class_weight = "balanced",max_depth = 5, n_estimators = 100)
    rf.fit(X,y)
    rfe = RFECV(rf,min_features_to_select =1, cv=3)
    rfe.fit(X,y)
    rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
    rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
    rfe_score = rfe_score.sort_values("Score", ascending = False)
    print('---------Boruta Score on Balanced dataset----------')
    rfe_score
    features_selected = rfe_score.copy()
    
threshold = st.number_input('Input a threshold value for minimum score (0 - 1.0)') 
if not threshold:
    st.warning('Please input a valid score.')
    st.stop()
    st.success('Thank you for inputting a valid score.')

features_selected_index = features_selected.index[features_selected['Score'] < float(threshold)].tolist()


#Dimensionality reduction
features.drop(features.columns[features_selected_index],axis=1,inplace=True)



##### Split train test dataset Split train test dataset
X = features.drop("DECISION",1)
y = features["DECISION"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)


if st.checkbox('Show feature selected data',value=False):
        st.write(features)
        st.write("Number of rows:" , np.shape(features)[0])
        st.write("Number of columns:" , np.shape(features)[1])


st.sidebar.title('Select a technique:')
selection = st.sidebar.multiselect(' ',['Association Rule Mining','Classification','Clustering'])


if "Classification" in selection:
    ########### Classification ###########
    st.title('Classification')
    
    classification_algo = ['ADA Boosting','Decision Tree','Gradient Boosting','Logistic Regression',
                           'Naive Bayes','Random Forest Classification','Support Vector Machine',
                           'XG Boost']
    selection_classifier = st.multiselect('Select a classifier: ', classification_algo)
    
    if "ADA Boosting" in selection_classifier:
        st.markdown("## ADA Boosting")
        abc = AdaBoostClassifier(n_estimators=50,
                             learning_rate=1)
        #fit model
        ada = abc.fit(X_train, y_train)
        
        #y prediction
        y_pred = ada.predict(X_test)
        
        #proba
        prob_ADA = ada.predict_proba(X_test)
        prob_ADA = prob_ADA[:, 1]
        
        confusion_majority=confusion_matrix(y_test, y_pred)
        auc_ADA = roc_auc_score(y_test, prob_ADA)
        st.write('Accuracy on test set= {:.3f}'. format(accuracy_score(y_test, y_pred)*100),'%')
        #Apply cross validation
        abc_cv = cross_val_score(abc,X,y,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (10 folds cross validation)= {:.3f}'. format(abc_cv*100),'%')
        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)*100),'%')
        st.write('Recall= {:.3f}'. format(recall_score(y_test, y_pred)*100),'%')
        st.write('F1= {:.3f}'. format(f1_score(y_test, y_pred)*100),'%')
        st.write('AUC: {:.3f}'. format(auc_ADA*100),'%')
        st.write('Majority classifier Confusion Matrix\n', confusion_majority)
        
        st.write('Majority TN= ', confusion_majority[0][0])
        st.write('Majority FP=', confusion_majority[0][1])
        st.write('Majority FN= ', confusion_majority[1][0])
        st.write('Majority TP= ', confusion_majority[1][1])
    
        st.write('****************')     
    if "Decision Tree" in selection_classifier:
        st.markdown("## Decision Tree Classifier")
        model = DecisionTreeClassifier()
        #fit data
        model=model.fit(X_train,y_train)
        #y prediction
        y_pred=model.predict(X_test)
        #proba
        prob_DT = model.predict_proba(X_test)
        prob_DT = prob_DT [:, 1]
        
        confusion_majority=confusion_matrix(y_test, y_pred)
        auc_DT = roc_auc_score(y_test, prob_DT)
        st.write('Accuracy on test set= {:.3f}'. format(accuracy_score(y_test, y_pred)*100),'%')
        #Apply cross validation
        dt_cv = cross_val_score(model,X,y,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (10 folds cross validation)= {:.3f}'. format(dt_cv*100),'%')
        st.write('Precision= {:.3f}'.format(precision_score(y_test, y_pred)*100),'%')
        st.write('Recall= {: 3f}'. format(recall_score(y_test, y_pred)*100),'%')
        st.write('F1= {:.3f}'. format(f1_score(y_test, y_pred)*100),'%')
        st.write('AUC: {:.3f}'. format(auc_DT*100),'%')
        
        st.write('Majority classifier Confusion Matrix\n', confusion_majority)
        
        st.write('Majority TN= ', confusion_majority[0][0])
        st.write('Majority FP=', confusion_majority[0][1])
        st.write('Majority FN= ', confusion_majority[1][0])
        st.write('Majority TP= ', confusion_majority[1][1])
        
        
     
        st.write('****************')
    
    if "Gradient Boosting" in selection_classifier:
        st.markdown("## Gradient Boosting")
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
        #fit model
        gb = gb_clf.fit(X_train, y_train)
        #y prediction
        y_pred = gb.predict(X_test)
        #st.write(y_pred)
        #proba
        prob_GB = gb.predict_proba(X_test)
        prob_GB = prob_GB[:, 1]
        
        confusion_majority=confusion_matrix(y_test, y_pred)
        auc_GB = roc_auc_score(y_test, prob_GB)
        st.write('Accuracy on test set= {:.3f}'. format(accuracy_score(y_test, y_pred)*100),'%')
        #Apply cross validation
        gb_cv = cross_val_score(gb_clf,X,y,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (10 folds cross validation)= {:.3f}'. format(gb_cv*100),'%')
        st.write('Precision= {:.3f}'.format(precision_score(y_test, y_pred)*100),'%')
        st.write('Recall= {:.3f}'. format(recall_score(y_test, y_pred)*100),'%')
        st.write('F1= {:.3f}'. format(f1_score(y_test, y_pred)*100),'%')
        st.write('AUC: {:.3f}'. format(auc_GB*100),'%')
        
        st.write('Majority classifier Confusion Matrix\n', confusion_majority)
        
        st.write('Majority TN= ', confusion_majority[0][0])
        st.write('Majority FP=', confusion_majority[0][1])
        st.write('Majority FN= ', confusion_majority[1][0])
        st.write('Majority TP= ', confusion_majority[1][1])
    
        
        
        st.write('****************')
        
    if "Logistic Regression" in selection_classifier:
        st.markdown("## Logistic Regression")
        logreg = LogisticRegression()
        #fit data
        logreg.fit(X_train,y_train)
        #y prediction
        y_pred = logreg.predict(X_test)
        #proba
        prob_logreg = logreg.predict_proba(X_test)
        prob_logreg = prob_logreg [:, 1]
        
        
        confusion_majority=confusion_matrix(y_test, y_pred)
        auc_logreg = roc_auc_score(y_test, prob_logreg)
        st.write('Accuracy on test set= {:.3f}'. format(accuracy_score(y_test, y_pred)*100),'%')
        #Apply cross validation
        logreg_cv = cross_val_score(logreg,X,y,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (10 folds cross validation)= {:.3f}'. format(logreg_cv*100),'%')
        st.write('Precision= {:.3f}'.format(precision_score(y_test, y_pred)*100),'%')
        st.write('Recall= {:.3f}'. format(recall_score(y_test, y_pred)*100),'%')
        st.write('F1= {:.3f}'. format(f1_score(y_test, y_pred)*100),'%')
        st.write('AUC: {:.3f}'. format(auc_logreg*100),'%')
        
        st.write('Mjority classifier Confusion Matrix\n', confusion_majority)
        
        st.write('Majority TN= ', confusion_majority[0][0])
        st.write('Majority FP=', confusion_majority[0][1])
        st.write('Majority FN= ', confusion_majority[1][0])
        st.write('Majority TP= ', confusion_majority[1][1])
     
        st.write('****************')
        
    if "Naive Bayes" in selection_classifier:
        st.markdown("## Naive Bayes")
        nb=GaussianNB()
        #fit data
        nb.fit(X_train,y_train)
        #y prediction
        y_pred=nb.predict(X_test)
        
        #Proba
        prob_NB = nb.predict_proba(X_test)
        prob_NB = prob_NB[:,1]
        
        #Auc
        auc_NB= roc_auc_score(y_test, prob_NB)
        
        # Provide model accuracy, confusion matrix, <TN,TP,FP,FN>, Precision, Recall, F1, Accuracy
        confusion_majority=confusion_matrix(y_test, y_pred)
        st.write("Accuracy on test set: {:.3f}".format(nb.score(X_test, y_test)*100),'%')
        #Apply cross validation
        NB_cv = cross_val_score(nb,X,y,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (10 folds cross validation)= {:.3f}'. format(NB_cv*100),'%')
        st.write('Precision= {:.3f}'.format(precision_score(y_test, y_pred)*100),'%')
        st.write('Recall= {:.3f}'. format(recall_score(y_test, y_pred)*100),'%')
        st.write('F1= {:.3f}'. format(f1_score(y_test, y_pred)*100),'%')
        st.write('AUC: {:.3f}'. format(auc_NB*100),'%')
        
        st.write('Majority classifier Confusion Matrix\n', confusion_majority)
        
        st.write('Majority TN= ', confusion_majority[0][0])
        st.write('Majority FP=', confusion_majority[0][1])
        st.write('Majority FN= ', confusion_majority[1][0])
        st.write('Majority TP= ', confusion_majority[1][1])
        
                       
        st.write('****************')
        
    if "Random Forest Classification" in selection_classifier: 
        st.markdown("## Random Forest Classification")
        rf = RandomForestClassifier(random_state=10)
        #fit data
        rf.fit(X_train, y_train)
        #y prediction
        
        y_pred = rf.predict(X_test)
        
        #proba
        prob_RF = rf.predict_proba(X_test)
        prob_RF = prob_RF[:, 1]
        
        confusion_majority=confusion_matrix(y_test, y_pred)
        
        st.write('Accuracy on test set= {:.3f}'. format(accuracy_score(y_test, y_pred)*100),'%')
        #Apply cross validation
        RF_cv = cross_val_score(rf,X,y,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (10 folds cross validation)= {:.3f}'. format(RF_cv*100),'%')
        st.write('Precision= {:.3f}'.format(precision_score(y_test, y_pred)*100),'%')
        st.write('Recall= {:.3f}'. format(recall_score(y_test, y_pred)*100),'%')
        st.write('F1= {:.3f}'. format(f1_score(y_test, y_pred)*100),'%')   
        auc_RF = roc_auc_score(y_test, prob_RF)
        st.write('AUC: {:.3f}'. format(auc_RF*100),'%')
        
        st.write('Majority classifier Confusion Matrix\n', confusion_majority)
        
        st.write('Majority TN= ', confusion_majority[0][0])
        st.write('Majority FP=', confusion_majority[0][1])
        st.write('Majority FN= ', confusion_majority[1][0])
        st.write('Majority TP= ', confusion_majority[1][1])
    
        
        
        st.write('****************')
        
    if "Support Vector Machine" in selection_classifier:
        st.markdown("## Support Vector Machine")
        model = SVC(kernel = "linear", probability=True, gamma='auto')
        #fit data
        model.fit(X_train, y_train)
        #y prediction
        y_pred = model.predict(X_test)
        #proba
        prob_SVM = model.predict_proba(X_test)
        prob_SVM = prob_SVM[:, 1]
        
        confusion_majority=confusion_matrix(y_test, y_pred)
        auc_SVM = roc_auc_score(y_test, prob_SVM)
        st.write('Accuracy on test set= {:.3f}'. format(accuracy_score(y_test, y_pred)*100),'%')
        #Apply cross validation
        svm_cv = cross_val_score(model,X,y,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (10 folds cross validation)= {:.3f}'. format(svm_cv*100),'%')
    
        st.write("AUC: %.2f" % auc_SVM)
        
        st.write('Majority classifier Confusion Matrix\n', confusion_majority)
        
        st.write('Majority TN= ', confusion_majority[0][0])
        st.write('Majority FP=', confusion_majority[0][1])
        st.write('Majority FN= ', confusion_majority[1][0])
        st.write('Majority TP= ', confusion_majority[1][1])
        
        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
        st.write('****************')
    
    if "XG Boost" in selection_classifier:  
        st.markdown("## XG Boosting")
        xgb_clf = XGBClassifier()
        #fit data
        xgb_clf.fit(X_train, y_train)
        #y prediction
        y_pred=xgb_clf.predict(X_test)
        
        #proba
        prob_XG = xgb_clf.predict_proba(X_test)
        prob_XG = prob_XG [:, 1]
        
        confusion_majority=confusion_matrix(y_test, y_pred)
        auc_XG = roc_auc_score(y_test, prob_XG)
        st.write('Accuracy on test set= {:.3f}'. format(accuracy_score(y_test, y_pred)*100),'%')
        #Apply cross validation
        xg_cv = cross_val_score(xgb_clf,X,y,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (10 folds cross validation)= {:.3f}'. format(xg_cv*100),'%')
    
        st.write("AUC: %.2f" % auc_XG)
        
        st.write('Majority classifier Confusion Matrix\n', confusion_majority)
        
    
        st.write('Majority TN= ', confusion_majority[0][0])
        st.write('Majority FP=', confusion_majority[0][1])
        st.write('Majority FN= ', confusion_majority[1][0])
        st.write('Majority TP= ', confusion_majority[1][1])
        
        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
        st.write('****************')
    
    
    #Ploting
    
    st.markdown("## ROC Plot Comparison")
    roc_curve_plot = plt.figure(figsize=(15,9))
    
    if "ADA Boosting" in selection_classifier:
        fpr_ADA, tpr_ADA, thresholds_ADA = roc_curve(y_test, prob_ADA)
        plt.plot(fpr_ADA, tpr_ADA, color='red', label='ADA')
        prec_ADA, rec_ADA, thresholds_ADA = precision_recall_curve(y_test, prob_ADA)
    
    if "Decision Tree" in selection_classifier:   
        fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, prob_DT)
        plt.plot(fpr_DT, tpr_DT, color='purple', label='DT')
        prec_DT, rec_DT, thresholds_DT = precision_recall_curve(y_test, prob_DT)
    
    if "Gradient Boosting" in selection_classifier:
        fpr_GB, tpr_GB, thresholds_GB = roc_curve(y_test, prob_GB)
        plt.plot(fpr_GB, tpr_GB, color='orange', label='GB')
        prec_GB, rec_GB, thresholds_GB = precision_recall_curve(y_test, prob_GB)
    
    if "Logistic Regression" in selection_classifier:   
        fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_test, prob_logreg) 
        plt.plot(fpr_logreg, tpr_logreg, color='green', label='LogReg') 
        prec_logreg, rec_logreg, thresholds_logreg = precision_recall_curve(y_test, prob_logreg)
    
    if "Naive Bayes" in selection_classifier:   
        fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_NB)
        plt.plot(fpr_NB, tpr_NB, color='yellow', label='NB') 
        prec_NB, rec_NB, thresholds_NB = precision_recall_curve(y_test, prob_NB)
    
    if "Random Forest Classification" in selection_classifier: 
        fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, prob_RF)
        plt.plot(fpr_RF, tpr_RF, color='blue', label='RF')
        prec_RF, rec_RF, thresholds_RF = precision_recall_curve(y_test, prob_RF) 
    
    if "Support Vector Machine" in selection_classifier:
        fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, prob_SVM)
        plt.plot(fpr_SVM, tpr_SVM, color='pink', label='SVM')
        prec_SVM, rec_SVM, thresholds_SVM = precision_recall_curve(y_test, prob_SVM)
    
    if "XG Boost" in selection_classifier:   
        fpr_XG, tpr_XG, thresholds_XG = roc_curve(y_test, prob_XG)
        plt.plot(fpr_XG, tpr_XG, color='brown', label='XG')
        prec_XG, rec_XG, thresholds_XG = precision_recall_curve(y_test, prob_XG)
    
    
    plt.plot([0, 1], [0, 1], color='black', linestyle='--',)
    plt.xlabel('False Positive Rate',fontsize = 20)
    plt.ylabel('True Positive Rate',fontsize = 20)
    plt.title('Receiver Operating Characteristic (ROC) Curve',fontsize = 27)
    plt.legend(fontsize = 12)
    st.pyplot(roc_curve_plot)
    
    st.markdown("## Precision-Recall Comparison")
    p_r_plot = plt.figure(figsize=(15,9))
    
    if "ADA Boosting" in selection_classifier:
        prec_ADA, rec_ADA, thresholds_ADA = precision_recall_curve(y_test, prob_ADA)
        plt.plot(prec_ADA, rec_ADA, color='red', label='ADA')
        
    if "Decision Tree" in selection_classifier:   
        prec_DT, rec_DT, thresholds_DT = precision_recall_curve(y_test, prob_DT)
        plt.plot(prec_DT, rec_DT, color='purple', label='DT')
        
    if "Gradient Boosting" in selection_classifier:
        prec_GB, rec_GB, thresholds_GB = precision_recall_curve(y_test, prob_GB)
        plt.plot(prec_GB, rec_GB, color='orange', label='GB')
        
    if "Logistic Regression" in selection_classifier:   
        prec_logreg, rec_logreg, thresholds_logreg = precision_recall_curve(y_test, prob_logreg)
        plt.plot(prec_logreg, rec_logreg, color='green', label='LogReg') 
        
    if "Naive Bayes" in selection_classifier:   
        prec_NB, rec_NB, thresholds_NB = precision_recall_curve(y_test, prob_NB)
        plt.plot(prec_NB, rec_NB, color='yellow', label='NB') 
        
    if "Random Forest Classification" in selection_classifier: 
        prec_RF, rec_RF, thresholds_RF = precision_recall_curve(y_test, prob_RF) 
        plt.plot(prec_RF, rec_RF, color='blue', label='RF')
        
    if "Support Vector Machine" in selection_classifier:
        prec_SVM, rec_SVM, thresholds_SVM = precision_recall_curve(y_test, prob_SVM)
        plt.plot(prec_SVM, rec_SVM, color='pink', label='SVM')
        
    if "XG Boost" in selection_classifier:   
        prec_XG, rec_XG, thresholds_XG = precision_recall_curve(y_test, prob_XG)
        plt.plot(prec_XG, rec_XG, color='brown', label='XG')
    
    plt.plot([1, 0], [0.1, 0.1], color='black', linestyle='--')
    plt.xlabel('Recall',fontsize = 20)
    plt.ylabel('Precision',fontsize = 20)
    plt.title('Precision-Recall Curve',fontsize = 27)
    plt.legend()
    st.pyplot(p_r_plot)
    
    
    X_all = features.drop("DECISION",1)
    y_all = features["DECISION"]
    
    ############# FINALISE MODE #############
    ############ ADA BOOST ############
    ada_FINAL = AdaBoostClassifier(n_estimators=50,learning_rate=1)
    #fit model
    ada_FINAL = ada_FINAL.fit(X_all, y_all)
    
    ############ Decision tree ############
    dt_FINAL = DecisionTreeClassifier()
    #fit data
    dt_FINAL=dt_FINAL.fit(X_all,y_all)
    
    ############ Gradient BOOST ############
    gb_FINAL = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
    #fit model
    gb_FINAL = gb_FINAL.fit(X_train, y_train)
    
    ############ Logistic Regression ############
    logreg_FINAL = LogisticRegression()
    #fit data
    logreg_FINAL.fit(X_all,y_all)
    
    ############ Naive Bayes ############
    nb_FINAL=GaussianNB()
    #fit data
    nb_FINAL.fit(X_all,y_all)
    
    ############ Random Forest Classification ############
    rf_FINAL = RandomForestClassifier(random_state=10)
    #fit data(entire dataset)
    rf_FINAL.fit(X_all, y_all)
    
    ############ SVM############
    svm_FINAL= SVC(kernel = "linear", probability=True, gamma='auto')
    #fit data
    svm_FINAL.fit(X_all, y_all)
    
    ############ XG BOOST ############
    xgb_FINAL = XGBClassifier()
    #fit data
    xgb_FINAL.fit(X_all, y_all)
    
    
    
    
    ##### PROMPT USER INPUT ######
    st.sidebar.header('Prediction')
    cc_exceed = st.sidebar.slider('Enter credit card exceed (months)', 1, 7, 1)
    emp_type = st.sidebar.selectbox("Enter employment type", cleaning_df['EMPLOYMENT_TYPE'].unique())
    loan_amt = st.sidebar.number_input('Enter amount of loan') 
    tenure_year = st.sidebar.slider('Enter loan tenure year', 10, 24, 19)
    more_than_one = st.sidebar.selectbox("Do you have more than one product?", ['Yes','No'])
    cc_types = st.sidebar.selectbox("Enter Credit Card Types", ['Normal','Gold','Platinum'])
    dependent = st.sidebar.slider('Enter number of dependents.', 2, 6, 2)
    financial_freedom = st.sidebar.slider('Enter years to financial freedom.', 5, 19, 15)
    cc_facility = st.sidebar.selectbox('Enter number of credit cards on hand.', [2,3,4,5,6])
    num_property =  st.sidebar.selectbox('Enter number of property owned.', [2,3,4,5])
    bank_product =  st.sidebar.selectbox('Enter number of bank product(s) owned.', [1,2,3,4,5])
    loan_to_approve =  st.sidebar.selectbox('Enter number of loan to approve.', [1,2,3])
    property_type =  st.sidebar.selectbox('Enter property type.', cleaning_df['PROPERTY_TYPE'].unique())
    property_completion = st.sidebar.slider('Enter years to property completion', 10, 13, 10)
    state =  st.sidebar.selectbox('Enter your state.', cleaning_df['STATE'].unique())
    side =  st.sidebar.selectbox('Enter number of side income.',[1,2,3])
    monthly_salary = st.sidebar.number_input('Enter monthly salary')
    sum_loan = st.sidebar.number_input('Enter total sum of loan')
    join_income = st.sidebar.number_input('Enter income for join application')
    score = st.sidebar.slider('Enter rating assessed by bank', 6,9,9)
    
    
    
    
    
    
    
    #### CREATE SELECTED FEATURE DATA FRAME ###
    data_input = {'TOTAL_SUM_OF_LOAN':  [sum_loan],'MONTHLY_SALARY':  [monthly_salary],'LOAN_TENURE_YEAR':  [tenure_year],
                  'LOAN_AMOUNT':  [loan_amt],'SCORE':  [score],'NUMBER_OF_LOAN_TO_APPROVE':  [loan_to_approve],
                  'TOTAL_INCOME_FOR_JOIN_APPLICATION':  [join_income],'NUMBER_OF_SIDE_INCOME':  [side],
    
            }
    
    input_record =  pd.DataFrame (data_input, columns = ['LOAN_AMOUNT','LOAN_TENURE_YEAR','NUMBER_OF_LOAN_TO_APPROVE','NUMBER_OF_SIDE_INCOME','MONTHLY_SALARY','TOTAL_SUM_OF_LOAN','TOTAL_INCOME_FOR_JOIN_APPLICATION','SCORE'])
    
    input_record['TOTAL_SUM_OF_LOAN'] = input_record['TOTAL_SUM_OF_LOAN'].astype(np.int64)
    input_record['MONTHLY_SALARY'] = input_record['MONTHLY_SALARY'].astype(np.int64)
    input_record['LOAN_TENURE_YEAR'] = input_record['LOAN_TENURE_YEAR'].astype(np.int64)
    input_record['LOAN_AMOUNT'] = input_record['LOAN_AMOUNT'].astype(np.int64)
    input_record['SCORE'] = input_record['SCORE'].astype(np.int64)
    input_record['NUMBER_OF_LOAN_TO_APPROVE'] = input_record['NUMBER_OF_LOAN_TO_APPROVE'].astype(np.int64)
    input_record['TOTAL_INCOME_FOR_JOIN_APPLICATION'] = input_record['TOTAL_INCOME_FOR_JOIN_APPLICATION'].astype(np.int64)
    input_record['NUMBER_OF_SIDE_INCOME'] = input_record['NUMBER_OF_SIDE_INCOME'].astype(np.int64)
    
    X_input = cleaning_df[['LOAN_AMOUNT','LOAN_TENURE_YEAR','NUMBER_OF_LOAN_TO_APPROVE','NUMBER_OF_SIDE_INCOME',
                           'MONTHLY_SALARY', 'TOTAL_SUM_OF_LOAN',  'TOTAL_INCOME_FOR_JOIN_APPLICATION',
                               'SCORE' ]]
    y_input = cleaning_df["DECISION"]
    
    X_input_new = X_input.append({'TOTAL_SUM_OF_LOAN':  sum_loan,'MONTHLY_SALARY':  monthly_salary,'LOAN_TENURE_YEAR':  tenure_year,
                  'LOAN_AMOUNT':  loan_amt,'SCORE':  score,'NUMBER_OF_LOAN_TO_APPROVE':  loan_to_approve,
                  'TOTAL_INCOME_FOR_JOIN_APPLICATION':  join_income,'NUMBER_OF_SIDE_INCOME':  side}, ignore_index=True)
    
    min_max_scaler= MinMaxScaler()
    normalize_X_input_new = min_max_scaler.fit_transform(X_input_new)
    features = X_input.columns
    
    
    #NORMALIZED WITH INPUT INCLUDED
    normalize_X_input_new = pd.DataFrame(normalize_X_input_new, columns = features)
    
    
    
    #new record normalized
    to_predict = normalize_X_input_new.tail(1)
    
    ##### HARD CODE NUMBER OF INCOME PROBLEM
    to_predict.iloc[0]['NUMBER_OF_SIDE_INCOME'] = "{:.2f}".format(to_predict.iloc[0]['NUMBER_OF_SIDE_INCOME'] )
    if to_predict.iloc[0]['NUMBER_OF_SIDE_INCOME'] == 0.33:
        to_predict.iloc[0]['NUMBER_OF_SIDE_INCOME'] = 0
    if to_predict.iloc[0]['NUMBER_OF_SIDE_INCOME']== 0.67:
        to_predict.iloc[0]['NUMBER_OF_SIDE_INCOME'] = 0.5
    
    
    #st.write(input_record)
    #st.write(to_predict)
    #st.write(X_all)
    
    st.title("Classification results ")
    st.write("Based on the above analysis, we have come to conclude that the Top 3 classifier with above 80% accuracy(with 10 fold cross validation) are:")
    st.markdown("#### 1. ADA Boost")
    st.markdown("#### 2. Gradient Boost")
    st.markdown("#### 3. Random Forest Classification")
    st.write(" ")
    
    if (st.button('Predict')):
        
        #Display new record
        st.header("User input data")
        st.write("Months of credit card exceeds: ", cc_exceed)
        st.write("Employment type: ", emp_type)
        st.write("Loan amount: ", loan_amt)
        st.write("Months of credit card exceeds: ", tenure_year)
        st.write("More than one product: ", more_than_one)
        st.write("Credit card type: ", cc_types)
        st.write("Number of dependents: ", dependent)
        st.write("Years to financial freedom: ", financial_freedom)
        st.write("Number of credit card facility: ", cc_facility)
        st.write("Number of properties: ", num_property)
        st.write("Number of bank products: ", bank_product)
        st.write("Number of loan(s) to approve: ", loan_to_approve)
        st.write("Property type: ", property_type)
        st.write("Years to property for completion: ", property_completion)
        st.write("State: ", state)
        st.write("Number of side income: ", side)
        st.write("Monthly salary: ", monthly_salary)
        st.write("Total sum of laon: ", sum_loan)
        st.write("Total income for join applications: ", join_income)
        st.write("Score(customer rating by bank): ", score)
        st.success('Classification model predicting....')
        #st.write(to_predict)
        
        
        ### Predict using ADA ######
        st.header('ADA Boosting prediction')
        y_pred_ADA = ada_FINAL.predict(X_all)
        #st.write(y_pred_ADA)
        #Apply cross validation
        abc_cv = cross_val_score(ada_FINAL,X_all,y_all,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (cross validated)= {:.3f}'. format(abc_cv*100),'%')
        prediction_ADA=ada_FINAL.predict(to_predict)
        #st.write(prediction_ADA)
        if prediction_ADA[0] == 0:
            st.success('ACCEPT')
        if prediction_ADA[0] == 1:
            st.error('REJECT')
            
        ### Predict using GB ######
        st.header('Gradient Boosting prediction')
        y_pred_GB = gb_FINAL.predict(X_all)
        #st.write(y_pred_GB)
        #Apply cross validation
        gb_cv = cross_val_score(gb_FINAL,X_all,y_all,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (cross validated)= {:.3f}'. format(gb_cv*100),'%')
        prediction_GB=gb_FINAL.predict(to_predict)
        #st.write(prediction_GB)
        if prediction_GB[0] == 0:
            st.success('ACCEPT')
        if prediction_GB[0] == 1:
            st.error('REJECT')
        
        ### Predict using DECISION TREE ######
        st.header('Decision Tree prediction')
        y_pred_DT = dt_FINAL.predict(X_all)
        #st.write(y_pred_DT)
        #Apply cross validation
        dt_cv = cross_val_score(dt_FINAL,X_all,y_all,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (cross validated)= {:.3f}'. format(dt_cv*100),'%')
        prediction_DT=dt_FINAL.predict(to_predict)
        #st.write(prediction_DT)
        if prediction_DT[0] == 0:
            st.success('ACCEPT')
        if prediction_DT[0] == 1:
            st.error('REJECT')
        
        ### Predict using LR ######
        st.header('Logistic Regression prediction')
        y_pred_LR = logreg_FINAL.predict(X_all)
        #st.write(y_pred_LR)
        #Apply cross validation
        logreg_cv = cross_val_score(logreg_FINAL,X_all,y_all,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (cross validated)= {:.3f}'. format(logreg_cv*100),'%')
        prediction_LR=logreg_FINAL.predict(to_predict)
        #st.write(prediction_LR)
        if prediction_LR[0] == 0:
            st.success('ACCEPT')
        if prediction_LR[0] == 1:
            st.error('REJECT')
        
        ### Predict using Naive Bayes######
        st.header('Naive Bayes prediction')
        y_pred_NB = nb_FINAL.predict(X_all)
        #st.write(y_pred_NB)
        #Apply cross validation
        NB_cv = cross_val_score(nb_FINAL,X_all,y_all,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (cross validated)= {:.3f}'. format(NB_cv*100),'%')
        prediction_NB=nb_FINAL.predict(to_predict)
        #st.write(prediction_NB)
        if prediction_NB[0] == 0:
            st.success('ACCEPT')
        if prediction_NB[0] == 1:
            st.error('REJECT')
        
        ### Predict using RFC ######
        st.header('Random Forest Classification prediction')
        y_pred_RF = rf_FINAL.predict(X_all)
        #st.write(y_pred_RF)
        RF_cv = cross_val_score(rf_FINAL,X_all,y_all,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (cross validated)= {:.3f}'. format(RF_cv*100),'%')
        prediction_RF=rf_FINAL.predict(to_predict)
        #st.write(prediction_RF)
        if prediction_RF[0] == 0:
            st.success('ACCEPT')
        if prediction_RF[0] == 1:
            st.error('REJECT')
            
        
        ### Predict using SVM######
        st.header('Support Vector Machine prediction')
        y_pred_SVM = svm_FINAL.predict(X_all)
        #st.write(y_pred_GB)
        #Apply cross validation
        svm_cv = cross_val_score(svm_FINAL,X_all,y_all,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (cross validated)= {:.3f}'. format(svm_cv*100),'%')
        prediction_SVM=svm_FINAL.predict(to_predict)
        #st.write(prediction_SVM)
        if prediction_SVM[0] == 0:
            st.success('ACCEPT')
        if prediction_SVM[0] == 1:
            st.error('REJECT')
            
        ### Predict using XB ######
        st.header('XG Boosting prediction')
        y_pred_XGB = xgb_FINAL.predict(X_all)
        #st.write(y_pred_XGB)
        #Apply cross validation
        xg_cv = cross_val_score(xgb_FINAL,X_all,y_all,cv = 10,scoring= "accuracy").mean()
        st.write('Accuracy (cross validated)= {:.3f}'. format(xg_cv*100),'%')
        prediction_XGB=xgb_FINAL.predict(to_predict)
        #st.write(prediction_XGB)
        if prediction_XGB[0] == 0:
            st.success('ACCEPT')
        if prediction_XGB[0] == 1:
            st.error('REJECT')
            
            
            
            
if "Clustering" in selection:
######################## Clustering ###################
        #UPON CLICKING BUTTON PREDICT    
    

    st.title('Clustering')
    
    st.success('Clustering algorithms running....')
    
    cleaning_df_cluster = cleaning_df.copy()
            
    #Label Encoding to transfer categorical to numerical representation
    cleaning_df_cluster['EMPLOYMENT_TYPE'] = LabelEncoder().fit_transform(cleaning_df_cluster.EMPLOYMENT_TYPE)
    cleaning_df_cluster['MORE_THAN_ONE_PRODUCTS'] = LabelEncoder().fit_transform(cleaning_df_cluster.MORE_THAN_ONE_PRODUCTS)
    cleaning_df_cluster['CREDIT_CARD_TYPES'] = LabelEncoder().fit_transform(cleaning_df_cluster.CREDIT_CARD_TYPES)
    cleaning_df_cluster['PROPERTY_TYPE'] = LabelEncoder().fit_transform(cleaning_df_cluster.PROPERTY_TYPE)
    cleaning_df_cluster['STATE'] = LabelEncoder().fit_transform(cleaning_df_cluster.STATE)
    cleaning_df_cluster['DECISION'] = LabelEncoder().fit_transform(cleaning_df_cluster.DECISION)    
            
            
    X = cleaning_df_cluster.drop("DECISION",1)
    y = cleaning_df_cluster["DECISION"]
    colnames = X.columns
    
    
    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 11)
    feat_selector = BorutaPy(rf, n_estimators="auto", random_state = 5)
    feat_selector.fit(X.values, y.values.ravel())
    
    boruta_score_cluster = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
    boruta_score_cluster = pd.DataFrame(list(boruta_score_cluster.items()), columns=['Features', 'Score'])
    
    boruta_score_cluster = boruta_score_cluster.sort_values("Score", ascending = False)
    
    
    #Dimensionality reduction
    boruta_features_cluster =cleaning_df_cluster.copy()
    boruta_features_cluster.drop(cleaning_df_cluster.columns[[5,10,1,11,6,15,13,9,4]], axis = 1, inplace = True)
    
    
    boruta_features_cluster["DECISION"]=cleaning_df_cluster["DECISION"]
    
    X_cluster=boruta_features_cluster.copy()
    
    
    
    
       
    ### Normalize
    data_scaled_X = normalize(X_cluster)
    data_scaled_X = pd.DataFrame(data_scaled_X, columns=X_cluster.columns)
        
    st.write('Clustering is performed using the same feature selection method as above.')
    st.write('Analyzing the SSE plot below, we conclude that the elbow is at k = 3. Hence, K-means clustering is performed using 3 clusters.')
    st.header('K-means')
    #To show SSE, and find the optimum K value (ie elbow value)
    distortions = []
    for i in range(1,11):                  #k=1 till k=10
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(data_scaled_X)
        distortions.append(km.inertia_)
    
    sse_fig = plt.figure(figsize=(15,9)) 
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()
    st.pyplot(sse_fig)
    
    #k-means model
    km = KMeans(n_clusters = 3, random_state =12)
    km.fit(data_scaled_X)
    
    labels1 = km.labels_
    
    
    
    
    #merge cluster with selected columns used in clustering 
    cluster_clustering = X_cluster.copy()
    cluster_clustering['CLUSTER']=km.labels_
    
    #shiloutte score
    st.markdown("## Silhoutte Score for K-means(n=3) :" )
    st.write(str(silhouette_score(data_scaled_X, labels1)))
        
    
    cluster_clustering['PROPERTY_TYPE'] = cluster_clustering['PROPERTY_TYPE'].astype(str)
    
    cluster_clustering['STATE'] = cluster_clustering['STATE'].astype(str)
    
    cluster_clustering['DECISION'] = cluster_clustering['DECISION'].astype(str)
    
    cluster_clustering['CLUSTER'] = cluster_clustering['CLUSTER'].astype(str)
    
    #convert integer representation to categorical data [PROPERTY_TYPE]
    cluster_clustering['PROPERTY_TYPE']=cluster_clustering['PROPERTY_TYPE'].replace("0", "Flat")
    cluster_clustering['PROPERTY_TYPE']=cluster_clustering['PROPERTY_TYPE'].replace("1", "Condominium")
    cluster_clustering['PROPERTY_TYPE']=cluster_clustering['PROPERTY_TYPE'].replace("2", "Bungalow")
    cluster_clustering['PROPERTY_TYPE']=cluster_clustering['PROPERTY_TYPE'].replace("3", "Not Specified")
    cluster_clustering['PROPERTY_TYPE']=cluster_clustering['PROPERTY_TYPE'].replace("4", "Terrace")
    #convert integer representation to categorical data [STATE]
    cluster_clustering['STATE']=cluster_clustering['STATE'].replace("0", "Johor")
    cluster_clustering['STATE']=cluster_clustering['STATE'].replace("2", "Kuala Lumpur")
    cluster_clustering['STATE']=cluster_clustering['STATE'].replace("7", "Selangor")
    cluster_clustering['STATE']=cluster_clustering['STATE'].replace("4", "Pulau Pinang")
    cluster_clustering['STATE']=cluster_clustering['STATE'].replace("3", "Negeri Sembilan")
    cluster_clustering['STATE']=cluster_clustering['STATE'].replace("5", "Sabah")
    cluster_clustering['STATE']=cluster_clustering['STATE'].replace("6", "Sarawak")
    cluster_clustering['STATE']=cluster_clustering['STATE'].replace("8", "Terengganu")
    cluster_clustering['STATE']=cluster_clustering['STATE'].replace("1", "Kedah")
    #convert integer representation to categorical data [DECISION]
    cluster_clustering['DECISION']=cluster_clustering['DECISION'].replace("1", "Reject")
    cluster_clustering['DECISION']=cluster_clustering['DECISION'].replace("0", "Accept")
    #convert integer representation to categorical data [CLUSTER]
    cluster_clustering['CLUSTER']=cluster_clustering['CLUSTER'].replace("0", "Cluster 1")
    cluster_clustering['CLUSTER']=cluster_clustering['CLUSTER'].replace("1", "Cluster 2")
    cluster_clustering['CLUSTER']=cluster_clustering['CLUSTER'].replace("2", "Cluster 3")
    
    #PLOT
    k1 = plt.figure(figsize=(15,9)) 
    sns.scatterplot(x="LOAN_AMOUNT", y="TOTAL_SUM_OF_LOAN", hue=cluster_clustering.CLUSTER.tolist(), data=cluster_clustering)
    plt.title("Cluster visualisation on LOAN_AMOUNT and TOTAL_SUM_OF_LOAN")
    plt.show()
    st.pyplot(k1)
    
    k2 = plt.figure(figsize=(15,9)) 
    sns.scatterplot(x="MONTHLY_SALARY", y="TOTAL_SUM_OF_LOAN", hue=cluster_clustering.CLUSTER.tolist(), data=cluster_clustering)
    plt.title("Cluster visualisation on MONTHLY_SALARY and TOTAL_SUM_OF_LOAN")
    plt.show()
    st.pyplot(k2)
    
    k3 = plt.figure(figsize=(15,9)) 
    sns.scatterplot(x="TOTAL_INCOME_FOR_JOIN_APPLICATION", y="TOTAL_SUM_OF_LOAN", hue=cluster_clustering.CLUSTER.tolist(), data=cluster_clustering)
    plt.title("Cluster visualisation on TOTAL_INCOME_FOR_JOIN_APPLICATION and TOTAL_SUM_OF_LOAN")
    plt.show()
    st.pyplot(k3)
    
    k4 = plt.figure(figsize=(15,9)) 
    sns.scatterplot(x="LOAN_AMOUNT", y="MONTHLY_SALARY", hue=cluster_clustering.CLUSTER.tolist(), data=cluster_clustering)
    plt.title("Cluster visualisation on LOAN_AMOUNT and MONTHLY_SALARY")
    plt.show()
    st.pyplot(k4)
    
    
    
    
    
    st.markdown("## K-means Analysis")
    cluster1_decision= cluster_clustering.groupby(['CLUSTER','DECISION'])['DECISION'].count()
    
    k_img1 = Image.open('Error_k1.jpg')
    st.image(k_img1,  use_column_width=True)
    
    st.write(cluster1_decision)
    
    
    
    
    #Calculate percentage of Decision
    perc     = cluster_clustering[["CLUSTER", "DECISION"]]
    
    c1_a   = ((perc["CLUSTER"] == 'Cluster 1') & (perc["DECISION"] =='Accept'))
    c1_a   = perc[c1_a]
    c1_a   = len(c1_a.index)
    c1_r   = ((perc["CLUSTER"] == 'Cluster 1') & (perc["DECISION"] =='Reject'))
    c1_r   = perc[c1_r]
    c1_r   = len(c1_r.index)
    
    
    pc1_a = (c1_a/(c1_a+c1_r)) * 100
    pc1_r = (c1_r/(c1_a+c1_r)) * 100
    
    c2_a   = ((perc["CLUSTER"] == 'Cluster 2') & (perc["DECISION"] =='Accept'))
    c2_a   = perc[c2_a]
    c2_a   = len(c2_a.index)
    c2_r   = ((perc["CLUSTER"] == 'Cluster 2') & (perc["DECISION"] =='Reject'))
    c2_r   = perc[c2_r]
    c2_r   = len(c2_r.index)
    
    pc2_a = (c2_a/(c2_a+c2_r)) * 100
    pc2_r = (c2_r/(c2_a+c2_r)) * 100
    
    c3_a   = ((perc["CLUSTER"] == 'Cluster 3') & (perc["DECISION"] =='Accept'))
    c3_a   = perc[c3_a]
    c3_a   = len(c3_a.index)
    c3_r   = ((perc["CLUSTER"] == 'Cluster 3') & (perc["DECISION"] =='Reject'))
    c3_r   = perc[c3_r]
    c3_r   = len(c3_r.index)
    
    pc3_a = (c3_a/(c3_a+c3_r)) * 100
    pc3_r = (c3_r/(c3_a+c3_r)) * 100
    
    
    st.write('Cluster 1 (Accepted) = ',str(round(pc1_a,2)) ,'%')
    st.write('Cluster 1 (Rejected) = ',str(round(pc1_r,2)),'%')
    
    
    st.write('Cluster 2 (Accepted) = ' ,str(round(pc2_a,2)),'%')
    st.write('Cluster 2 (Rejected) = ',str(round(pc2_r,2)),'%')
    
    
    st.write('Cluster 3 (Accepted) = '+str(round(pc3_a,2)),'%')
    st.write('Cluster 3 (Rejected) = '+str(round(pc3_r,2))+'%')
    
    st.header('Agglomerative clustering')
    st.write('Analyzing the dendogram below, we conclude there are 3 clusters under the threshold with y =6')
    
    
    agg_fig1 = plt.figure(figsize=(10, 7)) 
    plt.title("Dendrograms")  
    
    dend = shc.dendrogram(shc.linkage(data_scaled_X, method='ward'))
    
    plt.axhline(y=6, color='r', linestyle='--')     
    plt.show()
    st.pyplot(agg_fig1)
    
    
    agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    label_agg = agg.fit_predict(data_scaled_X)
    
    cluster_clustering2 = boruta_features_cluster.copy()
    cluster_clustering2['CLUSTER']=agg.labels_
    labels2 = agg.labels_
    
    st.markdown("## Silhoutte Score for Agglomerative clustering(n=3) :" )
    st.write(str(silhouette_score(data_scaled_X, labels2)))
    
    cluster_clustering2['PROPERTY_TYPE'] = cluster_clustering2['PROPERTY_TYPE'].astype(str)
    
    cluster_clustering2['STATE'] = cluster_clustering2['STATE'].astype(str)
    
    cluster_clustering2['DECISION'] = cluster_clustering2['DECISION'].astype(str)
    
    cluster_clustering2['CLUSTER'] = cluster_clustering2['CLUSTER'].astype(str)
    
    #convert integer representation to categorical data [PROPERTY_TYPE]
    cluster_clustering2['PROPERTY_TYPE']=cluster_clustering2['PROPERTY_TYPE'].replace("0", "Flat")
    cluster_clustering2['PROPERTY_TYPE']=cluster_clustering2['PROPERTY_TYPE'].replace("1", "Condominium")
    cluster_clustering2['PROPERTY_TYPE']=cluster_clustering2['PROPERTY_TYPE'].replace("2", "Bungalow")
    cluster_clustering2['PROPERTY_TYPE']=cluster_clustering2['PROPERTY_TYPE'].replace("3", "Not Specified")
    cluster_clustering2['PROPERTY_TYPE']=cluster_clustering2['PROPERTY_TYPE'].replace("4", "Terrace")
    
    #convert integer representation to categorical data [DECISION]
    cluster_clustering2['DECISION']=cluster_clustering2['DECISION'].replace("1", "Reject")
    cluster_clustering2['DECISION']=cluster_clustering2['DECISION'].replace("0", "Accept")
    
    cluster_clustering2['STATE']=cluster_clustering2['STATE'].replace("0", "Johor")
    cluster_clustering2['STATE']=cluster_clustering2['STATE'].replace("2", "Kuala Lumpur")
    cluster_clustering2['STATE']=cluster_clustering2['STATE'].replace("7", "Selangor")
    cluster_clustering2['STATE']=cluster_clustering2['STATE'].replace("4", "Pulau Pinang")
    cluster_clustering2['STATE']=cluster_clustering2['STATE'].replace("3", "Negeri Sembilan")
    cluster_clustering2['STATE']=cluster_clustering2['STATE'].replace("5", "Sabah")
    cluster_clustering2['STATE']=cluster_clustering2['STATE'].replace("6", "Sarawak")
    cluster_clustering2['STATE']=cluster_clustering2['STATE'].replace("8", "Terengganu")
    cluster_clustering2['STATE']=cluster_clustering2['STATE'].replace("1", "Kedah")
    
    #convert integer representation to categorical data [CLUSTER]
    cluster_clustering2['CLUSTER']=cluster_clustering2['CLUSTER'].replace("0", "Cluster 1")
    cluster_clustering2['CLUSTER']=cluster_clustering2['CLUSTER'].replace("1", "Cluster 2")
    cluster_clustering2['CLUSTER']=cluster_clustering2['CLUSTER'].replace("2", "Cluster 3")
    
    
    #PLOTING
    cluster2_decision= cluster_clustering2.groupby(['CLUSTER','DECISION'])["DECISION"].count()
    agg_img1 = Image.open('Error_agg1.jpg')
    st.image(agg_img1,  use_column_width=True)
    
    cluster2_decision
    
    #Calculate percentage of Decision
    perc     = cluster_clustering2[["CLUSTER", "DECISION"]]
    
    c1_a   = ((perc["CLUSTER"] == 'Cluster 1') & (perc["DECISION"] =='Accept'))
    c1_a   = perc[c1_a]
    c1_a   = len(c1_a.index)
    c1_r   = ((perc["CLUSTER"] == 'Cluster 1') & (perc["DECISION"] =='Reject'))
    c1_r   = perc[c1_r]
    c1_r   = len(c1_r.index)
    
    
    pc1_a = (c1_a/(c1_a+c1_r)) * 100
    pc1_r = (c1_r/(c1_a+c1_r)) * 100
    
    c2_a   = ((perc["CLUSTER"] == 'Cluster 2') & (perc["DECISION"] =='Accept'))
    c2_a   = perc[c2_a]
    c2_a   = len(c2_a.index)
    c2_r   = ((perc["CLUSTER"] == 'Cluster 2') & (perc["DECISION"] =='Reject'))
    c2_r   = perc[c2_r]
    c2_r   = len(c2_r.index)
    
    pc2_a = (c2_a/(c2_a+c2_r)) * 100
    pc2_r = (c2_r/(c2_a+c2_r)) * 100
    
    c3_a   = ((perc["CLUSTER"] == 'Cluster 3') & (perc["DECISION"] =='Accept'))
    c3_a   = perc[c3_a]
    c3_a   = len(c3_a.index)
    c3_r   = ((perc["CLUSTER"] == 'Cluster 3') & (perc["DECISION"] =='Reject'))
    c3_r   = perc[c3_r]
    c3_r   = len(c3_r.index)
    
    pc3_a = (c3_a/(c3_a+c3_r)) * 100
    pc3_r = (c3_r/(c3_a+c3_r)) * 100
    
    
    st.write('Cluster 1 (Accepted) ='+str(round(pc1_a,2)) +'%')
    st.write('Cluster 1 (Rejected) ='+str(round(pc1_r,2))+'%')
    
    
    st.write('Cluster 2 (Accepted) ='+str(round(pc2_a,2))+'%')
    st.write('Cluster 2 (Rejected) ='+str(round(pc2_r,2))+'%')
    
    
    st.write('Cluster 3 (Accepted) ='+str(round(pc3_a,2))+'%')
    st.write('Cluster 3 (Rejected) ='+str(round(pc3_r,2))+'%')
    

if "Association Rule Mining" in selection:
####################### ARM #####################
    st.title("Association Rule Mining(ARM)")
    st.markdown("## Apriori algorithm")
    
    #Prompt input
    min_Sup = st.number_input('Enter minimum support')
    st.write("Minimum support: ",str(min_Sup))
    
    min_Conf = st.number_input('Enter minimum confidence')
    st.write("Minimum confidence: ", str(min_Conf))
    
    min_Lift = st.number_input('Enter minimum lift')
    st.write("Minimum lift: ",str(min_Lift))
    
    
    if (st.button('Run')):
        st.success('Apriori algorithm running....')
        ###### RUN ONLY ONE TIME
        armcopy = cleaning_df.copy()
        armcopy['CREDIT_CARD_EXCEED_MONTHS'] = 'CCEM_' + armcopy['CREDIT_CARD_EXCEED_MONTHS'].astype(str)
        armcopy['LOAN_TENURE_YEAR'] = 'LTY_' + armcopy['LOAN_TENURE_YEAR'].astype(str)
        armcopy['NUMBER_OF_DEPENDENTS'] = 'NOD_' + armcopy['NUMBER_OF_DEPENDENTS'].astype(str)
        armcopy['YEARS_TO_FINANCIAL_FREEDOM'] = 'YTFF_' + armcopy['YEARS_TO_FINANCIAL_FREEDOM'].astype(str)
        armcopy['NUMBER_OF_CREDIT_CARD_FACILITY'] = 'NOCF_' + armcopy['NUMBER_OF_CREDIT_CARD_FACILITY'].astype(str)
        armcopy['NUMBER_OF_PROPERTIES'] = 'NOP_' + armcopy['NUMBER_OF_PROPERTIES'].astype(str)
        armcopy['NUMBER_OF_LOAN_TO_APPROVE'] = 'NOLTA_' + armcopy['NUMBER_OF_LOAN_TO_APPROVE'].astype(str)
        armcopy['YEARS_FOR_PROPERTY_TO_COMPLETION'] = 'YFPTC_' + armcopy['YEARS_FOR_PROPERTY_TO_COMPLETION'].astype(str)
        armcopy['NUMBER_OF_SIDE_INCOME'] = 'NOSI_' + armcopy['NUMBER_OF_SIDE_INCOME'].astype(str)
        armcopy['SCORE'] = 'S_' + armcopy['SCORE'].astype(str)
        armcopy['NUMBER_OF_BANK_PRODUCTS'] = 'NOBP_' + armcopy['NUMBER_OF_BANK_PRODUCTS'].astype(str)
        
        
        arm_df = armcopy
        records = []
        for i in range(0, 2350):
            records.append([str(arm_df.values[i,j]) for j in range(0, 21)])
        
        
        #Apriori algorithm
        association_rules = apriori(records,min_support=min_Sup,min_confidence=min_Conf,min_lift=min_Lift,min_length=2) 
        association_results = list(association_rules)
        
        st.write("### Number of rules: ", str(len(association_results)))
        
        cnt = 0
        for item in association_results:
            cnt += 1
            # first index of the inner list
            # Contains base item and add item
            pair = item[0] 
            items = [x for x in pair]
            st.write("### (Rule " + str(cnt) + ") " + items[0] + " -> " + items[1])
        
            #second index of the inner list
            st.write("Support: " + str(round(item[1],3)))
        
            #third index of the list located at 0th
            #of the third index of the inner list
        
            st.write("Confidence: " + str(round(item[2][0][2],4)))
            st.write("Lift: " + str(round(item[2][0][3],4)))
            st.write("=====================================")