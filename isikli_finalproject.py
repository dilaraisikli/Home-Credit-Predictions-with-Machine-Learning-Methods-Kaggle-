import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc

def Read_dataframes():
    #read bureau, previous application, installments payments, application train and test dataframes  
    #credit_card_balance = pd.read_csv("C:/Users/dilar/Documents/homecredit/credit_card_balance.csv")
    #pos_cash_balance = pd.read_csv("C:/Users/dilar/Documents/homecredit/POS_CASH_balance.csv")
    data_bureau = pd.read_csv("C:/Users/dilar/Documents/homecredit/bureau.csv")
    data_installments_payments = pd.read_csv("C:/Users/dilar/Documents/homecredit/installments_payments.csv")
    data_previous_application = pd.read_csv("C:/Users/dilar/Documents/homecredit/previous_application.csv")
    data_application_train = pd.read_csv("C:/Users/dilar/Documents/homecredit/application_train.csv")
    data_application_test =  pd.read_csv("C:/Users/dilar/Documents/homecredit/application_test.csv")
    return(data_bureau,data_installments_payments,data_previous_application,data_application_train,data_application_test)
    

def Merge_Dataframes(data_bureau,data_installments_payments,data_previous_application,data_application_train,data_application_test):
    #credit_card_balance_agg= credit_card_balance.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':['mean','std', 'min','max'],'AMT_BALANCE':['mean','std', 'min','max'],'AMT_CREDIT_LIMIT_ACTUAL':['mean','std', 'min','max'],'AMT_DRAWINGS_ATM_CURRENT':['mean','std', 'min','max'],'AMT_DRAWINGS_CURRENT':['mean','std', 'min','max'],'AMT_DRAWINGS_OTHER_CURRENT':['mean','std', 'min','max'],'AMT_DRAWINGS_POS_CURRENT':['mean','std', 'min','max'],'AMT_INST_MIN_REGULARITY':['mean','std', 'min','max'],'AMT_PAYMENT_CURRENT':['mean','std', 'min','max'],'AMT_PAYMENT_TOTAL_CURRENT':['mean','std', 'min','max'],'AMT_RECEIVABLE_PRINCIPAL':['mean','std', 'min','max'],'AMT_RECIVABLE':['mean','std', 'min','max'],'AMT_TOTAL_RECEIVABLE':['mean','std', 'min','max'],'CNT_DRAWINGS_ATM_CURRENT':['mean','std', 'min','max'],'CNT_DRAWINGS_CURRENT':['mean','std', 'min','max'],'CNT_DRAWINGS_OTHER_CURRENT':['mean','std', 'min','max'],'CNT_DRAWINGS_POS_CURRENT':['mean','std', 'min','max'],'CNT_INSTALMENT_MATURE_CUM':['mean','std', 'min','max'],'SK_DPD':['mean','std', 'min','max'],'SK_DPD_DEF':['mean','std', 'min','max'],'NAME_CONTRACT_STATUS':lambda x: x.mode()[0]})
    #pos_cash_balance_agg = pos_cash_balance.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE':['mean','std', 'min','max'],'CNT_INSTALMENT':['mean','std', 'min','max'],'CNT_INSTALMENT_FUTURE':['mean','std', 'min','max'],'SK_DPD':['mean','std', 'min','max'],'SK_DPD_DEF':['mean','std', 'min','max'],'NAME_CONTRACT_STATUS': lambda x: x.mode()[0]})
    #find mean,std,min and max values of numerical columns and find mode for categorical columns
    bureau_agg = data_bureau.groupby('SK_ID_CURR').agg({'DAYS_CREDIT':['mean','std', 'min','max'], 'CREDIT_DAY_OVERDUE':['mean','std', 'min','max'], 'DAYS_CREDIT_ENDDATE':['mean','std', 'min','max'], 'DAYS_ENDDATE_FACT':['mean','std', 'min','max'] , 'AMT_CREDIT_MAX_OVERDUE':['mean','std', 'min','max'] , 'AMT_CREDIT_SUM':['mean','std', 'min','max'], 'AMT_CREDIT_SUM_DEBT':['mean','std', 'min','max'],'AMT_CREDIT_SUM_LIMIT':['mean','std', 'min','max'], 'AMT_CREDIT_SUM_OVERDUE':['mean','std', 'min','max'], 'DAYS_CREDIT_UPDATE':['mean','std', 'min','max'], 'AMT_ANNUITY':['mean','std', 'min','max'], 'CREDIT_ACTIVE':lambda x: x.mode()[0] ,'CREDIT_CURRENCY':lambda x: x.mode()[0],'CNT_CREDIT_PROLONG': lambda x: x.mode()[0],'CREDIT_TYPE':lambda x: x.mode()[0]})
    #find mean,std,min and max values of numerical columns and find mode for categorical columns
    installments_payments_agg = data_installments_payments.groupby('SK_ID_CURR').agg({'NUM_INSTALMENT_VERSION':['mean','std', 'min','max'], 'NUM_INSTALMENT_NUMBER':['mean','std', 'min','max'] , 'DAYS_INSTALMENT':['mean','std', 'min','max'] , 'DAYS_ENTRY_PAYMENT':['mean','std', 'min','max'], 'AMT_INSTALMENT':['mean','std', 'min','max'],'AMT_PAYMENT':['mean','std', 'min','max']})
    ##replace nan values for DAYS_EMPLOYED
    data_previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    data_previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    data_previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    data_previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    data_previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    #find mean,std,min and max values of numerical columns and find mode for categorical columns
    new_data_previous_application = data_previous_application.groupby('SK_ID_CURR').agg({'AMT_ANNUITY':['mean','std', 'min','max'],'AMT_APPLICATION':['mean','std', 'min','max'],'AMT_CREDIT':['mean','std', 'min','max'],'AMT_DOWN_PAYMENT':['mean','std', 'min','max'],'AMT_GOODS_PRICE':['mean','std', 'min','max'],'HOUR_APPR_PROCESS_START':['mean','std', 'min','max'],'NFLAG_LAST_APPL_IN_DAY':['mean','std', 'min','max'],'RATE_DOWN_PAYMENT':['mean','std', 'min','max'],'RATE_INTEREST_PRIMARY':['mean','std', 'min','max'],'RATE_INTEREST_PRIVILEGED':['mean','std', 'min','max'],'DAYS_DECISION':['mean','std', 'min','max'],'SELLERPLACE_AREA':['mean','std', 'min','max'],'CNT_PAYMENT':['mean','std', 'min','max'],'DAYS_FIRST_DRAWING':['mean','std', 'min','max'],'DAYS_FIRST_DUE':['mean','std', 'min','max'],'DAYS_LAST_DUE_1ST_VERSION':['mean','std', 'min','max'],'DAYS_LAST_DUE':['mean','std', 'min','max'],'DAYS_TERMINATION':['mean','std', 'min','max'],'NFLAG_INSURED_ON_APPROVAL':['mean','std', 'min','max'],'NAME_CONTRACT_TYPE':lambda x: x.mode()[0] ,'WEEKDAY_APPR_PROCESS_START':lambda x: x.mode()[0],'FLAG_LAST_APPL_PER_CONTRACT':lambda x: x.mode()[0],'NAME_CASH_LOAN_PURPOSE':lambda x: x.mode()[0] ,'NAME_CONTRACT_STATUS':lambda x: x.mode()[0] ,'NAME_PAYMENT_TYPE':lambda x: x.mode()[0] ,'CODE_REJECT_REASON':lambda x: x.mode()[0] ,'NAME_CLIENT_TYPE':lambda x: x.mode()[0],'NAME_GOODS_CATEGORY':lambda x: x.mode()[0],'NAME_PORTFOLIO':lambda x: x.mode()[0],'NAME_PRODUCT_TYPE': lambda x: x.mode()[0],'CHANNEL_TYPE':lambda x: x.mode()[0],'NAME_SELLER_INDUSTRY': lambda x: x.mode()[0] ,'NAME_YIELD_GROUP':lambda x: x.mode()[0],'PRODUCT_COMBINATION':lambda x: x.mode()[0]})	
    #merge all dataframes with train dataframe (considering SK_ID_CURR)
    data_application_train1 = pd.merge(data_application_train, installments_payments_agg, on = 'SK_ID_CURR', how = 'left')
    data_application_train2 = pd.merge(data_application_train1, bureau_agg, on = 'SK_ID_CURR', how = 'left')
    data_application_train3 = pd.merge(data_application_train2, new_data_previous_application, on = 'SK_ID_CURR', how = 'left')
    #data_application_train4 = pd.merge(data_application_train3, pos_cash_balance_agg, on = 'SK_ID_CURR', how = 'left')
    #data_application_train5 = pd.merge(data_application_train3, credit_card_balance_agg, on = 'SK_ID_CURR', how = 'left')
    #delete target and SK_ID_CURR from merged train dataframe
    del data_application_train3 ['TARGET']
    del data_application_train3['SK_ID_CURR']
    #merge all dataframes with test dataframe (considering SK_ID_CURR)
    data_application_test1 = pd.merge(data_application_test, installments_payments_agg, on = 'SK_ID_CURR', how = 'left')
    data_application_test2 = pd.merge(data_application_test1, bureau_agg, on = 'SK_ID_CURR', how = 'left')
    data_application_test3 = pd.merge(data_application_test2, new_data_previous_application, on = 'SK_ID_CURR', how = 'left')
    #data_application_test4 = pd.merge(data_application_test3, pos_cash_balance_agg, on = 'SK_ID_CURR', how = 'left')
    #data_application_test5 = pd.merge(data_application_test3, credit_card_balance_agg, on = 'SK_ID_CURR', how = 'left')
    #delete SK_ID_CURR from merged test dataframe
    del data_application_test3['SK_ID_CURR']
    #concatenation for test and train dataframes
    data_app_train_test=pd.concat([data_application_train3,data_application_test3])
    #replace nan values for DAYS_EMPLOYED
    data_app_train_test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    #create new columns with using exist columns
    data_app_train_test['ANN_RATIO_CRED'] = data_app_train_test['AMT_CREDIT'] / data_app_train_test['AMT_ANNUITY']
    data_app_train_test['DAYS_EMPLOYED_BIRTH'] = data_app_train_test['DAYS_EMPLOYED'] / data_app_train_test['DAYS_BIRTH']
    data_app_train_test['INCOME_ANN'] = data_app_train_test['AMT_ANNUITY'] / data_app_train_test['AMT_INCOME_TOTAL']
    data_app_train_test['GOODS_RATIO_CRED'] = data_app_train_test['AMT_CREDIT'] / data_app_train_test['AMT_GOODS_PRICE']
    data_app_train_test['RATEOFPAY'] = data_app_train_test['AMT_ANNUITY'] / data_app_train_test['AMT_CREDIT']
    data_app_train_test['INCOME_CRED'] = data_app_train_test['AMT_INCOME_TOTAL'] / data_app_train_test['AMT_CREDIT']
    data_app_train_test['CAR_RAT_EMP_RATIO'] = data_app_train_test['OWN_CAR_AGE'] / data_app_train_test['DAYS_EMPLOYED']
    data_app_train_test['AMT_CRED_INCOME_RAT'] = data_app_train_test['AMT_CREDIT'] / data_app_train_test['AMT_INCOME_TOTAL']
    return(data_app_train_test)
    
def Correlation(cor_data,threshold):
    #find correlation on them
    correlations = cor_data.corr().apply(abs)
    #create two lists to keep columns names which have correlation over 0.9
    listI=[]
    listJ=[]
    for i in range(len(correlations)):
        for j in range( len(correlations)):
            if ((correlations.iloc[i,j]) > threshold) and (correlations.iloc[i,j] != 1):
                if correlations.index[i] not in listI and correlations.index[i] not in listJ:
                    listI.append(correlations.index[i])
                if correlations.columns[j] not in listI and correlations.columns[j] not in listJ:
                    listJ.append(correlations.columns[j])
    #delete columns names which have correlation over 0.9
    for col in listI:
            del cor_data[col]
    for col in listJ:
            del cor_data[col]

    return(cor_data)

def Find_Missing_Values(miss_data,missing_percent):
    listMiss=[]
    deleteMissColNames=[]
    #find missing data in dataframe
    mis_val = miss_data.isnull().sum()
    #find percentage of missing data in dataframe
    mis_val_percent = 100 * miss_data.isnull().sum() / len(miss_data)
    #calculate percantage of missing value for each columns and keep them which have over 0.60
    for i in range(len(mis_val)):
        if (mis_val[i] != 0 and mis_val_percent[i]>missing_percent ):
            listMiss.append([miss_data.columns[i],mis_val[i],mis_val_percent[i]])
            deleteMissColNames.append(miss_data.columns[i])
    #delete columns which have missing values over 0.60
    for col in deleteMissColNames:
            del miss_data[col]
    return(miss_data)

def Normalization_Encoding(data_nor_encod):
    #find numerical columns in dataframe
    num_cols = list(data_nor_encod._get_numeric_data().columns)
    #create list with all columns name
    cols=list(data_nor_encod.columns)
    #remove numeric columns to find categorical columns
    for i in range(len(num_cols)):
        if num_cols[i] in cols:
            cols.remove(num_cols[i])
    #create 2 dataframes for categorical and numerical 
    numericDf = data_nor_encod[num_cols].copy()
    catecDf = data_nor_encod[cols].copy()
    
    #use minmaxscaler function to make normalization
    scaler = MinMaxScaler()
    numericDf[num_cols] = scaler.fit_transform(numericDf)
    #use dummies function to one hot encoding
    encoding_catecDf=pd.get_dummies(catecDf)
    #merge categorical and numerical columns dataframe
    result = pd.concat([encoding_catecDf, numericDf], axis=1, sort=False)
    return(result)

def Feature_Importance(application_train,application_test,encod_nor_extant_corr_data_app_train_test):
    # cross validation with k = 2 to have more precision on the features scores
    n_splits = 2 
    k_fold = KFold(n_splits = n_splits, shuffle = True)
    zeroNum = []
    
    sep_train = encod_nor_extant_corr_data_app_train_test.iloc[:307511]
    sep_test = encod_nor_extant_corr_data_app_train_test.iloc[307511]
    ID = application_test['SK_ID_CURR']
    target= application_train['TARGET']
    
    train = sep_train.to_numpy()
    target = target.to_numpy()
    test = sep_test.to_numpy()
    
    # array to save features scores
    feature_importances = np.zeros(train.shape[1])
    
    # cross validation loop
    for train_indices, valid_indices in k_fold.split(train):
        train_features, train_labels = train[train_indices], target[train_indices]
        valid_features, valid_labels = train[valid_indices], target[valid_indices]
        
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', boosting_type='goss',
                                       class_weight = 'balanced', learning_rate = 0.05, 
                                       reg_alpha = 0.1, reg_lambda = 0.1, n_jobs = -1 )
        
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], 
                  early_stopping_rounds = 100, verbose = 200)
        # at each step of the cross validation the feature importances are saved in this array; we take the average of this on the 2 folds
        feature_importances += model.feature_importances_/ n_splits
        # clean memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # delete feature with 0 importance
    for i in range(len(feature_importances)):
        if feature_importances[i] == 0:
            zeroNum.append(i)
    
    train = np.delete(train, zeroNum,1)
    test = np.delete(test, zeroNum, 1)
    return(train,test,target,ID) 


def Learning_Model(train,test,target,ID):
    # LGBM model with 5-fold cross validation
    n_splits = 5
    k_fold = KFold(n_splits = n_splits, shuffle = True)
    
    test_predictions = np.zeros(target.shape[0])
    feature_importances = np.zeros(train.shape[1])
    test_predictions = np.zeros(test.shape[0])
    
    # arrays for validation and train AUC scores
    valid_scores = []
    train_scores = []
    
    # model loop
    for train_indices, valid_indices in k_fold.split(train):
        train_features, train_labels = train[train_indices], target[train_indices]
        valid_features, valid_labels = train[valid_indices], target[valid_indices]
        
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', boosting_type='goss',
                                       class_weight = 'balanced', learning_rate = 0.02, 
                                       reg_alpha = 0.1, reg_lambda = 0.1, n_jobs = -1 )
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], 
                  early_stopping_rounds = 100, verbose = 200)
        
        # save best iteration
        best_iteration = model.best_iteration_
        # feature imporatance scores
        feature_importances += model.feature_importances_/ n_splits
        # apply model on test set, the final result is given by the average scores on the 5 folds
        test_predictions += model.predict_proba(test, num_iteration = best_iteration)[:, 1] / n_splits
        
        # train and validation scores
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
    
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # clean memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    print('valid_scr:', sum(valid_scores)/5)
    # create submission file
    submission = pd.DataFrame({'SK_ID_CURR': list(ID), 'TARGET': test_predictions})
    submission.to_csv("C:/Users/dilar/Documents/homecredit/submission.csv", index=False)

#read application train and test dataframes  
data_application_train = pd.read_csv("C:/Users/dilar/Documents/homecredit/application_train.csv")
data_application_test =  pd.read_csv("C:/Users/dilar/Documents/homecredit/application_test.csv")

Merge_Dataframes(Read_dataframes())
Correlation(Merge_Dataframes(),0.95)
Find_Missing_Values(Correlation(),0.60)   
Normalization_Encoding(Find_Missing_Values())
Learning_Model(Feature_Importance(data_application_train,data_application_test,Normalization_Encoding()))     