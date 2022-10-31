# coding: utf-8

import os
import pandas as pd
import numpy as np 
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN, KMeansSMOTE, SMOTEN, SMOTENC, SVMSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from collections import namedtuple
import logging
from log import init_logging
from utils import send_inference_msg_to_slack, timing


class Model(object):
    _defaults = {
    }
    
    SCORE_TUPLE = namedtuple('SCORE_TUPLE', 'feature, preprocess, sampling, model, iter, precision, time')
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
        
    def __init__(self, dataset, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        init_logging('model')
        self.init_data(dataset)
        self.init_hyperparameter(dataset)

    @timing
    def init_data(self, dataset):
        labelY = dataset.Feature_Dict['labelY']
        display(dataset.data_df[labelY].value_counts().to_frame())
        self.dataset_Tr = dataset.data_df[dataset.data_df['sar_flag']!=2]
        self.dataset_TsPub = dataset.data_df[dataset.data_df['sar_flag']==2]
        self.dataset_Tr.fillna(0, inplace=True)
        self.dataset_TsPub.fillna(0, inplace=True)
        print('dataset_Tr:', self.dataset_Tr.shape)
        print('dataset_TsPub:', self.dataset_TsPub.shape)

    def init_hyperparameter(self, dataset):
        pipePreMinMax = Pipeline([('sc', MinMaxScaler()), ('pca', PCA(n_components=0.99999))])
        pipePreStd = Pipeline([('sc', StandardScaler()), ('pca', PCA(n_components=0.9))])
        pipePreMMStd = Pipeline([('sc1', MinMaxScaler()), ('sc2', StandardScaler()), ('pca', PCA(n_components=0.9))])
        #pipe_list = zip([pipePreMinMax, pipePreStd, pipePreMMStd], ['MinMaxScaler','StandardScaler','MinMaxScaler+StandardScaler'])
        pipe_list = zip([pipePreMinMax], ['MinMaxScaler'])
        self.pipe_list = list(pipe_list)

        pipeSpAdasyn = imbPipeline([('sp', ADASYN())])
        pipeSpSmoten = imbPipeline([('sp', SMOTEN())])
        pipeSpSvmsmote = imbPipeline([('sp', SVMSMOTE())])
        pipeSpRandom = imbPipeline([('sp', RandomOverSampler())])
        pipeSpSmote = imbPipeline([('sp', SMOTE())])
        pipeSpBSmote = imbPipeline([('sp', BorderlineSMOTE())])
        pipeSpTom = imbPipeline([('sp', TomekLinks())])
        pipeSpSmoteTom = imbPipeline([('sp1', BorderlineSMOTE()), ('sp2', TomekLinks())])
        sp_list = zip([pipeSpAdasyn, pipeSpSmoten, pipeSpSvmsmote, pipeSpRandom, pipeSpSmote, pipeSpBSmote, pipeSpTom, pipeSpSmoteTom], ['adaysn', 'Smoten', 'Svmsmote', 'random', 'smote','bsmote', 'tom', 'SmoteTom'])
        #sp_list = zip([pipeSpSmote],['smote'])
        self.sp_list = list(sp_list)

        knn = KNeighborsClassifier(3)
        svc = SVC(kernel="rbf", C=0.025, probability=True)
        nusvc = NuSVC(probability=True)
        dtree = DecisionTreeClassifier()
        rfc = RandomForestClassifier()
        adb = AdaBoostClassifier()
        gdc = GradientBoostingClassifier()
        gnb = GaussianNB()
        lda = LinearDiscriminantAnalysis()
        qda = QuadraticDiscriminantAnalysis()
        sgdc = SGDClassifier(loss='log', max_iter=10)
        modelLR = LogisticRegression()
        modelXGBC = XGBClassifier(use_label_encoder=False, tree_method='gpu_hist', gpu_id=0, eval_metric='logloss')
        modelMLPC = MLPClassifier()
        #model_list = zip([qda, lda, gnb, gdc, adb, rfc, dtree, knn, modelLR, modelXGBC, modelMLPC], ['qda', 'lda', 'gnb', 'gdc', 'adb', 'rfc', 'dtree', 'knn', 'LogisticRegression', 'XGBClassifier', 'MLPClassifier'])
        model_list = zip([modelLR, sgdc], ['LR', 'SGDC'])
        self.model_list = list(model_list)

        Feature = dataset.Feature_Dict
        Feature['feature_AccumDiff']
        featureD = Feature['feature_ADate']
        featurePD = Feature['feature_Profile']+Feature['feature_ADate']
        featurePDA = Feature['feature_Profile']+Feature['feature_ADate']+Feature['feature_Accum']
        featurePDAY = Feature['feature_Profile']+Feature['feature_ADate']+Feature['feature_Accum']+Feature['feature_Year']
        featureD_Diff = Feature['feature_ADateDiff']
        featurePD_Diff = Feature['feature_Profile']+Feature['feature_ADateDiff']
        featurePDA_Diff = Feature['feature_Profile']+Feature['feature_ADateDiff']+Feature['feature_AccumDiff']
        featurePDAY_Diff = Feature['feature_Profile']+Feature['feature_ADateDiff']+Feature['feature_AccumDiff']\
                                                                                        +Feature['feature_Year']
        featureD_Composite = Feature['feature_ADate']+Feature['feature_ADateDiff']
        featurePD_Composite = Feature['feature_Profile']+Feature['feature_ADate']+Feature['feature_ADateDiff']
        featurePDA_Composite = Feature['feature_Profile']+Feature['feature_ADate']+Feature['feature_Accum']\
                                +Feature['feature_ADateDiff']+Feature['feature_AccumDiff']
        featurePDAY_Composite = Feature['feature_Profile']+Feature['feature_ADate']+Feature['feature_Accum']\
        +Feature['feature_ADateDiff']+Feature['feature_AccumDiff']+Feature['feature_Year']
        
        feature_list = [featureD, featurePD, featurePDA, featurePDAY, featureD_Diff, featurePD_Diff, featurePDA_Diff,\
                        featurePDAY_Diff, featureD_Composite, featurePD_Composite, featurePDA_Composite, featurePDAY_Composite]
        feature_list = zip(feature_list,['D', 'PD', 'PDA', 'PDAY', 'D_Diff', 'PD_Diff', 'PDA_Diff', \
                                         'PDAY_Diff','D_Composite','PD_Composite', 'PDA_Composite', 'PDAY_Composite'])
        #feature_list = zip([featurePDA], ['PDA'])
        self.feature_list = list(feature_list)

        score_list_tmp=[]
        for featureX, featureName in self.feature_list:
            for i in range(4):
                for pipePre, pipePreName in self.pipe_list:
                    for pipeSp, pipeSpName in self.sp_list:
                        for model, modelName in self.model_list:
                            score_list_tmp.append(self.SCORE_TUPLE(featureName, pipePreName, pipeSpName, modelName, i, 0,0))
        print('hyperparameter combinations:',len(score_list_tmp))
        
    

    
    def get_recallN_Precision(self, y_predProb, y_true):
        y_pred_df = pd.DataFrame(list(zip(y_predProb, y_true)), columns=['predProb','trueLabel'])
        y_pred_df.sort_values(by='predProb', ascending=False, inplace=True)
        y_pred_df['idx']=list(range(len(y_pred_df)))
        idx = y_pred_df[y_pred_df['trueLabel']==1]['idx'].iloc[-2]
        precision_score = (sum(y_true)-1)/idx
        return precision_score
    
    def train_add_hat(self, x, features):
        import numpy as np
        import pandas as pd
        df = x.copy() 
        q95_dict = {} 
        q5_dict = {} 
        for col in features:
            q95 = np.percentile(df[col], 90)
            q5 = np.percentile(df[col], 5) 
            q95_dict[col] = q95 
            q5_dict[col] = q5 
            b = np.array(df[col])
            c = list(map(lambda x: q95 if x > q95 else x, b))
            c = list(map(lambda x: q5 if x < q5 else x, c))        
            df = df.drop(col, axis=1)
            df[col] = c 
        return df, q95_dict, q5_dict

    # 使用同一标准处理测试集
    def add_hat(self, x, features, q95_dict, q5_dict):
        import numpy as np
        import pandas as pd
        df = x.copy()
        len_d = len(df.index)  # 测试集大小
        for col in features:
            q95 = q95_dict[col]
            q5 = q5_dict[col]
            b = np.array(df[col])
            c = list(map(lambda x:q95 if x > q95 else x, b))
            c = list(map(lambda x: q5 if x < q5 else x, c))        
            df = df.drop(col, axis=1)
            df[col] = c
        return df        
        
    