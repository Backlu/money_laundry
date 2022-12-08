# coding: utf-8

import pandas as pd
import numpy as np 
from collections import namedtuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sns
from log import init_logging
from autogluon.tabular import TabularPredictor


class RegressionModel(object):
    _defaults = {
        'feature_type' : 'feature_all',
        'eps' : 3, 
        'min_samples':5,
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
        
    def __init__(self, dataset, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        init_logging('ClusterModel')
        self.dataset = dataset
        
    def inference(self):
        data = self.dataset.data_df.copy()
        features = self.dataset.Feature_Dict[self.feature_type]
        self.data_val = self.verify_val_precision(data, features)
        self.data_ts = self.predict_test(data, features)
        return self.data_ts

    
    def predict_test(self, data, features):
        data['data_type'] = data['sar_flag'].map(lambda x: 'tr' if x<2 else 'ts')
        predictor = self.model_training(data, features)
        data = self.pred_sar_prob(data, features, predictor)
        return data
        
    def verify_val_precision(self, data, features):
        '''
        Validataion alert_date
            - tr: 1~332
            - val: 333~364
        - Testing alert_date
            - tr: 1~364
            - ts: 365~393            
        '''
        data_val = self.seperate_val_data(data, (333,365))
        #train model
        predictor = self.model_training(data_val, features)
        data_val = self.pred_sar_prob(data_val, features, predictor)
        precision_score = self.get_recallN_Precision(data_val)
        logging.info(f'val precision: {precision_score}')
        return data_val
        
    def seperate_val_data(self, data, val_range=(333,365)):
        data['alert_date']=data.index.get_level_values(1)
        data_val = data[data['alert_date'] < val_range[1]].copy()
        data_val['data_type'] = data_val['alert_date'].map(lambda x: 'tr' if x<333 else 'ts')
        data_val['sar_flag_raw'] = data_val['sar_flag']
        data_val['sar_flag'] = data_val.apply(lambda r: 2 if r.alert_date >= val_range[0] else r.sar_flag, axis=1)
        return data_val        
        
    def model_training(self, data, features):
        data_tr = data[data['data_type']=='tr']
        data_tr = data_tr[data_tr['sar_prob_clustering']>0]
        logging.info(f'data_tr qty: {len(data_tr)}')
        predictor = TabularPredictor(label='sar_flag').fit(data_tr[features+['sar_flag']])
        return predictor        
    
    def pred_sar_prob(self, data, features, predictor):
        data['sar_prob_reg'] = predictor.predict(data[features])
        data['sar_prob'] = data['sar_prob_reg']*data['sar_prob_clustering']
        return data

    def get_recallN_Precision(self, data):
        data_verify = data[data['data_type']=='ts'].copy()
        data_verify.sort_values(by='sar_prob', ascending=False, inplace=True)
        data_verify['idx']=list(range(len(data_verify)))
        idx = data_verify[data_verify['sar_flag_raw']==1]['idx'].iloc[-2]
        precision_score = (data_verify['sar_flag_raw'].sum()-1)/idx
        return precision_score    
    


