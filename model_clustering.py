# coding: utf-8

import pandas as pd
import numpy as np 
from collections import namedtuple
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from log import init_logging


class ClusterModel(object):
    _defaults = {
        'feature_type' : 'feature_all',
        'eps' : 3, 
        'min_samples':5,
        'pca_n_components': 0.9,
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
        data = self.clustering(data, features, eps=self.eps, min_samples=self.min_samples)
        self.data_val = self.verify_val_precision(data)
        data_ts = self.pred_sar_prob(data)
        return data_ts

    
    def clustering(self, data, features, eps, min_samples):
        pipePre = Pipeline([('sc', MinMaxScaler()), ('pca', PCA(n_components=self.pca_n_components))]) 
        pipeTsne = Pipeline([('tsne', TSNE())]) 
        x_tr = data[features]
        y_tr = data['sar_flag']
        x_tr_pre = pipePre.fit_transform(x_tr.fillna(0))
        x_embedded = pipeTsne.fit_transform(x_tr_pre)
        model = DBSCAN(eps=eps, min_samples=min_samples).fit(x_embedded)
        data['cluster_label'] = model.labels_
        data['embedding'] = list(x_embedded)
        return data
    
    def verify_val_precision(self, data):
        '''
        Validataion alert_date
            - tr: 1~332
            - val: 333~364
        - Testing alert_date
            - tr: 1~364
            - ts: 365~393            
        '''
        data_val = self.seperate_val_data(data, (333,365))
        data_val = self.pred_sar_prob(data_val)
        precision_score = self.get_recallN_Precision(data_val)
        logging.info(f'val precision: {precision_score}')
        return data_val
        
    
    def pred_sar_prob(self, data):
        '''
        Algorithm:
            - 群內sar_flag機率 (sar qty/cluster_qty)        
        '''
        get_sar_prob = lambda x: sum(x==1)/sum(x<=1) if (sum(x==1)>0) else 0
        data_agg = data.groupby('cluster_label').agg({'sar_flag':get_sar_prob})
        sar_prob = data_agg['sar_flag'].to_dict()
        data['sar_prob'] = data['cluster_label'].map(lambda x: sar_prob[x])
        return data

    def seperate_val_data(self, data, val_range=(333,365)):
        data['alert_date']=data.index.get_level_values(1)
        data_val = data[data['alert_date'] < val_range[1]].copy()
        data_val['data_type'] = data_val['alert_date'].map(lambda x: 'tr' if x<val_range[0] else 'ts')
        data_val['sar_flag_raw'] = data_val['sar_flag'].copy()
        data_val['sar_flag'] = data_val.apply(lambda r: 2 if r.alert_date >= val_range[0] else r.sar_flag, axis=1)
        return data_val

    def get_recallN_Precision(self, data):
        data_verify = data[data['data_type']=='ts'].copy()
        data_verify.sort_values(by='sar_prob', ascending=False, inplace=True)
        data_verify['idx'] = list(range(len(data_verify)))
        idx = data_verify[data_verify['sar_flag_raw']==1]['idx'].iloc[-2]
        precision_score = (data_verify['sar_flag_raw'].sum()-1)/idx
        return precision_score
    
    def plot_clustering(self, data):
        x_embedded = np.array([(x[0], x[1]) for x in data['embedding']])
        sar_flag = data['sar_flag']
        cluster_label = data['cluster_label']
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        palette = sns.color_palette("bright", len(np.unique(sar_flag)))
        sns.scatterplot(x_embedded[:,0], x_embedded[:,1], hue=sar_flag, legend='full', palette=palette)
        plt.xlim(-100,100)
        plt.ylim(-100,100)    
        plt.subplot(122)
        label_nunique = len(set(cluster_label))
        palette = sns.color_palette("bright", label_nunique)
        sns.scatterplot(x_embedded[:,0], x_embedded[:,1], hue=cluster_label, legend=False, palette=palette)
        plt.title(f'label_nunique:{label_nunique}')
        plt.xlim(-100,100)
        plt.ylim(-100,100)    
        
    def hyperparameter_search(self):
        ACC_INFO = namedtuple('ACC_INFO', 'eps, min_samples, score')
        acc_list=[]
        for eps in [1, 1.5, 2, 2.5, 3, 3,5, 4]:
            for min_samples in [5,10, 15, 20]:
                data = dataset.data_df.copy()
                #data['alert_date']=data.index.get_level_values(1)
                feature = dataset.Feature_Dict['feature_all'].copy()
                #feature.append('alert_date')
                data = clustering(data, feature, eps=eps, min_samples=min_samples)
                score, data_agg = assess_score(data)
                acc_list.append(ACC_INFO(eps, min_samples, score))
                print(f'eps:{eps}, min_samples:{min_samples}, score:{score:.5f}')
        acc_df = pd.DataFrame(acc_list)
        return acc_df

