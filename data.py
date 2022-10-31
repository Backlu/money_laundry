# coding: utf-8

import os
import pandas as pd
import re
import numpy as np
import datetime
from datetime import date
from itertools import product
from joblib import Parallel, delayed
import logging
from log import init_logging
from utils import send_inference_msg_to_slack, timing


class Data(object):
    _defaults = {
        'rolling_window':7
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
        
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        init_logging('data')
        self.init_data()
        self.custinfo_preprocess()
        self.featuring_alertDate()
        self.featuring_accumulate()
        self.featuring_integrate()

    @timing
    def init_data(self):
        self.ccba_df = pd.read_csv('data/dataset1/public_train_x_ccba_full_hashed.csv')
        self.cdtx_df = pd.read_csv('data/dataset1/public_train_x_cdtx0001_full_hashed.csv')
        self.custinfo_df = pd.read_csv('data/dataset1/public_train_x_custinfo_full_hashed.csv')
        self.dp_df = pd.read_csv('data/dataset1/public_train_x_dp_full_hashed.csv')
        self.remit_df = pd.read_csv('data/dataset1/public_train_x_remit1_full_hashed.csv')
        self.tr_alertX_df = pd.read_csv('data/dataset1/train_x_alert_date.csv')
        self.tr_sarY_df = pd.read_csv('data/dataset1/train_y_answer.csv')
        self.public_alertX_df = pd.read_csv('data/dataset1/public_x_alert_date.csv')        
    
    @timing
    def custinfo_preprocess(self):
        '''
        將alert_date, sar_flag merge到custinfo
        '''
        logging.info('custinfo_preprocess')
        #mapping alert_date & sar flag
        alertDate_dict = self.tr_alertX_df.set_index('alert_key').to_dict()['date']
        sarFlag_dict = self.tr_sarY_df.set_index('alert_key').to_dict()['sar_flag']
        self.public_alertX_df['sar_flag']=2
        alertDatePublc_dict = self.public_alertX_df.set_index('alert_key').to_dict()['date']
        sarFlagPublic_dict = self.public_alertX_df.set_index('alert_key').to_dict()['sar_flag']
        self.custinfo_df['alert_date'] = self.custinfo_df['alert_key'].map(lambda x: alertDate_dict[x] if x in alertDate_dict \
                                                                 else alertDatePublc_dict[x] )
        self.custinfo_df['sar_flag'] = self.custinfo_df['alert_key'].map(lambda x: sarFlag_dict[x] if x in sarFlag_dict \
                                                               else sarFlagPublic_dict[x])
        self.custinfo_df['sar_flag_nunique'] = self.custinfo_df.groupby(['cust_id','alert_date'])['sar_flag'].transform('nunique')
        invalid_data_qty = sum(self.custinfo_df['sar_flag_nunique']>1)
        print(f'invalid_data_qty: {invalid_data_qty}')
        custinfo_valid = self.custinfo_df[self.custinfo_df['sar_flag_nunique']==1]
        custinfo_valid.drop_duplicates(['cust_id','alert_date','sar_flag'], inplace=True)
        custinfo_valid = custinfo_valid.set_index(['cust_id','alert_date'])
        
        new_cols = []
        for c in custinfo_valid.columns:
            if c in ['alert_key', 'sar_flag', 'sar_flag_nunique']:
                new_cols.append(c)
            else:
                new_cols.append(c+'_Profile')
        custinfo_valid.columns=new_cols
        self.custinfo_valid = custinfo_valid
        
    @timing    
    def featuring_alertDate(self):
        logging.info('featuring_alertDate')
        #cdtx
        cdtxDate_ = self.cdtx_df.groupby(['cust_id','date']).agg(cdtxAmt_ADate=('amt', sum), cdtxCnt_ADate=('date', 'count'))
        cdtxDate_ntd = self.cdtx_df[self.cdtx_df['cur_type']==47].groupby(['cust_id','date']).agg(cdtxAmtNTD_ADate=('amt', sum),\
                                                                                             cdtxCntNTD_ADate=('date', 'count'))
        cdtxDate_fc = self.cdtx_df[self.cdtx_df['cur_type']!=47].groupby(['cust_id','date']).agg(cdtxAmtFC_ADate=('amt', sum),\
                                                                                       cdtxCntFC_ADate=('date', 'count'))
        cdtxDate_tw = self.cdtx_df[self.cdtx_df['country']==130].groupby(['cust_id','date']).agg(cdtxAmtTW_ADate=('amt', sum),\
                                                                                            cdtxCntTW_ADate=('date', 'count'))
        cdtxDate_f = self.cdtx_df[self.cdtx_df['country']!=130].groupby(['cust_id','date']).agg(cdtxAmtF_ADate=('amt', sum),\
                                                                                           cdtxCntF_ADate=('date', 'count'))
        self.cdtxDate_df = pd.concat([cdtxDate_, cdtxDate_ntd, cdtxDate_fc, cdtxDate_tw, cdtxDate_f], axis=1)

        #dp
        self.dp_df['amt'] = self.dp_df['tx_amt']*self.dp_df['exchg_rate']
        dpDate_ = self.dp_df.groupby(['cust_id','tx_date']).agg(dpAmt_ADate=('tx_amt', sum), dpCnt_ADate=('tx_date', 'count'))
        dpDate_CR = self.dp_df[self.dp_df['debit_credit']=='CR'].groupby(['cust_id','tx_date']).agg(\
                                              dpAmtCR_ADate=('tx_amt', sum), dpCntCR_ADate=('tx_date', 'count')) 
        dpDate_DB = self.dp_df[self.dp_df['debit_credit']=='CB'].groupby(['cust_id','tx_date']).agg(\
                                              dpAmtDB_ADate=('tx_amt', sum), dpCntDB_ADate=('tx_date', 'count'))
        dpDate_CC = self.dp_df[(self.dp_df['tx_type']==1)&\
                               (self.dp_df['info_asset_code']==12)].groupby(['cust_id','tx_date']).agg(\
                               dpAmtCC_ADate=('tx_amt', sum), dpCntCC_ADate=('tx_date', 'count'))
        dpDate_NCC = self.dp_df[~((self.dp_df['tx_type']==1)&\
                             (self.dp_df['info_asset_code']==12))].groupby(['cust_id','tx_date']).agg(\
                              dpAmtNCC_ADate=('tx_amt', sum), dpCntNCC_ADate=('tx_date', 'count'))
        dpDate_CBank = self.dp_df[self.dp_df['cross_bank']==1].groupby(['cust_id','tx_date']).agg(\
                              dpAmtCBank_ADate=('tx_amt', sum), dpCntCBank_ADate=('tx_date', 'count'))
        dpDate_InBank = self.dp_df[self.dp_df['cross_bank']==0].groupby(['cust_id','tx_date']).agg(\
                              dpAmtInBank_ADate=('tx_amt', sum), dpCntInBank_ADate=('tx_date', 'count'))
        dpDate_ATM = self.dp_df[self.dp_df['ATM']==1].groupby(['cust_id','tx_date']).agg(dpAmtATM_ADate=('tx_amt', sum),\
                                                                                         dpCntATM_ADate=('tx_date', 'count'))
        dpDate_NATM = self.dp_df[self.dp_df['ATM']==0].groupby(['cust_id','tx_date']).agg(dpAmtNATM_ADate=('tx_amt', sum),\
                                                                                          dpCntNATM_ADate=('tx_date', 'count'))
        dpDate_branchNunique = self.dp_df.groupby(['cust_id','tx_date']).agg(dpBranchNunique_ADate=('txbranch', 'nunique'))
        self.dpDate_df = pd.concat([dpDate_, dpDate_CR, dpDate_DB, dpDate_CC, dpDate_NCC, dpDate_CBank, dpDate_InBank,\
                                    dpDate_ATM, dpDate_NATM, dpDate_branchNunique], axis=1)

        #remit
        self.remitDate_df = self.remit_df.groupby(['cust_id','trans_date']).agg(remitAmt_ADate=('trade_amount_usd', sum),\
                                                                           remitCnt_ADate=('trans_no', 'count'))
        
        #ccba
        ccbaProfile1 = self.ccba_df.replace(0, np.nan).groupby('cust_id').agg(
                                       ccbalupayAmt_Year=('lupay', np.mean),\
                                       ccbausgamAmt_Year=('usgam', np.mean),\
                                       ccbacycamAax_Year=('cycam', np.max),\
                                       ccbaclamtAmt_Year=('clamt', np.mean),\
                                       ccbacsamtAmt_Year=('csamt', np.mean),\
                                       ccbainamtAmt_Year=('inamt', np.mean),\
                                       ccbacucsmAmt_Year=('cucsm', np.mean),\
                                       ccbacucahAmt_Year=('cucah', np.mean),\
                                      )
        ccbaProfile2 = self.ccba_df.groupby('cust_id').agg(ccbalupayCnt_Year=('lupay', np.count_nonzero),\
                                       ccbausgamCnt_Year=('usgam', np.count_nonzero),\
                                       ccbaclamtCnt_Year=('clamt', np.count_nonzero),\
                                       ccbacsamtCnt_Year=('csamt', np.count_nonzero),\
                                       ccbainamtCnt_Year=('inamt', np.count_nonzero),\
                                       ccbacucsmCnt_Year=('cucsm', np.count_nonzero),\
                                       ccbacucahCnt_Year=('cucah', np.count_nonzero),\
                                       ccbabyymmCnt_Year=('byymm', np.count_nonzero),\
                                      )
        self.ccbaProfile_df = pd.concat([ccbaProfile1, ccbaProfile2], axis=1)        
        
    @timing        
    def featuring_accumulate(self):
        logging.info('featuring_accumulate')
        self.cdtx_feature = self._accum_featuring_parallel(self.cdtxDate_df, 'date')
        self.dp_feature = self._accum_featuring_parallel(self.dpDate_df, 'tx_date')
        self.remit_feature = self._accum_featuring_parallel(self.remitDate_df, 'trans_date')
    
    def _accum_featuring_parallel(self, dateFeature, date_colName='date'):
        idx_custid, _ = zip(*dateFeature.index)
        idx_custid = np.unique(idx_custid)
        accu_feature_list = Parallel(n_jobs=-2)(delayed(get_accum_feature)(dateFeature.loc[(cust_id)], \
                                       cust_id, date_colName, self.rolling_window) for cust_id in idx_custid) 
        accu_feature = pd.concat(accu_feature_list)
        return accu_feature

    @timing    
    def featuring_integrate(self, drop_nan_alertkey=True):
        logging.info('featuring_integrate')
        self.custProfile_df = pd.merge(self.custinfo_valid.reset_index(level=1), self.ccbaProfile_df,\
                                  left_index=True, right_index=True, how='left')
        self.custProfile_df.set_index('alert_date', append=True, inplace=True)
        data_df = pd.concat([self.custProfile_df, self.cdtx_feature, self.dp_feature, self.remit_feature], axis=1)
        data_df['sar_flag']=data_df.apply(lambda r: r['sar_flag'] if r['alert_key']==r['alert_key'] else -1, axis=1)
        if drop_nan_alertkey:
            data_df = data_df[data_df['sar_flag']!=-1]
        
        Feature_Dict = {}
        Feature_Dict['labelY'] = 'sar_flag'
        Feature_Dict['feature_all'] = [x for x in data_df.columns if x not in ['alert_key', 'sar_flag', 'sar_flag_nunique']]
        Feature_Dict['feature_Profile'] = [x for x in data_df.columns if x.split('_')[-1]=='Profile']
        Feature_Dict['feature_ADate'] = [x for x in data_df.columns if x.split('_')[-1]=='ADate']
        Feature_Dict['feature_ADateDiff'] = [x for x in data_df.columns if x.split('_')[-1]=='ADateDiff']
        Feature_Dict['feature_Accum'] = [x for x in data_df.columns if x.split('_')[-1]=='Accum']
        Feature_Dict['feature_AccumDiff'] = [x for x in data_df.columns if x.split('_')[-1]=='AccumDiff']
        Feature_Dict['feature_Year'] = [x for x in data_df.columns if x.split('_')[-1]=='Year']
        assert len(Feature_Dict['feature_all'])==len(Feature_Dict['feature_Profile']+Feature_Dict['feature_ADate']\
                                                     +Feature_Dict['feature_ADateDiff']+Feature_Dict['feature_Accum']\
                                                     +Feature_Dict['feature_AccumDiff']+Feature_Dict['feature_Year'])
        self.data_df = data_df
        self.Feature_Dict = Feature_Dict

    def display_cust_data(self, cust_id):
        display(custinfo_df[custinfo_df['cust_id']==cust_id].style.set_caption('custinfo_df'))
        display(ccba_df[ccba_df['cust_id']==cust_id].style.set_caption('ccba_df'))
        display(cdtx_df[cdtx_df['cust_id']==cust_id].style.set_caption('cdtx_df'))
        display(dp_df[dp_df['cust_id']==cust_id].style.set_caption('dp_df'))
        display(remit_df[remit_df['cust_id']==cust_id].style.set_caption('remit_df'))
    

def get_accum_feature(cust_data, cust_id, date_colName, rolling_window):
    '''
    ** 這個function要放在class外面, joblib.Parallel才會比較快.  放在class內會很慢. 
    Process:
        - 1. 組合cust_id * dates
        - 2. merge with date feature
        - 3. calculate accum feature
        - 4. merge date and accum features    
    '''
    dates = list(range(cust_data.index.min(), cust_data.index.max()+1))
    all_custDate = pd.DataFrame(list(product([cust_id], dates)),\
                                columns=['cust_id', date_colName]).set_index(['cust_id', date_colName])
    cust_fullDate = pd.merge(cust_data, all_custDate, how='outer', left_index=True, right_index=True)
    cust_fullDate.sort_index(ascending=True, inplace=True)
    cust_fullDate.fillna(0, inplace=True)
    cust_recentAccum = cust_fullDate.rolling(window=rolling_window, min_periods=1).mean().shift()
    cust_recentAccum2W = cust_fullDate.rolling(window=rolling_window*2, min_periods=1).mean().shift()
    cust_recentAccumDiff = cust_recentAccum*2 - cust_recentAccum2W
    cust_fullDateDiff = cust_fullDate - cust_recentAccum

    new_cols = []
    for c in cust_recentAccum.columns:
        new_cols.append(c.replace('_ADate', '_Accum'))
    cust_recentAccum.columns=new_cols

    new_cols = []
    for c in cust_recentAccumDiff.columns:
        new_cols.append(c.replace('_ADate', '_AccumDiff'))
    cust_recentAccumDiff.columns=new_cols

    new_cols = []
    for c in cust_fullDateDiff.columns:
        new_cols.append(c.replace('_ADate', '_ADateDiff'))
    cust_fullDateDiff.columns=new_cols        

    feature = pd.merge(cust_data, cust_fullDateDiff, how='left', left_index=True, right_index=True)
    feature = pd.merge(feature, cust_recentAccum, how='left', left_index=True, right_index=True)
    feature = pd.merge(feature, cust_recentAccumDiff, how='left', left_index=True, right_index=True)

    return feature        
