#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 19:33:46 2018

@author: yi
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def profiling_ftr(data, data_table):
    for feature in data.columns:
        if not feature.startswith('FTR'):
            continue
        if feature == 'FTR51':
            continue
        mean_feature = data.groupby('PERSONID')[feature]\
                        .mean().\
                        to_frame(name=feature+'_MEAN').reset_index()
                        
        std_feature = data.groupby('PERSONID')[feature]\
                        .std().\
                        to_frame(name=feature+'_STD').reset_index()
        max_feature = data.groupby('PERSONID')[feature]\
                        .max().\
                        to_frame(name=feature+'_MAX').reset_index()
        min_feature = data.groupby('PERSONID')[feature]\
                        .min().\
                        to_frame(name=feature+'_MIN').reset_index()
        data_table = data_table.merge(mean_feature, on='PERSONID', how='left')
        data_table = data_table.merge(std_feature, on='PERSONID', how='left')
        data_table = data_table.merge(max_feature, on='PERSONID', how='left')
        data_table = data_table.merge(min_feature, on='PERSONID', how='left')
        
        data_table[feature+'_MEANSTDR'] =\
            data_table[feature+'_MEAN']/data_table[feature+'_STD']
        data_table[feature+'_MEANSTDR'] =\
            data_table[feature+'_MEANSTDR'].replace(np.inf, 0).fillna(0)
            
    return data_table


def profiling_person_daily(data, data_table):
    daily_app = data.groupby(['PERSONID', 'CREATETIME'])['APPLYNO']\
                    .nunique().to_frame('DAILY_APP_NO').reset_index()
    unique_day = data.groupby('PERSONID')['CREATETIME']\
        .nunique().to_frame(name='UNIQUE_DAY').reset_index()
    data_table = data_table.merge(unique_day, on='PERSONID', how='left')
    for feature in data.columns:
        if not feature.startswith('FTR'):
            continue
        if feature == 'FTR51':
            continue
        
        mean_feature = data.groupby(['PERSONID', 'CREATETIME'])[feature]\
                        .mean().\
                        to_frame(name=feature+'_MEAN_DAY').reset_index()
                        
        std_feature = data.groupby(['PERSONID', 'CREATETIME'])[feature]\
                        .std().\
                        to_frame(name=feature+'_STD_DAY').reset_index()
        max_feature = data.groupby(['PERSONID', 'CREATETIME'])[feature]\
                        .max().\
                        to_frame(name=feature+'_MAX_DAY').reset_index()
        min_feature = data.groupby(['PERSONID', 'CREATETIME'])[feature]\
                        .min().\
                        to_frame(name=feature+'_MIN_DAY').reset_index()
        daily_app = daily_app.merge(mean_feature, on=['PERSONID', 'CREATETIME'],
                        how='left')
        daily_app = daily_app.merge(std_feature, on=['PERSONID', 'CREATETIME'],
                        how='left')
        daily_app = daily_app.merge(max_feature, on=['PERSONID', 'CREATETIME'],
                        how='left')
        daily_app = daily_app.merge(min_feature, on=['PERSONID', 'CREATETIME'],
                        how='left')
    for feature in daily_app:
        if daily_app[feature].dtype==object:
            continue
        max_daily_feature = daily_app.groupby('PERSONID')[feature]\
                            .max()\
                            .to_frame(name=feature+'_MAX')
        min_daily_feature = daily_app.groupby('PERSONID')[feature]\
                            .max()\
                            .to_frame(name=feature+'_MIN')
        data_table = data_table.merge(max_daily_feature, on='PERSONID',
                         how='left')
        data_table = data_table.merge(min_daily_feature, on='PERSONID',
                         how='left')
    return data_table


def profiling_ftr51(data, data_table):
    FTR51_ser = data['FTR51'].apply(lambda x: x.rstrip().split(','))
    items_dict = dict()
    for items in FTR51_ser:
        for item in items:
            if item in items_dict:
                items_dict[item] += 1
            else:
                items_dict[item] = 1
    sorted_items = sorted(items_dict, key = items_dict.get)[-100:]
    def process_list(xs):
        for idx, x in enumerate(xs):
            if x not in sorted_items:
                xs[idx] = 'OTHER'
        return xs
    FTR51_ser = FTR51_ser.apply(process_list)
    FTR51_str_list = [' '.join(x) for x in FTR51_ser]
    vect = CountVectorizer(lowercase=False, max_features=101)
    X_trans = vect.fit_transform(FTR51_str_list)

    X_dense=X_trans.todense()
    X_df = pd.DataFrame(X_dense, columns =
                        sorted(vect.vocabulary_, key=vect.vocabulary_.get))
    data_combine = pd.concat([data, X_df], axis=1)
    
    for item in sorted_items+['OTHER']:
        sum_item = data_combine.groupby('PERSONID')[item]\
            .sum().to_frame(name=item+'_SUM').reset_index()
        data_table = data_table.merge(sum_item,
                                      on='PERSONID',
                                      how='left').fillna(0)
    return data_table
    

def process_features(data, data_table):
    person_count = data.groupby('PERSONID')['PERSONID']\
        .count().to_frame(name='NUM_RECORDS').reset_index()
    data_table = data_table.merge(person_count, on='PERSONID', how='left')
    print('Step1')
    data_table = profiling_ftr(data, data_table)
    print('Step2')
    data_table = profiling_person_daily(data, data_table)
    print('Step3')
    data_table = profiling_ftr51(data, data_table)
    return data_table
    

if __name__ == '__main__':
    train = pd.read_csv('train.tsv', sep='\t')
    train_id = pd.read_csv('train_id.tsv', sep='\t')
    train_table = process_features(train, train_id)
    print(train_table.shape)