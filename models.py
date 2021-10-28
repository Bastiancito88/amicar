from operator import sub
import numpy as np
import pandas as pd 
import feather 
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from plottinglib import custom_confusion_matrix, custom_roc_curve
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, cohen_kappa_score
from sklearn.feature_selection import chi2
import re


import lightgbm as lgb


pd.options.mode.chained_assignment = None

class AmicarDataset():
    def __init__(self, X = None, Y = None, path = '', path_train = '', path_test = '', dataset_name = '', seed = 0, scoring = 'balanced_accuracy', n_folds = 5):
        
        self.seed = seed
        self.path = path
        self.path_train = path_train
        self.path_test = path_test
        self.dataset_name = dataset_name
        self.n_folds = n_folds
        self.scoring = scoring
        self.X = X
        self.Y = Y
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_sub = None
        self.y_sub = None

        self.xgboost = XGBClassifier(tree_method = 'gpu_hist', 
                                    use_label_encoder= False)

        self.lgbm =  LGBMClassifier()

        self.column_transformer = None
        self.cols_to_transform = None

        self.numeric_cols = None
        self.categorical_cols = None

        self.label_encoder = LabelEncoder()

        self.mood = { '1':  self.xgboost, '2' : self.lgbm , '3' : None, '4' : None}

    def read_dataset(self):

        if self.path != '':
            try:
                data = feather.read_dataframe(self.path)
                data = data.rename(columns = lambda x: re.sub('\W+', ' ', x).strip())  

                self.Y = data.ETIQUETA
                self.X = data[data.columns.difference(['ETIQUETA'])]

                print('Loaded dataset')
                print('n samples = ', data.shape[0])
                print('n features = ', data.shape[1] - 1)
            except:
                print('File not found !')
        else:
            print('no path !')

    def read_test(self):

        if self.path_test != '':
            try:
                data = feather.read_dataframe(self.path_test)
                self.y_test = data.ETIQUETA
                self.X_test = data[data.columns.difference(['ETIQUETA'])]
            except:
                print('File not found !')
        else:
            print('no path !')

    def read_train(self):

        if self.path_train != '':
            try:
                data = feather.read_dataframe(self.path_train)
                self.y_train = data.ETIQUETA
                self.X_train = data[data.columns.difference(['ETIQUETA'])]
            except:
                print('File not found !')
        else:
            print('no path !')


    def read_folder(self):

        # assumes that de data is contained in varius dataframes inside one folder
        if self.path != '':
            try:

                data_frames_list = []

                for path in self.path:
                    data = feather.read_dataframe(path)
                    data = data.rename(columns = lambda x: re.sub('\W+', ' ', x).strip())  
                    
                    data_frames_list.append(data)

                all_data = pd.concat(data_frames_list, ignore_index= True)

                self.Y = all_data.ETIQUETA
                self.X = all_data[all_data.columns.difference(['ETIQUETA'])]

                print('Loaded dataset')
                print('n samples = ', all_data.shape[0])
                print('n features = ', all_data.shape[1] -1)
            except:
                print('File not found !')
        else:
            print('no path !')

    def subsampling(self, size = 0.5):


        X_train = self.X_train
        Y_train = self.y_train
        y_train = self.label_encoder.transform(self.y_train)



        majority_indexes = np.where(y_train == 0)[0]
        minority_indexes = np.where(y_train == 1)[0]

        random_indexes = np.random.choice(majority_indexes, int(size*len(majority_indexes)), replace=False)
        final_indexes = np.concatenate((random_indexes, minority_indexes), axis = 0)

        X_sub = X_train.iloc[final_indexes.tolist()]
        Y_sub = Y_train.iloc[final_indexes.tolist()]


        self.X_sub = X_sub
        self.y_sub = Y_sub

    '''split dataset into Train and test for all classifiers'''
    def split_dataset(self, test_size, stratify = True):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X,  self.Y,
                                                            test_size=test_size,
                                                            stratify = self.Y if stratify else False)

        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

    ''' MinMaxScaler only in numeric features '''
    def fit_normalize_data(self):
        
        X_train = self.X_train
        Y_train = self.y_train

        cols_to_transform = []

        for i in X_train.columns:
            if X_train[i].max() > 1:
                cols_to_transform.append(i)

        cols_to_keep = list(set(X_train.columns.tolist()) - set(cols_to_transform))


        min_max = True

        if min_max:
            ct_train = ColumnTransformer([('features_to_process', MinMaxScaler(), cols_to_transform)])

        # save trasnformer 
        ct_train.fit(X_train)
        self.label_encoder.fit(Y_train)
        self.column_transformer = ct_train
        self.numeric_cols = cols_to_transform
        self.categorical_cols = cols_to_keep

    '''normalize new data'''
    def normalize_data(self, train = True, subsample = False):


        if subsample:
            X = self.X_sub if train  else self.X_test
        else:
            X = self.X_train if train  else self.X_test

        cols_to_transform = self.numeric_cols
        features = self.column_transformer.transform(X)
        scaled_features = pd.DataFrame(features,  index=X.index,  columns= cols_to_transform )
        X = X.drop(columns = cols_to_transform)
        X = pd.merge(X, scaled_features, left_index=True, right_index=True, how = 'inner')

        return X


    ''' Test functions over data '''

    def get_chi2(self):

        X = self.X_train[self.categorical_cols]
        Y = self.label_encoder.transform(self.y_train)

        chi_value, p_values = chi2(X, Y)

        return chi_value, p_values

    def get_kappa_cohen(self):

        y_1 = self.forward(mood = '1')
        y_2 = self.forward(mood = '2')

        kappa_score = cohen_kappa_score(y_1, y_2)

        return kappa_score

    ''' XGBOOST rutine '''
    def fit_xgboost(self, subsample = False):

        from collections import Counter
        counter_dic = Counter(self.y_sub if subsample else self.y_train)
        scale_pos_weight = counter_dic['NO_PREPAGO']/counter_dic['PREPAGO']

        X_train = self.normalize_data( subsample= subsample)
        X_train = X_train.fillna(-1)
        y_train = self.label_encoder.transform(self.y_sub) if subsample else self.label_encoder.transform(self.y_train)



        print(X_train.shape)
        print(y_train.shape)
        
        params = {'max_leaves' : [2, 3, 4],
          'scale_pos_weight' : [scale_pos_weight, np.sqrt(scale_pos_weight)],
          'max_depth': [5, 10, 15],
          'eval_metric' : ['auc']}

        folds = self.n_folds

       
        skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = self.seed)   
        grid_search = GridSearchCV(self.xgboost,
                             param_grid = params,
                             scoring = self.scoring, 
                             n_jobs = -1, 
                             cv = skf.split(X_train, y_train),
                             verbose = 3)

        grid_search.fit(X_train, y_train)

        #print(grid_search.best_estimator_)
        self.xgboost = grid_search.best_estimator_
        self.mood['1'] = grid_search.best_estimator_

    ''' LGBM rutine '''
    def fit_lgbm(self, subsample = False):

        from collections import Counter
        counter_dic = Counter(self.y_train)
        scale_pos_weight = counter_dic['NO_PREPAGO']/counter_dic['PREPAGO']

        X_train = self.normalize_data( subsample= subsample)
        X_train = X_train.fillna(-1)
        y_train = self.label_encoder.transform(self.y_sub) if subsample else self.label_encoder.transform(self.y_train)

        params = {'num_leaves': [ 2, 3, 4], 
         'objective': ['binary'],
         'max_depth' : [5, 10, 15],
         'metric' : ['auc'],
          'scale_pos_weight' : [scale_pos_weight, np.sqrt(scale_pos_weight)]
         }

        folds = self.n_folds

        skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = self.seed)



        grid_search = GridSearchCV(self.lgbm,
                             param_grid = params,
                             scoring = self.scoring, 
                             n_jobs = -1, 
                             cv = skf.split(X_train, y_train),
                             verbose = 3)


        grid_search.fit(X_train, y_train)
        self.lgbm = grid_search.best_estimator_
        self.mood['2'] = grid_search.best_estimator_
    
    ''' get predictions'''
    def forward(self, mood = '1'):
        # get normalized data
        x = self.normalize_data( train = False)
        x = x.fillna(-1)
        return self.mood[mood].predict(x)

    ''' get proba '''
    def forward_proba(self, mood = '1'):
        # get normalized data
        x = self.normalize_data( train = False)
        x = x.fillna(-1)
        return self.mood[mood].predict_proba(x)

    ''' classification report '''
    def classification_report(self, mood = '1'):

        x = self.normalize_data( train = False)
        x = x.fillna(-1)
        y_pred = self.mood[mood].predict(x)
        y_test = self.y_test

        print(classification_report(self.label_encoder.transform(y_test), y_pred))

    ''' function to make some plots: confusion matrix, ROC curve'''
    def check_folder(self):

        isExist = os.path.exists('./results/' + self.dataset_name)
        if not isExist:
            os.makedirs('./results/' + self.dataset_name)


    def get_confusion_matrix(self, mood = '1'):

        self.check_folder()

        y_pred = self.forward(mood = mood)
        y_test = self.label_encoder.transform(self.y_test)
        plt.style.use('bmh')
        ax, fig = custom_confusion_matrix( y_test, y_pred, self.label_encoder.classes_,
                          normalize= True,
                          title= 'Confusion Matrix',
                          plot_size = (5,5),
                          cmap= plt.cm.Blues)

        plt.grid(False)
        plt.savefig('./results/' + self.dataset_name + '/CM_'+ str(mood)+ '_.png', format = 'png')
        plt.close()

    def get_roc_curve(self, mood = '1'):

        self.check_folder()

        y_score = self.forward_proba(mood = mood)
        y_test = self.label_encoder.transform(self.y_test)

        plt.style.use('ggplot')
        plt.rcParams['axes.facecolor'] = 'white'

        custom_roc_curve(y_test, y_score)

        plt.grid(False)
        plt.savefig('./results/' + self.dataset_name + '/ROC_'+ str(mood)+ '_.png', format = 'png')
        plt.close()

    def get_feature_importance(self, mood = '1'):

        self.check_folder()
        fig, ax = plt.subplots(1,1,figsize= ( 12, 9))
        
        if mood == '1':
            ax = xgb.plot_importance(self.xgboost, ax = ax, max_num_features=10)
            
        elif mood == '2':
            ax = lgb.plot_importance(self.lgbm, max_num_features=10
                         , figsize= ( 12, 9))

        new_label_axi = []
        for y_label in ax.get_yticklabels():
            new_axi = y_label
            new_axi.set_text( new_axi.get_text().replace('_', ' \n ').title())
            new_label_axi.append(new_axi)

        ax.set_yticklabels(new_label_axi)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('Ranking de Caracteristicas')
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize = 11, multialignment='center')
        plt.grid(False)
        plt.savefig('./results/' + self.dataset_name + '/FI_'+ str(mood)+ '_.png', format = 'png')
        plt.close()
    

    def save_models(self):

        self.check_folder()

        joblib.dump( self.xgboost, './results/' + self.dataset_name + '/xgboost.pkl')
        joblib.dump( self.lgbm, './results/' + self.dataset_name + '/lgbm.pkl')

        data_test = self.X_test
        data_test['ETIQUETA'] = self.y_test

        data_train = self.X_train
        data_train['ETIQUETA'] = self.y_train

        # save train and test data

        with open( './results/' + self.dataset_name + '/data_test.feather', 'wb') as f:
            feather.write_dataframe(data_test, f)

        with open( './results/' + self.dataset_name + '/data_train.feather', 'wb') as f:
            feather.write_dataframe(data_train, f)

        # save object with trained classifiers
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_sub = None
        self.X_sub = None
        # save model 
        joblib.dump(self, './results/' + self.dataset_name + '/models.pkl' )



        












