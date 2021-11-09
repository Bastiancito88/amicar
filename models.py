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
from imblearn.ensemble import BalancedRandomForestClassifier
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
    def __init__(self, X = None, Y = None, path = '', path_train = '', path_test = '', dataset_name = '', seed = 0, scoring = 'balanced_accuracy', n_folds = 2):
        
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

        self.rf = RandomForestClassifier(class_weight = 'balanced')


        self.brf = BalancedRandomForestClassifier(class_weight = 'balanced')

        self.column_transformer = None
        self.cols_to_transform = None

        self.numeric_cols = None
        self.categorical_cols = None

        self.label_encoder = LabelEncoder()

        self.mood = { '1':  self.xgboost, '2' : self.lgbm , '3' : self.rf, '4' : self.brf}

        self.feature_importance = {'1' : None, '2' : None}

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

                print(data.columns)

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
        y_3 = self.forward(mood = '3')
        y_4 = self.forward(mood = '4')

        results_clf = [y_1, y_2, y_3, y_4]


        kappa_matrix = np.zeros((4,4))

        for i, y_right in enumerate(results_clf):
            for j, y_left in enumerate(results_clf):
                kappa_score = cohen_kappa_score(y_right, y_left)

                kappa_matrix[i, j] = kappa_score

        return kappa_matrix

    ''' XGBOOST rutine '''
    def fit_xgboost(self, subsample = False):

        from collections import Counter
        counter_dic = Counter(self.y_sub if subsample else self.y_train)
        scale_pos_weight = counter_dic['NO_PREPAGO']/counter_dic['PREPAGO']

        X_train = self.normalize_data( subsample= subsample)
        #X_train = X_train.fillna(-1)
        y_train = self.label_encoder.transform(self.y_sub) if subsample else self.label_encoder.transform(self.y_train)



        print(X_train.shape)
        print(y_train.shape)
        
        params = {'max_leaves' : [2, 3, 4],
          'scale_pos_weight' : [scale_pos_weight, np.sqrt(scale_pos_weight)],
          'max_depth': [5, 10, 15],
          'max_delta_step': [1, 10, 20],
          'eval_metric' : ['auc', 'aucpr', 'error'],
          }

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
        self.feature_importance['1'] = self.xgboost.feature_importances_
    ''' LGBM rutine '''
    def fit_lgbm(self, subsample = False):

        from collections import Counter
        counter_dic = Counter(self.y_train)
        scale_pos_weight = counter_dic['NO_PREPAGO']/counter_dic['PREPAGO']

        X_train = self.normalize_data( subsample= subsample)
        X_train = X_train.fillna(-1)
        y_train = self.label_encoder.transform(self.y_sub) if subsample else self.label_encoder.transform(self.y_train)

        params = {'num_leaves': [ 2, 3, 10, 20], 
         'objective': ['binary'],
         'max_depth' : [5, 10, 15],
         'metric' : ['auc', 'aucpr', 'error'],
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
        self.feature_importance['2'] = self.lgbm.feature_importances_


    def fit_rf(self, subsample = False):

        from collections import Counter
        #counter_dic = Counter(self.y_train)
        #scale_pos_weoght = counter_dic['NO_PREPAGO'] / counter_dic['PREPAGO']
        X_train = self.normalize_data(subsample=subsample)
        X_train = X_train.fillna(-1)
        y_train = self.label_encoder.transform(self.y_sub) if subsample else self.label_encoder.transform(self.y_train)


        # Number of trees in random forest
        n_estimators = [100, 150] #[int(x) for x in np.linspace(start = 100, stop = 200, num = 15)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt', 'log2']
        # Maximum number of levels in tree
        max_depth = [5, 10, 15] # [int(x) for x in np.linspace(3, 40, num = 15)]
       
        # Minimum number of samples required to split a node
        min_samples_split = [2, 4] #falta un 6
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2] # le falta un 4
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        grid_search = GridSearchCV(estimator = self.rf,
                                     scoring = 'balanced_accuracy', param_grid = param_grid, 
                                    cv = 2,
                                     n_jobs = -1, 
                                     verbose = 2)


        grid_search.fit(X_train, y_train)
        self.rf = grid_search.best_estimator_
        self.mood['3'] = grid_search.best_estimator_
        try:
            self.feature_importance['3'] = self.rf.feature_importances_
        except:
            pass

    def fit_brf(self, subsample = False):

        from collections import Counter
        #counter_dic = Counter(self.y_train)
        #scale_pos_weoght = counter_dic['NO_PREPAGO'] / counter_dic['PREPAGO']
        X_train = self.normalize_data(subsample=subsample)
        X_train = X_train.fillna(-1)
        y_train = self.label_encoder.transform(self.y_sub) if subsample else self.label_encoder.transform(self.y_train)

        # Number of trees in random forest
        n_estimators =  [100, 150]  #[int(x) for x in np.linspace(start = 100, stop = 200, num = 15)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt', 'log2']
        # Maximum number of levels in tree
        max_depth =[5, 10, 15] # [int(x) for x in np.linspace(3, 40, num = 15)]
       
        # Minimum number of samples required to split a node
        min_samples_split = [2, 4, 6]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        grid_search = GridSearchCV(estimator = self.brf, scoring = 'balanced_accuracy', param_grid = param_grid, 
                          cv = 2, n_jobs = -1, verbose = 2)

        grid_search.fit(X_train, y_train)
        self.brf = grid_search.best_estimator_
        self.mood['4'] = grid_search.best_estimator_
        try:
            self.feature_importance['4'] = self.brf.feature_importances_
        except:
            pass

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

    '''Sensibilidad de Tasa'''
    def sensibilidad_tasa(self, mood = '1', prepago = True, percent = 0.4):

        tasa_list = ['Tasa', 'Tasa_Amicar', 'Tasa_Piso_Ofertada', 'PORCENTAJE_PIE']

        x = self.normalize_data( train = False)
        y_pred = self.mood[mood].predict(x)
        y_test = self.label_encoder.transform(self.y_test)

        """indices donde fue bien clasificado el cliente"""
        correct_classification = np.where((y_pred == y_test) & (y_pred == (1 if prepago else 0)))[0]
        x_prueba = x.iloc[correct_classification.tolist()]
        y_prueba = y_pred[correct_classification.tolist()]

        """introducir cambios en la variable tasa en un tanto % con respecto a su valor"""
        x_prueba[tasa_list[0]] = x_prueba[tasa_list[0]]*percent
        nueva_clasificacion = self.mood[mood].predict(x_prueba)

        print('cantidad de clientes prepagos en un inicio = ', len(y_prueba))
        print('cantidad de clientes prepagos despues de una variacion de Tasa  =', len(np.where(nueva_clasificacion == 1)[0]))

    ''' function to make some plots: confusion matrix, ROC curve'''
    def check_folder(self):

        isExist = os.path.exists('./results/' + self.dataset_name)
        if not isExist:
            os.makedirs('./results/' + self.dataset_name)

    def name_clf(self, mood = '1'):
        if mood == '1':
            return '_XGB'
        elif mood == '2':
            return '_LGBM'
        elif mood == '3':
            return '_RF'
        elif mood == '4':
            return '_BRF'
        else:
            print('NAME NOT FOUND !')
            return None

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

        clf_name = self.name_clf(mood = mood) 
        plt.grid(False)
        plt.savefig('./results/' + self.dataset_name + '/CM_'+ self.dataset_name + clf_name + '.png', format = 'png')
        plt.close()

    def get_roc_curve(self, mood = '1'):

        self.check_folder()

        y_score = self.forward_proba(mood = mood)
        y_test = self.label_encoder.transform(self.y_test)

        plt.style.use('ggplot')
        plt.rcParams['axes.facecolor'] = 'white'

        custom_roc_curve(y_test, y_score)

        clf_name = self.name_clf(mood = mood) 
        plt.grid(False)
        plt.savefig('./results/' + self.dataset_name + '/ROC_'+ self.dataset_name + clf_name + '.png', format = 'png')
        plt.close()

    def get_feature_importance(self, mood = '1'):

        self.check_folder()
        fig, ax = plt.subplots(1,1,figsize= ( 12, 9))
        
        if mood == '1':
            ax = xgb.plot_importance(self.xgboost, ax = ax, max_num_features=10)
            
        elif mood == '2':
            ax = lgb.plot_importance(self.lgbm, max_num_features=10
                         , figsize= ( 12, 9))

        elif mood == '3':

            import matplotlib.pyplot as plt
            
            feat_importances = pd.Series(self.rf.feature_importances_, index= self.X_train.columns)
            ax = feat_importances.nlargest(10).plot(kind='barh')

        elif mood == '4':
            ax = None

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
        clf_name = self.name_clf(mood = mood) 
        plt.grid(False)
        plt.savefig('./results/' + self.dataset_name + '/FI_'+ self.dataset_name + clf_name + '.png', format = 'png')
        plt.close()


    def get_distribution(self, feature_name = 'Tasa'):

        self.check_folder()

        x_train = self.X_train
        y_train = self.label_encoder.transform(self.y_train)

        fig, ax = plt.subplots(1,1,figsize= ( 8, 5))

        from matplotlib import cm
        plt.style.use('bmh')

        bin = 25
        colors =  ['darkorange', 'darkblue']

        tasa_list = ['Tasa', 'Tasa_Amicar', 'Tasa_Piso_Ofertada', 'PORCENTAJE_PIE']

        for i, class_name in enumerate(self.label_encoder.classes_):

            data = x_train.iloc[ np.where(y_train == i)[0]][tasa_list[0]].values.tolist()
            n, bins, patches = ax.hist( data , bin, density=False,  facecolor= colors[i] , alpha=0.4, label = '{}'.format(class_name))
            h_max = np.max(n)
            for item in patches:
                item.set_height(item.get_height()/h_max)
            
            print(n)
            ax.vlines(np.mean(data), 0, 1,  colors = colors[i] , lw = 3)
        
        ax.legend()
        ax.set_ylim([0, 1])
        plt.grid(False)
        plt.savefig('prueba.png', format = 'png')
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



        












