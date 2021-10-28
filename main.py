import numpy as np
import argparse
import glob
import feather
import re
from models import AmicarDataset
import pandas as pd

if __name__ == '__main__':

    pd.options.mode.chained_assignment = None

    # lectura de ficheros disponibles 

    datasets_folders = glob.glob('datasets/*')
    cnt = 0
    for dataset in datasets_folders:


        if cnt == 0:

            cnt+=1
        
            dataset_name = re.sub('\W+',' ',  dataset.partition("datasets")[2]).strip() #re.sub('[^A-Za-z0-9]+', ' ',  dataset.partition("datasets")[2])
            folder_path = glob.glob(dataset + '/*')

            models = AmicarDataset(path= folder_path, dataset_name = dataset_name)
            models.read_folder()
            models.split_dataset(test_size= 0.8)
            models.fit_normalize_data()
            #models.subsampling(size = 0.99)

            models.fit_xgboost(subsample=False)
            models.fit_lgbm(subsample=False)

            models.classification_report(mood= '2')
            models.get_confusion_matrix(mood= '2')
            models.get_roc_curve(mood = '2')
            models.get_feature_importance(mood= '2')

            models.classification_report(mood= '1')
            models.get_confusion_matrix(mood= '1')
            models.get_roc_curve(mood = '1')
            models.get_feature_importance(mood= '1')

            kappa_value = models.get_kappa_cohen()
            print('*'*10)
            print('kappa value =', kappa_value)
            print('*'*10)

            models.save_models()
        else:
            pass



    #models.classification_report(mood= '1')
    #models.get_confusion_matrix(mood= '1')
    #models.get_roc_curve(mood = '1')



    

    
