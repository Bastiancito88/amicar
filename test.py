import numpy as np
import argparse
import glob
import feather
import re
import joblib
from xgboost.training import train
from models import AmicarDataset
import pandas as pd


if __name__ == '__main__':


    results_folders = glob.glob('results/*/')

    print(results_folders)

    for result in results_folders:


        
        print('*'*10)
        print(result)
        print('*'*10)
        
        # path amicar model 
        model_path = glob.glob(result + '*models.pkl')[0]
        # path old data train y test
        test_path = glob.glob(result + '*_test.feather')[0]
        train_path = glob.glob(result + '*_train.feather')[0]
        # loading model 
        amicar_model = joblib.load(model_path)
        amicar_model.path_train = train_path
        amicar_model.path_test = test_path

        # read datasets
        amicar_model.read_train()
        amicar_model.read_test()

        # get some metrics 
        kappa_value = amicar_model.get_kappa_cohen()
        print('*'*10)
        print('kappa value =', kappa_value)
        print('*'*10)

        amicar_model.classification_report(mood= '2')
        amicar_model.classification_report(mood= '2')