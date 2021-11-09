


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix,  precision_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import (OneHotEncoder, label_binarize, LabelEncoder )

def mean_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          plot_size = None,
                          cmap=plt.cm.Blues):

    """ 
    # y_true : [y_true_sim_1, y_true_sim_2, .... y_pred_sim_k]    
    # y_pred : [y_pred_sim_1, y_pred_sim_2, .... y_pred_sim_k]
    # classes : ['label_1', label_2, ...] len(classes) = n_classes
    # title : -----
    # plot_size : (h, w)
    # cmap : plt.cm.Blues,  [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    """
                              
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    
    n_sim = len(y_true)  

    cnt = 0
    cm_list = []

    for y_true_idx, y_pred_idx in zip(y_true, y_pred):
        if cnt == 0:
            cm = confusion_matrix(y_true_idx, y_pred_idx)
        else:
            cm += confusion_matrix(y_true_idx, y_pred_idx)

        cm_list.append(confusion_matrix(y_true_idx, y_pred_idx))

        cnt += 1

    cm = np.mean(cm_list, axis = 0)
    cm_std = np.std(cm_list, axis = 0)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true_idx, y_pred_idx)]

    if normalize:

        normalize_factor = cm.sum(axis=1)[:, np.newaxis]

        cm = cm.astype('float') / normalize_factor
        cm_std = cm_std.astype('float') / normalize_factor

        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

        cm = np.floor(cm).astype('int')

    fig, ax = plt.subplots(figsize=plot_size,  dpi=80)
    
    if normalize:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin= 0, vmax=1)
    
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            if i != j :
                ax.text(j, i, format(cm[i, j], fmt) ,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            elif i == j:

                ax.text(j, i, format(cm[i, j], fmt) + r'$\pm$' + format(cm_std[i,j], '.2f') ,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
                
             
    fig.tight_layout()
    
    return ax, fig

def custom_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          plot_size = None,
                          cmap=plt.cm.Blues):
  
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        normalize_factor = cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / normalize_factor
        cm = cm*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        cm = np.floor(cm).astype('int')

    fig, ax = plt.subplots(figsize=plot_size,  dpi=80)
    if normalize:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin= np.min(cm), vmax= np.max(cm))
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = (cm.max() - cm.min() ) / 2.  + cm.min()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt) + r'$\%$' ,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return ax, fig

def custom_roc_curve(y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 2

    enc = OneHotEncoder(sparse= False)

    # onehot encoding y proba 
    
    y_test = enc.fit_transform( y_test.reshape(-1,1)) 

    #y_score = best_model.predict_proba(X_test)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize = (10, 6))
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
            lw=lw, label='ROC curve 1 (area = %0.2f)' % roc_auc[1])

    plt.plot(fpr[0], tpr[0], color = 'darkblue',
            lw=lw, label='ROC curve 0 (area = %0.2f)' % roc_auc[1])

    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.title('ROC curve', fontsize = 14)
    plt.legend(loc= "lower right", fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    