import os
import numpy as np
import pandas as pd
import scipy
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.utils import resample
from matplotlib import pyplot as plt

def get_data(din):
    bf = pd.read_csv(din,sep='\t')
    genes = bf['Hugo_Symbol']
    cols = list(bf.columns)
    cols.remove('Entrez_Gene_Id')
    cols.remove('Hugo_Symbol')
    bf = bf[cols]
    bf = bf.transpose()
    bf.columns = genes
    return bf

def get_residuals(data,U):
    I = np.identity(data.shape[1])
    z = data.dot(I - U.dot(U.T))
    residuals = np.power(z,2).sum(axis=1) 
    return residuals

def get_score_threshold(X_train, X_test):
    
    scaler = StandardScaler().fit(X_train)
    X_train_nom,X_test_nom = scaler.transform(X_train), scaler.transform(X_test)        
    pca = PCA()
    pca.fit(X_train_nom)
    
    S,U = pca.explained_variance_, pca.components_.T
    cs = S.cumsum()
    K = int(np.where(cs >= cs[-1] * keep_info)[0][0] + 1)
    U = U[:,:K]
    
    residuals = get_residuals(X_test_nom, U)
    
    c_beta = scipy.stats.norm.ppf(1 - beta)
    theta1 = sum(S[K+1:])
    theta2 = sum(S[K+1:]**2)
    theta3 = sum(S[K+1:]**3)
    h0 = 1 - ((2*theta1*theta3)/(3*theta2*theta2))
    Qbeta = theta1 * (((c_beta*np.sqrt(2*theta2*h0*h0)/theta1) + 1 + ((theta2*h0*(h0-1))/(theta1*theta1)) )**(1/h0))
    
    return residuals, Qbeta

random_state = np.random.RandomState(42)

opts = [['breast','brca'], ['liver','lihc'], ['lung','luad'],['prostate','prad'],['stomach','stad'],['thyroid','thca']]
# opts = [ opts[-1] ] 
dirin = 'data/'

keep_info = .1
beta = .999

for p in range(1,10):
    percent = p*10
    print("\npercent:", percent)
    result = []
    for opt in opts:
        din1 = dirin + opt[0] + '-rsem-fpkm-gtex.txt'
        din2 = dirin + opt[1] + '-rsem-fpkm-tcga.txt'
        din3 = dirin + opt[1] + '-rsem-fpkm-tcga-t.txt'
        if os.path.isfile(din1) and os.path.isfile(din2) and os.path.isfile(din3):
            normal1 = get_data(din1)
            normal2 = get_data(din2)
            abnormal = get_data(din3)
            common_gene = set(list(normal1.columns)).intersection(set(list(normal2.columns))).intersection(set(list(abnormal.columns)))
            common_gene = list(common_gene)
            normal1 = normal1[common_gene]
            normal2 = normal2[common_gene]
            abnormal = abnormal[common_gene]
            X_train_full = normal1.values
            X_test = pd.concat( [normal2, abnormal] ).values
            y_test = np.zeros(len(X_test), dtype=int)
            y_test[len(normal2):] = 1
            
            
            ret = []
            for seed in range(10):
                X_train = resample(X_train_full, replace=False, 
                                        n_samples= int((float(percent)/100)*len(X_train_full)), random_state=seed)
                
                anomal_score,threshold = get_score_threshold(X_train,X_test)
                AUC = metrics.roc_auc_score(y_test, anomal_score)
                y_pred =  (anomal_score > threshold).astype('int').ravel()
                F1 = metrics.f1_score(y_test, y_pred)
                Accuracy = metrics.accuracy_score(y_test, y_pred)
                Precision = metrics.precision_score(y_test, y_pred)
                Recall = metrics.recall_score(y_test, y_pred)
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
                Specificity = tn / (tn+fp)
                score = [F1,Precision,Recall,Specificity,Accuracy,AUC]
                ret += [ score ]
            ret = np.array(ret)
            ret = list(np.mean(ret, axis=0)) 
            ret = [round(e,3) for e in ret]            
            ret = [opt[0], len(X_train), len(normal2), len(abnormal)] + [keep_info,beta] + ret
            result += [ ret ]
            print(ret[0],ret[-6:])     


    with open('results/result_' + str(percent) + '.csv','w') as f:
        f.write('dataset,GTEx(N),TCGA(N),TCGA(C),keep_info,beta,F1,Precision,Recall,Specificity,Accuracy,AUC\n')
        for e in result:
            st = ','.join(map(str,e)) + '\n'
            f.write(st)