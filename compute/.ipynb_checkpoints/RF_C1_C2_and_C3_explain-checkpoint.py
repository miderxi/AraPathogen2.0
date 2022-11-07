"""
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from zzd.utils.assess import multi_scores
import sys
import pickle
from sklearn.model_selection import train_test_split

class Features:
    def __init__(self,info:list):
        feature_ac = np.load("../features/AC/ara_and_eff_AC.pkl",allow_pickle=True)                #shape (210,)
        feature_ct = np.load("../features/CT/ara_and_eff_CT.pkl",allow_pickle=True)                #shape (343,)
        feature_dpc = np.load("../features/DPC/ara_and_eff_DPC.pkl",allow_pickle=True)             #shape (400,)
        feature_cksaap = np.load("../features/CKSAAP/ara_and_eff_CKSAAP.pkl",allow_pickle=True)    #shape (1200,)
        feature_EsmMean = np.load("../features/esm-mean/ara_and_eff_esm-mean.pkl",allow_pickle=True)             #shape(1280,)
        feature_EsmMsa = np.load("../features/esm-msa/ara_and_eff_esm-msa.pkl",allow_pickle=True) #shape (768,)

        feature_prottrans = np.load("../features/prottrans/protein_embs.pkl",allow_pickle=True)  #shape (1024,)
        feature_prottrans.update(np.load("../features/prottrans/lost_protein_embs.pkl",allow_pickle=True))  #shape (1024,)
        feature_doc2vec = np.load("../features/doc2vec/ara_and_eff_doc2vec.pkl",allow_pickle=True) #shape (400,)

        class Feature_AraNetProperty:
            def __init__(self):
                self.flag_foldn=None
                self.data = [np.load(f"../features/AraNet_property/ara_and_eff_AraNet_property_C1_fold{i}.pkl",allow_pickle=True) 
                                for i in range(5)] #shape (12,)
            def __getitem__(self,index):
                return self.data[self.flag_foldn][index] if index in self.data[self.flag_foldn].keys() else np.zeros(12,np.float32) 
        
        class Feature_AraNetNode2vec:
            def __init__(self):
                self.data = np.load("../features/AraNet_node2vec/AraNet_node2vec.pkl",allow_pickle=True)  #shape (256,)
            def __getitem__(self,index):
                return self.data[index] if index in self.data.keys() else np.zeros(256,np.float32) 
        
        class Feature_GeneExpression:
            def __init__(self):
                self.data = np.load("../features/gene_expression/geo.pkl",allow_pickle=True) #shape (117, )
            def __getitem__(self,index):
                return self.data[index] if index in self.data.keys() else np.zeros(111,np.float32) 

        class Feature_sublocation:
            def __init__(self):
                self.data = np.load("../features/sublocation/sublocaiton.pkl",allow_pickle=True) #shape (117, )
            def __getitem__(self,index):
                return self.data[index] if index in self.data.keys() else np.zeros(34,np.float32)


        class Feature_DMINode2vec:
            def __init__(self):
                self.data = np.load("../features/dmi_node2vec/ara_and_eff_dmi_node2vec.pkl",allow_pickle=True)
            def __getitem__(self,index):
                return self.data[index] if index in self.data.keys() else np.zeros(128,np.float32) 
        
        #class Feature_pssm_emb():
        #    def __init__(self):
        #        self.flag_foldn=None
        #        self.data =[np.load(f"../features/pssm_emb/pssm_emb_train_{i}.pkl",allow_pickle=True) for i in range(10)]
        #    def __getitem__(self,index):
        #        return self.data[self.flag_foldn][index] 

        #class Feature_onehot_emb():
        #    def __init__(self):
        #        self.flag_foldn=None
        #        self.data =[np.load(f"../features/onehot_emb/onehot_emb_train_{i}.pkl",allow_pickle=True) for i in range(10)]
        #    def __getitem__(self,index):
        #        return self.data[self.flag_foldn][index] 
        #
        #class Feature_AraEffSeqNet():
        #    def __init__(self):
        #        self.data = np.load(f"../features/AraEffSeqNet/AraEffSeqNet.pkl",allow_pickle=True)
        #    def __getitem__(self,index):
        #        return self.data[index] if index in self.data.keys() else np.zeros(10,np.float32)

        #class Feature_AraEffSeqNet_node2vec():
        #    def __init__(self):
        #        self.data = np.load(f"../features/AraEffSeqNet_node2vec/AraEffSeqNet_node2vec.pkl",allow_pickle=True)
        #    def __getitem__(self,index):
        #        return self.data[index] if index in self.data.keys() else np.zeros(256,np.float32)

        self.info = info
        self.features={
            'ac':feature_ac,
            'ct':feature_ct,
            'dpc':feature_dpc,
            'cksaap':feature_cksaap,
            'AraNetProperty':Feature_AraNetProperty(),
            'EsmMean':feature_EsmMean,
            'EsmMsa':feature_EsmMsa,
            'prottrans':feature_prottrans,
            'doc2vec':feature_doc2vec,
            'AraNetNode2vec':Feature_AraNetNode2vec(),
            'DMINode2vec':Feature_DMINode2vec(),
            'GeneExpression':Feature_GeneExpression(),
            #'pssm':Feature_pssm_emb(),
            #'onehot':Feature_onehot_emb(),
            'sublocation':Feature_sublocation(),
            #'AraEffSeqNet':Feature_AraEffSeqNet(),
            #'AraEffSeqNet_node2vec':Feature_AraEffSeqNet_node2vec(),
            }
    
    def get(self,index,foldn=None):
        self.features['AraNetProperty'].flag_foldn=foldn
        #self.features['pssm'].flag_foldn=foldn
        #self.features['onehot'].flag_foldn=foldn
        tmp = np.hstack([self.features[i][index] for i in self.info])
        return tmp


# parameters
print(sys.argv)
info_list = sys.argv[1:] if len(sys.argv)>1 else None
output_dir = f"../output/preds/C123_RF_"+"_".join(info_list)
os.makedirs(output_dir,exist_ok=True)

# read
c1_files = [f'../data/5folds_C1_C2_and_C3/C1_fold{i}.txt' for i in range(5)] 
c2_files = [f'../data/5folds_C1_C2_and_C3/C2_fold{i}.txt' for i in range(5)] 
c3_files = [f'../data/5folds_C1_C2_and_C3/C3_fold{i}.txt' for i in range(5)] 

features = Features(info=info_list)

c1_scores = []
c2_scores = []
c3_scores = []

for foldn in range(1):
    c1 = np.genfromtxt(c1_files[foldn],str)
    c2 = np.genfromtxt(c2_files[foldn],str)
    c3 = np.genfromtxt(c3_files[foldn],str)

    X_c1, y_c1 = c1[:,:2], c1[:,2].astype(np.float32)
    X_c2, y_c2 = c2[:,:2], c2[:,2].astype(np.float32)
    X_c3, y_c3 = c3[:,:2], c3[:,2].astype(np.float32)

    X_c1_train, X_c1_test, y_c1_train,y_c1_test = train_test_split(X_c1, y_c1, train_size=0.9, random_state=0,shuffle=True)
    
    x_c1_train = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c1_train ]) 
    x_c1_test = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c1_test ]) 
    x_c2 = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c2 ]) 
    x_c3 = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c3 ])

    model = RandomForestClassifier(n_estimators=500,n_jobs=-1, random_state=0)
    model.fit(x_c1_train,y_c1_train)

    y_c1_test_pred = model.predict_proba(x_c1_test)[:,1]
    y_c2_pred = model.predict_proba(x_c2)[:,1]
    y_c3_pred = model.predict_proba(x_c3)[:,1]

    c1_score = multi_scores(y_c1_test, y_c1_test_pred, show=True)
    c2_score = multi_scores(y_c2, y_c2_pred, show=True,show_index=False)
    c3_score = multi_scores(y_c3, y_c3_pred, show=True,show_index=False)
    
    c1_scores.append(c1_score)
    c2_scores.append(c2_score)
    c3_scores.append(c3_score)


import lime
from lime import lime_tabular

feature_names = ['degree','betweenness','closeness','pagerank','eccentricity','eigenvector','transitivity','taget_C1_min','taget_C1_max','target_rgene_min','target_rgene_max','target_rgene_unip_min','target_rgene_unip_max']

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(x_c1_train),
    feature_names=[f"A_{i}"for i in feature_names] + [f"B_{i}"for i in feature_names],
    class_names=['neg', 'pos'],
    mode='classification'
)

exp = explainer.explain_instance(
    data_row=x_c1_test[0],
    predict_fn=model.predict_proba
)

exp.show_in_notebook(show_table=True)





























    #with open(f"{output_dir}/c1_pred_{foldn}.txt","w") as f:
    #    for line in np.hstack([X_c1_test,y_c1_test.reshape(-1,1),y_c1_test_pred.reshape(-1,1)]):
    #        line = "\t".join(line) + "\n"
    #        f.write(line)

    #with open(f"{output_dir}/c2_pred_{foldn}.txt","w") as f:
    #    for line in np.hstack([X_c2, y_c2.reshape(-1,1), y_c2_pred.reshape(-1,1)]):
    #        line = "\t".join(line) + "\n"
    #        f.write(line)

    #with open(f"{output_dir}/c3_pred_{foldn}.txt","w") as f:
    #    for line in np.hstack([X_c3, y_c3.reshape(-1,1), y_c3_pred.reshape(-1,1)]):
    #        line = "\t".join(line) + "\n"
    #        f.write(line)

    #with open(f"{output_dir}/c1_score_{foldn}.txt","w") as f:
    #        f.write("TP	TN	FP	FN	PPV	TPR	TNR	Acc	mcc	f1	AUROC	AUPRC	AP\n")
    #        f.write("\t".join([str(i) for i in c1_score]))

    #with open(f"{output_dir}/c2_score_{foldn}.txt","w") as f:
    #        f.write("TP	TN	FP	FN	PPV	TPR	TNR	Acc	mcc	f1	AUROC	AUPRC	AP\n")
    #        f.write("\t".join([str(i) for i in c2_score]))

    #with open(f"{output_dir}/c3_score_{foldn}.txt","w") as f:
    #        f.write("TP	TN	FP	FN	PPV	TPR	TNR	Acc	mcc	f1	AUROC	AUPRC	AP\n")
    #        f.write("\t".join([str(i) for i in c3_score]))



    #save model
    #model_file_name = "../outputs/model_state/RF_" + "_".join(info_list)+f"_foldn_{foldn}.pkl"
    #with open(model_file_name,"wb") as f:
    #    pickle.dump(model,f)

#c1_scores = np.array(c1_scores)
#fmat =  [1, 1,  1,  1,  3,  3,  3,  3,  3,  3,  3,      3,      3]
#with open(f"{output_dir}/c1_average_score.txt",'w') as f:
#    line1 = "TP TN  FP  FN  PPV TPR TNR Acc mcc f1  AUROC   AUPRC   AP\n"
#    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c1_scores.mean(0),c1_scores.std(0))])
#    print(line1)
#    print(line2)
#    f.write(line1)
#    f.write(line2)
#
#c2_scores = np.array(c2_scores)
#with open(f"{output_dir}/c2_average_score.txt",'w') as f:
#    line1 = "TP TN  FP  FN  PPV TPR TNR Acc mcc f1  AUROC   AUPRC   AP\n"
#    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c2_scores.mean(0),c2_scores.std(0)) ])
#    print(line1)
#    print(line2)
#    f.write(line1)
#    f.write(line2)
#
#
#c3_scores = np.array(c3_scores)
#with open(f"{output_dir}/c3_average_score.txt",'w') as f:
#    line1 = "TP TN  FP  FN  PPV TPR TNR Acc mcc f1  AUROC   AUPRC   AP\n"
#    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c3_scores.mean(0),c3_scores.std(0))])
#    print(line1)
#    print(line2)
#    f.write(line1)
#    f.write(line2)
#
#
