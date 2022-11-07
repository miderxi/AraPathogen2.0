import tensorflow as tf
from tensorflow import keras
import sys
import os
import numpy as np

from zzd.utils.assess import multi_scores
import pickle
from sklearn.model_selection import train_test_split
from RF_features import Features_C123 as Features


def dense_model(input_shape):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(keras.layers.Dense(256, activation="relu", name="layer1"))
    model.add(keras.layers.Dense(128, activation="relu", name="layer2"))
    model.add(keras.layers.Dense(64, activation="relu", name="layer3"))
    #model.add(layers.Dense(64, activation="relu", name="layer4"))
    model.add(keras.layers.Dense(32, activation="relu", name="layer5"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1,  activation="sigmoid", name="layer6"))
    model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'])
    return model

def train(model, x, y):
    callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callback2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0,
            mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    model.fit(x,y,
            epochs=30,batch_size=16,validation_split=0.1,
            callbacks=[callback1,callback2],
            shuffle=True,
            workers=15)
    return model



#1. parameters
print(sys.argv)
info_list = sys.argv[1:] if len(sys.argv)>1 else None
output_dir = f"../output/preds/10folds_C1223_NN_"+"_".join(info_list)
os.makedirs(output_dir,exist_ok=True)

#2. load data
#2.1 load C1 C2 and C3 file
c1_files = [f'../data/10folds_C1223/C1_fold{i}.txt' for i in range(10)] 
c2_host_unseen_files = [f'../data/10folds_C1223/C2h_fold{i}.txt' for i in range(10)] 
c2_pathogen_unseen_files = [f"../data/10folds_C1223/C2p_fold{i}.txt" for i in range(10)]
c3_files = [f'../data/10folds_C1223/C3_fold{i}.txt' for i in range(10)] 

#2.2 load features
features = Features(info=info_list)

c1_scores = []
c2_host_unseen_scores = []
c2_pathogen_unseen_scores = []
c3_scores = []
for foldn in range(10):
    
    print(f"fold{foldn}: load file => ",end="");sys.stdout.flush();
    c1 = np.genfromtxt(c1_files[foldn],str)
    c2_host_unseen = np.genfromtxt(c2_host_unseen_files[foldn],str)
    c2_pathogen_unseen = np.genfromtxt(c2_pathogen_unseen_files[foldn],str)
    c3 = np.genfromtxt(c3_files[foldn],str)

    print("encode file =>",end="");sys.stdout.flush();
    X_c1, y_c1 = c1[:,:2], c1[:,2].astype(np.float32)
    X_c2_host_unseen,y_c2_host_unseen =  c2_host_unseen[:,:2], c2_host_unseen[:,2].astype(np.float32)
    X_c2_pathogen_unseen,y_c2_pathogen_unseen =  c2_pathogen_unseen[:,:2], c2_pathogen_unseen[:,2].astype(np.float32)
    X_c3, y_c3 = c3[:,:2], c3[:,2].astype(np.float32)

    X_c1_train, X_c1_test, y_c1_train,y_c1_test = train_test_split(X_c1, y_c1, train_size=0.9, random_state=0,shuffle=True)
    
    x_c1_train = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c1_train ]) 
    x_c1_test = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c1_test ]) 
    x_c2_host_unseen = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c2_host_unseen ]) 
    x_c2_pathogen_unseen = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c2_pathogen_unseen ]) 
    x_c3 = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_c3 ])
    
    
    print("training==>",end="");sys.stdout.flush();
    model = dense_model(x_c1_train.shape[1])
    
    callback1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=False)
    callback2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0,
            mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    model.fit(x_c1_train,y_c1_train,
            epochs=30,batch_size=16,validation_data=[x_c1_test,y_c1_test],
            callbacks=[callback1,callback2],
            verbose=True,
            shuffle=True,
            workers=15)

    print("predicting..")
    y_c1_test_pred = model.predict(x_c1_test)
    y_c2_host_unseen_pred = model.predict(x_c2_host_unseen)
    y_c2_pathogen_unseen_pred = model.predict(x_c2_pathogen_unseen)
    y_c3_pred = model.predict(x_c3)

    c1_score = multi_scores(y_c1_test, y_c1_test_pred, show=True,threshold=0.09)
    c2_host_unseen_score = multi_scores(y_c2_host_unseen, y_c2_host_unseen_pred, show=True,show_index=False,threshold=0.09)
    c2_pathogen_unseen_score = multi_scores(y_c2_pathogen_unseen, y_c2_pathogen_unseen_pred, show=True,show_index=False,threshold=0.09)
    c3_score = multi_scores(y_c3, y_c3_pred, show=True,show_index=False,threshold=0.09)
    
    c1_scores.append(c1_score)
    c2_host_unseen_scores.append(c2_host_unseen_score)
    c2_pathogen_unseen_scores.append(c2_pathogen_unseen_score)
    c3_scores.append(c3_score)

    #save pred result
    with open(f"{output_dir}/c1_pred_{foldn}.txt","w") as f:
        for line in np.hstack([X_c1_test,y_c1_test.reshape(-1,1),y_c1_test_pred.reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c2_host_unseen_pred_{foldn}.txt","w") as f:
        for line in np.hstack([X_c2_host_unseen, y_c2_host_unseen.reshape(-1,1), y_c2_host_unseen_pred.reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c2_pathogen_unseen_pred_{foldn}.txt","w") as f:
        for line in np.hstack([X_c2_pathogen_unseen, y_c2_pathogen_unseen.reshape(-1,1), y_c2_pathogen_unseen_pred.reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c3_pred_{foldn}.txt","w") as f:
        for line in np.hstack([X_c3, y_c3.reshape(-1,1), y_c3_pred.reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    #save pred score
    with open(f"{output_dir}/c1_score_{foldn}.txt","w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n")
            f.write("\t".join([str(i) for i in c1_score]))

    with open(f"{output_dir}/c2_score_{foldn}.txt","w") as f:
            f.write(f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n")
            f.write("\t".join([str(i) for i in c2_host_unseen_score]))

    with open(f"{output_dir}/c2_score_{foldn}.txt","w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n")
            f.write("\t".join([str(i) for i in c2_pathogen_unseen_score]))

    with open(f"{output_dir}/c3_score_{foldn}.txt","w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n")
            f.write("\t".join([str(i) for i in c3_score]))

    #save model
    #if len(info_list) == 1:
    #    model_file_name = "../output/model_state/C1223_RF_" + "_".join(info_list)+f"_foldn_{foldn}.pkl"
    #    with open(model_file_name,"wb") as f:
    #        pickle.dump(model,f)

print("10 fold C1223 average")
c1_scores = np.array(c1_scores)
fmat =  [1, 1,  1,  1,  3,  3,  3,  3,  3,  3,  3,      3,      3]
with open(f"{output_dir}/c1_average_score.txt",'w') as f:
    line1 = f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c1_scores.mean(0),c1_scores.std(0))])
    print(line1,end="")
    print('\t'.join([f'{a:.{_}f}' for (_,a) in zip(fmat,c1_scores.mean(0))]))
    f.write(line1)
    f.write(line2)

c2_host_unseen_scores = np.array(c2_host_unseen_scores)
with open(f"{output_dir}/c2_average_score.txt",'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c2_host_unseen_scores.mean(0),c2_host_unseen_scores.std(0)) ])
    #print(line1)
    print('\t'.join([f'{a:.{_}f}' for (_,a) in zip(fmat,c2_host_unseen_scores.mean(0))]))
    f.write(line1)
    f.write(line2)

c2_pathogen_unseen_scores = np.array(c2_pathogen_unseen_scores)
with open(f"{output_dir}/c2_average_score.txt",'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c2_pathogen_unseen_scores.mean(0),c2_pathogen_unseen_scores.std(0)) ])
    #print(line1)
    print('\t'.join([f'{a:.{_}f}' for (_,a) in zip(fmat,c2_pathogen_unseen_scores.mean(0))]))
    f.write(line1)
    f.write(line2)

c3_scores = np.array(c3_scores)
with open(f"{output_dir}/c3_average_score.txt",'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c3_scores.mean(0),c3_scores.std(0))])
    #print(line1)
    print('\t'.join([f'{a:.{_}f}' for (_,a) in zip(fmat,c3_scores.mean(0))]))
    f.write(line1)
    f.write(line2)









