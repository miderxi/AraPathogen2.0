"""
"""
import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from zzd.utils.assess import multi_scores
import pickle
from sklearn.model_selection import train_test_split
from RF_features import Features_C123 as Features
np.set_printoptions(suppress=True)

import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
from zzd.utils.assess import mean_accuray

#1. Define Moldel Function
class cnn_block(nn.Module):
    def __init__(self,input_shape=20):
        super(cnn_block,self).__init__()
        self.norm0 = nn.BatchNorm1d(20)
        
        self.conv1 = nn.Conv1d(input_shape, 32, kernel_size=3, stride=1,padding=1)
        self.norm1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4,padding=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1,padding=1)
        self.norm2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4,padding=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1,padding=1)
        self.norm3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4,padding=2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1,padding=1)
        self.norm4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(4,padding=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*9,64)
        #self.fc2 = nn.Linear(128,128)

    def forward(self,x):
        #x = self.norm0(x)
        
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        x = self.pool1(x)
            
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.norm3(x)
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.norm4(x)
        x = self.pool4(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return x

class siamese_cnn(nn.Module):
    def __init__(self,device='cpu'):
        super(siamese_cnn, self).__init__()
        self.cnn_block = cnn_block().to(device)
        self.fc1 = nn.Linear(128,32)
        self.fc2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1,x2):
        x1 = self.cnn_block(x1)
        x2 = self.cnn_block(x2)

        x = torch.cat((x1,x2),1)
        x = F.relu(self.fc1(x))

        #x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

def weights_init(m):
    if isinstance(m, (torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')


#2. Define Data Function
class Features:
    """ 
    input protein feature type, 
    return PSSM array with shape (2000,20) index by protein ID    
    """
    def __init__(self,info:list=['pssm']):
        feature_pssm = np.load("../features/pssm/pssm.pkl",allow_pickle=True)
        self.info = info
        self.features={
            'pssm':feature_pssm,
            }
            
    def __getitem__(self,index:str):
        tmp = np.zeros((2000,20),dtype=np.float32)
        tmp2 = self.features[self.info[0]][index][:2000]
        tmp[:len(tmp2)] = tmp2
        return tmp


class ppis_dataset():
    """ 
    input ppis list, 
    return one PSSM pair by number index
    """
    def __init__(self, ppis, features):
        self.ppis = ppis
        self.features = features

    def __len__(self):
        return self.ppis.__len__()

    def __getitem__(self, index):
        sub_ppis = self.ppis[index].reshape(-1,3)
        x1 = np.stack([self.features[i].T for i in sub_ppis[:,0]]) #shape(n_sample,chanel,length)
        x2 = np.stack([self.features[i].T for i in sub_ppis[:,1]]) #shape(n_sample,chanel,length)
        y = np.array(sub_ppis[:,-1],dtype=np.float32)

        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)
        y = torch.from_numpy(y)
        return x1,x2,y


class ppis_batchset():
    """
    Input ppis dataset object,
    Return batch encoded data 
    """
    def __init__(self,ppis_dataset,batch_size=32):
        self.ppis_dataset = ppis_dataset
        self.batch_size = batch_size
        self.indexes = np.arange(self.ppis_dataset.__len__()//self.batch_size)

    def __len__(self):
        return self.ppis_dataset.__len__()//self.batch_size

    def __getitem__(self,index):
        x1,x2,y = self.ppis_dataset[self.indexes[index]*self.batch_size:(self.indexes[index]+1)*self.batch_size]
        return x1,x2,y

    def shuffle(self):
        np.random.shuffle(self.indexes)



#3. parameters
output_dir = f"../output/preds/10folds_C1223_CNN_pssm"
os.makedirs(output_dir,exist_ok=True)
device='cuda'
BatchSize=32
Epochs=50
train_acc_cut_off=0.999
LeaningRate=0.001
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


#2. load data
#2.1 load C1 C2 and C3 file
c1_files = [f'../data/10folds_C1223/C1_fold{i}.txt' for i in range(10)] 
c2h_files = [f'../data/10folds_C1223/C2h_fold{i}.txt' for i in range(10)] 
c2p_files = [f"../data/10folds_C1223/C2p_fold{i}.txt" for i in range(10)]
c3_files = [f'../data/10folds_C1223/C3_fold{i}.txt' for i in range(10)] 


#2.2 load features
features = Features(info=['pssm'])


c1_scores = []
c2h_scores = []
c2p_scores = []
c3_scores = []

for foldn in range(10):
    np.random.seed(0)
    #(1) prepare train and test file
    print(f"fold{foldn}: load file => ",end="");sys.stdout.flush();
    c1_tb  = np.genfromtxt(c1_files[foldn],str)
    c2h_tb = np.genfromtxt(c2h_files[foldn],str)
    c2p_tb = np.genfromtxt(c2p_files[foldn],str)
    c3_tb = np.genfromtxt(c3_files[foldn],str)

    X_c1, y_c1  = c1_tb[:,:2],  c1_tb[:,2].astype(np.float32)
    X_c2h,y_c2h = c2h_tb[:,:2], c2h_tb[:,2].astype(np.float32)
    X_c2p,y_c2p = c2p_tb[:,:2], c2p_tb[:,2].astype(np.float32)
    X_c3, y_c3  = c3_tb[:,:2],  c3_tb[:,2].astype(np.float32)

    #(2)split c1 to trian and test
    X_c1_train, X_c1_test, y_c1_train,y_c1_test = train_test_split(X_c1, y_c1, train_size=0.9, random_state=0,shuffle=True)
    c1_train_tb =  np.hstack([X_c1_train,y_c1_train.reshape(-1,1)])
    c1_test_tb = np.hstack([X_c1_test,   y_c1_test.reshape(-1,1)])

    #(3) encode file
    print("dataset =>",end="");sys.stdout.flush();
    x_c1_train_data = ppis_dataset(c1_train_tb,features)
    x_c1_test_data = ppis_dataset(c1_test_tb,features)
    x_c2h_data = ppis_dataset(c2h_tb,features)
    x_c2p_data = ppis_dataset(c2p_tb,features)
    x_c3_data = ppis_dataset(c3_tb,features)

    #(4) batch dataset
    x_c1_train_batch = ppis_batchset(x_c1_train_data)
    x_c2_test_batch = ppis_batchset(x_c1_test_data)
    x_c2h_batch = ppis_batchset(x_c2h_data)
    x_c2p_batch = ppis_batchset(x_c2p_data)
    x_c3_batch  = ppis_batchset(x_c3_data)

    #(5) init model
    model = siamese_cnn(device)
    model.to(device)
    model.apply(weights_init)   #kaiminghe parameters init
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

    # trainning
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch_i in range(Epochs):#epochs iterate
        start = time.time()
        epoch_loss = np.empty(0,np.float32)
        epoch_y = np.empty(0,np.float32)
        epoch_y_pred = np.empty(0,np.float32)
        epoch_acc = 0
        
        x_c1_train_batch.shuffle()
        for batch_idx in range(len(x_c1_train_batch)):#batch iterate
            x1,x2,y = x_c1_train_batch[batch_idx]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            #forward
            y_pred = model(x1,x2) #shape(n_sample,1)
            y_pred = y_pred.squeeze(dim=-1)
            loss = criterion(y_pred, y)
            
            #backword
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            #assess by acc and loss
            epoch_y      = np.append(epoch_y,      y.to('cpu').data.numpy())
            epoch_y_pred = np.append(epoch_y_pred, y_pred.to('cpu').detach().data.numpy())
            epoch_acc    = (np.round(epoch_y_pred) == np.round(epoch_y)).mean()
            epoch_loss   = np.append(epoch_loss,loss.item())
            
            print(f"\rfoldn:{foldn} epoch:{epoch_i}  {batch_idx*32} / {len(x_c1_train_data)} {time.time()-start:.1f}s "+
                  f" loss:{epoch_loss.mean():.4f}  acc:{epoch_acc:.4f}",end="")
            sys.stdout.flush()
        print("")
         
        with torch.no_grad():
            x1,x2,y_test =x_c1_test_data[:]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            y_test_pred = model(x1,x2) 
        
        multi_scores(y_test,y_test_pred.to('cpu').detach().data.numpy(),show=True,show_index=False)

        #Ealystop by acc
        if epoch_acc > train_acc_cut_off:
            print("")
            break 
        
    #train end
    #c1,c2h,c2p,c3 assess start
    with torch.no_grad():
        x1,x2,y_test =x_c1_test_data[:]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        c1_test_pred = model(x1,x2) 
        
        tmp = multi_scores(y_test,y_test_pred.to('cpu').detach().data.numpy(), show=True, show_index=True)
        c1_scores.append(tmp)

    with torch.no_grad():
        x1,x2,y_test =x_c2h_data[:]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        c2h_pred = model(x1,x2) 
        
        tmp = multi_scores(y_test,c2h_pred.to('cpu').detach().data.numpy(), show=True, show_index=False)
        c2h_scores.append(tmp)


    with torch.no_grad():
        x1,x2,y_test =x_c2p_data[:]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        c2p_pred = model(x1,x2) 
        
        tmp = multi_scores(y_test,c2p_pred.to('cpu').detach().data.numpy(), show=True, show_index=False)
        c2p_scores.append(tmp)

    with torch.no_grad():
        x1,x2,y_test =x_c3_data[:]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        c3_pred = model(x1,x2) 
        
        tmp = multi_scores(y_test,c3_pred.to('cpu').detach().data.numpy(), show=True, show_index=False)
        c3_scores.append(tmp)

    #save c1-test,c2h,c2p and c3 pred resutl
    with open(f"{output_dir}/c1_pred_{foldn}.txt","w") as f:
        for line in np.hstack([c1_test_tb, c1_test_pred.to('cpu').detach().data.numpy().reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c2_host_unseen_pred_{foldn}.txt","w") as f:
        for line in np.hstack([c2h_tb, c2h_pred.to('cpu').detach().data.numpy().reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)
    

    with open(f"{output_dir}/c2_pathogen_unseen_pred_{foldn}.txt","w") as f:
        for line in np.hstack([c2p_tb, c2p_pred.to('cpu').detach().data.numpy().reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/c3_pred_{foldn}.txt","w") as f:
        for line in np.hstack([c3_tb, c3_pred.to('cpu').detach().data.numpy().reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    #save model
    torch.save(model.state_dict(), f'../output/model_state/cnn_pssm_foldn_{foldn}.state_dict')
    
    #save pred score
    with open(f"{output_dir}/c1_score_{foldn}.txt","w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n")
            f.write("\t".join([str(i) for i in c1_scores[-1]]))

    with open(f"{output_dir}/c2h_score_{foldn}.txt","w") as f:
            f.write(f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n")
            f.write("\t".join([str(i) for i in c2h_scores[-1]]))

    with open(f"{output_dir}/c2p_score_{foldn}.txt","w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n")
            f.write("\t".join([str(i) for i in c2p_scores[-1]]))

    with open(f"{output_dir}/c3_score_{foldn}.txt","w") as f:
            f.write("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n")
            f.write("\t".join([str(i) for i in c3_scores[-1]]))

    #save model
    model_file_name = f"../output/model_state/C1223_CNN_pssm_foldn{foldn}.pkl"
    with open(model_file_name,"wb") as f:
        pickle.dump(model,f)


print("5 fold C1223 average")
c1_scores = np.array(c1_scores)
fmat =  [1, 1,  1,  1,  3,  3,  3,  3,  3,  3,  3,      3,      3]
with open(f"{output_dir}/c1_average_score.txt",'w') as f:
    line1 = f"TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c1_scores.mean(0),c1_scores.std(0))])
    print(line1,end="")
    print('\t'.join([f'{a:.{_}f}' for (_,a) in zip(fmat,c1_scores.mean(0))]))
    f.write(line1)
    f.write(line2)

c2_host_unseen_scores = np.array(c2h_scores)
with open(f"{output_dir}/c2_average_score.txt",'w') as f:
    line1 = "TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC\tAP\n"
    line2 = '\t'.join([f'{a:.{_}f}±{b:.{_}f}' for (_,a,b) in zip(fmat,c2_host_unseen_scores.mean(0),c2_host_unseen_scores.std(0)) ])
    #print(line1)
    print('\t'.join([f'{a:.{_}f}' for (_,a) in zip(fmat,c2_host_unseen_scores.mean(0))]))
    f.write(line1)
    f.write(line2)

c2_pathogen_unseen_scores = np.array(c2p_scores)
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


