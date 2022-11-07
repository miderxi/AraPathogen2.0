from Bio import SeqIO

#1. prepare sequences
#(1) sequences of uniprot
seqs_list =[(i.id,str(i.seq)) for i in  SeqIO.parse("../../../data/sequences/ara_and_eff.fasta","fasta")]

#[(i.id,str(i.seq)) for i in SeqIO.parse("./swissprot_30-5000_50.fasta","fasta")]
#(2) sequences of know sequence
#(3) sequences of effectors, download from ncbi and uniprot by keyword search
#[seqs_list.append((i.id,str(i.seq))) for i in  SeqIO.parse("./ncbi_uniprot_effector.fasta","fasta")]
#(4) sequences of arabidopsis thaliana, download from ncbi and uniprot by keyword search
#[seqs_list.append((i.id,str(i.seq))) for i in  SeqIO.parse("./ncbi_arabidopsis_thaliana.fasta","fasta")]

#2.build word to train
#(1) create k-mer (2-mer)
with open("./swis.txt","w") as f:
    for idx,(t_id,t_seq) in enumerate(seqs_list):
        seq=[]
        for  i in range(len(t_seq)-1):
            seq.append(t_seq[i:i+2])
        seq = " ".join(seq)
        seq = seq.lower()
        seq = seq.encode("ascii","ignore")
        f.write(f"_*{idx} {seq}\n")


#(2) save index file,index is the number,value is the id.
with open("./swis_id2num.txt","w") as f:
    for idx,(t_id,t_seq) in enumerate(seqs_list):
        f.write(f"{t_id} {idx}\n")

#3.trainning model
from Bio import SeqIO
import numpy as np
import word2vec
import pickle

#(1)trainning
word2vec.doc2vec('./swis.txt', './swis_doc2vec-vectors2.bin', cbow=0, size=100, window=10, negative=5,
                 hs=0, sample='1e-4', threads=12, iter_=20, min_count=1, binary=True, verbose=True)

#4.extract feature
#(1).prepare candicate id for extracte
#ids = [i.id  for i in  SeqIO.parse("../../../data/ppis_total.fasta","fasta")]
#[ids.append(i.id)  for i in  SeqIO.parse("../../../data/ind_2021/ppis_ind.fasta","fasta")]
#
##(2).read number to id file
#ids2num = {k:v for k,v in np.genfromtxt("./swis_id2num.txt",str)}
#
##(3) load model
#model = word2vec.load('./swis_doc2vec-vectors2.bin')
#
##(4) extract embeding vector
#total_features = {}
#for t_id in ids:
#    total_features[t_id] = model[f"_*{ids2num[t_id]}"]
#
##(5) save to disk.
#with open("../../doc2vec/doc2vec_400.pkl","wb") as f:
#    pickle.dump(total_features,f)
#
