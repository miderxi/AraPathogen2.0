from Bio import SeqIO
import numpy as np
import word2vec
import pickle

#4.extract feature
#(1).prepare candicate id for extracte
ids = [i.id  for i in  SeqIO.parse("../../../data/sequences/ara_and_eff.fasta","fasta")]
ids_set = set(ids)

#(2).read number to id file
ids2num = dict()
for prot_idx, doc2vec_idx in np.genfromtxt("./swis_id2num.txt",str):  #left is protein id,right is index of doc2vec model embeding vector.
    if prot_idx not in ids2num.keys() and prot_idx in ids_set:
        ids2num[prot_idx] = doc2vec_idx

#(3) load model
model = word2vec.load('./swis_doc2vec-vectors2.bin')

#(4) extract embeding vector
total_features = {}
for prot_idx,doc2vec_index in ids2num.items():
    total_features[prot_idx] = model[f"_*{doc2vec_index}"]

#(5) save to disk.
with open("../../doc2vec/ara_and_eff_2kmer_doc2vec.pkl","wb") as f:
    pickle.dump(total_features,f)


