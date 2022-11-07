from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import itertools
import os
import string
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt
import matplotlib as mpl
from Bio import SeqIO
import biotite.structure as bs
import biotite.structure.io as bsio
from tqdm import tqdm
import pandas as pd

import esm
import time
import pickle
import scipy.sparse as sp

torch.set_grad_enabled(False)
from Bio import SeqIO


#0.funciotns
def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa

    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]


#1.load data and model
#(1) data

#(2) model
msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval().cuda()
msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()


def run_inputs(inputs):
    msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
    msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)   #

    with torch.no_grad():
        #output_contact = msa_transformer.predict_contacts(msa_transformer_batch_tokens)[0].cpu() #make prediction
        out_emb = msa_transformer(msa_transformer_batch_tokens,repr_layers=[12], need_head_weights=True)
        temp = out_emb['representations'][12][0][:,1:,:].cpu().numpy()
    return temp


#2.1compute

white_set1 = {i.id 
                for i in SeqIO.parse("../../../data/sequences/ara_and_eff.fasta","fasta") }
#white_set2 = {_ for _ in np.genfromtxt("./only_need_list.txt",str)}
black_set = {file_name.split(".")[0] for file_name in os.listdir("./esm-msa-emb/")}
files = [file_name 
            for file_name in os.listdir("./hhblits_msa/") 
                if file_name.split(".")[0] in white_set1  and file_name.split(".")[0] not in black_set ]

print(files)

#print(white_set)
print(len(files),len(white_set1))
os.makedirs("./esm-msa-emb/",exist_ok=True)
for idx,file_name in enumerate(files):
    # idx,file_name = 0,files[0]
    protein_id = file_name.split(".")[0]
    input_file = f"./hhblits_msa/{file_name}"
    output_file = f"./esm-msa-emb/{protein_id}.pkl"#scipy sparse matrix

    if not os.path.exists(output_file):
        start = time.time()
        inputs = read_msa(input_file) #read msa
        inputs = greedy_select(inputs, num_seqs=64) #subset of msa
        print(f'{idx}/{len(files)} {file_name}')
        
        #!!! input file 
        origin_seq = [str(i.seq) for i in SeqIO.parse(f"/tmp/esm_msa_seqs/{protein_id}.fasta","fasta")][0]

        if len(inputs[0][1]) <= 1000:
            temp = run_inputs(inputs)   #turn msa to vector 
            
            if len(origin_seq)==len(inputs[0][1]) and len(inputs[0][1])==temp.shape[1]:
                pass
            else:
                print(f"warning:  origin_fasta {len(origin_seq)} != a3m, {len(inputs[0][1])} {temp.shape}")

            temp = temp.mean(0).mean(0)
            with open(output_file,'wb') as f:
                pickle.dump(temp,f)
            print(f'ok  {time.time()-start:.1f}s {len(origin_seq)} {len(inputs[0][1])} {temp.shape}')

        elif len(inputs[0][1]) <= 2000: #split into two part
            inputs1= [[ _[0], _[1][:1000]]      for _ in inputs]
            inputs2= [[ _[0], _[1][1000:2000]]  for _ in inputs]
            temp1 = run_inputs(inputs1)
            temp2 = run_inputs(inputs2)
            temp = np.concatenate((temp1,temp2),axis=1)
            print(temp.shape)

            if len(origin_seq)==len(inputs[0][1]) and len(inputs[0][1])==temp.shape[1]:
                pass
            else:
                print(f"warning:  origin_fasta {len(origin_seq)} != a3m, {len(inputs[0][1])} {temp.shape}")
                
            temp = temp.mean(0).mean(0)
            with open(output_file,'wb') as f:
                pickle.dump(temp,f)
            print(f'ok {time.time()-start:.1f}s {len(origin_seq)} {len(inputs[0][1])} {temp.shape}')

        elif len(inputs[0][1]) < 3000: #split into three part
            inputs1= [[_[0],_[1][   0:1000]]for _ in inputs]
            inputs2= [[_[0],_[1][1000:2000]]for _ in inputs]
            inputs3= [[_[0],_[1][2000:3000]]for _ in inputs]

            temp1 = run_inputs(inputs1)
            temp2 = run_inputs(inputs2)
            temp3 = run_inputs(inputs3)

            temp = np.concatenate((temp1,temp2,temp3),axis=1)
            
            if len(origin_seq)==len(inputs[0][1]) and len(inputs[0][1])==temp.shape[1]:
                pass
            else:
                print(f"warning:  origin_fasta {len(origin_seq)} != a3m, {len(inputs[0][1])} {temp.shape}")
            
            temp = temp.mean(0).mean(0)
            with open(output_file,'wb') as f:
                pickle.dump(temp,f)
            print(f'ok {time.time()-start:.1f}s {len(origin_seq)} {len(inputs[0][1])} {temp.shape}')

        elif len(inputs[0][1]) < 4000: #split into thor part
            inputs1= [[_[0],_[1][   0:1000]]for _ in inputs]
            inputs2= [[_[0],_[1][1000:2000]]for _ in inputs]
            inputs3= [[_[0],_[1][2000:3000]]for _ in inputs]
            inputs4= [[_[0],_[1][3000:4000]]for _ in inputs]

            temp1 = run_inputs(inputs1)
            temp2 = run_inputs(inputs2)
            temp3 = run_inputs(inputs3)
            temp4 = run_inputs(inputs4)

            temp = np.concatenate((temp1,temp2,temp3,temp4),axis=1)
            
            if len(origin_seq)==len(inputs[0][1]) and len(inputs[0][1])==temp.shape[1]:
                pass
            else:
                print(f"warning:  origin_fasta {len(origin_seq)} != a3m, {len(inputs[0][1])} {temp.shape}")
            
            temp = temp.mean(0).mean(0)
            with open(output_file,'wb') as f:
                pickle.dump(temp,f)
            print(f'ok {time.time()-start:.1f}s {len(origin_seq)} {len(inputs[0][1])} {temp.shape}')


        elif len(inputs[0][1]) < 5000: #split into thor part
            inputs1= [[_[0],_[1][   0:1000]]for _ in inputs]
            inputs2= [[_[0],_[1][1000:2000]]for _ in inputs]
            inputs3= [[_[0],_[1][2000:3000]]for _ in inputs]
            inputs4= [[_[0],_[1][3000:4000]]for _ in inputs]
            inputs5= [[_[0],_[1][4000:5000]]for _ in inputs]

            temp1 = run_inputs(inputs1)
            temp2 = run_inputs(inputs2)
            temp3 = run_inputs(inputs3)
            temp4 = run_inputs(inputs4)
            temp5 = run_inputs(inputs5)

            temp = np.concatenate((temp1,temp2,temp3,temp4,temp5),axis=1)
            
            if len(origin_seq)==len(inputs[0][1]) and len(inputs[0][1])==temp.shape[1]:
                pass
            else:
                print(f"warning:  origin_fasta {len(origin_seq)} != a3m, {len(inputs[0][1])} {temp.shape}")
            
            temp = temp.mean(0).mean(0)
            with open(output_file,'wb') as f:
                pickle.dump(temp,f)
            print(f'ok {time.time()-start:.1f}s {len(origin_seq)} {len(inputs[0][1])} {temp.shape}')



        elif len(inputs[0][1]) < 6000: #split into thor part
            inputs1= [[_[0],_[1][   0:1000]]for _ in inputs]
            inputs2= [[_[0],_[1][1000:2000]]for _ in inputs]
            inputs3= [[_[0],_[1][2000:3000]]for _ in inputs]
            inputs4= [[_[0],_[1][3000:4000]]for _ in inputs]
            inputs5= [[_[0],_[1][4000:5000]]for _ in inputs]
            inputs6= [[_[0],_[1][5000:6000]]for _ in inputs]

            temp1 = run_inputs(inputs1)
            temp2 = run_inputs(inputs2)
            temp3 = run_inputs(inputs3)
            temp4 = run_inputs(inputs4)
            temp5 = run_inputs(inputs5)
            temp6 = run_inputs(inputs6)


            temp = np.concatenate((temp1,temp2,temp3,temp4,temp5,temp6),axis=1)
            
            if len(origin_seq)==len(inputs[0][1]) and len(inputs[0][1])==temp.shape[1]:
                pass
            else:
                print(f"warning:  origin_fasta {len(origin_seq)} != a3m, {len(inputs[0][1])} {temp.shape}")
            
            temp = temp.mean(0).mean(0)
            with open(output_file,'wb') as f:
                pickle.dump(temp,f)
            print(f'ok {time.time()-start:.1f}s {len(origin_seq)} {len(inputs[0][1])} {temp.shape}')


