import sys
import networkx as nx
from node2vec import Node2Vec
import pandas as pd
import pickle
import numpy as np
#1. generate ara net features

# load graph
def main(input_file,output_file):
    graph = nx.read_edgelist(input_file, create_using=nx.DiGraph(), nodetype=None)

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=256, walk_length=40, num_walks=400, workers=12)  # Use temp_folder for big graphs

    # Embed nodes
    model = node2vec.fit(window=20, min_count=1, batch_words=8, workers=16)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Save embeddings for later use
    model.wv.save_word2vec_format("./ara_node2vec_feature.txt")

    AraNet_node2vec_dict = {line[0]:np.array(line[1:],np.float32) for line in np.genfromtxt("./ara_node2vec_feature.txt",str,delimiter=" ",skip_header=1)}
    with open(output_file,"wb") as f:
        pickle.dump(AraNet_node2vec_dict,f)

    #2. conver uniprot id to tair id
    #uniprot2AGI = {i:j.split(";")[0].split(".")[0]
    #        for (i,j) in pd.read_table("../../A2_id_conversion/Uniprot2AGI.20180728.txt").to_numpy()}
    #
    #node_feature =  pd.read_table("/tmp/ara_ut_node2vec_feature.txt").to_numpy()
    #
    #
    #loss=[]
    #with open("./ara_node2vec_feature.txt",'w') as f:
    #    for line in node_feature:
    #        line = line[0].split(" ")
    #        if line[0] in uniprot2AGI.keys():
    #            line[0] = uniprot2AGI[line[0]]
    #            f.write("\t".join(line) + "\n")
    #        else:
    #            loss.append(line[0])
    #
    #
    # Save model for later use
    # model.save(EMBEDDING_MODEL_FILENAME)



script_path = sys.path[0]
input_file  = f"{script_path}/arabidopsis_thaliana_ppis.txt"
output_file = f"{script_path}/../../AraNet_node2vec/AraNe_node2vec_256.pkl"

#input_file  = f"{script_path}/arabidopsis_thaliana_ppis.txt"
#output_file = f"{script_path}/../../AraNet_node2vec/AraNet_node2vec_64.pkl"

main(input_file=input_file, output_file=output_file)

#perference
#https://github.com/eliorc/node2vec?msclkid=c933ee24b8a211ec904820bbb5a3efe7
#https://radimrehurek.com/gensim/index.html
