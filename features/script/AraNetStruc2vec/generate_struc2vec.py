import networkx as nx
from GraphEmbedding.ge.models import Struc2Vec
import pickle

def run1():
    G = nx.read_edgelist('./arabidopsis_thaliana_ppis.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])

    model = Struc2Vec(G,walk_length=5, num_walks=200, workers=12, opt3_num_layers=4,verbose=40)
    model.train(embed_size=256, window_size = 10, iter = 50, workers=12)
    embeddings = model.get_embeddings()# get embedding vectors

    with open("../../AraNetStruc2vec/AraNetStruc2vec_256.pkl","wb") as f:
        pickle.dump(embeddings,f)



run1()
