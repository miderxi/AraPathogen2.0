import os
import time
from Bio import SeqIO

#1.create filedir

os.makedirs("/tmp/esm_msa_seqs",exist_ok=True)
os.system("rm -rf /tmp/esm_msa_seqs/*.fasta")
os.makedirs("./hhblits_msa/",exist_ok=True)

#2. read sequeces and write in fasta format
seqs = {i.id:str(i.seq) for i in SeqIO.parse("../../../data/sequences/ara_and_eff_only_need.fasta","fasta")}
seqs = {i.id:str(i.seq) for i in SeqIO.parse("../../../data/sequences/ara_and_eff.fasta","fasta")}

for k,v in seqs.items():
    with open(f"/tmp/esm_msa_seqs/{k}.fasta","w") as f:
        f.write(f">{k}\n")
        f.write(f"{v}\n")

#3.read all fasta format
fasta_files = os.listdir("/tmp/esm_msa_seqs")
fasta_files = [i for i in fasta_files if i[-5:]=="fasta"]

#4. black list
blast_list = set()
for file_name in  os.listdir("./hhblits_msa/"):
    if os.path.getsize(f"./hhblits_msa/{file_name}") > 100000:
        blast_list.add(file_name.split(".a3m")[0])

#for file_name in  os.listdir("./hhblits_msa_train/"):
#    if os.path.getsize(f"./hhblits_msa_train/{file_name}") > 100000:
#        blast_list.add(file_name.split(".a3m")[0])

for idx,base_name in enumerate(fasta_files):
    temp_id = base_name.split(".")[0]
    file_name = f"/tmp/esm_msa_seqs/{base_name}"
    out_file_name = f"./hhblits_msa/{temp_id}.a3m"
    
    if temp_id not in blast_list and not os.path.exists(out_file_name):
        #cmd = f" hhblits -cpu 16 -i {file_name} -d /home/v2/db/uniclust30_2018_08/uniclust30_2018_08 -oa3m {out_file_name} -n 3"
        #cmd = f" hhblits -cpu 16 -i {file_name} -d /home/v2/db/uniclust30_2022_02/UniRef30_2022_02 -oa3m {out_file_name} -n 3"
        cmd = f" hhblits -cpu 16 -i {file_name} -d /home/v2/db/uniclust30_2018_08/uniclust30_2018_08 -oa3m {out_file_name} -n 3"
        
        #a = list(os.popen(f"cat {file_name}|wc -c"))[0]
        #print(f"{idx}/{len(fasta_files)} length:{a}")

        start = time.time()
        print(cmd)
        os.system(cmd)
        #time.sleep(0.1)

        print(f"{idx}/{len(fasta_files)}  {time.time()-start:.1f}s done!")
    else:
        print(f"{idx}/{len(fasta_files)} done!")



