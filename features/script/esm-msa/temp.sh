#take long time

#generate hhblits result
python get_esm-msa_step1_hhblits.py

#generate esm-msa embedding
python get_esm-msa_step2_run_model.py

#concate esm-msa embbeding
python get_esm-msa_step3_concate.py



