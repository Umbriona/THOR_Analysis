from Bio import SeqIO
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os, sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')



from tqdm import tqdm
tqdm.pandas()

def load_fasta(file):
    df_wt = {"id":[],"seq_wt":[],"pOGT_wt":[], "id_wt":[]}
    df_var = {"id":[],"seq_var":[],"pOGT_var":[]}
    for idx, rec in enumerate(SeqIO.parse(file, "fasta")):
        wt_id, typ, rep = rec.id.split('_')
        if idx % 2 == 0:
            assert typ == "wt", f"type not is: {typ}"
            df_wt["id"].append(f"{wt_id}_{rep}")
            df_wt["id_wt"].append(f"{wt_id}")
            df_wt["seq_wt"].append(str(rec.seq))
            df_wt["pOGT_wt"].append(float(rec.description.split(' ')[-1]))
        else:
            assert typ == "variant", f"type not is: {typ}"
            df_var["id"].append(f"{wt_id}_{rep}")
            df_var["seq_var"].append(str(rec.seq))
            df_var["pOGT_var"].append(float(rec.description.split(' ')[-1]))
    df_wt = pd.DataFrame(df_wt)
    df_var = pd.DataFrame(df_var)
    df = df_wt.merge(df_var, on = "id", how = "inner")
    return df

def load_features(file):
    with open(file, "rb") as f:
        df = pickle.load(f) 
    return df

def load_PF2DF(files):
    variant = {"id":[], "seq":[], "wt_seq":[]}
    for idx_outer, file in enumerate(files):
        for idx_inner, rec in enumerate(SeqIO.parse(file, "fasta")):
            variant["id"].append(f"{rec.id}")
            variant["wt_seq"].append(rec.id.split('_')[0])
            variant["seq"].append(str(rec.seq))    
    variant = pd.DataFrame(variant)
    return variant



#@jit
def id2wt(q_seq, wt_seq):
    count = 0
    for res_q, res_wt in zip(q_seq, wt_seq):
        if res_q == res_wt:
            count +=1
    return count/len(q_seq)

def get_og(_id):
    if _id in hmap_id_og:
        return hmap_id_og[_id]
    else:
        return "Null"


