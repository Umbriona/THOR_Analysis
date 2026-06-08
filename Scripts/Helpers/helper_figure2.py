from Bio import SeqIO
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os, sys, re

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict, Counter

from scipy.stats import fisher_exact, binomtest, chisquare
from statsmodels.stats.multitest import multipletests
import umap
import pickle


from tqdm import tqdm
tqdm.pandas()

##############################################################################
####################### COG Temperature shift ################################
##############################################################################

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

##############################################################################
#################### AA frequency and Enrichment #############################
##############################################################################

AAS = 'ACDEFGHIKLMNPQRSTVWY'

def convert_table(seq, w):    
    aas = 'ACDEFGHIKLMNPQRSTVWYX'
    dict_ = {i:aa for i, aa in enumerate(aas)}
    seq_str = "".join([dict_[res] for res in seq[w==1]])
    return seq_str 

def load_fasta_aasub(file):
    df_wt = {"id":[],"seq":[],"pOGT":[]}
    for idx, rec in enumerate(SeqIO.parse(file, "fasta")):
        wt_id = rec.id
        df_wt["id"].append(f"{wt_id}")
        df_wt["seq"].append(str(rec.seq))
        df_wt["pOGT"].append(float(rec.description.split(' ')[-1]))

    df_wt = pd.DataFrame(df_wt)
    return df_wt

def add_aa2dict(seq, df_aa):
    for res in seq:
        if res not in AAS:
            continue
        df_aa["AA_count"][res] += 1
        df_aa["count"]+= 1

##############################################################################
############################## AA Substitutions ##############################
##############################################################################

# ============================================================
# Settings
# ============================================================

AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AAS)
aa2int = {aa: i for i, aa in enumerate(AAS)}
int2aa = {i: aa for aa, i in aa2int.items()}

LOW_LABEL = "Uniprot Enzymes"   # mesophilic, <45 C
HIGH_LABEL = "IMG Enzymes"      # thermophilic, >=60 C


# ============================================================
# Header parsing
# ============================================================

def get_group_from_description_wt(description):
    parts = [x.strip() for x in description.split("|")]
    label = parts[-1]
    if label == LOW_LABEL:
        return "low"
    elif label == HIGH_LABEL:
        return "high"
    return None


def get_organism_from_description_wt(description):
    """
    Tries to recover an organism identifier from the header.

    Preferred format:
        >seq_id | organism_name | Uniprot Enzymes
    Then organism_name is used.

    Fallback:
        use the first field before '|', which is usually the seq id.
    """
    parts = [x.strip() for x in description.split("|")]
    if len(parts) >= 3:
        return parts[-2]
    return parts[0]


# ============================================================
# Consensus collapsing per organism, per alignment
# ============================================================

def consensus_sequence_wt(seqs):
    """
    Most frequent non-gap, standard aa per column.
    If no standard aa is observed at a column, return '-'.
    """
    if len(seqs) == 0:
        return None

    L = len(seqs[0])
    cons = []

    for pos in range(L):
        col = [s[pos] for s in seqs if s[pos] in AA_SET]
        if len(col) == 0:
            cons.append("-")
        else:
            cons.append(Counter(col).most_common(1)[0][0])

    return "".join(cons)


def load_alignment_records_wt(alignment_file, collapse_by_organism=True):
    """
    Returns a list of dicts:
        {
          'group': 'low' or 'high',
          'organism': str,
          'seq': aligned sequence string
        }

    If collapse_by_organism=True, multiple sequences from the same organism
    within the same group/alignment are collapsed to one consensus sequence,
    which is closer to the paper. :contentReference[oaicite:3]{index=3}
    """
    raw = []
    for rec in SeqIO.parse(alignment_file, "fasta"):
        group = get_group_from_description_wt(rec.description)
        if group is None:
            continue

        seq = str(rec.seq)
        organism = get_organism_from_description_wt(rec.description)

        raw.append({
            "group": group,
            "organism": organism,
            "seq": seq
        })

    if len(raw) == 0:
        return []

    L = len(raw[0]["seq"])
    for r in raw:
        if len(r["seq"]) != L:
            raise ValueError(f"Different sequence lengths found in {alignment_file}")

    if not collapse_by_organism:
        return raw

    grouped = defaultdict(list)
    for r in raw:
        grouped[(r["group"], r["organism"])].append(r["seq"])

    collapsed = []
    for (group, organism), seqs in grouped.items():
        collapsed.append({
            "group": group,
            "organism": organism,
            "seq": consensus_sequence_wt(seqs)
        })

    return collapsed


# ============================================================
# Per-position residue association testing
# ============================================================

def test_position_amino_acid_wt(records, pos, aa):
    """
    2x2 Fisher exact test:

                 aa present   aa absent
        low          a           b
        high         c           d

    Returns:
        pval, oddsratio, counts
    """
    a = b = c = d = 0

    for r in records:
        res = r["seq"][pos]
        if res not in AA_SET:
            continue

        present = (res == aa)
        if r["group"] == "low":
            if present:
                a += 1
            else:
                b += 1
        else:
            if present:
                c += 1
            else:
                d += 1

    n_low = a + b
    n_high = c + d

    # not enough usable residues at this position
    if n_low == 0 or n_high == 0:
        return np.nan, np.nan, (a, b, c, d)

    # aa never appears or always appears -> still a valid Fisher table,
    # but if both groups are identical it adds little
    table = np.array([[a, b], [c, d]])
    oddsratio, pval = fisher_exact(table, alternative="two-sided")
    return pval, oddsratio, (a, b, c, d)


def find_temperature_associated_residues_wt(
    records,
    correction="bonferroni",
    alpha=0.01,
    min_group_size=5
):
    """
    Closer to the paper's first stage:
    test every position x amino acid for temperature association. :contentReference[oaicite:4]{index=4}

    Returns:
        sig_low:  dict[pos] -> set(amino acids significantly enriched in low-T)
        sig_high: dict[pos] -> set(amino acids significantly enriched in high-T)
        test_df:  list of test results
    """
    if len(records) == 0:
        return defaultdict(set), defaultdict(set), []

    low_n = sum(r["group"] == "low" for r in records)
    high_n = sum(r["group"] == "high" for r in records)
    if low_n < min_group_size or high_n < min_group_size:
        return defaultdict(set), defaultdict(set), []

    L = len(records[0]["seq"])
    tests = []

    for pos in range(L):
        for aa in AAS:
            pval, oddsratio, counts = test_position_amino_acid_wt(records, pos, aa)
            a, b, c, d = counts

            tests.append({
                "pos": pos,
                "aa": aa,
                "pval": pval,
                "oddsratio": oddsratio,
                "low_present": a,
                "low_absent": b,
                "high_present": c,
                "high_absent": d
            })

    valid_idx = [i for i, t in enumerate(tests) if np.isfinite(t["pval"])]
    if len(valid_idx) == 0:
        return defaultdict(set), defaultdict(set), tests

    raw_pvals = [tests[i]["pval"] for i in valid_idx]

    if correction == "bonferroni":
        reject, p_adj, _, _ = multipletests(raw_pvals, alpha=alpha, method="bonferroni")
    elif correction == "fdr_bh":
        reject, p_adj, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_bh")
    else:
        raise ValueError("correction must be 'bonferroni' or 'fdr_bh'")

    sig_low = defaultdict(set)
    sig_high = defaultdict(set)

    for idx_local, idx_global in enumerate(valid_idx):
        t = tests[idx_global]
        t["p_adj"] = p_adj[idx_local]
        t["reject"] = bool(reject[idx_local])

        if not t["reject"]:
            continue

        low_freq = t["low_present"] / max(t["low_present"] + t["low_absent"], 1)
        high_freq = t["high_present"] / max(t["high_present"] + t["high_absent"], 1)

        if low_freq > high_freq:
            sig_low[t["pos"]].add(t["aa"])
        elif high_freq > low_freq:
            sig_high[t["pos"]].add(t["aa"])

    return sig_low, sig_high, tests


# ============================================================
# Build Fig. 5B-like substitution matrix
# ============================================================

def count_standard_residues_at_pos_wt(records, pos, group):
    counts = np.zeros(len(AAS), dtype=int)
    for r in records:
        if r["group"] != group:
            continue
        aa = r["seq"][pos]
        if aa in aa2int:
            counts[aa2int[aa]] += 1
    return counts


def build_substitution_matrix_from_significant_positions_wt(records, sig_low, sig_high):
    """
    Build a low-T -> high-T substitution matrix, using only positions where
    at least one low-associated residue and one high-associated residue were found.

    Rows    = low-T residue (Uniprot / mesophilic)
    Columns = high-T residue (IMG / thermophilic)

    Conserved / same-residue contributions are excluded, so diagonal stays 0.
    """
    mat = np.zeros((len(AAS), len(AAS)), dtype=float)
    pair_signs = defaultdict(list)

    positions = sorted(set(sig_low.keys()) | set(sig_high.keys()))
    for pos in positions:
        low_aas = sig_low.get(pos, set())
        high_aas = sig_high.get(pos, set())

        if len(low_aas) == 0 or len(high_aas) == 0:
            continue

        low_counts = count_standard_residues_at_pos_wt(records, pos, "low")
        high_counts = count_standard_residues_at_pos_wt(records, pos, "high")

        low_total = low_counts.sum()
        high_total = high_counts.sum()
        if low_total == 0 or high_total == 0:
            continue

        low_freq = low_counts / low_total
        high_freq = high_counts / high_total

        # Only use residues that were found significant in the low/high groups
        pos_mat = np.zeros_like(mat)
        for aa_low in low_aas:
            i = aa2int[aa_low]
            for aa_high in high_aas:
                j = aa2int[aa_high]
                if i == j:
                    continue
                # contribution based on product of group-specific frequencies
                pos_mat[i, j] = low_freq[i] * high_freq[j]

        s = pos_mat.sum()
        if s == 0:
            continue

        # normalize so each informative position contributes equally
        pos_mat = pos_mat / s
        mat += pos_mat

        # directional vote for reciprocal-pair significance
        for a in range(len(AAS)):
            for b in range(a + 1, len(AAS)):
                ab = pos_mat[a, b]
                ba = pos_mat[b, a]
                if ab > ba:
                    pair_signs[(a, b)].append(+1)
                elif ba > ab:
                    pair_signs[(a, b)].append(-1)

    np.fill_diagonal(mat, 0.0)
    return mat, pair_signs


# ============================================================
# Across many alignments
# ============================================================

def analyze_alignment_paper_like_wt(
    alignment_file,
    collapse_by_organism=True,
    correction="bonferroni",
    alpha=0.01,
    min_group_size=5
):
    records = load_alignment_records_wt(alignment_file, collapse_by_organism=collapse_by_organism)
    if len(records) == 0:
        return np.zeros((len(AAS), len(AAS))), defaultdict(list), []

    sig_low, sig_high, tests = find_temperature_associated_residues_wt(
        records,
        correction=correction,
        alpha=alpha,
        min_group_size=min_group_size
    )

    sub_mat, pair_signs = build_substitution_matrix_from_significant_positions_wt(
        records, sig_low, sig_high
    )
    return sub_mat, pair_signs, tests


def analyze_many_alignments_paper_like_wt(
    files,
    collapse_by_organism=True,
    correction="bonferroni",
    alpha=0.01,
    min_group_size=5
):
    total_mat = np.zeros((len(AAS), len(AAS)), dtype=float)
    all_pair_signs = defaultdict(list)
    all_tests = []
    n_used = 0

    for f in tqdm(files):
        try:
            sub_mat, pair_signs, tests = analyze_alignment_paper_like_wt(
                f,
                collapse_by_organism=collapse_by_organism,
                correction=correction,
                alpha=alpha,
                min_group_size=min_group_size
            )
            total_mat += sub_mat
            for k, v in pair_signs.items():
                all_pair_signs[k].extend(v)
            all_tests.extend(tests)
            n_used += 1
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return total_mat, all_pair_signs, all_tests, n_used


# ============================================================
# Reciprocal-direction significance for '+'
# ============================================================

def compute_preferred_direction_plus(pair_signs_all, correction="fdr_bh", alpha=0.05, min_n=10):
    """
    Test whether low->high is preferred over the reciprocal high->low for each pair.

    For one unordered pair (A,G), we compare counts of:
        A->G  versus  G->A

    This mirrors the Fig. 5B style interpretation of reciprocal changes. :contentReference[oaicite:5]{index=5}
    """
    n = len(AAS)
    sig_plus = np.zeros((n, n), dtype=bool)

    pair_list = []
    pvals = []

    for (a, b), signs in pair_signs_all.items():
        n_info = len(signs)
        if n_info < min_n:
            continue

        n_ab = np.sum(np.array(signs) == +1)
        n_ba = np.sum(np.array(signs) == -1)

        pval = binomtest(k=n_ab, n=n_ab + n_ba, p=0.5, alternative="two-sided").pvalue
        pair_list.append((a, b, n_ab, n_ba))
        pvals.append(pval)

    if len(pvals) == 0:
        return sig_plus

    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=correction)

    for keep, (a, b, n_ab, n_ba) in zip(reject, pair_list):
        if not keep:
            continue
        if n_ab > n_ba:
            sig_plus[a, b] = True
        elif n_ba > n_ab:
            sig_plus[b, a] = True

    return sig_plus


# ============================================================
# Normalize + plot
# ============================================================

def matrix_to_relative_frequency(mat):
    total = mat.sum()
    if total == 0:
        return mat.copy()
    return mat / total


def plot_substitution_heatmap_wt(sub_freq, sig_plus=None, title=None):
    muted = sns.color_palette("muted")
    muted_orange = muted[1]   # often the orange in muted

    #cmap = sns.light_palette(muted_orange, as_cmap=True)
    cmap = sns.blend_palette(["#ffffff", muted_orange], as_cmap=True)
    mm = 1 / 25.4
    fig_w = 55 * mm
    fig_h = 55 * mm
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    im = ax.imshow(sub_freq, cmap=cmap, aspect="auto", interpolation="nearest",vmin = 0, vmax= 0.019)

    ax.set_xticks(range(len(AAS)))
    ax.set_yticks(range(len(AAS)))
    ax.set_xticklabels(list(AAS), fontsize= 5.5)
    ax.set_yticklabels(list(AAS), fontsize= 5.5)

    #ax.set_xlabel("Thermophilic / IMG residue")
    #ax.set_ylabel("Mesophilic / Uniprot residue")
   # ax.set_title(title or "Temperature-associated substitution frequencies")

    #cbar = plt.colorbar(im, ax=ax)
    #cbar.set_label("Relative frequency")

    if sig_plus is not None:
        for i in range(len(AAS)):
            for j in range(len(AAS)):
                if sig_plus[i, j]:
                    ax.text(j, i, "+", ha="center", va="center",
                            color="black", fontsize=5.5, fontweight="bold")

    plt.tight_layout()






# ============================================================
# Core counting
# ============================================================

def build_paired_substitution_matrix_thor(
    df,
    wt_col="seq_wt",
    var_col="seq_var",
    position_mask=None,
    ignore_nonstandard=True
):
    """
    Count paired substitutions from wildtype -> variant.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain wt_col and var_col.
    wt_col, var_col : str
        Column names for wildtype and variant sequences.
    position_mask : None or array-like of bool
        If provided, only positions with True are counted.
        Length must match sequence length.
    ignore_nonstandard : bool
        If True, ignore positions where either residue is not one of the 20 standard aa.

    Returns
    -------
    sub_counts : (20, 20) ndarray
        Counts of wt residue -> variant residue.
        Diagonal forced to 0.
    pair_signs : dict[(a,b)] -> list of +1/-1
        For unordered pair (a,b):
          +1 means a->b observed
          -1 means b->a observed
    n_pairs_used : int
        Number of sequence pairs successfully processed.
    total_substitutions : int
        Number of non-conserved substitutions counted.
    """
    sub_counts = np.zeros((len(AAS), len(AAS)), dtype=int)
    pair_signs = defaultdict(list)

    n_pairs_used = 0
    total_substitutions = 0

    for idx, row in df.iterrows():
        wt = row[wt_col]
        var = row[var_col]

        if pd.isna(wt) or pd.isna(var):
            continue

        wt = str(wt)
        var = str(var)

        if len(wt) != len(var):
            print(f"Skipping row {idx}: sequence lengths differ ({len(wt)} vs {len(var)})")
            continue

        L = len(wt)

        if position_mask is not None:
            if len(position_mask) != L:
                raise ValueError(
                    f"position_mask length ({len(position_mask)}) does not match sequence length ({L})"
                )

        n_pairs_used += 1

        for pos, (aa_wt, aa_var) in enumerate(zip(wt, var)):
            if position_mask is not None and not position_mask[pos]:
                continue

            if ignore_nonstandard:
                if aa_wt not in AA_SET or aa_var not in AA_SET:
                    continue

            # Ignore conserved positions so diagonal stays zero
            if aa_wt == aa_var:
                continue

            if aa_wt not in aa2int or aa_var not in aa2int:
                continue

            i = aa2int[aa_wt]
            j = aa2int[aa_var]

            sub_counts[i, j] += 1
            total_substitutions += 1

            a, b = sorted((i, j))
            if i == a and j == b:
                pair_signs[(a, b)].append(+1)  # a->b
            else:
                pair_signs[(a, b)].append(-1)  # b->a

    np.fill_diagonal(sub_counts, 0)
    return sub_counts, pair_signs, n_pairs_used, total_substitutions


def counts_to_relative_frequency_thor(sub_counts):
    total = sub_counts.sum()
    if total == 0:
        return sub_counts.astype(float)
    return sub_counts / total

def compute_preferred_direction_matrix_thor(pair_signs_all, alpha=0.05, min_n=10, correction="fdr_bh"):
    """
    Test whether one direction is preferred over the reverse.

    For each unordered pair (a,b), test:
      count(a->b) vs count(b->a)

    Uses an exact binomial test and multiple-testing correction.

    Returns
    -------
    sig_pref : (20,20) bool ndarray
        sig_pref[i,j] = True means i->j is significantly preferred.
    pval_matrix : (20,20) ndarray
        Raw p-values.
    qval_matrix : (20,20) ndarray
        Adjusted p-values.
    ninfo_matrix : (20,20) ndarray
        Number of informative substitutions for each unordered pair.
    """
    n = len(AAS)
    sig_pref = np.zeros((n, n), dtype=bool)
    pval_matrix = np.full((n, n), np.nan)
    qval_matrix = np.full((n, n), np.nan)
    ninfo_matrix = np.zeros((n, n), dtype=int)

    tests = []
    pair_meta = []

    for (a, b), signs in pair_signs_all.items():
        signs = np.asarray(signs)
        n_info = len(signs)
        ninfo_matrix[a, b] = n_info
        ninfo_matrix[b, a] = n_info

        if n_info < min_n:
            continue

        n_ab = np.sum(signs == +1)
        n_ba = np.sum(signs == -1)

        pval = binomtest(k=n_ab, n=n_ab + n_ba, p=0.5, alternative="two-sided").pvalue

        pval_matrix[a, b] = pval
        pval_matrix[b, a] = pval

        tests.append(pval)
        pair_meta.append((a, b, n_ab, n_ba))

    if len(tests) == 0:
        return sig_pref, pval_matrix, qval_matrix, ninfo_matrix

    reject, qvals, _, _ = multipletests(tests, alpha=alpha, method=correction)

    for keep, qval, (a, b, n_ab, n_ba) in zip(reject, qvals, pair_meta):
        qval_matrix[a, b] = qval
        qval_matrix[b, a] = qval

        if not keep:
            continue

        if n_ab > n_ba:
            sig_pref[a, b] = True
        elif n_ba > n_ab:
            sig_pref[b, a] = True

    return sig_pref, pval_matrix, qval_matrix, ninfo_matrix

def plot_substitution_heatmap_thor(sub_freq, sig_pref=None, title="WT → Variant substitution frequencies"):
    mm = 1 / 25.4
    fig_w = 55 * mm
    fig_h = 55 * mm
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    muted = sns.color_palette("muted")
    muted_orange = muted[1]   # often the orange in muted

    #cmap = sns.light_palette(muted_orange, as_cmap=True)
    cmap = sns.blend_palette(["#ffffff", muted_orange], as_cmap=True)
    
    im = ax.imshow(sub_freq, cmap=cmap, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(AAS)))
    ax.set_yticks(range(len(AAS)))
    ax.set_xticklabels(list(AAS),fontsize= 5.5)
    ax.set_yticklabels(list(AAS),fontsize= 5.5)

   # ax.set_xlabel("Variant residue")
   # ax.set_ylabel("Wildtype residue")
   # ax.set_title(title)

    #cbar = plt.colorbar(im, ax=ax, shrink=0.7)
 #   cbar.set_label("Relative frequency", fontsize=5.5)
    #cbar.ax.tick_params(labelsize=5.5)

    if sig_pref is not None:
        for i in range(len(AAS)):
            for j in range(len(AAS)):
                if sig_pref[i, j]:
                    ax.text(j, i, "+", ha="center", va="center",
                            color="black", fontsize=5.5, fontweight="bold")

    plt.tight_layout()


def substitution_table_thor(sub_counts, sub_freq):
    rows = []
    for i in range(len(AAS)):
        for j in range(len(AAS)):
            if i == j:
                continue
            if sub_counts[i, j] == 0:
                continue
            rows.append({
                "wt_res": int2aa[i],
                "var_res": int2aa[j],
                "substitution": f"{int2aa[i]}->{int2aa[j]}",
                "count": int(sub_counts[i, j]),
                "frequency": float(sub_freq[i, j])
            })

    out = pd.DataFrame(rows).sort_values(["count", "frequency"], ascending=[False, False])
    return out.reset_index(drop=True)

