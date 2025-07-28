from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import time
from scipy import sparse
import warnings
from scipy.stats import hypergeom
import statsmodels.stats.multitest as smm
from joblib import Parallel, delayed
from itertools import combinations
import random
import argparse
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def positive_int(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"Threads must be ≥1 (got {value})")
    return ivalue

def read_tsv_file(file_path):
    df = pd.read_csv(file_path, sep='\t', dtype={'ID': str})
    if 'ID' not in df.columns:
        raise ValueError("TSV file must contain an 'ID' column")
    dup_ids = df['ID'][df['ID'].duplicated()]
    if not dup_ids.empty:
        unique_dups = dup_ids.unique()
        print(f"Warning: {len(unique_dups)} duplicate ID(s) found: {', '.join(unique_dups)}")
    return df


def _extract_canonical_blocks(arr: np.ndarray, cb: int):
    n = arr.size
    if n < cb:
        return set()
    win = np.lib.stride_tricks.sliding_window_view(arr, window_shape=cb)
    return {min(tuple(w), tuple(w[::-1])) for w in win}


def find_conserved_blocks(df, cb: int, min_samples: int, n_jobs: int = 1):
    """
    df: DataFrame with 'ID' and 'Block_Content'
    cb: window size
    min_samples: threshold for a block to be 'conserved'
    n_jobs: 1 for serial, -1 or >1 to parallelize extraction
    """
    ids = df['ID'].tolist()
    contents = df['block_Content'].tolist()
    arrays = [np.fromstring(s, dtype=int, sep=',') for s in contents]

    # Extract all blocks and count them
    counter = Counter()

    if n_jobs == 1:
        sample_block_sets = []
        for arr in tqdm(arrays, desc="Extracting blocks of size {cb}."):
            blks = _extract_canonical_blocks(arr, cb)
            sample_block_sets.append(blks)
            counter.update(blks)
    else:
        sample_block_sets = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_extract_canonical_blocks)(arr, cb) for arr in arrays
        )
        for blks in sample_block_sets:
            counter.update(blks)

    # Build the conserved‐block to column index map
    block_to_idx = {}
    for blk, ct in counter.items():
        if ct >= min_samples:
            block_to_idx[blk] = len(block_to_idx)

    # Build the presence matrix
    row_idx, col_idx = [], []
    final_sample_sets = {}

    for i, sid in enumerate(tqdm(ids, desc="Building matrix")):
        blks = sample_block_sets[i]
        conserved_blks = {b for b in blks if b in block_to_idx}
        final_sample_sets[sid] = conserved_blks
        for b in conserved_blks:
            j = block_to_idx[b]
            row_idx.append(i)
            col_idx.append(j)

    n_rows = len(ids)
    n_cols = len(block_to_idx)
    if col_idx:
        max_j = max(col_idx)
        if max_j >= n_cols:
            warnings.warn(
                f"max column index {max_j} >= allocated columns {n_cols}; "
                f"expanding to {max_j + 1}"
            )
            n_cols = max_j + 1

    presence = sparse.coo_matrix(
        (np.ones(len(row_idx), int), (row_idx, col_idx)),
        shape=(n_rows, n_cols),
        dtype=int
    ).tocsr()

    conserved_ids = [ids[i] for i in sorted(set(row_idx))]

    return {
        'all_samples': ids,
        'sample_sets': final_sample_sets,
        'block_counts': counter,
        'block_to_idx': block_to_idx,
        'conserved_ids': conserved_ids,
        'presence_matrix': presence,
        'total_windows': sum(len(s) for s in sample_block_sets),
        'unique_blocks': len(counter),
        'conserved_blocks': len(block_to_idx),
        'samples_with_conserved': len(conserved_ids),
    }


def build_bipartite_graph(sample_sets: dict, block_to_idx: dict) -> nx.Graph:
    G = nx.Graph()

    # Genome nodes
    for sid in sorted(sample_sets):
        G.add_node(sid, type='genome', bipartite=0)

    # Cluster block nodes
    for blk in sorted(block_to_idx):
        name = "_".join(map(str, blk))
        G.add_node(name, type='cluster_block', bipartite=1)
        # record the block itself, if you still want that
        G.nodes[name]['clusters'] = ",".join(map(str, blk))

    # edges
    for sid, blks in sample_sets.items():
        for blk in sorted(blks):
            if blk in block_to_idx:
                G.add_edge(sid, "_".join(map(str, blk)))

    return G


def detect_syntenome(G: nx.Graph,
                     sample_sets: dict) -> dict:
    """
    sample_sets: { sample_id: set(block1, block2, …) }
    G: bipartite graph

    1) Build sample–sample weighted projection by
       iterating over each block’s sample list.
    2) Run Louvain on that smaller graph.
    3) Write the 'syntenome' attribute back onto G’s sample nodes.
    """
    # Invert sample_sets → block → [samples]
    block_to_samples = defaultdict(list)
    for sid, blks in sample_sets.items():
        for blk in blks:
            block_to_samples[blk].append(sid)

    # Accumulate pairwise co-occurrence counts
    weight_dict = defaultdict(int)
    for blk, samples in tqdm(block_to_samples.items(), desc="Projecting samples"):
        for u, v in combinations(samples, 2):
            weight_dict[(u, v)] += 1

    # Build the sample–sample graph
    S = nx.Graph()
    S.add_nodes_from(sample_sets.keys())
    for (u, v), w in weight_dict.items():
        S.add_edge(u, v, weight=w)

    # Louvain
    try:
        import community.community_louvain as community_louvain
    except ImportError:
        raise ImportError("Install python-louvain (`pip install python-louvain`) to detect syntenome.")
    partition = community_louvain.best_partition(S, weight='weight', random_state=SEED)

    nx.set_node_attributes(G, partition, 'syntenome')
    return partition


def build_block_jaccard_graph(G_full: nx.Graph,
                              sample_sets: dict,
                              threshold: float = 0.0) -> nx.Graph:
    """
    From the full bipartite graph G_full and its sample→blocks map,
    build a block–block graph where edge weights = Jaccard index,
    and only edges with jaccard >= threshold are kept.
    """

    # 1) Invert to block set(samples)
    block_to_samps = defaultdict(set)
    for samp, blks in sample_sets.items():
        for blk in blks:
            block_to_samps[blk].add(samp)

    # 2) Pre‑compute the string names and copy over the block nodes
    blk2name = {
        blk: "_".join(map(str, blk))
        for blk in block_to_samps
    }
    Gj = nx.Graph()
    for node, data in G_full.nodes(data=True):
        if data.get('type') == 'cluster_block':
            Gj.add_node(node, **data)

    # 3) Compute pairwise Jaccard and add edges
    for b1, b2 in combinations(block_to_samps, 2):
        s1, s2 = block_to_samps[b1], block_to_samps[b2]
        union = s1 | s2
        if not union:
            continue
        jci = len(s1 & s2) / len(union)
        if jci >= threshold:
            Gj.add_edge(blk2name[b1], blk2name[b2], jaccard=jci)

    return Gj


def enrich_block_in_syntenome(G: nx.Graph,
                              partition: dict,
                              sig_figs: int = 3) -> pd.DataFrame:
    """
    For each syntenome and each block node in G, perform a one-tailed
    hypergeometric test (enrichment/depletion), then FDR-correct.
    """
    # collect all genome samples present in partition
    all_samples = [n for n, d in G.nodes(data=True) if d.get('type') == 'genome']
    missing = set(all_samples) - set(partition)
    if missing:
        warnings.warn(f"Skipping {len(missing)} genomes missing in partition: {sorted(missing)}")
    samples = [s for s in all_samples if s in partition]
    M = len(samples)

    data = []
    # for each community
    for com in sorted(set(partition[s] for s in samples)):
        comm_samps = [s for s in samples if partition[s] == com]
        N = len(comm_samps)

        # test each block
        for blk, node_data in G.nodes(data=True):
            if node_data.get('type') != 'cluster_block':
                continue
            # counts in pop & in community
            K = sum(1 for s in samples if G.has_edge(s, blk))
            k = sum(1 for s in comm_samps if G.has_edge(s, blk))

            # choose tail
            if k / N > K / M:
                p = hypergeom.sf(k - 1, M, K, N)
                direction = 'enriched'
            elif k / N < K / M:
                p = hypergeom.cdf(k, M, K, N)
                direction = 'depleted'
            else:
                p = 1.0
                direction = 'neutral'

            data.append({
                'syntenome': com,
                'block': blk,
                'n_in_syn': k,
                'n_in_pop': K,
                'syn_size': N,
                'pop_size': M,
                'direction': direction,
                'p_value': p
            })

    df = pd.DataFrame(data)
    # FDR correction
    rej, qvals, _, _ = smm.multipletests(df['p_value'], alpha=0.05, method='fdr_bh')
    df['q_value'] = qvals
    df['significant'] = rej

    # round to sig figs
    fmt = lambda x: float(f"{x:.{sig_figs}g}")
    df['p_value'] = df['p_value'].map(fmt)
    df['q_value'] = df['q_value'].map(fmt)

    return df

def write_jaccard_tsv(G: nx.Graph, tsv_path: str):
    """
    Write every undirected edge in G as three-column TSV:
    block1<TAB>block2<TAB>jaccard_score
    """
    with open(tsv_path, 'w') as out:
        out.write("block1\tblock2\tjaccard_score\n")
        for u, v, data in G.edges(data=True):
            # ensure a consistent ordering
            b1, b2 = sorted([u, v])
            j = data.get('jaccard', 0.0)
            out.write(f"{b1}\t{b2}\t{j:.3f}\n")


def write_graphml(G: nx.Graph, output_file: str):
    nx.write_graphml(G, output_file)

def write_network_csv(G: nx.Graph, path: str):
    df = nx.to_pandas_edgelist(G)
    # ensure the right column names
    df = df.rename(columns={'source':'source','target':'target'})
    df.to_csv(path, index=False)

def write_metadata_csv(G: nx.Graph, path: str, node_type: str = None):
    records = []
    for n, data in G.nodes(data=True):
        if node_type and data.get('type') != node_type:
            continue
        rec = {'id': n}
        rec.update(data)
        records.append(rec)
    pd.DataFrame.from_records(records).to_csv(path, index=False)


def main(input_file, output: str, cluster_block: int, min_samples: int, jaccard: int, threads: int, format: str):
    start_time = time.time()

    # ensure output subdirs
    net_dir = os.path.join(output, 'networks')
    ana_dir = os.path.join(output, 'analysis_data')
    os.makedirs(net_dir, exist_ok=True)
    os.makedirs(ana_dir, exist_ok=True)

    # Load cluster blocks
    print("Reading cluster block input…")
    df = read_tsv_file(input_file)

    # Conserved block finder (computationally intesive if set to 1 with large data set)
    results = find_conserved_blocks(df, cluster_block, min_samples, n_jobs=threads)

    # Block filter (removes genomes with no conserved blocks)
    orig = len(results['sample_sets'])
    results['sample_sets'] = {
        sid: blks
        for sid, blks in results['sample_sets'].items()
        if blks
    }
    dropped = orig - len(results['sample_sets'])
    if dropped:
        print(f"Dropping {dropped} isolated genome(s) with no conserved blocks")

    results['all_samples'] = list(results['sample_sets'])

    # General stats to std out
    print(f"Total genome counts: {len(df)}")
    print(f"Total cluster block windows extracted:                 {results['total_windows']}")
    print(f"Total unique cluster blocks found:             {results['unique_blocks']}")
    print(f"Blocks ≥ {min_samples}:                    {results['conserved_blocks']}")
    print(f"Genomes with ≥1 conserved block:         {results['samples_with_conserved']}")

    # Genome to cluster block
    G = build_bipartite_graph(
        sample_sets=results['sample_sets'],
        block_to_idx=results['block_to_idx'],
    )

    # Genome to genome (pairwise block similarity)
    partition = detect_syntenome(
        G,
        sample_sets=results['sample_sets']
    )

    # Cluster block to cluster block
    J = build_block_jaccard_graph(
        G_full=G,
        sample_sets=results['sample_sets'],
        threshold=jaccard
    )

    # Enrichment
    block_enrich = enrich_block_in_syntenome(G, partition)

    enriched_map = (
        block_enrich
        .query("significant & direction=='enriched'")
        .groupby('syntenome')['block']
        .apply(set)
        .to_dict()
    )

    # Outputs
    if format == 'graphml':
        nx.write_graphml(G, os.path.join(net_dir, 'genome_to_cb.graphml'))
        nx.write_graphml(J, os.path.join(net_dir, 'cluster_block_jaccard.graphml'))
    else:  # CSV
        write_network_csv(G, os.path.join(net_dir, 'genome_to_cb_network.csv'))
        write_metadata_csv(G, os.path.join(net_dir, 'genome_to_cb_metadata.csv'), node_type='genome')
        write_network_csv(J, os.path.join(net_dir, 'cluster_block_jaccard_network.csv'))
        write_metadata_csv(J, os.path.join(net_dir, 'cluster_block_jaccard_metadata.csv'), node_type='cluster_block')

    write_jaccard_tsv(J, os.path.join(ana_dir, "block_jaccard_scores.tsv"))
    block_enrich.to_csv(os.path.join(ana_dir, 'block_enrichment_scores.tsv'), sep='\t', index=False)

    # genome enriched blocks
    out_df = pd.DataFrame([
        {
            'ID': sample,
            'enriched_Cluster_Blocks': f"({', '.join(enriched)})" if (enriched := [
                nbr for nbr in G.neighbors(sample)
                if G.nodes[nbr].get('type') == 'cluster_block' and nbr in enriched_map.get(partition[sample], set())
            ]) else 'None',
            'Syntenome': partition[sample]
        }
        for sample in sorted(partition)
    ])
    out_df.to_csv(os.path.join(ana_dir, 'genome_enriched_blocks.tsv'), sep='\t', index=False)

    print(f"Completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze genome synteny using cluster block composition.')
    parser.add_argument('-i', '--input', required=True, help='Input TSV file with cluster block data')
    parser.add_argument('-o', '--output', default='job_output', help='Output prefix')
    parser.add_argument('-cb', '--cluster_block', type=int, default=3, help='Cluster block size. Default = 3')
    parser.add_argument('-min', '--min_genomes', type=int, default=5,
                        help='Minimum genomes for conserved cluster‑block. Default = 5')
    parser.add_argument('-jci', '--jaccard', type=float, default=0.25,
                        help='Jaccard threshold for cluster block co‑occurrence visual (Setting too low could cause very noisy networks). Default = 0.25')
    parser.add_argument(
        '-t', '--threads', type=positive_int, default=1, metavar='N',
        help='Number of worker threads (must be ≥1; default: %(default)s)'
    )
    parser.add_argument('-f','--format', choices=['graphml','csv'], default='csv',
                        help='Output format for networks. Recommend csv for Cosmograph and gramphml for Cytoscape. (default: %(default)s)')

    args = parser.parse_args()
    main(args.input, args.output, args.cluster_block, args.min_genomes, args.jaccard, args.threads, args.format)
