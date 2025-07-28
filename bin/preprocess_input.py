from pathlib import Path
import argparse
import subprocess
import tempfile
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from tempfile import NamedTemporaryFile

def run_prodigal(input_fasta: Path, peptide_fasta: Path) -> None:
    """
    Sanitize headers and run Prodigal to predict ORFs and output peptides.
    Replaces underscores in contig IDs with pipes to preserve full ID.
    """
    with NamedTemporaryFile(mode='w+', delete=False, suffix=".fasta") as tmp_fasta:
        for rec in SeqIO.parse(input_fasta, "fasta"):
            rec.id = rec.id.replace('_', '|')
            rec.description = ''
            SeqIO.write(rec, tmp_fasta, 'fasta')
        tmp_fasta_path = tmp_fasta.name

    cmd = ['prodigal', '-i', tmp_fasta_path, '-a', str(peptide_fasta), '-q']
    print(f"Running Prodigal: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_prodigal(peptide_fasta: Path, metadata_tsv: Path) -> None:
    records = []
    metadata = []

    for rec in SeqIO.parse(str(peptide_fasta), 'fasta'):
        desc_parts = rec.description.split(' # ')
        if len(desc_parts) < 4:
            continue
        original_id, start, end, strand_frame = desc_parts[0].replace('|', '_'), desc_parts[1], desc_parts[2], desc_parts[3]

        try:
            sf = int(strand_frame)
        except ValueError:
            continue

        orf_start, orf_end = (end, start) if sf < 0 else (start, end)

        if '_' in original_id:
            contig_id, gene_idx = original_id.rsplit('_', 1)
        else:
            continue

        clean_id = f"{contig_id}_{orf_start}_{orf_end}_{gene_idx}"
        seq_str = str(rec.seq).replace('*', '')
        rec.seq = Seq(seq_str)
        rec.id = clean_id
        rec.description = ''
        records.append(rec)
        metadata.append((contig_id, clean_id, gene_idx))

    SeqIO.write(records, str(peptide_fasta), 'fasta')

    with open(metadata_tsv, 'w') as tf:
        tf.write('ID\tORF_ID\tORF_Position\n')
        for seq_id, orf_id, pos in metadata:
            tf.write(f"{seq_id}\t{orf_id}\t{pos}\n")


def run_mmseqs_clustering(peptide_fasta: Path, cluster_tsv: Path, min_seq_id: float = 0.3) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        result_prefix = tmp_path / "mmseqs_res"

        cmd = [
            "mmseqs", "easy-cluster",
            str(peptide_fasta),
            str(result_prefix),
            str(tmp_path),
            "--min-seq-id", str(min_seq_id),
            "-c", "0.8",
            "--cov-mode", "0"
        ]
        print(f"Running MMseqs2: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        cluster_file = result_prefix.parent / f"{result_prefix.name}_cluster.tsv"

        cluster_map = {}
        cluster_id = 0
        seen = set()

        with open(cluster_file) as f:
            for line in f:
                rep, mem = line.strip().split('\t')
                if rep not in seen:
                    cluster_id += 1
                    seen.add(rep)
                    cluster_map[rep] = (cluster_id, "rep")
                if mem != rep:
                    cluster_map[mem] = (cluster_id, "mem")

        with open(cluster_tsv, 'w') as out:
            out.write("ORF_ID\tmmseqs_cluster\tmembership\n")
            for orf_id, (cid, role) in cluster_map.items():
                out.write(f"{orf_id}\t{cid}\t{role}\n")

        print(f"Cluster results written to: {cluster_tsv}")


def append_cluster_to_metadata(metadata_path: Path, cluster_path: Path) -> None:
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    cluster_df = pd.read_csv(cluster_path, sep='\t')
    merged = metadata_df.merge(cluster_df, how='left', on='ORF_ID')
    merged['mmseqs_cluster'] = merged['mmseqs_cluster'].fillna('singleton')
    merged['membership'] = merged['membership'].fillna('rep')
    merged.to_csv(metadata_path, sep='\t', index=False)


def block_formation(df: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for block_id, unique_id in enumerate(df['ID'].unique()):
        id_rows = df[df['ID'] == unique_id].sort_values(by='ORF_Position')
        cluster_order = ','.join(map(str, id_rows['mmseqs_cluster'].tolist()))
        blocks.append({'block_ID': block_id, 'ID': unique_id, 'block_Content': cluster_order})
    return pd.DataFrame(blocks)


def main():
    parser = argparse.ArgumentParser(description='Run Prodigal, cluster ORFs, and preprocess metadata.')
    parser.add_argument('-i', '--input', required=True, help='Path to input FASTA file')
    parser.add_argument('-o', '--out-prefix', default='prodigal_output', help='Output file prefix')
    parser.add_argument('--min-seq-id', type=float, default=0.3, help='MMseqs minimum sequence identity')
    args = parser.parse_args()

    input_fasta = Path(args.input)
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    output_dir = project_dir / 'output' / f"{args.out_prefix}_preprocess_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    peptide_fasta = output_dir / f"{args.out_prefix}_orfs.pep"
    metadata_tsv = output_dir / f"{args.out_prefix}_orf_metadata.tsv"
    cluster_tsv = output_dir / f"{args.out_prefix}_mmseqs_cluster_membership.tsv"
    block_tsv = output_dir / f"{args.out_prefix}_cluster_blocks.tsv"

    run_prodigal(input_fasta, peptide_fasta)
    parse_prodigal(peptide_fasta, metadata_tsv)
    run_mmseqs_clustering(peptide_fasta, cluster_tsv, min_seq_id=args.min_seq_id)
    append_cluster_to_metadata(metadata_tsv, cluster_tsv)

    metadata_df = pd.read_csv(metadata_tsv, sep='\t')
    block_df = block_formation(metadata_df)
    block_df.to_csv(block_tsv, sep='\t', index=False)

    print(f"Peptide FASTA: {peptide_fasta}")
    print(f"Metadata TSV: {metadata_tsv}")
    print(f"Cluster Membership TSV: {cluster_tsv}")
    print(f"Cluster Blocks TSV: {block_tsv}")


if __name__ == '__main__':
    main()
