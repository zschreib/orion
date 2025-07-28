#!/usr/bin/env python3
import argparse
import subprocess
import shutil
import sys
import logging
from pathlib import Path

# Version
__version__ = '0.80-beta'

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: preprocess FASTA and run ORION into one results folder"
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Path to input FASTA file. Nucleotide genomes/contigs.'
    )
    parser.add_argument(
        '-p', '--prefix', required=True,
        help='Base name for results directory (creates <prefix>_results)'
    )
    parser.add_argument(
        '--min-seq-id', type=float, default=0.3,
        help='MMseqs2 min amino acid sequence identity for clustering. Default = 0.3.'
    )
    parser.add_argument(
        '-cb', '--cluster-block', type=int, default=3,
        help='Cluster block size for ORION. Default = 3'
    )
    parser.add_argument(
        '-min', '--min-genomes', type=int, default=5,
        help='Minimum genomes for conserved cluster‑block. Default = 5'
    )
    parser.add_argument(
        '-jci', '--jaccard', type=float, default=0.25,
        help='Jaccard threshold for cluster block co‑occurrence visual. Default = 0.25.'
    )
    parser.add_argument(
        '-t', '--threads', type=int, default=1,
        help='Number of worker threads for ORION. Default = 1'
    )
    parser.add_argument(
        '-f', '--format', choices=['graphml','csv'], default='csv',
        help='Output format for networks. Recommend csv for Cosmograph or graphml for Cytoscape.'
    )
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {__version__}',
        help='Show program version and exit.'
    )
    args = parser.parse_args()

    # setup results dir and log
    results_dir = Path(f"{args.prefix}_results")
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f"{args.prefix}_job_logs.txt"

    # configure logging
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Log user flags
    logger.info("\nUSER INPUT FLAGS")
    logger.info("#" * 20)
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info("#" * 20 + "\n")

    # create subdirs
    pre_dir = results_dir / 'preprocess_output'
    orion_dir = results_dir / 'orion_output'
    net_dir = orion_dir / 'networks'
    ana_dir = orion_dir / 'analysis_data'
    for d in (pre_dir, net_dir, ana_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocessing
    pre_script = Path(__file__).resolve().parent / 'bin' / 'preprocess_input.py'
    cmd_pre = [sys.executable, str(pre_script),
               '-i', args.input,
               '-o', args.prefix,
               '--min-seq-id', str(args.min_seq_id)]
    logger.info(f"Running preprocessing: {' '.join(cmd_pre)}")
    res_pre = subprocess.run(cmd_pre, capture_output=True, text=True)
    logger.info("\nPREPROCESSING OUTPUT")
    logger.info("#" * 20)
    logger.info(res_pre.stdout)
    logger.info(res_pre.stderr)
    logger.info("#" * 20 + "\n")
    if res_pre.returncode != 0:
        logger.error(f"Error: Preprocessing failed (exit code {res_pre.returncode})")
        sys.exit(res_pre.returncode)

    # move preprocess outputs
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir
    default_pre = project_dir / 'output' / f"{args.prefix}_preprocess_output"
    if default_pre.exists():
        for item in default_pre.iterdir():
            if item.is_file():
                shutil.move(str(item), pre_dir)
        try:
            default_pre.rmdir()
            (project_dir / 'output').rmdir()
        except Exception:
            pass

    # locate cluster file
    cluster_file = pre_dir / f"{args.prefix}_cluster_blocks.tsv"
    if not cluster_file.exists():
        logger.error(f"Error: cluster blocks file not found at {cluster_file}")
        sys.exit(1)

    # Step 2: ORION analysis
    orion_script = script_dir / 'bin' / 'orion.py'
    cmd_orion = [sys.executable, str(orion_script),
                 '-i', str(cluster_file),
                 '-o', str(orion_dir),
                 '-cb', str(args.cluster_block),
                 '-min', str(args.min_genomes),
                 '-jci', str(args.jaccard),
                 '-t', str(args.threads),
                 '-f', args.format]
    logger.info(f"Running ORION: {' '.join(cmd_orion)}")
    res_ori = subprocess.run(cmd_orion, capture_output=True, text=True)
    logger.info("\nORION OUTPUT")
    logger.info("#" * 20)
    logger.info(res_ori.stdout)
    logger.info(res_ori.stderr)
    logger.info("#" * 20 + "\n")
    if res_ori.returncode != 0:
        logger.error(f"Error: ORION failed (exit code {res_ori.returncode})")
        sys.exit(res_ori.returncode)

    logger.info("Pipeline completed successfully.")

if __name__ == '__main__':
    main()
