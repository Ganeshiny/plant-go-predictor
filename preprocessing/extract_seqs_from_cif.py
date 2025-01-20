'''
No clustering, use all the sequences

'''
from Bio.Data.IUPACData import protein_letters_3to1 
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import gzip
import os
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import argparse
import networkx as nx
import obonet
import csv 
import numpy as np

exp_evidence_codes = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'CURATED'])
root_terms = set(['GO:0008150', 'GO:0003674', 'GO:0005575'])
domain_terms = 'GO:0008150'

def get_seqs(fname):
    with gzip.open(fname, "rt") as handle:
        parser = MMCIFParser()
        pdb_id = os.path.split(fname)[1].split(".")[0] 
        structure = parser.get_structure(pdb_id, handle)
        chains = {f"{pdb_id}_{chain.id}":seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}
    return chains

def write_seqs_from_cifdir(dirpath, fname):
    structure_dir = Path(dirpath)
    seqs_file = open(fname, "w")
    for file in structure_dir.glob("*"):
        chain_dir = get_seqs(file)
        for key in chain_dir:
            #unknown_percentage = chain_dir[key].count("X")/len(chain_dir[key])
            #print(f"seq:{chain_dir[key]}, percentage:{unknown_percentage}")
            #if unknown_percentage <= 0.2:
            seqs_file.write(f">{key}\n{chain_dir[key]}\n")
    return seqs_file


#write_seqs_from_cifdir("preprocessing/data/structure_files")

'''def load_cluster_file(fname):
    representatives = []
    with open(fname) as cluster_file:
        for line in cluster_file:
            representatives.append(line.strip().split("\t")[0])
    return set(representatives)'''

def read_seqs_file(seqs_file):
    pdb2seq = {}
    with open(seqs_file, "r") as fasta_handle:
        for line in fasta_handle:
            if ">" in line:
                key  = line.strip().replace(">", "")
            else:
                unknown_percentage = line.strip().count("X")/len(line.strip())
                if unknown_percentage <= 0.2:
                    pdb2seq[key] = line.strip() 
    return pdb2seq

            #unknown_percentage = chain_dir[key].count("X")/len(chain_dir[key])
            #print(f"seq:{chain_dir[key]}, percentage:{unknown_percentage}")
            #if unknown_percentage <= 0.2:

'''def write_seqs_file(pdb2seq, chains):
    with open("clustered_seqs.txt", "w") as handle:
        for chain in chains:   
            if chain in pdb2seq.keys():
                unknown_percentage = pdb2seq[chains].count("X")/len(pdb2seq[chains])
                print(f"seq:{pdb2seq[chains]}, percentage:{unknown_percentage}")
                if unknown_percentage <= 0.2:
                    handle.write(f">{chain}\n{pdb2seq[chain]}\n")
    return handle'''

def load_go_graph(fname):
    """
    Load the Gene Ontology graph and return it.
    """
    go_graph = obonet.read_obo(fname)
    print(f"Loaded GO graph with {len(go_graph.nodes)} nodes and {len(go_graph.edges)} edges.")
    return go_graph

def create_subgraph(go_graph, parent_term):
    """
    Create a subgraph based on the given parent term.
    
    Parameters:
    go_graph: The full Gene Ontology graph.
    parent_term: The GO term from which the subgraph should be created.
    
    Returns:
    A subgraph that includes the parent term and all its descendants.
    """
    if parent_term not in go_graph:
        raise ValueError(f"Parent term {parent_term} not found in the GO graph.")
    
    # Get all descendants of the parent term (including the parent term itself)
    child_terms = nx.ancestors(go_graph, parent_term)
    child_terms.add(parent_term)  # Include the parent term in the subgraph

    # Create the subgraph containing only the parent term and its descendants
    subgraph = go_graph.subgraph(child_terms).copy()

    print(f"Created subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
    return subgraph

def read_sifts(sifts_fname, pdb_chains, go_graph):
    pdb2go = {}
    go2info = {}

    with open(sifts_fname, mode='rt') as tsvfile:
        for _ in range(2):
            next(tsvfile)
        for line in tsvfile:
            pdb = line.strip().upper().split("\t")[0]
            chain = line.strip().split("\t")[1]
            evidence = line.strip().split("\t")[4]
            go_id = line.strip().split("\t")[5]
            pdb_chain = pdb + '_' + chain
            #print(pdb_chain)
            if (pdb_chain in pdb_chains) and (go_id in go_graph) and (go_id not in root_terms):
                if pdb_chain not in pdb2go:
                    pdb2go[pdb_chain] = {'goterms': [go_id], 'evidence': [evidence]}
                namespace = go_graph.nodes[go_id]['namespace']
                go_ids = nx.descendants(go_graph, go_id)
                go_ids.add(go_id)
                go_ids = go_ids.difference(root_terms)
                for go in go_ids:
                    pdb2go[pdb_chain]['goterms'].append(go)
                    pdb2go[pdb_chain]['evidence'].append(evidence)
                    name = go_graph.nodes[go]['name']
                    if go not in go2info:
                        go2info[go] = {'ont': namespace, 'goname': name, 'pdb_chains': set([pdb_chain])}
                    else:
                        go2info[go]['pdb_chains'].add(pdb_chain)
    return pdb2go, go2info

def write_output_files(fname, pdb2go, go2info, pdb2seq):
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    selected_goterms = {ont: set() for ont in onts}
    selected_proteins = set()
    for goterm in go2info:
        prots = go2info[goterm]['pdb_chains']
        num = len(prots)
        namespace = go2info[goterm]['ont']
        #print(f"Goterm: {goterm}, Num: {num}, Namespace: {namespace}")
        if (num > 10) and (num <= 5000):
            selected_goterms[namespace].add(goterm)
            selected_proteins = selected_proteins.union(prots)

    selected_goterms_list = {ont: list(selected_goterms[ont]) for ont in onts}
    selected_gonames_list = {ont: [go2info[goterm]['goname'] for goterm in selected_goterms_list[ont]] for ont in onts}

    for ont in onts:
        print ("###", ont, ":", len(selected_goterms_list[ont]))

    sequences_list = []
    protein_list = []

    with open(fname + "pdb2sequences.fasta", 'wt', newline='')  as out_file:
        for key in pdb2seq.keys():   
            #unknown_percentage = pdb2seq[key].count("X")/len(pdb2seq[key])
            #print(f"seq:{pdb2seq[key]}, percentage:{unknown_percentage}")
            #if unknown_percentage <= 0.2:
            out_file.write(f">{key}\n{pdb2seq[key]}\n")

    with open(fname + 'pdb2go.tsv', 'wt', newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for ont in onts:
            tsv_writer.writerow(["### GO-terms (%s)" % (ont)])
            tsv_writer.writerow(selected_goterms_list[ont])
            tsv_writer.writerow(["### GO-names (%s)" % (ont)])
            tsv_writer.writerow(selected_gonames_list[ont])
        tsv_writer.writerow(["### PDB-chain", "GO-terms (molecular_function)", "GO-terms (biological_process)", "GO-terms (cellular_component)"])
        for chain in selected_proteins:
            if chain in pdb2seq:
                goterms = set(pdb2go[chain]['goterms'])
                if len(goterms) > 2:
                    # selected goterms
                    mf_goterms = goterms.intersection(set(selected_goterms_list[onts[0]]))
                    bp_goterms = goterms.intersection(set(selected_goterms_list[onts[1]]))
                    cc_goterms = goterms.intersection(set(selected_goterms_list[onts[2]]))
                    if len(mf_goterms) > 0 or len(bp_goterms) > 0 or len(cc_goterms) > 0:
                        sequences_list.append(SeqRecord(Seq(pdb2seq[chain], protein_letters_3to1), id=chain, description="nrPDB"))
                        protein_list.append(chain)
                        tsv_writer.writerow([chain, ','.join(mf_goterms), ','.join(bp_goterms), ','.join(cc_goterms)])
                    else:
                        print(f"Chain {chain} not found in pdb2seq.")

    np.random.seed(1234)
    np.random.shuffle(protein_list)
    print ("Total number PDB Chains annotated =%d" % (len(protein_list)))
    #print(f"Sample protein chain : {protein_list[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sifts', type=str, default='preprocessing/data/pdb_chain_go.tsv_2024-06-25', help="SIFTS annotation files.")
    parser.add_argument('-struc_dir', type= str, default='preprocessing/data/structure_files', help= 'directory containing cif files')
    #parser.add_argument('-clu', type=str, default='preprocessing/data/mmseqs2_clusters_0.5_seq_id_0.8_cov.tsv', help="mmseqs2_cluster_results")
    parser.add_argument('-seqs', type=str, default='preprocessing/data/seqs_from_structure_dir.fasta', help="sequences from cif directory")
    parser.add_argument('-obo', type=str, default='preprocessing/data/go-basic_2024-06-25.obo', help="gene ontology basic.obo file")
    parser.add_argument('-out', type=str, default='preprocessing/data/', help="output files")    
    args = parser.parse_args()

    #write_seqs_from_cifdir(args.struc_dir, args.seqs)
    #repr = load_cluster_file(args.clu)

    #pdb2seq = pdb_2_seq(args.seqs)
    #write_seqs_file(pdb2seq,repr)

    #f = open("output test.txt", "w")
    #f.write(str(read_sifts(args.sifts, repr, go_graph)))

    ids = read_seqs_file(args.seqs).keys()
    print(ids)
    go_graph = load_go_graph(args.obo)
    #sub_go_graph = create_subgraph(go_graph, domain_terms)
    pdb2seq = read_seqs_file(args.seqs)
    pdb2go, go2info = read_sifts(args.sifts, ids, go_graph)
    write_output_files(args.out, pdb2go, go2info, pdb2seq)