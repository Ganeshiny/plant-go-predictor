import os
import csv
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import pickle

#Inializing here
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
model = BertModel.from_pretrained('Rostlab/prot_bert_bfd')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

def seq2onehot(seq):
    """
    Convert sequence to one-hot encoding.
    """
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_embed = {char: idx for idx, char in enumerate(chars)}
    vocab_one_hot = np.eye(len(chars), dtype=int)
    seqs_x = np.array([vocab_one_hot[vocab_embed[v]] for v in seq])
    return seqs_x

def seq2protbert(seq):
    # Tokenize the sequence
    seq = ' '.join(seq)
    inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True, padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        # Get ProtBERT embeddings
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        
    # ProtBERT embeddings to numpy
    embeddings = embeddings.detach().cpu().numpy()
    attention_mask = attention_mask.detach().cpu().numpy()
    
    features = []
    for seq_num in range(len(embeddings)):
        seq_len = (attention_mask[seq_num] == 1).sum() 
        if seq_len > 2: 
            seq_emd = embeddings[seq_num][1:seq_len-1]  # without [CLS] and [SEP]
            features.append(seq_emd)  
    return np.array(features)

class PDB_Dataset(Dataset):
    def __init__(self, root, annot_file, num_shards=20, selected_ontology=None, transform=None, pre_transform=None, model = None, annot_pkl_file = "preprocessing/data/annot_dict.pkl"):
        self.annot_pkl_file = annot_pkl_file
        annot_data = self.annot_file_reader(annot_file, self.annot_pkl_file)
        self.model = model
        self.prot2annot = annot_data[0]
        self.gonames = annot_data[2][selected_ontology]
        self.goterms = annot_data[1][selected_ontology]
        self.prot_list = [prot_id for prot_id in annot_data[3] if os.path.exists(os.path.join(root, f'{prot_id}.npz'))]
        self.npz_dir = root
        self.num_shards = num_shards
        self.selected_ontology = selected_ontology
        self.y_labels = annot_data[1][selected_ontology]
        self.transform = transform
        self.pre_transform = pre_transform
        super(PDB_Dataset, self).__init__(root, transform, pre_transform)
    
    def annot_file_reader(self, annot_filename, output_filename):
        onts = ['molecular_function', 'biological_process', 'cellular_component']
        prot2annot = {}
        goterms = {ont: [] for ont in onts}
        gonames = {ont: [] for ont in onts}
        prot_list = []
        with open(annot_filename, mode='r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')

            # molecular function
            next(reader, None)  # skip the headers
            goterms[onts[0]] = next(reader)
            next(reader, None)  # skip the headers
            gonames[onts[0]] = next(reader)

            # biological process
            next(reader, None)  # skip the headers
            goterms[onts[1]] = next(reader)
            next(reader, None)  # skip the headers
            gonames[onts[1]] = next(reader)

            # cellular component
            next(reader, None)  # skip the headers
            goterms[onts[2]] = next(reader)
            next(reader, None)  # skip the headers
            gonames[onts[2]] = next(reader)

            next(reader, None)  # skip the headers
            for row in reader:
                prot, prot_goterms = row[0], row[1:]
                prot2annot[prot] = {ont: [] for ont in onts}
                for i in range(3):
                    goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if
                                    goterm != '']
                    prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]), dtype=np.int64)
                    prot2annot[prot][onts[i]][goterm_indices] = 1.0
                prot_list.append(prot)

            # Save the results to a file
        output_data = {
            'prot2annot': prot2annot,
            'goterms': goterms,
            'gonames': gonames,
            'prot_list': prot_list
        }
        output_file = open(output_filename, 'wb')
        pickle.dump(output_data, output_file)

        print(f"Output saved to {output_filename}")
        return prot2annot, goterms, gonames, prot_list

    @property
    def num_classes(self):
        return len(self.prot2annot[self.prot_list[0]][self.selected_ontology]) if self.selected_ontology else None

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.prot_list))]
    
    @property
    def num_classes(self):
        return len(self.prot2annot[self.prot_list[0]][self.selected_ontology]) if self.selected_ontology else None

    def process(self):
        data_list = []
        for index, prot_id in tqdm(enumerate(self.prot_list), total=len(self.prot_list)):
            data = self._load_data(prot_id, self.prot2annot)
            if data:
                if self.pre_transform:
                    data = self.pre_transform(data)
                data_list.append(data)
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))
        return data_list

    def _load_data(self, prot_id, prot2annot):
        pdb_file = os.path.join(self.npz_dir, f'{prot_id}.npz')
        if not os.path.isfile(pdb_file):
            print(f"File not found: {pdb_file}")
            return None
        
        cmap = np.load(pdb_file)
        sequence = str(cmap['seqres'])
        
        # One-hot encoding
        onehot_features = torch.tensor(seq2onehot(sequence), dtype=torch.float).squeeze(0)
        # ProtBERT embeddings
        protbert_features = torch.tensor(seq2protbert(sequence), dtype=torch.float).squeeze(0)
        
        adjacency_info = self._get_adjacency_info(cmap['C_alpha'])
        labels = self._get_labels(prot_id, prot2annot)
        length = torch.tensor(len(sequence), dtype=torch.long)

        if self.selected_ontology:
            labels = labels.get(self.selected_ontology, torch.zeros(len(self.goterms), dtype=torch.long))

        # Data objects
        onehot_data = Data(x=onehot_features, edge_index=adjacency_info, u=prot_id, y=labels, length=length)
        protbert_data = Data(x=protbert_features, edge_index=adjacency_info, u=prot_id, y=labels, length=length)

        if self.model == "protBERT":
            return protbert_data
        else:
            return onehot_data

    def _get_labels(self, prot_id, prot2annot):
        labels = {
            ont: torch.tensor(prot2annot[prot_id][ont], dtype=torch.long).unsqueeze(0)
            for ont in ['molecular_function', 'biological_process', 'cellular_component']
        }

        for ont, label in labels.items():
            if label.numel() == 0:
                labels[ont] = torch.zeros(1, dtype=torch.long)

        return labels

    def _get_adjacency_info(self, distance_matrix, threshold=8.0):
        adjacency_matrix = (distance_matrix <= threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)
        edge_indices = np.nonzero(adjacency_matrix)

        coo_matrix = sp.coo_matrix((np.ones_like(edge_indices[0]), edge_indices))
        return torch.tensor([coo_matrix.row, coo_matrix.col], dtype=torch.long)

    def len(self):
        return len(self.prot_list)
    
    def get(self, idx):
        prot_id = self.prot_list[idx]
        data = self._load_data(prot_id, self.prot2annot)
        return data


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Dataset Setup
root = 'preprocessing/data/structure_files/tmp_cmap_files'
annot_file = 'preprocessing/data/pdb2go.tsv'
num_shards = 20

torch.manual_seed(12345)
pdb_protBERT_dataset = PDB_Dataset(root, annot_file, num_shards=num_shards, selected_ontology="biological_process", model="protBERT")

print(pdb_protBERT_dataset[0].x[0].shape, pdb_protBERT_dataset[0].edge_index.shape, pdb_protBERT_dataset[0].y.shape, pdb_protBERT_dataset[0].length)
