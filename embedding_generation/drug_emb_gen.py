import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import Data
from rdkit import Chem


class DrugEmbeddingGenerator:
    def __init__(self,
                 drug_id_path,
                 drug_info_path,
                 output_dir,
                 device=None,
                 drug_encoder_path='DeepChem/ChemBERTa-77M-MTR',
                 max_length=512):

        self.drug_id_path = drug_id_path
        self.drug_info_path = drug_info_path
        self.output_dir = output_dir
        self.max_length = max_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if os.path.exists(drug_encoder_path):
            self.drug_encoder = AutoModel.from_pretrained(drug_encoder_path)
            self.drug_tokenizer = AutoTokenizer.from_pretrained(drug_encoder_path)
        else:
            self.drug_encoder = AutoModel.from_pretrained(drug_encoder_path)
            self.drug_tokenizer = AutoTokenizer.from_pretrained(drug_encoder_path)

        self.drug_encoder = self.drug_encoder.to(self.device)
        os.makedirs(output_dir, exist_ok=True)

    def is_valid_smiles(self, smiles):
        if pd.isna(smiles) or not smiles:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def generate_token_edges(self, S_trim):
        n = len(S_trim)
        edges = []
        for i in range(n - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        return np.array(edges).T

    def generate_embeddings(self, drug_ids, drug_info_df):
        missing_smiles_drugs = []
        filtered_df = drug_info_df[drug_info_df["DrugID"].isin(drug_ids)].copy()

        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
            drug_id = row['DrugID']
            smiles = row['Canonical_SMILES']

            if not self.is_valid_smiles(smiles):
                missing_smiles_drugs.append(drug_id)
                continue

            output_path = os.path.join(self.output_dir, f"{drug_id}_embedded.pt")
            if os.path.exists(output_path):
                continue

            try:
                S_trim = self.drug_tokenizer.encode(
                    smiles,
                    truncation=True,
                    max_length=self.max_length
                )
                tokens = torch.tensor(S_trim).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    M_seq = self.drug_encoder(tokens).last_hidden_state

                edges = self.generate_token_edges(S_trim)
                node_ids = [True] * len(S_trim)

                data = {
                    'embeddings': Data(
                        x=M_seq.squeeze(0).cpu(),
                        edge_index=torch.tensor(edges, dtype=torch.long)
                    ),
                    'Drug_ID': drug_id,
                    'SMILES': smiles,
                    'node_ids': np.array(node_ids, dtype=bool),
                }

                torch.save(data, output_path)

            except Exception:
                missing_smiles_drugs.append(drug_id)

        return len(filtered_df), missing_smiles_drugs

    def run(self):
        drug_ids_data = np.load(self.drug_id_path, allow_pickle=True)
        if isinstance(drug_ids_data, dict):
            drug_ids = list(drug_ids_data.keys())
        elif isinstance(drug_ids_data, np.ndarray):
            drug_ids = list(drug_ids_data[0].keys())

        drug_info_df = pd.read_csv(self.drug_info_path, sep='\t')
        processed_count, missing_smiles_drugs = self.generate_embeddings(drug_ids, drug_info_df)

        return processed_count, missing_smiles_drugs


def main():
    drug_id_path = r"..\DMAPLM\embedding_generation\drug_id.npy"
    drug_info_path = r"..\DMAPLM\embedding_generation\1.General Information of Drug.tsv"
    output_dir = r"..\appoved\emb\drug_embeddings"
    drug_encoder_path = "DeepChem/ChemBERTa-77M-MTR"

    generator = DrugEmbeddingGenerator(
        drug_id_path=drug_id_path,
        drug_info_path=drug_info_path,
        output_dir=output_dir,
        drug_encoder_path=drug_encoder_path
    )

    processed_count, missing_smiles_drugs = generator.run()


if __name__ == "__main__":
    main()