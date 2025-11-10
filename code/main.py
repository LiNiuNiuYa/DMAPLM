import argparse
import sys
import os
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
import warnings
import time

warnings.filterwarnings('ignore')

sys.path.append('/')
from loader import data_preparation
from Att import AttentionWeightedPooling
from contrastive_learning import ContrastiveFeatureEnhancer


def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest Experiments")
    parser.add_argument('--dataset', default='appoved')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--disease_TopK', type=int, default=4)
    parser.add_argument('--drug_TopK', type=int, default=4)

    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--min_samples_split', type=int, default=5)
    parser.add_argument('--min_samples_leaf', type=int, default=2)
    parser.add_argument('--max_features', type=str, default='sqrt')

    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--drug_emb_path', type=str,
                        default='../appoved/emb/drug_embeddings')
    parser.add_argument('--disease_emb_path', type=str,
                        default='../appoved/emb')
    parser.add_argument('--pooling_method', type=str, default='attention',
                        choices=['mean', 'attention'])
    parser.add_argument('--use_contrastive', action='store_true')
    return parser.parse_args()


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def process_drug_embedding(emb_data, pooling_method='attention'):
    import torch
    M_drug = None

    if isinstance(emb_data, dict) and 'embeddings' in emb_data:
        embedding_obj = emb_data['embeddings']
        if hasattr(embedding_obj, 'x'):
            M_seq = embedding_obj.x
            if len(M_seq.shape) == 3:
                M_seq = M_seq.squeeze(0)
            if pooling_method == 'attention' and M_seq.shape[0] > 1:
                pooling_module = AttentionWeightedPooling(embed_dim=M_seq.shape[1])
                M_drug, _ = pooling_module(M_seq)
                M_drug = M_drug.detach().cpu().numpy()
            else:
                M_drug = M_seq.mean(dim=0).detach().numpy()
    else:
        import torch
        if isinstance(emb_data, torch.Tensor):
            M_seq = emb_data.detach()
            if len(M_seq.shape) == 3:
                M_seq = M_seq.squeeze(0)
            if len(M_seq.shape) == 2 and M_seq.shape[1] == 384:
                if pooling_method == 'attention' and M_seq.shape[0] > 1:
                    pooling_module = AttentionWeightedPooling(embed_dim=M_seq.shape[1])
                    M_drug, _ = pooling_module(M_seq)
                    M_drug = M_drug.detach().cpu().numpy()
                else:
                    M_drug = M_seq.mean(dim=0).numpy()
            elif len(M_seq.shape) == 1 and M_seq.shape[0] == 384:
                M_drug = M_seq.numpy()
    return M_drug


def load_embeddings(args):
    import torch
    D_all_diseases = np.load(f"../dataset/{args.dataset}/emb/disease_embeddings.npy")
    drug_idx_to_id_path = os.path.join("../appoved", "drug_idx_to_id.npy")
    if os.path.exists(drug_idx_to_id_path):
        drug_idx_to_id = np.load(drug_idx_to_id_path, allow_pickle=True).item()
        drug_id_to_idx = {v: k for k, v in drug_idx_to_id.items()}
    else:
        drug_id_to_idx = {f"DRUG_{i}": i for i in range(args.n_drugs)}

    M_all_drugs = np.zeros((args.n_drugs, 384))
    if os.path.exists(args.drug_emb_path):
        for file_name in os.listdir(args.drug_emb_path):
            if file_name.endswith(".pt"):
                emb_path = os.path.join(args.drug_emb_path, file_name)
                try:
                    drug_id = file_name.split("_embedded.pt")[0]
                    if drug_id in drug_id_to_idx:
                        drug_idx = drug_id_to_idx[drug_id]
                        emb_data = torch.load(emb_path, map_location='cpu')
                        M_drug = process_drug_embedding(emb_data, args.pooling_method)
                        if M_drug is not None and M_drug.shape == (384,):
                            M_all_drugs[drug_idx] = M_drug
                except Exception:
                    continue
    return M_all_drugs, D_all_diseases


def prepare_enhanced_data(train_mask, test_mask, A, M_all_drugs, D_all_diseases, use_contrastive=False):
    n_diseases, n_drugs = A.shape
    train_pairs, test_pairs = [], []
    for i in range(n_diseases):
        for j in range(n_drugs):
            if train_mask[i, j]:
                y = 1 if A[i, j] > 0 else 0
                train_pairs.append((j, i, y))
            elif test_mask[i, j]:
                y = 1 if A[i, j] > 0 else 0
                test_pairs.append((j, i, y))

    enhancer = ContrastiveFeatureEnhancer(use_contrastive=use_contrastive)
    X_train, y_train = enhancer.fit_and_transform_train(M_all_drugs, D_all_diseases, train_pairs, train_mask)
    X_test, y_test = enhancer.transform_test(M_all_drugs, D_all_diseases, test_pairs)
    return X_train, y_train, X_test, y_test


def run_rf_experiment(args, train_mask, test_mask, A, M_all_drugs, D_all_diseases, fold_idx):
    X_train, y_train, X_test, y_test = prepare_enhanced_data(train_mask, test_mask, A, M_all_drugs, D_all_diseases, args.use_contrastive)
    if len(X_train) == 0 or len(X_test) == 0:
        return

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    print(f"Fold {fold_idx + 1} done. Predictions shape: {y_pred.shape}")


def main():
    args = parse_args()
    setup_seed(args.seed)


    print("Random Forest Training Started")

    disease_adj, drug_adj, A, all_train_mask, all_test_mask, pos_weight = data_preparation(args)
    M_all_drugs, D_all_diseases = load_embeddings(args)

    for fold_idx in range(args.n_splits):
        run_rf_experiment(args, all_train_mask[fold_idx], all_test_mask[fold_idx], A, M_all_drugs, D_all_diseases, fold_idx)


    print("All folds completed.")


if __name__ == "__main__":
    main()
