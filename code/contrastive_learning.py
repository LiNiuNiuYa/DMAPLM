import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class DrugDiseaseContrastiveLearning:
    def __init__(self, drug_dim=384, disease_dim=384, projection_dim=128,
                 temperature=0.1, negative_ratio=3, device='cpu'):
        self.drug_dim = drug_dim
        self.disease_dim = disease_dim
        self.proj_dim = projection_dim
        self.tau = temperature
        self.neg_ratio = negative_ratio
        self.device = device

        self.f_drug = nn.Sequential(
            nn.Linear(drug_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        ).to(device)

        self.f_disease = nn.Sequential(
            nn.Linear(disease_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        ).to(device)

        self.f_interact = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim // 2)
        ).to(device)

    def forward(self, M_drug, D_disease):
        if not isinstance(M_drug, torch.Tensor):
            M_drug = torch.tensor(M_drug, dtype=torch.float32).to(self.device)
        if not isinstance(D_disease, torch.Tensor):
            D_disease = torch.tensor(D_disease, dtype=torch.float32).to(self.device)

        Z_drug = self.f_drug(M_drug)
        Z_disease = self.f_disease(D_disease)

        Z_concat = torch.cat([Z_drug, Z_disease], dim=-1)
        Z_interact = self.f_interact(Z_concat)

        M_concat = torch.cat([M_drug, D_disease], dim=-1)

        X_enhanced = torch.cat([
            M_concat,
            Z_drug,
            Z_disease,
            Z_interact
        ], dim=-1)

        return X_enhanced.detach().cpu().numpy()

    def contrastive_loss(self, M_drug, D_disease, y):
        if not isinstance(M_drug, torch.Tensor):
            M_drug = torch.tensor(M_drug, dtype=torch.float32).to(self.device)
        if not isinstance(D_disease, torch.Tensor):
            D_disease = torch.tensor(D_disease, dtype=torch.float32).to(self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32).to(self.device)

        Z_drug = F.normalize(self.f_drug(M_drug), dim=-1)
        Z_disease = F.normalize(self.f_disease(D_disease), dim=-1)

        S = torch.matmul(Z_drug, Z_disease.T) / self.tau

        batch_size = Z_drug.shape[0]
        targets = torch.arange(batch_size).to(self.device)

        pos_indices = torch.where(y.flatten() > 0)[0]

        if len(pos_indices) > 0:
            S_pos = S[pos_indices]
            targets_pos = pos_indices
            loss = F.cross_entropy(S_pos, targets_pos)
        else:
            loss = torch.tensor(0.0, requires_grad=True).to(self.device)

        return loss

    def fit_contrastive(self, M_drugs, D_diseases, A,
                        epochs=50, learning_rate=0.001, batch_size=512):
        print("Starting contrastive learning training...")

        optimizer = torch.optim.Adam(
            list(self.f_drug.parameters()) +
            list(self.f_disease.parameters()) +
            list(self.f_interact.parameters()),
            lr=learning_rate
        )

        pos_indices = np.where(A > 0)
        pos_drugs = pos_indices[1]
        pos_diseases = pos_indices[0]

        neg_drugs, neg_diseases = self._generate_negative_samples(
            A, len(pos_drugs) * self.neg_ratio
        )

        all_drugs = np.concatenate([pos_drugs, neg_drugs])
        all_diseases = np.concatenate([pos_diseases, neg_diseases])
        y_all = np.concatenate([
            np.ones(len(pos_drugs)),
            np.zeros(len(neg_drugs))
        ])

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            indices = np.random.permutation(len(all_drugs))

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i + batch_size]

                drug_idx_batch = all_drugs[batch_idx]
                disease_idx_batch = all_diseases[batch_idx]
                y_batch = y_all[batch_idx]

                M_batch = M_drugs[drug_idx_batch]
                D_batch = D_diseases[disease_idx_batch]

                optimizer.zero_grad()
                loss = self.contrastive_loss(M_batch, D_batch, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        print("Contrastive learning training completed")

    def _generate_negative_samples(self, A, n_samples):
        n_diseases, n_drugs = A.shape
        neg_drugs = []
        neg_diseases = []

        while len(neg_drugs) < n_samples:
            drug_idx = np.random.randint(0, n_drugs)
            disease_idx = np.random.randint(0, n_diseases)

            if A[disease_idx, drug_idx] == 0:
                neg_drugs.append(drug_idx)
                neg_diseases.append(disease_idx)

        return np.array(neg_drugs), np.array(neg_diseases)

    def enhance_features(self, M_drugs, D_diseases, drug_indices, disease_indices):
        M_selected = M_drugs[drug_indices]
        D_selected = D_diseases[disease_indices]

        X_enhanced = self.forward(M_selected, D_selected)

        return X_enhanced

    def get_projected_embeddings(self, M_drugs, D_diseases):
        with torch.no_grad():
            if not isinstance(M_drugs, torch.Tensor):
                M_drugs = torch.FloatTensor(M_drugs).to(self.device)
            if not isinstance(D_diseases, torch.Tensor):
                D_diseases = torch.FloatTensor(D_diseases).to(self.device)

            Z_drugs = self.f_drug(M_drugs).cpu().numpy()
            Z_diseases = self.f_disease(D_diseases).cpu().numpy()

        return Z_drugs, Z_diseases


class ContrastiveFeatureEnhancer:
    def __init__(self, drug_dim=384, disease_dim=384, use_contrastive=True):
        self.drug_dim = drug_dim
        self.disease_dim = disease_dim
        self.use_contrastive = use_contrastive
        self.contrastive_module = None
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        self.train_drug_indices = None
        self.train_disease_indices = None
        self.drug_idx_map = None
        self.disease_idx_map = None

    def fit_and_transform_train(self, M_all_drugs, D_all_diseases, train_pairs, train_mask):
        if self.use_contrastive:
            actual_drug_dim = M_all_drugs.shape[1]
            actual_disease_dim = D_all_diseases.shape[1]

            print(f"Detected embedding dimensions - Drug: {actual_drug_dim}, Disease: {actual_disease_dim}")

            self.contrastive_module = DrugDiseaseContrastiveLearning(
                drug_dim=actual_drug_dim,
                disease_dim=actual_disease_dim,
                projection_dim=128,
                temperature=0.25
            )

            self.train_drug_indices = np.unique([pair[0] for pair in train_pairs])
            self.train_disease_indices = np.unique([pair[1] for pair in train_pairs])

            print(
                f"Training set contains {len(self.train_drug_indices)} drugs and {len(self.train_disease_indices)} diseases")

            M_train_drugs = M_all_drugs[self.train_drug_indices]
            D_train_diseases = D_all_diseases[self.train_disease_indices]

            self.drug_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(self.train_drug_indices)}
            self.disease_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(self.train_disease_indices)}

            A_train_subset = np.zeros((len(self.train_disease_indices), len(self.train_drug_indices)))
            remapped_train_pairs = []

            for drug_idx, disease_idx, label in train_pairs:
                new_drug_idx = self.drug_idx_map[drug_idx]
                new_disease_idx = self.disease_idx_map[disease_idx]
                A_train_subset[new_disease_idx, new_drug_idx] = label
                remapped_train_pairs.append((new_drug_idx, new_disease_idx, label))

            self.contrastive_module.fit_contrastive(
                M_train_drugs, D_train_diseases, A_train_subset,
                epochs=30, batch_size=256
            )

            drug_indices = [self.drug_idx_map[pair[0]] for pair in train_pairs]
            disease_indices = [self.disease_idx_map[pair[1]] for pair in train_pairs]
            y = [pair[2] for pair in train_pairs]

            X_enhanced = self.contrastive_module.enhance_features(
                M_train_drugs, D_train_diseases, drug_indices, disease_indices
            )

            X_enhanced = self.scaler.fit_transform(X_enhanced)
            self.scaler_fitted = True

        else:
            X_enhanced = []
            y = []

            for drug_idx, disease_idx, label in train_pairs:
                x = np.concatenate([
                    M_all_drugs[drug_idx],
                    D_all_diseases[disease_idx]
                ])
                X_enhanced.append(x)
                y.append(label)

            X_enhanced = np.array(X_enhanced)
            X_enhanced = self.scaler.fit_transform(X_enhanced)
            self.scaler_fitted = True

        return X_enhanced, np.array(y)


    def transform_test(self, M_all_drugs, D_all_diseases, test_pairs):
        if not self.scaler_fitted:
            raise ValueError("Must call fit_and_transform_train on training data first")

        if self.use_contrastive:
            if self.contrastive_module is None:
                raise ValueError("Contrastive module not initialized")

            X_enhanced_list = []
            y = []

            for drug_idx, disease_idx, label in test_pairs:
                feature = self.contrastive_module.enhance_features(
                    M_all_drugs, D_all_diseases, [drug_idx], [disease_idx]
                )
                X_enhanced_list.append(feature[0])
                y.append(label)

            if len(X_enhanced_list) == 0:
                raise ValueError("No test samples to process!")

            X_enhanced = np.array(X_enhanced_list)

        else:
            X_enhanced = []
            y = []

            for drug_idx, disease_idx, label in test_pairs:
                if drug_idx < len(M_all_drugs) and disease_idx < len(D_all_diseases):
                    x = np.concatenate([
                        M_all_drugs[drug_idx],
                        D_all_diseases[disease_idx]
                    ])
                    X_enhanced.append(x)
                    y.append(label)

            if len(X_enhanced) == 0:
                raise ValueError("No valid test samples!")

            X_enhanced = np.array(X_enhanced)

        X_enhanced = self.scaler.transform(X_enhanced)

        return X_enhanced, np.array(y)

    def get_enhanced_embeddings(self, M_drugs, D_diseases):
        if not self.use_contrastive or self.contrastive_module is None:
            return M_drugs, D_diseases

        return self.contrastive_module.get_projected_embeddings(M_drugs, D_diseases)