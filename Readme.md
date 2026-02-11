# DMAPLM: A Multimodal Pretrained Framework for Computational Drug Repositioning

![DMAPLM Framework](model.png)

## Abstract

DMAPLM is a lightweight dual-encoder framework for predicting drug-disease associations using pretrained language models. The framework employs **ChemBERTa-2** for molecular encoding and **BioBERT** for disease text encoding, with contrastive learning and attention-weighted pooling for enhanced representations. A Random Forest classifier predicts associations based on the multimodal features.

**Performance:** AUROC = 0.8919, AUPR = 0.9116 (5-fold CV)

## Dataset

- **Source:** DrugMAP 2.0 (https://idrblab.org/drugmap)
- **Diseases:** 1,455
- **Drugs:** 2,622
- **Associations:** 5,993
- **Density:** 0.16%

## Installation

```bash
conda env create -f environment.yml
conda activate dmaplm
conda config --env --set channel_priority strict
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Quick Start

### Step 1: Generate Embeddings

```bash
python embedding_generation/drug_emb_gen.py
python embedding_generation/disease_emb_gen.py
```

Output files:
- `dataset/appoved/emb/drug_embeddings/*.pt`
- `dataset/appoved/emb/disease_embeddings.npy`

### Step 2: Train Model

```bash
cd code
python main_final.py --dataset appoved --n_estimators 250 --max_depth 25 --use_contrastive
```

**Key parameters:**
- `--n_estimators 250`: Number of Random Forest trees
- `--max_depth 25`: Maximum tree depth
- `--use_contrastive`: Enable contrastive learning (required for best performance)

### Step 3: Cold-Start Evaluation

```bash
python main_final.py --cold_start C1 --use_contrastive  # New drugs
python main_final.py --cold_start C2 --use_contrastive  # New diseases
python main_final.py --cold_start C3 --use_contrastive  # Both new
```

## Expected Results

### 5-Fold Cross-Validation
- AUROC: 0.8919
- AUPR: 0.9116
- F1-score: 0.8161
- Accuracy: 0.8244

### Cold-Start Performance
- C1 (New Drugs): AUROC = 0.8150, AUPR = 0.8338
- C2 (New Diseases): AUROC = 0.7805, AUPR = 0.8098
- C3 (Both): AUROC = 0.7456, AUPR = 0.7355

## Requirements

- Python 3.10.16
- PyTorch 1.13.1 (CUDA 11.7)
- GPU: ≥8GB VRAM (recommended)
- RAM: ≥16GB
- Disk: ~11GB

## Citation

```bibtex
@article{chen2025dmaplm,
  title={DMAPLM: A Multimodal Pretrained Framework for Computational Drug Repositioning},
  author={Chen, Hailin and Li, Zhongling},
  year={2025}
}
```

## Acknowledgements

Supported by:
- National Natural Science Foundation of China (62562031)
- Jiangxi Provincial Natural Science Foundation, China (20242BAB25083)

## Contact

- Email: chenhailin@ecjtu.edu.cn
- GitHub: https://github.com/LiNiuNiuYa/DMAPLM

## License

MIT License
