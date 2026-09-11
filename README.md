# Graph Neural Network Drug Repurposing for Spaceflight-Induced Muscle Atrophy
Multi-dataset consensus differential expression across 7 NASA GeneLab experiments, Ensembl-validated ortholog mapping, and a CTD+ChEMBL cross-validated drug-gene knowledge graph, used to train a heterogeneous GraphSAGE link-prediction model that prioritizes drug repurposing candidates for spaceflight-induced muscle atrophy.

# Spaceflight Differential Gene Expression & Graph Neural Network Drug Discovery Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-HeteroGraph-3D8BD3)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computational biology and Graph Neural Network (GNN) framework designed to identify therapeutic drug candidates that counteract spaceflight-induced physiological stress. 

This repository processes multi-study RNA-seq/microarray differential gene expression (DEG) datasets from NASA GeneLab, normalizes and weights consensus directional trends, maps mouse-to-human orthologs via Ensembl homology REST APIs, constructs a heterogeneous knowledge graph, and trains a **GraphSAGE Link Prediction / Node Representation Model** to prioritize candidate therapeutics.

---

##  Key Features

- **Direction-Aware DEG Extraction**: Automatically parses complex contrast strings e.g., `(Spaceflight)v(Ground Control)` vs. `(Control)v(Flight)`, systematically re-orienting $\log_2(\text{fold-change})$ signs to maintain true physiological trajectory.
- **Robust Multi-Dataset Consensus**: Combines across multi-study experiments using dataset support counts, cross-study direction agreement, and FDR-adjusted $q$-values ($q < 0.05$).
- **Cross-Species Orthology Resolver**: Robust mouse-to-human mapping engine utilizing multi-stage Ensembl REST API lookup and `mygene` fallback for clean HGNC symbol output.
- **Biomedical Knowledge Graph Construction**: Integrates functional interactions from **CTD** (Comparative Toxicogenomics Database) and targeted chemical bioactivity data from **ChEMBL**.
- **PyG Heterogeneous Graph Neural Network**: Employs GraphSAGE (`SAGEConv`) with margin-based ranking loss and hard-negative sampling to embed complex chemical-gene biological networks.

---

## Installation & Requirements

Ensure you have Python 3.8+ installed. You can install all required scientific computing, bioinformatics, and deep learning dependencies directly:

```bash
pip install pandas numpy scikit-learn requests mygene torch torch-geometric
