# 			**QuICPTMs: A Multi-modal Framework Integrating Mamba and Dynamic Geometric Graph Attention for Precise Protein S-sulfhydration Site Prediction**

**1.Catalog description**

​	data: Data storage

​	src: Main code storage

​	model/embedding: Weight storage

------

**2.Requirements**

```
pandas

scikit-learn

transformers

matplotlib

umap-learn
```

Please download torch version>=2.0 or above to avoid version conflicts

Core Model Dependencies:

Mamba-SSM: Required for the sequence branch.
```
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```
PyTorch Geometric: Required for the DyGeo-GAT structure branch.
```
pip install torch_geometric
```
------

**3.PLM model & Embedding**

This framework utilizes ESM-2 for sequence feature extraction and ProstT5 for structure-aware node features. Please refer to their official repositories for generating embeddings if not using our pre-computed data.
ESM-2 : [facebookresearch/esm](https://github.com/facebookresearch/esm)
ProstT5: [mheinzinger/ProstT5](https://github.com/mheinzinger/ProstT5)

------

**4.Onedrive/Huggingface  description**

Due to GitHub's file size limit, we have uploaded the model weights and embeddings to OneDrive and the Hugging Face repository.

------

**5.How to Run****

Just run src/train.py to start training

```python
python train.py
```

