import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

# Dependency checks with cleaner warnings
try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Mamba not installed. Please run: pip install mamba-ssm causal-conv1d")

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    raise ImportError("torch_geometric not installed. Please run: pip install torch_geometric")


def get_pad_mask(embedding_tensor, pad_value=0):
    """Generates a padding mask based on the embedding tensor."""
    pad_mask = torch.all(embedding_tensor == pad_value, dim=-1)
    return pad_mask


class DyGeo_GAT_Module(nn.Module):
    """
    Dynamic Geometric Attention (DyGeo-GAT) Module.
    Constructs dynamic graphs based on feature similarity and distance,
    then applies GATv2 convolution.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.edge_feature_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads)
        )

        self.gatv2_conv = GATv2Conv(
            in_channels=embed_dim,
            out_channels=self.head_dim,
            heads=num_heads,
            concat=True,
            edge_dim=num_heads,
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        residual = x

        # Dynamic Adjacency Construction
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        adjacency = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        adjacency = adjacency.mean(dim=1)

        if mask is not None:
            valid_mask_2d = (mask.unsqueeze(1) * mask.unsqueeze(2))
            fill_value = torch.finfo(adjacency.dtype).min
            adjacency.masked_fill_(~valid_mask_2d, fill_value)

        # Edge Feature Generation
        dist_matrix = torch.cdist(x, x, p=2)
        x_norm = F.normalize(x, p=2, dim=-1)
        sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2))
        edge_attr_dense = self.edge_feature_mlp(torch.stack([dist_matrix, sim_matrix], dim=-1))

        # PyG Batch Construction
        data_list = []
        for i in range(batch_size):
            valid_nodes = mask[i].sum().item() if mask is not None else seq_len
            adj_i = adjacency[i, :valid_nodes, :valid_nodes]
            edge_attr_i = edge_attr_dense[i, :valid_nodes, :valid_nodes]

            edge_index_i, _ = dense_to_sparse(adj_i)
            edge_attr_sparse_i = edge_attr_i[edge_index_i[0], edge_index_i[1]]

            data_list.append(Data(
                x=x[i, :valid_nodes],
                edge_index=edge_index_i,
                edge_attr=edge_attr_sparse_i
            ))

        pyg_batch = Batch.from_data_list(data_list)

        # GATv2 Convolution
        gat_output_flat = self.gatv2_conv(pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr)

        # Restore Batch Structure
        output = torch.zeros_like(x)
        node_counts = [int(data.num_nodes) for data in data_list]
        gat_outputs_split = torch.split(gat_output_flat, node_counts)

        for i, out_split in enumerate(gat_outputs_split):
            num_nodes = out_split.shape[0]
            output[i, :num_nodes] = out_split

        output = residual + self.dropout(output)
        output = self.layer_norm(output)

        if mask is not None:
            output = output.masked_fill(~mask.unsqueeze(-1), 0.0)

        return output, adjacency


class BottleneckAttentionBlock(nn.Module):
    """
    Bottleneck Attention Block incorporating both Self-Attention and Cross-Attention.
    """

    def __init__(self, embed_dim, modality_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, kdim=modality_dim, vdim=modality_dim,
                                                dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, queries, modality_features, modality_mask=None):
        # Self-Attention
        q = self.norm1(queries)
        self_attn_out, _ = self.self_attn(q, q, q)
        queries = queries + self.dropout1(self_attn_out)

        # Cross-Attention
        q = self.norm2(queries)
        cross_attn_out, _ = self.cross_attn(
            query=q,
            key=modality_features,
            value=modality_features,
            key_padding_mask=modality_mask
        )
        queries = queries + self.dropout2(cross_attn_out)

        # FFN
        q = self.norm3(queries)
        ffn_out = self.ffn(q)
        queries = queries + self.dropout3(ffn_out)

        return queries


class QDF_Net(nn.Module):
    """
    Querying-Distillation Fusion Network (QDF-Net).
    Fuses sequence and structure features using learnable queries.
    """

    def __init__(self, mamba_dim, struct_dim, num_queries, num_heads, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, struct_dim))

        self.seq_distiller = BottleneckAttentionBlock(struct_dim, mamba_dim, num_heads, dropout)
        self.struct_distiller = BottleneckAttentionBlock(struct_dim, struct_dim, num_heads, dropout)

        self.broadcaster = nn.MultiheadAttention(struct_dim, num_heads, kdim=struct_dim, vdim=struct_dim,
                                                 dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(struct_dim)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, x_seq, x_struct, mask=None):
        B = x_seq.size(0)
        queries = self.queries.repeat(B, 1, 1)
        key_padding_mask = ~mask if mask is not None else None

        # Distill information from both modalities
        queries = self.seq_distiller(queries, x_seq, key_padding_mask)
        queries = self.struct_distiller(queries, x_struct, key_padding_mask)

        # Broadcast fused information back to structure features
        broadcast_out, _ = self.broadcaster(
            query=x_struct,
            key=queries,
            value=queries
        )

        x_fused = self.norm_out(x_struct + self.dropout_out(broadcast_out))

        if mask is not None:
            x_fused = x_fused.masked_fill(~mask.unsqueeze(-1), 0.0)

        return x_fused


class QuICPTMsCore(nn.Module):
    """
    Core module for QuICPTMs.
    Encapsulates Mamba (Sequence), DyGeo-GAT (Structure), and QDF-Net (Fusion).
    """

    def __init__(self, esm_dim, str_dim, gnn_heads, d_state, d_conv, expand, dropout, device, num_queries):
        super().__init__()
        self.mamba = Mamba(
            d_model=esm_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dga = DyGeo_GAT_Module(embed_dim=str_dim, num_heads=gnn_heads, dropout=dropout)
        self.fusion = QDF_Net(
            mamba_dim=esm_dim,
            struct_dim=str_dim,
            num_queries=num_queries,
            num_heads=gnn_heads,
            dropout=dropout
        )

        self.seq_classifier_head = nn.Linear(esm_dim, 2)
        self.struct_classifier_head = nn.Linear(str_dim, 2)
        self.final_fused_dim = str_dim

    def forward(self, esm_embedding, str_nodes, mask, seq_lengths):
        # Feature Extraction
        seq_features = self.mamba(esm_embedding)
        str_features, dygeo_adyn = self.dga(str_nodes, mask=mask)

        # Fusion
        fused_features = self.fusion(seq_features, str_features, mask=mask)

        # Auxiliary Classification Heads
        seq_rep = seq_features.sum(dim=1) / seq_lengths
        str_rep = str_features.sum(dim=1) / seq_lengths

        logits_seq = self.seq_classifier_head(seq_rep)
        logits_struct = self.struct_classifier_head(str_rep)

        return fused_features, logits_seq, logits_struct, seq_features, str_features, dygeo_adyn


class QuICPTMs(nn.Module):
    """
    QuICPTMs: Query-based Integration of Context for Post-Translational Modifications.
    """

    def __init__(self, config):
        super(QuICPTMs, self).__init__()

        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        STR_EMBEDDING_DIM = 1024
        ESM_EMBEDDING_DIM = config.dim_embedding
        DROPOUT = config.dropout

        self.core = QuICPTMsCore(
            esm_dim=ESM_EMBEDDING_DIM,
            str_dim=STR_EMBEDDING_DIM,
            gnn_heads=config.gnn_heads,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            dropout=DROPOUT,
            device=device,
            num_queries=config.fusion_num_queries
        )

        FUSED_DIM = self.core.final_fused_dim

        self.classifier = nn.Sequential(
            nn.Linear(FUSED_DIM, FUSED_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(FUSED_DIM // 2, 2)
        )

    def forward(self, esm_embedding, str_nodes):
        esm_embedding = esm_embedding.to(torch.float32)
        str_nodes = str_nodes.to(torch.float32)

        pad_mask = get_pad_mask(esm_embedding)
        valid_nodes_mask = ~pad_mask
        seq_lengths = valid_nodes_mask.sum(dim=1).unsqueeze(-1).float().clamp(min=1.0)

        fused_features, logits_seq, logits_struct, seq_features, str_features, dygeo_adyn = self.core(
            esm_embedding,
            str_nodes,
            mask=valid_nodes_mask,
            seq_lengths=seq_lengths
        )

        pad_mask_expanded = pad_mask.unsqueeze(-1)
        fused_features = fused_features.masked_fill(pad_mask_expanded, 0.0)
        fused_embeddings = fused_features.sum(dim=1) / seq_lengths

        logits_fused = self.classifier(fused_embeddings)

        return {
            'logits_fused': logits_fused,
            'fused_embeddings': fused_embeddings,
            'logits_seq': logits_seq,
            'logits_struct': logits_struct,
            'seq_features': seq_features,
            'str_features': str_features,
            'dygeo_adyn': dygeo_adyn
        }