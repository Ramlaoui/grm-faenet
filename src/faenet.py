import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .gnn_utils import MessagePassing, GraphNorm, GaussianSmearing, dropout_edge, radius_graph, get_pbc_distances, scatter
from .modules.phys_embedding import PhysEmbedding

def swish(x):
    return x * torch.sigmoid(x)

class EmbeddingBlock(nn.Module):
    """Embedding block for the GNN
    Initialize nodes and edges' embeddings"""

    def __init__(
        self,
        num_gaussians,
        num_filters,
        hidden_channels,
        tag_hidden_channels,
        pg_hidden_channels,
        phys_embeds,
    ):
        super().__init__()
        self.use_tag = tag_hidden_channels > 0
        self.use_pg = pg_hidden_channels > 0

        # --- Node embedding ---

        # Phys embeddings
        self.phys_emb = PhysEmbedding(
            props=phys_embeds, props_grad=False, pg=self.use_pg
        )
        # With MLP
        phys_hidden_channels = self.phys_emb.n_properties

        # Period + group embeddings
        if self.use_pg:
            self.period_embedding = nn.Embedding(
                self.phys_emb.period_size, pg_hidden_channels
            )
            self.group_embedding = nn.Embedding(
                self.phys_emb.group_size, pg_hidden_channels
            )

        # Tag embedding
        if tag_hidden_channels:
            self.tag_embedding = nn.Embedding(3, tag_hidden_channels)

        # Main embedding
        self.emb = nn.Embedding(
            85,
            hidden_channels
            - tag_hidden_channels
            - phys_hidden_channels
            - 2 * pg_hidden_channels,
        )

        # MLP
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        # --- Edge embedding ---
        self.lin_e1 = nn.Linear(3, num_filters // 2)  # r_ij
        self.lin_e12 = nn.Linear(num_gaussians, num_filters - (num_filters // 2))  # d_ij

        self.emb.reset_parameters()
        if self.use_tag:
            self.tag_embedding.reset_parameters()
        if self.use_pg:
            self.period_embedding.reset_parameters()
            self.group_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e1.weight)
        self.lin_e1.bias.data.fill_(0)

    def forward(self, z, rel_pos, edge_attr, tag=None, subnodes=None):
        # --- Edge embedding --
        rel_pos = self.lin_e1(rel_pos)  # r_ij
        edge_attr = self.lin_e12(edge_attr)  # d_ij
        e = torch.cat((rel_pos, edge_attr), dim=1)
        e = swish(e)  # can comment out

        # --- Node embedding --

        # Create atom embeddings based on its characteristic number
        h = self.emb(z)

        if self.phys_emb.device != h.device:
            self.phys_emb = self.phys_emb.to(h.device)

        # Concat tag embedding
        if self.use_tag:
            h_tag = self.tag_embedding(tag)
            h = torch.cat((h, h_tag), dim=1)

        # Concat physics embeddings
        if self.phys_emb.n_properties > 0:
            h_phys = self.phys_emb.properties[z]
            h = torch.cat((h, h_phys), dim=1)

        # Concat period & group embedding
        if self.use_pg:
            h_period = self.period_embedding(self.phys_emb.period[z])
            h_group = self.group_embedding(self.phys_emb.group[z])
            h = torch.cat((h, h_period, h_group), dim=1)

        # MLP
        h = swish(self.lin(h))

        return h, e

class InteractionBlock(MessagePassing):
    """Interaction block for the GNN
    Updates node representations based on the message passing scheme"""

    def __init__(
        self,
        hidden_channels,
        num_filters,
        dropout_lin,
    ):
        super(InteractionBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.dropout_lin = float(dropout_lin)

        self.graph_norm = GraphNorm(
            hidden_channels 
        )

        self.lin_geom = nn.Linear(
            num_filters + 2 * hidden_channels, hidden_channels
        )
        self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        self.other_mlp = nn.Linear(hidden_channels, hidden_channels)

        nn.init.xavier_uniform_(self.lin_geom.weight)
        self.lin_geom.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.other_mlp.weight)
        self.other_mlp.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_h.weight)
        self.lin_h.bias.data.fill_(0)

    def forward(self, h, edge_index, e):
        # Define edge embedding

        if self.dropout_lin > 0:
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training
            )

        e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)

        e = swish(self.lin_geom(e))

        # --- Message Passing block --
        h = self.propagate(edge_index, x=h, W=e)  # propagate
        h = swish(self.graph_norm(h))
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training
        )
        h = swish(self.lin_h(h))

        h = F.dropout(
            h, p=self.dropout_lin, training=self.training
        )
        h = swish(self.other_mlp(h))

        return h

    def message(self, x_j, W, local_env=None):
        if local_env is not None:
            return W
        else:
            return x_j * W

class OutputBlock(nn.Module):
    def __init__(self, hidden_channels, dropout_lin):
        super().__init__()
        self.dropout_lin = float(dropout_lin)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

        self.w_lin = nn.Linear(hidden_channels, 1)

    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.lin1.weight)
    #     self.lin1.bias.data.fill_(0)
    #     nn.init.xavier_uniform_(self.lin2.weight)
    #     self.lin2.bias.data.fill_(0)
    #     nn.init.xavier_uniform_(self.w_lin.weight)
    #     self.w_lin.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, batch, data=None):
        alpha = self.w_lin(h)

        # MLP
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training
        )
        h = self.lin1(h)
        h = swish(h)
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training
        )
        h = self.lin2(h)

        h = h * alpha

        # Global pooling
        out = scatter(h, batch, dim=0, reduce="add")

        return out

class FAENet(nn.Module):
    r"""Frame Averaging GNN model FAENet.

    Args:
        cutoff (float): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        use_pbc (bool): Use of periodic boundary conditions.
            (default: `True`)
        max_num_neighbors (int): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: `40`)
        hidden_channels (int): Hidden embedding size.
            (default: `128`)
        tag_hidden_channels (int): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int): Hidden period and group embedding size.
            (default: :obj:`32`)
        num_interactions (int): The number of interaction (i.e. message passing) blocks.
            (default: :obj:`4`)
        num_gaussians (int): The number of gaussians :math:`\mu` to encode distance info.
            (default: :obj:`50`)
        num_filters (int): The size of convolutional filters.
            (default: :obj:`128`)
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        use_pbc: bool = True,
        max_num_neighbors: int = 40,
        num_gaussians: int = 50,
        num_filters: int = 128,
        hidden_channels: int = 128,
        tag_hidden_channels: int = 32,
        pg_hidden_channels: int = 32,
        phys_embeds: bool = True,
        num_interactions: int = 4,
        **kwargs,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.hidden_channels = hidden_channels
        self.max_num_neighbors = max_num_neighbors
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.num_interactions = num_interactions
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_embeds = phys_embeds
        self.tag_hidden_channels = tag_hidden_channels
        self.use_pbc = use_pbc

        self.dropout_edge = float(kwargs.get("dropout_edge") or 0)
        self.dropout_lin = float(kwargs.get("dropout_lin") or 0)

        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)

        # Embedding block
        self.embed_block = EmbeddingBlock(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.tag_hidden_channels,
            self.pg_hidden_channels,
            self.phys_embeds,
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    self.hidden_channels,
                    self.num_filters,
                    0
                )
                for i in range(self.num_interactions)
            ]
        )

        # Output block
        self.output_block = OutputBlock(
            self.hidden_channels,
            self.dropout_lin
        )

        # Skip co
        self.mlp_skip_co = nn.Linear((self.num_interactions + 1), 1)

    def forward(self, data):
        """Predicts any graph-level properties (e.g. energy) for 3D atomic systems.

        Args:
            data (data.Batch): Batch of graphs datapoints.
        Returns:
            dict: predicted properties for each graph (e.g. energy)
        """
        # Rewire the graph
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        energy_skip_co = []

        if self.use_pbc and hasattr(data, "cell"):
            assert z.dim() == 1 and z.dtype == torch.long

            if self.dropout_edge > 0:
                edge_index, edge_mask = dropout_edge(
                    data.edge_index,
                    p=self.dropout_edge,
                    training=self.training
                )

            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            rel_pos = out["distance_vec"]
            edge_attr = self.distance_expansion(edge_weight)
        else: # why is that an else?
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )
            rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
            edge_weight = rel_pos.norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)
            if self.dropout_edge > 0:
                edge_index, edge_mask = dropout_edge(
                    edge_index,
                    p=self.dropout_edge,
                    training=self.training
                )
                edge_weight = edge_weight[edge_mask]
                edge_attr = edge_attr[edge_mask]
                rel_pos = rel_pos[edge_mask]

        h, e = self.embed_block(z, rel_pos, edge_attr, data.tags)

        energy_skip_co = []
        for ib, interaction in enumerate(self.interaction_blocks):
            energy_skip_co.append(
                self.output_block(
                    h, edge_index, edge_weight, batch, data
                )
            )
            h = interaction(h, edge_index, e)

        energy = self.output_block(h, edge_index, edge_weight, batch, data=data)

        energy_skip_co.append(energy)
        energy = self.mlp_skip_co(torch.cat(energy_skip_co, dim=1))

        preds = {
            "energy": energy,
            "hidden_state": h,
        }

        return preds
