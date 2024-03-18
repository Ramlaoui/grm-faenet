import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .gnn_utils import MessagePassing, GraphNorm, GaussianSmearing, dropout_edge, radius_graph, get_pbc_distances, scatter
from .modules.phys_embedding import PhysEmbedding

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
        phys_hidden_channels,
        phys_embeds,
        act,
        second_layer_MLP,
    ):
        super().__init__()
        self.act = act
        self.use_tag = tag_hidden_channels > 0
        self.use_pg = pg_hidden_channels > 0
        self.use_mlp_phys = phys_hidden_channels > 0 and phys_embeds
        self.second_layer_MLP = second_layer_MLP

        # --- Node embedding ---

        # Phys embeddings
        self.phys_emb = PhysEmbedding(
            props=phys_embeds, props_grad=phys_hidden_channels > 0, pg=self.use_pg
        )
        # With MLP
        if self.use_mlp_phys:
            self.phys_lin = nn.Linear(self.phys_emb.n_properties, phys_hidden_channels)
        else:
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
        if self.second_layer_MLP:
            self.lin_2 = nn.Linear(hidden_channels, hidden_channels)

        # --- Edge embedding ---
        self.lin_e1 = nn.Linear(3, num_filters // 2)  # r_ij
        self.lin_e12 = nn.Linear(num_gaussians, num_filters - (num_filters // 2))  # d_ij

        if self.second_layer_MLP:
            self.lin_e2 = nn.Linear(num_filters, num_filters)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        if self.use_mlp_phys:
            nn.init.xavier_uniform_(self.phys_lin.weight)
        if self.use_tag:
            self.tag_embedding.reset_parameters()
        if self.use_pg:
            self.period_embedding.reset_parameters()
            self.group_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e1.weight)
        self.lin_e1.bias.data.fill_(0)
        if self.second_layer_MLP:
            nn.init.xavier_uniform_(self.lin_2.weight)
            self.lin_2.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_e2.weight)
            self.lin_e2.bias.data.fill_(0)

    def forward(self, z, rel_pos, edge_attr, tag=None, subnodes=None):
        # --- Edge embedding --
        rel_pos = self.lin_e1(rel_pos)  # r_ij
        edge_attr = self.lin_e12(edge_attr)  # d_ij
        e = torch.cat((rel_pos, edge_attr), dim=1)
        e = self.act(e)  # can comment out

        if self.second_layer_MLP:
            # e = self.lin_e2(e)
            e = self.act(self.lin_e2(e))

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
            if self.use_mlp_phys:
                h_phys = self.phys_lin(h_phys)
            h = torch.cat((h, h_phys), dim=1)

        # Concat period & group embedding
        if self.use_pg:
            h_period = self.period_embedding(self.phys_emb.period[z])
            h_group = self.group_embedding(self.phys_emb.group[z])
            h = torch.cat((h, h_period, h_group), dim=1)

        # MLP
        h = self.act(self.lin(h))
        if self.second_layer_MLP:
            h = self.act(self.lin_2(h))

        return h, e

class InteractionBlock(MessagePassing):
    """Interaction block for the GNN
    Updates node representations based on the message passing scheme"""

    def __init__(
        self,
        hidden_channels,
        num_filters,
        act,
        mp_type,
        complex_mp,
        graph_norm,
        dropout_lin,
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.mp_type = mp_type
        self.hidden_channels = hidden_channels
        self.complex_mp = complex_mp
        self.graph_norm = graph_norm
        self.dropout_lin = float(dropout_lin)
        if graph_norm:
            self.graph_norm = GraphNorm(
                hidden_channels if "updown" not in self.mp_type else num_filters
            )

        if self.mp_type == "simple":
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        elif self.mp_type == "updownscale":
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updownscale_base":
            self.lin_geom = nn.Linear(num_filters + 2 * hidden_channels, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updown_local_env":
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_up = nn.Linear(2 * num_filters, hidden_channels)

        else:  # base
            self.lin_geom = nn.Linear(
                num_filters + 2 * hidden_channels, hidden_channels
            )
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        if self.complex_mp:
            self.other_mlp = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.mp_type != "simple":
            nn.init.xavier_uniform_(self.lin_geom.weight)
            self.lin_geom.bias.data.fill_(0)
        if self.complex_mp:
            nn.init.xavier_uniform_(self.other_mlp.weight)
            self.other_mlp.bias.data.fill_(0)
        if self.mp_type in {"updownscale", "updownscale_base", "updown_local_env"}:
            nn.init.xavier_uniform_(self.lin_up.weight)
            self.lin_up.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_down.weight)
            self.lin_down.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.lin_h.weight)
            self.lin_h.bias.data.fill_(0)

    def forward(self, h, edge_index, e):
        # Define edge embedding

        if self.dropout_lin > 0:
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training
            )

        if self.mp_type in {"base", "updownscale_base"}:
            e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)

        if self.mp_type in {
            "updownscale",
            "base",
            "updownscale_base",
        }:
            e = self.act(self.lin_geom(e))

        # --- Message Passing block --

        if self.mp_type == "updownscale" or self.mp_type == "updownscale_base":
            h = self.act(self.lin_down(h))  # downscale node rep.
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training
            )
            h = self.act(self.lin_up(h))  # upscale node rep.

        elif self.mp_type == "updown_local_env":
            h = self.act(self.lin_down(h))
            chi = self.propagate(edge_index, x=h, W=e, local_env=True)
            e = F.dropout(
                e, p=self.dropout_lin, training=self.training
            )
            e = self.lin_geom(e)
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = torch.cat((h, chi), dim=1)
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training
            )
            h = self.lin_up(h)

        elif self.mp_type in {"base", "simple"}:
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training
            )
            h = self.act(self.lin_h(h))

        else:
            raise ValueError("mp_type provided does not exist")

        if self.complex_mp:
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training
            )
            h = self.act(self.other_mlp(h))

        return h

    def message(self, x_j, W, local_env=None):
        if local_env is not None:
            return W
        else:
            return x_j * W

class OutputBlock(nn.Module):
    def __init__(self, energy_head, hidden_channels, act, dropout_lin):
        super().__init__()
        self.energy_head = energy_head
        self.act = act
        self.dropout_lin = float(dropout_lin)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

        if self.energy_head == "weighted-av-final-embeds":
            self.w_lin = nn.Linear(hidden_channels, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.energy_head == "weighted-av-final-embeds":
            nn.init.xavier_uniform_(self.w_lin.weight)
            self.w_lin.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, batch, alpha, data=None):
        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        # MLP
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training
        )
        h = self.lin1(h)
        h = self.act(h)
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training
        )
        h = self.lin2(h)

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
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
        act (str): Activation function
            (default: `swish`)
        max_num_neighbors (int): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: `40`)
        hidden_channels (int): Hidden embedding size.
            (default: `128`)
        tag_hidden_channels (int): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int): Hidden period and group embedding size.
            (default: :obj:`32`)
        phys_embeds (bool): Do we include fixed physics-aware embeddings.
            (default: :obj: `True`)
        phys_hidden_channels (int): Hidden size of learnable physics-aware embeddings.
            (default: :obj:`0`)
        num_interactions (int): The number of interaction (i.e. message passing) blocks.
            (default: :obj:`4`)
        num_gaussians (int): The number of gaussians :math:`\mu` to encode distance info.
            (default: :obj:`50`)
        num_filters (int): The size of convolutional filters.
            (default: :obj:`128`)
        second_layer_MLP (bool): Use 2-layers MLP at the end of the Embedding block.
            (default: :obj:`False`)
        skip_co (str): Add a skip connection between each interaction block and
            energy-head. (`False`, `"add"`, `"concat"`, `"concat_atom"`)
        mp_type (str): Specificies the Message Passing type of the interaction block.
            (`"base"`, `"updownscale_base"`, `"updownscale"`, `"updown_local_env"`, `"simple"`):
        graph_norm (bool): Whether to apply batch norm after every linear layer.
            (default: :obj:`True`)
        complex_mp (bool); Whether to add a second layer MLP at the end of each Interaction
            (default: :obj:`True`)
        energy_head (str): Method to compute energy prediction
            from atom representations.
            (`None`, `"weighted-av-initial-embeds"`, `"weighted-av-final-embeds"`)
        regress_forces (str): Specifies if we predict forces or not, and how
            do we predict them. (`None` or `""`, `"direct"`, `"direct_with_gradient_target"`)
        force_decoder_type (str): Specifies the type of force decoder
            (`"simple"`, `"mlp"`, `"res"`, `"res_updown"`)
        force_decoder_model_config (dict): contains information about the
            for decoder architecture (e.g. number of layers, hidden size).
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        act: str = "swish",
        use_pbc: bool = True,
        complex_mp: bool = False,
        max_num_neighbors: int = 40,
        num_gaussians: int = 50,
        num_filters: int = 128,
        hidden_channels: int = 128,
        tag_hidden_channels: int = 32,
        pg_hidden_channels: int = 32,
        phys_hidden_channels: int = 0,
        phys_embeds: bool = True,
        num_interactions: int = 4,
        mp_type: str = "updownscale_base",
        graph_norm: bool = True,
        second_layer_MLP: bool = True,
        skip_co: str = "concat",
        energy_head: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.act = act
        self.complex_mp = complex_mp
        self.cutoff = cutoff
        self.energy_head = energy_head
        self.graph_norm = graph_norm
        self.hidden_channels = hidden_channels
        self.max_num_neighbors = max_num_neighbors
        self.mp_type = mp_type
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.num_interactions = num_interactions
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_embeds = phys_embeds
        self.phys_hidden_channels = phys_hidden_channels
        self.second_layer_MLP = second_layer_MLP
        self.skip_co = skip_co
        self.tag_hidden_channels = tag_hidden_channels
        self.use_pbc = use_pbc

        self.dropout_edge = float(kwargs.get("dropout_edge") or 0)
        self.dropout_lin = float(kwargs.get("dropout_lin") or 0)
        self.dropout_lowest_layer = kwargs.get("dropout_lowest_layer", "output") or ""
        self.first_trainable_layer = kwargs.get("first_trainable_layer", "") or ""

        if self.mp_type == "simple":
            self.num_filters = self.hidden_channels

        self.act = (
            (getattr(nn.functional, self.act) if self.act != "swish" else lambda x: torch.sigmoid(x) * x)
            if isinstance(self.act, str)
            else self.act
        )
        assert callable(self.act), (
            "act must be a callable function or a string "
            + "describing that function in torch.nn.functional"
        )

        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)

        # Embedding block
        self.embed_block = EmbeddingBlock(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.tag_hidden_channels,
            self.pg_hidden_channels,
            self.phys_hidden_channels,
            self.phys_embeds,
            self.act,
            self.second_layer_MLP,
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    self.hidden_channels,
                    self.num_filters,
                    self.act,
                    self.mp_type,
                    self.complex_mp,
                    self.graph_norm,
                    (
                        (
                            print(
                                f"🗑️ Setting dropout_lin for interaction block to {self.dropout_lin} ",
                                f"{i} / {self.num_interactions}",
                            )
                            or self.dropout_lin
                        )
                        if "inter" in self.dropout_lowest_layer
                        and (i >= int(self.dropout_lowest_layer.split("-")[-1]))
                        else 0
                    ),
                )
                for i in range(self.num_interactions)
            ]
        )

        # Output block
        self.output_block = OutputBlock(
            self.energy_head,
            self.hidden_channels,
            self.act,
            (
                (
                    print(
                        f"🗑️ Setting dropout_lin for output block to {self.dropout_lin}"
                    )
                    or self.dropout_lin
                )
                if (
                    "inter" in self.dropout_lowest_layer
                    or "output" in self.dropout_lowest_layer
                )
                else 0
            ),
        )

        # Energy head
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin = nn.Linear(self.hidden_channels, 1)

        # Skip co
        if self.skip_co == "concat":
            self.mlp_skip_co = nn.Linear((self.num_interactions + 1), 1)
        elif self.skip_co == "concat_atom":
            self.mlp_skip_co = nn.Linear(
                ((self.num_interactions + 1) * self.hidden_channels),
                self.hidden_channels,
            )

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

        # Use periodic boundary conditions
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
        else:
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )
            # edge_index = data.edge_index
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

        # Embedding block
        h, e = self.embed_block(z, rel_pos, edge_attr, data.tags)

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)
        else:
            alpha = None

        # Interaction blocks
        energy_skip_co = []
        for ib, interaction in enumerate(self.interaction_blocks):
            if self.skip_co == "concat_atom":
                energy_skip_co.append(h)
            elif self.skip_co:
                energy_skip_co.append(
                    self.output_block(
                        h, edge_index, edge_weight, batch, alpha, data
                    )
                )
            h = interaction(h, edge_index, e)

        # Atom skip-co
        if self.skip_co == "concat_atom":
            energy_skip_co.append(h)
            h = self.act(self.mlp_skip_co(torch.cat(energy_skip_co, dim=1)))


        energy = self.output_block(h, edge_index, edge_weight, batch, alpha, data=data)

        # Skip-connection
        energy_skip_co.append(energy)
        if self.skip_co == "concat":
            energy = self.mlp_skip_co(torch.cat(energy_skip_co, dim=1))
        elif self.skip_co == "add":
            energy = sum(energy_skip_co)

        preds = {
            "energy": energy,
            "hidden_state": h,
        }

        return preds
