import torch
from torch_scatter import scatter

class NL_model(torch.nn.Module):
    def __init__(self,
            layer_dict,
            energy_scaling_coeff=1.0,
            energy_shifting_coeff=0.0):
        super().__init__()
        self.node_attr_layer = layer_dict["elem_embed"]
        self.node_embedding_layer = layer_dict["first_atom_embed"]
        self.edge_attr_layer = layer_dict["edge_embed"]
        self.edge_embedding_layer = layer_dict["radial_basis"]
        self.conv_layers = torch.nn.ModuleList([layer_dict["conv:0"], layer_dict["conv:1"], layer_dict["conv:2"]])
        self.post_conv_layers = torch.nn.ModuleList([layer_dict["post_conv:0"],layer_dict["post_conv:1"]])
        self.register_buffer("energy_scaling_coeff", torch.tensor(energy_scaling_coeff))
        self.register_buffer("energy_shifting_coeff", torch.tensor(energy_shifting_coeff))


    def forward(self, x, pos, edge_index, period_vec, batch):
        # elem/node embedding
        x_attr = self.node_attr_layer(x.long().squeeze(-1))
        x_attr = x_attr.to(dtype=pos.dtype)
        # Linear map for embedding
        h = self.node_embedding_layer(x_attr)

        # Edge embedding
        edge_vec, edge_lengths, edge_sh = self.edge_attr_layer(pos, edge_index, period_vec)

        # Radial basis function
        edge_length_embedding = self.edge_embedding_layer(edge_lengths)

        # convolution
        for layer in self.conv_layers:
            h = layer(x_attr, h, edge_length_embedding, edge_sh, edge_index)

        # post convolution
        for layer in self.post_conv_layers:
            h = layer(h)

        h = h.squeeze()
        # energies, forces = h[:,0], h[:,1:4]
        # reduce energies per batch
        h = scatter(h, batch, dim=0)
        energies = h * self.energy_scaling_coeff + self.energy_shifting_coeff
        energies = energies.unsqueeze(-1)
        return energies