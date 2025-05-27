import torch
from e3nn.o3 import Irreps
from e3nn.util import jit
from layers import embedding as e
from layers import AtomwiseLinear
from layers import ConvNetLayer, InteractionBlock

from neuip_model import NL_model

import yaml
import sys

def gen_model(yaml_file, save=True):
    print(f"Yaml file: {yaml_file}")

    with open(yaml_file, "r") as stream:
        config = yaml.safe_load(stream)
    n_elems = len(config["elements"])

    layer_dict = {}
    node_embedding_dim_list = []

    # element node embedding onehot
    elem_embedding_layer = e.OneHotAtomEncoding(n_elems)
    node_attr_irrep = elem_embedding_layer.irreps_out
    layer_dict["elem_embed"] = elem_embedding_layer
    node_embedding_dim_list.append(elem_embedding_layer.irreps_out)

    # edge embeddings
    p = -1 if config["parity"] else 1
    edge_attr_irrep = Irreps([(1,(l,p**l)) for l in range(config["lmax"] + 1)])
    edge_embedding_layer = e.SphericalHarmonicEdgeAttrs(edge_attr_irrep)

    assert(edge_attr_irrep == edge_embedding_layer.irreps_out) # check if sh is same as requested
    layer_dict["edge_embed"] = edge_embedding_layer

    # Radial basis # to do rmin
    cutoff_kwargs = {"r_max": config["r_max"], "p": config["p"]}
    basis_kwargs = {"r_max": config["r_max"], "num_basis": config["n_radial_basis"]}
    radial_basis_layer = e.RadialBasisEdgeEncoding(basis_kwargs=basis_kwargs, cutoff_kwargs=cutoff_kwargs)
    edge_feature_irrep = radial_basis_layer.irreps_out
    layer_dict["radial_basis"] = radial_basis_layer

    # atomwise embedding
    node_feature_irrep = Irreps([(config["conv_feature_size"], (0,1))])
    first_atom_embedding = AtomwiseLinear(node_attr_irrep, node_feature_irrep)

    assert(node_feature_irrep == first_atom_embedding.irreps_out)
    layer_dict["first_atom_embed"] = first_atom_embedding
    node_embedding_dim_list.append(first_atom_embedding.irreps_out)

    # convolution layers
    node_feature_irrep_intermidiate = []
    conv_hidden_irrep = Irreps([(config["conv_feature_size"],(l,p)) for p in ((1, -1) if config["parity"] else (1,)) for l in range(config["lmax"] + 1)])
    invariant_neurons = config["radial_network_hidden_dim"]
    invariant_layers = config["radial_network_layers"]
    average_num_neigh = config["average_num_neigh"]
    conv_kw = {"invariant_layers": invariant_layers, "invariant_neurons": invariant_neurons, "avg_num_neighbors": average_num_neigh}

    last_node_irrep = node_feature_irrep

    for i in range(config["n_conv_layers"]):
        conv_layer = ConvNetLayer(last_node_irrep,
                conv_hidden_irrep,
                node_attr_irrep,
                edge_attr_irrep,
                edge_feature_irrep,
                convolution_kwargs=conv_kw)
        node_embedding_dim_list.append(conv_layer.irreps_out)
        layer_dict[f"conv:{i}"] = conv_layer
        last_node_irrep = conv_layer.irreps_out

    # final mappings
    for i in range(config["post_conv_layers"]):
        linear_post_layer = AtomwiseLinear(last_node_irrep, config["post_conv_irreps"][i])
        layer_dict[f"post_conv:{i}"] = linear_post_layer
        last_node_irrep = config["post_conv_irreps"][i]
        node_embedding_dim_list.append(last_node_irrep)

    model = NL_model(layer_dict, 
                    energy_scaling_coeff=config["scale"],
                    energy_shifting_coeff=config["shift"])
    print(model)
    if save:
        model = jit.script(model)
        model.save(f"{config['model_name']}.pt")
    return model


if __name__ == "__main__":
    if len(sys.argv)!=2:
        raise ValueError("Should provide only one argument, which is the yaml config file")    
    yaml_file = sys.argv[1]
    torch.manual_seed(42)
    model = gen_model(yaml_file)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model params: {pytorch_total_params}")

