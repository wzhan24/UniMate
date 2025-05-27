import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.resolver import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from math import sqrt

from .features import dist_emb, angle_emb, torsion_emb
from .transformer_backbone import Encoder, Encoder_cond
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


try:
    import sympy as sym
except ImportError:
    sym = None


class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish, use_node_features=True):
        super(init, self).__init__()
        self.act = act
        self.use_node_features = use_node_features
        if self.use_node_features:
            self.emb = Embedding(100, hidden_channels)
        else: # option to use no node features and a learned embedding vector for each node instead
            self.node_embedding = nn.Parameter(torch.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf,_,_ = emb
        if self.use_node_features:
            x = self.emb(x).view(x.size(0), -1)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        cat_emb = torch.cat([x[i], x[j], rbf0], dim=-1)
        e1 = self.act(self.lin(cat_emb))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial,
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i):
        _, e2 = e
        v = scatter(e2, i, dim=0)
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v

class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    if fc_num_layers == 0:
        return nn.Linear(in_dim, out_dim)

    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

'''class transformer(torch.nn.Module):
    def __init__(self,d_model,n_head,ffn_hidden,n_layers,drop_prob):
        super(transformer, self).__init__()
        self.lin1 = nn.Linear(3,d_model)
        self.encoder = Encoder(d_model, ffn_hidden, n_head, n_layers, drop_prob)
        self.lin2 = nn.Linear(d_model,3)
        #self.num_node = num_node
        self.d_model = d_model

    def forward(self,x):
        x0 = self.lin1(x.view(x.shape[0],-1,3))
        x1 = self.encoder(x0)
        x2 = self.lin2(x1)
        return x2.view(x.shape[0],-1)'''
    
class transformer(torch.nn.Module):
    def __init__(self,num_node,d_model,n_head,ffn_hidden,n_layers,drop_prob):
        super(transformer, self).__init__()
        self.lin1 = nn.Linear(num_node*3,num_node*d_model)
        self.encoder = Encoder(d_model, ffn_hidden, n_head, n_layers, drop_prob)
        # self.lin2 = nn.Linear(d_model,out_dim)
        #self.num_node = num_node
        self.d_model = d_model

    def forward(self,x):
        x0 = self.lin1(x)
        x1 = self.encoder(x0.view(x0.shape[0],-1,self.d_model))
        # x2 = self.lin2(x1)
        return x1

class transformer_cond(torch.nn.Module):
    def __init__(self,num_node,d_model,n_head,ffn_hidden,n_layers,drop_prob, cond_dim=21):
        super(transformer_cond, self).__init__()
        self.lin1 = nn.Linear(num_node*3,num_node*d_model)
        self.encoder = Encoder_cond(d_model, ffn_hidden, n_head, n_layers, drop_prob, num_node, cond_dim=cond_dim)
        # self.lin2 = nn.Linear(d_model,3)
        #self.num_node = num_node
        self.d_model = d_model

    def forward(self,x,cond):
        x0 = self.lin1(x)
        x1 = self.encoder(x0.view(x0.shape[0],-1,self.d_model),cond=cond)
        # x2 = self.lin2(x1)
        return x1

class res_mlp(torch.nn.Module):
    def __init__(self,num_node):
        super(res_mlp, self).__init__()
        hidden = 128
        self.lin1 = nn.Linear(num_node*3+12,hidden)
        #self.lin1 = nn.Linear((num_node-4)*3,hidden)
        self.lin2 = nn.Linear(hidden,hidden)
        self.lin3 = nn.Linear(hidden,hidden)
        self.lin4 = nn.Linear(hidden,hidden)
        self.lin5 = nn.Linear(hidden,hidden)
        self.lin6 = nn.Linear(hidden,hidden)
        self.lin7 = nn.Linear(hidden,num_node*3+12)
        #self.lin7 = nn.Linear(hidden,(num_node-4)*3)
        
    def forward(self,x):
        x0 = self.lin1(x)
        x1 = nn.ReLU()(x0)
        x2 = self.lin2(x1)
        x3 = nn.ReLU()(x2)
        x4 = x2 + x3
        x5 = self.lin3(x4)
        x6 = nn.ReLU()(x5)
        x7 = x5 + x6
        x8 = self.lin4(x7)
        x9 = nn.ReLU()(x8)
        x10 = x8 + x9
        x11 = self.lin5(x10)
        x12 = nn.ReLU()(x11)
        x12 = x10 + x11
        x13 = self.lin6(x12)
        x14 = nn.ReLU()(x13)
        x15 = x13 + x14
        x16 = self.lin7(x15)
        return x16

class aggregate_to_node(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act=swish, output_init='GlorotOrthogonal'):
        super(aggregate_to_node, self).__init__()
        self.act = act
        self.output_init = output_init

        self.lin_input = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_input.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i, dist_vec, node_num=None):
        e = self.lin_input(e)
        for lin in self.lins:
            e = self.act(lin(e))
        e = self.lin(e)
        e = e * (dist_vec / torch.norm(dist_vec, dim=1, keepdim=True))

        if node_num is not None:
            v = scatter(e, i, dim=0, dim_size=node_num, reduce='add')
        else:
            v = scatter(e, i, dim=0, reduce='add')


        return v
    
class encoderT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(encoderT, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index=None, batch=None):
        if edge_index is None:
            edge_index = []
            for i in range(torch.max(batch).item()+1):
                indices = torch.where(batch == i)[0]
                #print(indices)
                first_index = indices[0].item()
                last_index = indices[-1].item()
                edge_index.append([first_index, last_index])
            #edge_index = torch.tensor([[i, j] for i in range(x.shape[0]) for j in range(x.shape[0]) if i != j and batch[i]==batch[j]], dtype=torch.long).t().to(x.device)
            edge_index = torch.tensor(edge_index).to(x.device).t()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

# class encoderT(torch.nn.Module):
#     def __init__(self,d_model,n_head,ffn_hidden,n_layers,drop_prob):
#         super(encoderT, self).__init__()
#         self.lin1 = nn.Linear(3,d_model)
#         self.encoder = Encoder(d_model, ffn_hidden, n_head, n_layers, drop_prob)
#         # self.lin2 = nn.Linear(d_model,3)
#         #self.num_node = num_node
#         self.d_model = d_model

#     def forward(self,x):
#         x = self.lin1(x)
#         x = self.encoder(x)
#         return x

class decoderT(torch.nn.Module):
    def __init__(self,d_model,n_head,ffn_hidden,n_layers,drop_prob):
        super(decoderT, self).__init__()
        self.lin1 = nn.Linear(d_model,3)
        self.encoder = Encoder(d_model, ffn_hidden, n_head, n_layers, drop_prob)
        # self.lin2 = nn.Linear(d_model,3)
        #self.num_node = num_node
        self.d_model = d_model

    def forward(self,x,mask):
        x = self.encoder(x,mask)
        x = self.lin1(x)
        return x

class encoderP(torch.nn.Module):
    def __init__(self,d_model,property_dim):
        super(encoderP, self).__init__()
        self.lin1 = nn.Linear(property_dim,d_model*property_dim)
        self.lin2 = nn.Linear(d_model,d_model)
        self.lin3 = nn.Linear(d_model,d_model)
        self.property_dim = property_dim
        self.d_model = d_model

    def forward(self,x):
        x = self.lin1(x).view(-1,self.property_dim,self.d_model)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        x = self.lin3(x)
        return x
    
class decoderP(torch.nn.Module):
    def __init__(self,d_model,property_dim,latent_dim=64):
        super(decoderP, self).__init__()
        self.lin1 = nn.Linear(d_model*property_dim,latent_dim)
        self.lin2 = nn.Linear(latent_dim,latent_dim)
        self.lin3 = nn.Linear(latent_dim,property_dim)

    def forward(self,x):
        x = self.lin1(x.view(x.shape[0],-1))
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        x = self.lin3(x)
        return x

class encoderD(torch.nn.Module):
    def __init__(self,d_model):
        super(encoderD, self).__init__()
        self.lin1 = nn.Linear(1,d_model)
        self.lin2 = nn.Linear(d_model,d_model)
        self.lin3 = nn.Linear(d_model,d_model)

    def forward(self,x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        x = self.lin3(x)
        return x

class decoderD(torch.nn.Module):
    def __init__(self,d_model):
        super(decoderD, self).__init__()
        self.lin1 = nn.Linear(d_model,d_model)
        self.lin2 = nn.Linear(d_model,d_model)
        self.lin3 = nn.Linear(d_model,1)

    def forward(self,x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        x = self.lin3(x)
        return x

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.param = nn.Parameter(torch.randn(3, 3))

    def forward(self, x):
        # Example usage of the parameter
        return x @ self.param
    
class Codebook(torch.nn.Module):
    def __init__(self, num_tokens, token_dim, device):
        super(Codebook, self).__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.device = device
        self.codebook = nn.Parameter(torch.randn(num_tokens, token_dim)/10).to(device)

    def forward(self, x, mask=None, dropout=0.1):
        # Calculate the distance between x and each token in the codebook
        ori_shape = x.shape
        num_code = int(self.num_tokens * (1-dropout))
        random_indices = torch.randperm(self.num_tokens)[:num_code]
        random_indices = torch.sort(random_indices)[0]
        if len(ori_shape) == 2:
            ori_shape = torch.tensor([ori_shape[0],1,ori_shape[1]])
        x = x.view(-1,x.shape[-1])
        distances = torch.argmin(torch.norm(x.unsqueeze(1) - self.codebook[random_indices].unsqueeze(0),dim=-1),dim=1)
        distances = distances.view(ori_shape[0],-1)
        #print(distances)
        #print(random_indices)
        closest_indices = random_indices[distances.cpu()]
        closest_tokens = self.codebook[closest_indices]
        if mask is not None:
            rounding_loss = torch.sum(((closest_tokens - x.view(-1,ori_shape[1],x.shape[-1]))*mask) ** 2)/torch.sum(mask)
        else:
            rounding_loss = torch.mean((closest_tokens - x.view(-1,ori_shape[1],x.shape[-1])) ** 2)
        return closest_tokens, rounding_loss, closest_indices
    
    def cal_M(self, eps=1e-3):
        norms = torch.norm(self.codebook,dim=-1)
        cos_sim = torch.abs(torch.sum(self.codebook.unsqueeze(0)*self.codebook.unsqueeze(1),dim=-1))/(norms.unsqueeze(0)*norms.unsqueeze(1)+1e-8)
        C = cos_sim.unsqueeze(0) + cos_sim.unsqueeze(1) + cos_sim.unsqueeze(-1)
        M = torch.exp(-C/eps)
        return C, M
    
    def cal_TWD(self, FT, FD, FP, eps=1e-3, iter=10):
        C, M = self.cal_M(eps)
        u = torch.ones(self.num_tokens).to(self.device)
        v = torch.ones(self.num_tokens).to(self.device)
        w = torch.ones(self.num_tokens).to(self.device)
        for i in range(iter):
            u = FT/(torch.matmul(torch.matmul(M,w),v)+1e-8)
            v = FD/(torch.matmul(u,torch.matmul(M,w))+1e-8)
            w = FP/(torch.matmul(v,torch.matmul(u,M))+1e-8)
        trans_plan = torch.einsum('i,j,k->ijk', u, v, w) * M
        dw = torch.sum(C*trans_plan)
        return trans_plan, dw

    '''def cal_freq(self, indexT, indexD, indexP):
        FT = torch.zeros(self.num_tokens).to(self.device)
        FD = torch.zeros(self.num_tokens).to(self.device)
        FP = torch.zeros(self.num_tokens).to(self.device)
        for i in range(self.num_tokens):
            FT[i] += torch.sum(indexT==i)
            FD[i] += torch.sum(indexD==i)
            FP[i] += torch.sum(indexP==i)
        # FT = torch.bincount(indexT, minlength=self.num_tokens).float().to(self.device)
        # FD = torch.bincount(indexD, minlength=self.num_tokens).float().to(self.device)
        # FP = torch.bincount(indexP, minlength=self.num_tokens).float().to(self.device)
        return FT, FD, FP'''
    
    def cal_TWD_from_index(self,indexT, indexD, indexP,eps=1e-3,iter=10):
        FT = torch.zeros(self.num_tokens).to(self.device)
        FD = torch.zeros(self.num_tokens).to(self.device)
        FP = torch.zeros(self.num_tokens).to(self.device)
        for i in range(self.num_tokens):
            FT[i] += torch.sum(indexT==i)
            FD[i] += torch.sum(indexD==i)
            FP[i] += torch.sum(indexP==i)
        C, M = self.cal_M(eps)
        u = torch.ones(self.num_tokens).to(self.device)
        v = torch.ones(self.num_tokens).to(self.device)
        w = torch.ones(self.num_tokens).to(self.device)
        for i in range(iter):
            u = FT/(torch.matmul(torch.matmul(M,w),v)+1e-8)
            v = FD/(torch.matmul(u,torch.matmul(M,w))+1e-8)
            w = FP/(torch.matmul(v,torch.matmul(u,M))+1e-8)
        trans_plan = torch.einsum('i,j,k->ijk', u, v, w) * M
        dw = torch.sum(C*trans_plan)
        return trans_plan, dw

    
class TOT(torch.nn.Module):
    def __init__(self, num_tokens, token_dim, num_heads, num_layers):
        super(TOT, self).__init__()
        self.codebook = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.encoder = Encoder(token_dim, num_heads, num_layers)

    def forward(self, x):
        # Calculate the distance between x and each token in the codebook
        distances = torch.cdist(x, self.codebook)
        # Find the indices of the closest tokens
        closest_indices = torch.argmin(distances, dim=-1)
        # Retrieve the closest tokens
        closest_tokens = self.codebook[closest_indices]
        rounding_loss = torch.mean((closest_tokens - x) ** 2)
        return self.encoder(closest_tokens), rounding_loss
    

class PFtransformer(torch.nn.Module):
    def __init__(self,d_model,n_head,ffn_hidden,n_layers,drop_prob):
        super(PFtransformer, self).__init__()
        self.encoder = Encoder(d_model, ffn_hidden, n_head, n_layers, drop_prob)

    def forward(self,x,mask=None):
        x = self.encoder(x,mask)
        return x
    
