import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from .spherenet import SphereNetEncoder
from .Nequip.nequip_encoder import NequipEncoder
from .modules import build_mlp
from .coordgen import CoordGen


class MatGen(torch.nn.Module):
    def __init__(self, enc_backbone_params, dec_backbone_params, latent_dim, num_fc_hidden_layers, fc_hidden_dim, max_num_atoms, min_num_atoms,
        num_time_steps, noise_start, noise_end, cutoff, num_node, max_num_neighbors, logvar_clip=6.0, mu_clip=14.0,
        use_gpu=True, lattice_scale=True, pred_prop=False, use_multi_latent=False, coord_loss_type='per_node', score_norm=None,
        use_node_num_loss=True, property_dim=21, backbone_name='spherenet',codebook_num=16,device='cuda'):

        super(MatGen, self).__init__()
        self.lattice_scale = lattice_scale

        if backbone_name == 'spherenet':
            self.encoder_backbone = SphereNetEncoder(**enc_backbone_params)
        elif backbone_name == 'nequip':
            self.encoder_backbone = NequipEncoder(**enc_backbone_params)
        elif backbone_name == 'transformer':
            self.encoder_backbone = NequipEncoder(**enc_backbone_params)
        self.use_node_num_loss = use_node_num_loss
        self.pred_prop = pred_prop

        self.min_num_atoms = min_num_atoms
        self.max_num_atoms = max_num_atoms
        self.num_node = num_node

        latent_in_dim = enc_backbone_params['out_channels']
        latent_out_dim = latent_dim
        if self.pred_prop and use_multi_latent:
            latent_out_dim += latent_dim
        if self.use_node_num_loss and use_multi_latent:
            latent_out_dim += latent_dim
        self.fc_mu = nn.Linear(latent_in_dim, latent_out_dim)
        self.fc_var = nn.Linear(latent_in_dim, latent_out_dim)
        self.fc_lattice_mu = build_mlp(6, fc_hidden_dim, num_fc_hidden_layers, latent_dim)
        self.fc_lattice_log_var = build_mlp(6, fc_hidden_dim, num_fc_hidden_layers, latent_dim)
        self.fc_lattice = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, 6)

        if self.use_node_num_loss:
            self.fc_node_num = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, max_num_atoms-min_num_atoms+1)

        self.coordgen = CoordGen(dec_backbone_params, latent_dim, num_fc_hidden_layers, fc_hidden_dim, num_time_steps,
            noise_start, noise_end, cutoff, max_num_neighbors, num_node, loss_type=coord_loss_type, use_gpu=use_gpu, score_norm=score_norm,
            property_loss=pred_prop, backbone_name=backbone_name, property_dim=property_dim,codebook_num=codebook_num,device=device)

        self.cutoff = cutoff

        if use_gpu:
            self.encoder_backbone = self.encoder_backbone.to('cuda')
            self.fc_mu = self.fc_mu.to('cuda')
            self.fc_var = self.fc_var.to('cuda')
            self.fc_lattice_mu = self.fc_lattice_mu.to('cuda')
            self.fc_lattice_log_var = self.fc_lattice_log_var.to('cuda')
            self.fc_lattice = self.fc_lattice.to('cuda')

            if self.use_node_num_loss:
                # self.fc_elem_type = self.fc_elem_type.to('cuda')
                self.fc_node_num = self.fc_node_num.to('cuda')
                # self.fc_elem_type_num = self.fc_elem_type_num.to('cuda')



        if pred_prop:
            self.fc_prop = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, property_dim)
            if use_gpu:
                self.fc_prop = self.fc_prop.to('cuda')


        self.latent_dim = latent_dim
        self.logvar_clip = logvar_clip
        self.mu_clip = mu_clip
        self.use_gpu = use_gpu
        self.prop_normalizer = None
        self.lattice_normalizer = None
        self.use_multi_latent = use_multi_latent

    def encode(self, data_batch, temp=[0.5, 0.5, 0.5, 0.01]):
        hidden = self.encoder_backbone(data_batch)
        mu, log_var = self.fc_mu(hidden), self.fc_var(hidden)
        mu.clip_(min=-self.mu_clip, max=self.mu_clip)
        log_var.clip_(max=self.logvar_clip)

        if not self.use_multi_latent:
            std = torch.exp(0.5 * log_var) * temp[0]
            latent = torch.randn_like(std) * std + mu
            latent_node_num, latent_pos, latent_prop = latent, latent, latent
        else:
            cur_id = 0
            if self.use_node_num_loss:
                std_comp = torch.exp(0.5 * log_var[:, cur_id: cur_id+ self.latent_dim]) * temp[0]
                latent_node_num = torch.randn_like(std_comp) * std_comp + mu[:, cur_id:cur_id+self.latent_dim]
                cur_id += self.latent_dim
                std_pos = torch.exp(0.5 * log_var[:, cur_id :cur_id + self.latent_dim]) * temp[1]
                latent_pos = torch.randn_like(std_pos) * std_pos + mu[:, cur_id :cur_id +  self.latent_dim]
                cur_id += self.latent_dim
            else:
                latent_node_num = None
                std_pos = torch.exp(0.5 * log_var[:, cur_id: cur_id + self.latent_dim]) * temp[1]
                latent_pos = torch.randn_like(std_pos) * std_pos + mu[:, cur_id: cur_id +  self.latent_dim]
                cur_id += self.latent_dim

            if self.pred_prop:
                std_prop = torch.exp(0.5 * log_var[:, cur_id: cur_id + self.latent_dim]) * temp[2]
                latent_prop = torch.randn_like(std_prop) * std_prop + mu[:, cur_id: cur_id + self.latent_dim]
            else:
                latent_prop = None

        scaled_lengths = data_batch.lengths
        lengths_angles = torch.cat([scaled_lengths, data_batch.angles], dim=-1)
        if self.lattice_normalizer is not None:
            lengths_angles = self.lattice_normalizer.transform(lengths_angles)
        mu_lattice, log_var_lattice = self.fc_lattice_mu(lengths_angles), self.fc_lattice_log_var(lengths_angles)
        mu_lattice.clip_(min=-self.mu_clip, max=self.mu_clip)
        log_var_lattice.clip_(max=self.logvar_clip)
        std_lattice = torch.exp(0.5 * log_var_lattice) * temp[3]
        latent_lattice = torch.randn_like(std_lattice) * std_lattice + mu_lattice



        return mu, log_var, mu_lattice, log_var_lattice, latent_node_num, latent_pos, latent_lattice, latent_prop


    def __topk_mask(self, input, k):
        sorted, _ = torch.sort(input, dim=-1, descending=True)
        thres = sorted[torch.arange(input.shape[0], device=input.device), k-1].view(-1, 1)
        return (input >= thres).long().to(input.device)
    

    def __match_composition(self, elem_type_topk, target_elem_type, elem_num_pred, target_elem_num, num_elem_per_mat):
        idx = 0
        elem_type_match_num, elem_num_match_num, match_num = 0, 0, 0
        for i in range(len(num_elem_per_mat)):
            idx2 = idx + num_elem_per_mat[i].long()
            elem_type_match = (elem_type_topk[i] == target_elem_type[i]).min(-1).values
            elem_num_match = (elem_num_pred[idx : idx2] == target_elem_num[idx : idx2]).min(-1).values
            elem_type_match_num += elem_type_match
            elem_num_match_num += elem_num_match
            match_num += elem_type_match * elem_num_match
            idx = idx2
        return elem_type_match_num, elem_num_match_num, match_num


    def forward(self, data_batch, temp=[0.5, 0.5, 0.5, 0.01], eval=False, distance_reg=0.1):
        #print(data_batch.cart_coords.shape)
        loss_dict = {}
        #mu, log_var, mu_lattice, log_var_lattice, latent_node_num, latent_pos, latent_lattice, latent_prop = self.encode(data_batch, temp)
        latent_prop, latent_pos = None, None

        if self.use_node_num_loss:
            pass
        else:
            loss_dict['node_num_loss'] = torch.tensor(0.0)


        #pred_lengths_angles = self.fc_lattice(latent_lattice)

        #loss_dict['kld_loss'] = torch.mean(-0.5 * torch.sum(1.0 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0) + \
        #    torch.mean(-0.5 * torch.sum(1.0 + log_var_lattice - mu_lattice ** 2 - log_var_lattice.exp(), dim=1), dim=0)
        loss_dict['coord_loss'], _, _ = self.coordgen(None, data_batch.num_atoms, data_batch.node_feat, data_batch.frac_coords,data_batch.cart_coords,
                data_batch.lengths, data_batch.angles, data_batch.batch, density=data_batch.density.unsqueeze(-1), cond=data_batch.y.view(-1,12),edge_index=data_batch.edge_index)

        # print(data_batch.cart_coords.shape)
        # print(data_batch.batch.shape)
        # input()
        cut_idx, edge_prob = self.coordgen.predict_edge(data_batch.cart_coords,data_batch.batch)

        target_edge = torch.zeros_like(edge_prob).view(-1)
        #for time efficiency
        # di, dj = data_batch.edge_index
        # if (di > dj).sum() != (dj > di).sum():
        #     di = torch.cat([di,dj], dim=-1)
        #     dj = torch.cat([dj,di], dim=-1)
        # idxi = (cut_idx[0]==di.unsqueeze(0).t()).nonzero().t()
        # find_idxa = (cut_idx[1,idxi[1]] == dj[idxi[0]]).nonzero().view(-1)
        # target_edge[idxi[1,find_idxa]] = 1.0
        edge_loss = self.bce_loss(edge_prob.view(-1), target_edge)
        loss_dict['edge_pred_loss'] = edge_loss

        return loss_dict

    def bce_loss(self, pred, target, reduction='mean'):
        pos = torch.eq(target, 1).float()
        neg = torch.eq(target, 0).float()
        num_pos = pos.sum()
        num_neg = neg.sum()
        num_total = num_pos + num_neg
        alpha_pos = num_pos / num_total
        alpha_neg = num_neg / num_total
        weight = alpha_pos * neg + alpha_neg * pos
        return F.binary_cross_entropy(pred, target, weight=weight, reduction=reduction)

    @torch.no_grad()
    def process(self, data_batch, task_choices,latent_dim=32):
        return self.coordgen.process(data_batch, task_choices,latent_dim=latent_dim)

    @torch.no_grad()
    def generate(self, cond=None,density=None,latent_dim=128):

        # if not self.use_multi_latent:
        #     if self.use_gpu:
        #         prior = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]).cuda(), temperature[0] * torch.ones([self.latent_dim]).cuda())
        #     else:
        #         prior = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]), temperature[0] * torch.ones([self.latent_dim]))

        #     latent = prior.sample([num_gen])
        #     latent_comp, latent_pos, latent_lattice = latent, latent, latent
        #     latent_prop = latent
        # else:
        #     if self.use_gpu:
        #         prior1 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]).cuda(), temperature[0] * torch.ones([self.latent_dim]).cuda())
        #         prior2 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]).cuda(), temperature[1] * torch.ones([self.latent_dim]).cuda())
        #         prior3 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]).cuda(), temperature[2] * torch.ones([self.latent_dim]).cuda())
        #         prior4 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]).cuda(), temperature[3] * torch.ones([self.latent_dim]).cuda())
        #     else:
        #         prior1 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]), temperature[0] * torch.ones([self.latent_dim]))
        #         prior2 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]), temperature[1] * torch.ones([self.latent_dim]))
        #         prior3 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]), temperature[2] * torch.ones([self.latent_dim]))
        #         prior4 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]), temperature[3] * torch.ones([self.latent_dim]))
        #     # latent_comp, latent_pos, latent_lattice = prior1.sample([num_gen]), prior2.sample([num_gen]), prior3.sample([num_gen])
            # latent_prop = prior4.sample([num_gen])


        # if self.pred_prop:
        #     prediected_prop = self.fc_prop(latent_prop)
        #     if self.prop_normalizer:
        #         prediected_prop = self.prop_normalizer.inverse_transform(prediected_prop)
        # else:
        #     prediected_prop = latent_prop

        # if self.use_node_num_loss:
        #     pred_node_num = self.fc_node_num(latent_comp)
        #     pred_node_num = pred_node_num.argmax(dim=-1) + self.min_num_atoms
        #     num_atoms = pred_node_num

        #     # num_atoms_each = [6,8,10,12]
        #     # num_atoms = []
        #     # while len(num_atoms) < num_gen:
        #     #     num_atoms = num_atoms + num_atoms_each
        #     # num_gen = min(num_gen, len(num_atoms))
        #     # num_atoms = num_atoms[:num_gen]
        #     # num_atoms = torch.LongTensor(num_atoms).to(latent_comp.device)

        #     atom_types = torch.zeros((num_gen, 1), dtype=torch.long).to(latent_comp.device)
        #     atom_types = torch.repeat_interleave(atom_types, num_atoms, dim=0)
        # else:
        #     if num_atoms is None:
        #         num_atoms = torch.randint(min_num_atom, max_num_atom, [num_gen]).to(latent_comp.device)
        #     num_atoms = num_atoms.to(latent_comp.device)
        #     atom_types = torch.zeros((num_gen, 1), dtype=torch.long).to(latent_comp.device)
        #     atom_types = torch.repeat_interleave(atom_types, num_atoms, dim=0)

        # num_atoms = torch.ones((num_gen,), dtype=torch.long ).to(latent_comp.device) * self.num_node
        # atom_types = torch.zeros((num_gen*self.num_node, 1), dtype=torch.long).to(latent_comp.device)

        # lengths_angles = self.fc_lattice(latent_lattice)
        # if self.lattice_normalizer is not None:
        #     lengths_angles = self.lattice_normalizer.inverse_transform(lengths_angles)
        # lengths, angles = lengths_angles[:, :3], lengths_angles[:, 3:]
        # if self.lattice_scale:
        #     lengths = lengths * num_atoms.view(-1, 1).float()**(1/3)
        
        frac_coords, edge_index = self.coordgen.generate(cond=cond,density=density,latent_dim=latent_dim)

        return frac_coords, edge_index

    @torch.no_grad()
    def recon(self, data_batch, num_gen, temperature=[0.5, 0.5, 0.5,0.5, 0.01], coord_noise_start=0.01, coord_noise_end=10,
                 coord_num_diff_steps=10, coord_num_langevin_steps=100, coord_step_rate=1e-4, num_atoms=None,
                 min_num_atom=5, max_num_atom=50, threshold=0.6):

        mu, log_var, mu_lattice, log_var_lattice, latent_comp, latent_pos, latent_lattice, latent_prop = self.encode(data_batch,
                                                                                                        temperature)


        if self.use_node_num_loss:
            pred_node_num = self.fc_node_num(latent_comp)
            pred_node_num = pred_node_num.argmax(dim=-1) + self.min_num_atoms
            num_atoms = pred_node_num
            atom_types = torch.zeros((num_gen, 1), dtype=torch.long).to(latent_pos.device)
            atom_types = torch.repeat_interleave(atom_types, num_atoms, dim=0)
        else:
            if num_atoms is None:
                num_atoms = torch.randint(min_num_atom, max_num_atom, [num_gen]).to(latent_comp.device)
            num_atoms = num_atoms.to(latent_pos.device)
            atom_types = torch.zeros((num_gen, 1), dtype=torch.long).to(latent_pos.device)
            atom_types = torch.repeat_interleave(atom_types, num_atoms, dim=0)

        lengths_angles = self.fc_lattice(latent_lattice)
        if self.lattice_normalizer is not None:
            lengths_angles = self.lattice_normalizer.inverse_transform(lengths_angles)
        lengths, angles = lengths_angles[:, :3], lengths_angles[:, 3:]
        if self.lattice_scale:
            lengths = lengths * num_atoms.view(-1, 1).float() ** (1 / 3)



        if self.pred_prop:
            prediected_prop = self.fc_prop(latent_prop)
            if self.prop_normalizer:
                prediected_prop = self.prop_normalizer.inverse_transform(prediected_prop)
        else:
            prediected_prop = latent_prop

        # cutoff_ind, edge_prob = self.predict_edge(latents, num_atoms, atom_types, lengths, angles, cart_coords, batch, latent_prop)
        # edge_prob = edge_prob.view(-1)
        # edge_index = cutoff_ind[:,edge_prob > threshold ]
        # edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        # lattice_vec = lattice_params_to_matrix_torch(lengths, angles)
        # edge_num_per_node = scatter(torch.ones(size=(edge_index.shape[1],), device=edge_index.device).long(),
        #                             edge_index[0], dim_size=len(batch), reduce='sum')
        # edge_num_per_graph = scatter(edge_num_per_node, batch, reduce='sum')
        # to_jimages = calculate_to_jimages_efficient(cart_coords, edge_index, lattice_vec, num_bonds=edge_num_per_graph).to(latents.device)


        frac_coords, edge_index = self.coordgen.generate(latent_pos, num_atoms, atom_types, lengths, angles,
                                                         coord_noise_start,
                                                         coord_noise_end, coord_num_diff_steps,
                                                         coord_num_langevin_steps, temperature[-1], coord_step_rate,
                                                         threshold=threshold, latent_prop=latent_prop, edge_index=None, to_jimages=None, num_bonds=None)

        num_atoms = torch.ones((num_gen, )) * self.num_node

        return num_atoms, atom_types, lengths, angles, frac_coords, edge_index, prediected_prop
        