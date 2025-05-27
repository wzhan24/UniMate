import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from torch_cluster import radius_graph
from torch_scatter import scatter
from tqdm import tqdm
from traitlets import Int

#from .features import device
from .spherenet_light import SphereNetLightDecoder
from .Nequip.nequip_decoder import NequipDecoder
from .modules import build_mlp, aggregate_to_node, res_mlp, transformer, transformer_cond, PFtransformer, encoderT, decoderT, encoderP, decoderP, encoderD, decoderD, Codebook
import sys
sys.path.append("..")
from utils import get_pbc_cutoff_graphs, frac_to_cart_coords, cart_to_frac_coords, correct_cart_coords, \
    get_pbc_distances, align_gt_cart_coords, calculate_to_jimages_efficient, lattice_params_to_matrix_torch

EPS = 1e-8

class CoordGen(torch.nn.Module):
    def __init__(self, backbone_params, latent_dim, num_fc_hidden_layers, fc_hidden_dim, num_time_steps, noise_start, noise_end, cutoff, max_num_neighbors, num_node, 
                 loss_type='per_node', score_upper_bound=None, use_gpu=True, score_norm=None,
                 property_loss=False, backbone_name='spherenet', property_dim=21, codebook_num=16,device='cuda'):
        super(CoordGen, self).__init__()
        self.device = device
        self.property_loss = property_loss
        self.latent_dim = latent_dim
        if backbone_name == 'spherenet':
            self.backbone = SphereNetLightDecoder(**backbone_params)
        elif backbone_name == 'nequip':
            self.backbone = NequipDecoder(**backbone_params)
        elif backbone_name == 'transformer':
            self.backbone = PFtransformer(d_model=latent_dim, n_head=8, ffn_hidden=latent_dim, n_layers=3,
                                        drop_prob=0.1).to(self.device)
            #self.encoderT = encoderT(d_model=latent_dim, n_head=8, ffn_hidden=latent_dim, n_layers=3,drop_prob=0.1).to(self.device)
            self.encoderT = encoderT(in_channels=3,hidden_channels=latent_dim,out_channels=latent_dim).to(self.device)
            self.decoderT = decoderT(d_model=latent_dim, n_head=8, ffn_hidden=latent_dim, n_layers=3,drop_prob=0.1).to(self.device)
            self.encoderP = encoderP(d_model=latent_dim, property_dim=property_dim).to(self.device)
            self.decoderP = decoderP(d_model=latent_dim, property_dim=property_dim).to(self.device)
            self.encoderD = encoderD(d_model=latent_dim).to(self.device)
            self.decoderD = decoderD(d_model=latent_dim).to(self.device)
            self.codebook = Codebook(num_tokens=codebook_num,token_dim=latent_dim,device=device)
            self.codebook_num = codebook_num
        if not property_loss:
            property_dim = 0
        else:
            property_dim = latent_dim
        #self.fc_score = build_mlp(latent_dim + backbone_params['hidden_channels']+ property_dim, fc_hidden_dim, num_fc_hidden_layers, 1) #ini
        #self.fc_score = res_mlp(num_node)
        # self.fc_score = transformer(num_node=num_node,d_model=128,n_head=8,ffn_hidden=128,n_layers=3,drop_prob=0.1)
        self.fc_score = build_mlp(latent_dim, fc_hidden_dim, fc_num_layers=0, out_dim=3)
        # self.fc_score = aggregate_to_node(latent_dim + backbone_params['hidden_channels']+ property_dim, fc_hidden_dim, 3, num_fc_hidden_layers)
        if backbone_name == 'spherenet':
            self.edge_pred = SphereNetLightDecoder(**backbone_params)
        elif backbone_name == 'nequip':
            self.edge_pred = NequipDecoder(**backbone_params)
        elif backbone_name == 'transformer':
            self.edge_pred = transformer_cond(num_node=num_node, d_model=latent_dim, n_head=8, ffn_hidden=128, n_layers=3,
                                        drop_prob=0.1)

        # TODO: fc_edge_lin not used in current version
        self.fc_edge_lin = build_mlp(latent_dim, fc_hidden_dim, 1, latent_dim)

        self.fc_edge_prob = build_mlp(latent_dim, fc_hidden_dim, 0, latent_dim)
        self.binlin = nn.Bilinear(latent_dim, latent_dim, 1)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_node = num_node
        self.use_gpu = use_gpu
        self.score_norms = None
        if score_norm is not None:
            self.score_norms = torch.from_numpy(score_norm).float()
        
        sigmas = self._get_noise_params(num_time_steps, noise_start, noise_end)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)
        self.num_time_steps = num_time_steps
        self.pool = mp.Pool(16)
        self.score_upper_bound = score_upper_bound
        self.loss_type = loss_type
        if use_gpu:
            self.backbone = self.backbone.to(self.device)
            self.fc_score = self.fc_score.to(self.device)
            self.sigmas.data = self.sigmas.to(self.device)
            self.edge_pred = self.edge_pred.to(self.device)
            self.fc_edge_prob = self.fc_edge_prob.to(self.device)
            self.fc_edge_lin = self.fc_edge_lin.to(self.device)
            self.binlin = self.binlin.to(self.device)
            if score_norm is not None:
                self.score_norms = self.score_norms.to(self.device)
        self.create_posi_enc(dim=latent_dim,length=37)


    def _get_noise_params(self, num_time_steps, noise_start, noise_end):
        log_sigmas = np.linspace(np.log(noise_start), np.log(noise_end), num_time_steps)
        sigmas = np.exp(log_sigmas)
        sigmas = torch.from_numpy(sigmas).float()
        return sigmas

    def _center_coords(self, coords, batch):
        coord_center = scatter(coords, batch, reduce='mean', dim=0)
        return coords - coord_center[batch]

    '''def shrink_coords(self,ini_coords,batch_size):
        reshaped_coords = ini_coords.view(batch_size,-1,3)
        num_node = reshaped_coords.shape[1]
        result = torch.zeros((batch_size,5,3))
        for i in range(batch_size):
            corners, selected_corners = self.find_corners_index(reshaped_coords[i,:,:])
            for j in range(4):
                result[i,j,:] = reshaped_coords[i,selected_corners[j],:]
            position = 4
            for j in range(num_node):
                if j not in corners:
                    result[i,position,:] = reshaped_coords[i,j,:]
                    position += 1
        return result
        
    def find_corners_index(self,coords):
        corners = []  #index of 4 corners
        center = torch.mean(coords,dim=0)
        for i in range(9):
            if torch.norm(coords[i,:] - center) < 1e-6:
                continue
            corners.append(i)
        selected_corners = [0,1] #4 selected corners to represent the lattice
        one_axis = coords[0,:] - coords[1,:]
        for i in range(2,8):
            for j in range(2,8):
                if torch.norm((coords[i,:] - coords[j,:]) - one_axis) < 1e-6:
                    if torch.norm((coords[0,:]+coords[1,:]+coords[i,:]+coords[j,:])/4 - center) > 1e-6:
                        selected_corners.append(i)
                        for k in range(2,8):
                            if torch.norm(coords[j,:] + coords[k,:] - 2*center) < 1e-6:
                                selected_corners.append(k)
                                return corners, selected_corners
        return corners, [0,1,2,3]'''
    
    def find_corners_index(self,coords):
        corners = []  #index of 4 corners
        center = torch.mean(coords,dim=0)
        for i in range(self.num_node):
            if torch.norm(coords[i,:] - center) < 1e-6:
                continue
            corners.append(i)
        selected_corners = [corners[0]] #4 selected corners to represent the lattice
        for i in range(1,8):
            for j in range(1,8):
                for k in range(1,8):
                    if torch.norm((coords[corners[0],:]-coords[corners[i],:]-coords[corners[j],:]+coords[corners[k],:])) < 1e-6:
                        if torch.norm((coords[corners[0],:]+coords[corners[i],:]+coords[corners[j],:]+coords[corners[k],:])-4*center) > 1e-6:
                            for h in range(1,8):
                                if torch.norm((coords[corners[k],:]+coords[corners[h],:])-2*center) < 1e-6:
                                    return corners, selected_corners + [corners[i],corners[j],corners[h]]
        return corners, [0,1,2,3]

    def shrink_coords(self,ini_coords,batch_size):
            reshaped_coords = ini_coords.view(batch_size,-1,3)
            num_node = reshaped_coords.shape[1]
            result = torch.zeros((batch_size,5,3))
            for i in range(batch_size):
                corners, selected_corners = self.find_corners_index(reshaped_coords[i,:,:])
                #print(corners)
                #print(selected_corners)
                for j in range(4):
                    result[i,j,:] = reshaped_coords[i,selected_corners[j],:]
                position = 4
                for j in range(num_node):
                    if j not in corners:
                        result[i,position,:] = reshaped_coords[i,j,:]
                        position += 1
            return result.view(-1,3)

    def expend_coords(self,shrinked_coords):
        num_gen = shrinked_coords.shape[0]
        coords_to_add = torch.zeros((num_gen,4,3)).to(self.device)
        coords_to_add[:,0,:] = shrinked_coords[:,1,:] + shrinked_coords[:,2,:] - shrinked_coords[:,0,:]
        coords_to_add[:,1,:] = shrinked_coords[:,2,:] + shrinked_coords[:,3,:] - shrinked_coords[:,0,:]
        coords_to_add[:,2,:] = shrinked_coords[:,3,:] + shrinked_coords[:,1,:] - shrinked_coords[:,0,:]
        coords_to_add[:,3,:] = coords_to_add[:,0,:] + shrinked_coords[:,3,:] - shrinked_coords[:,0,:]
        return torch.cat([shrinked_coords,coords_to_add],dim=1)


    def forward(self, latents, num_atoms, atom_types, gt_frac_coords, gt_cart_coords, lengths, angles, batch, density=None, edge_index=None, to_jimages=None, num_bonds=None, distance_reg=None, latent_prop=None, cond=None):
        torch.set_printoptions(threshold=torch.inf)
        padded_coords, pad_mask = self.padding(gt_cart_coords, batch)
        #tokensT = self.encoderT(padded_coords.view(-1,padded_coords.shape[-1]),edge_index=edge_index).view(padded_coords.shape[0],-1,self.latent_dim)
        tokensT = self.encoderT(gt_cart_coords.view(-1,gt_cart_coords.shape[-1]),edge_index=edge_index)
        tokensT, pad_mask = self.padding_tokens(tokensT, batch)
        rounded_tokensT, loss_roundT,indexT = self.codebook(tokensT,pad_mask)
        recon_T = self.decoderT(rounded_tokensT*pad_mask,pad_mask)
        #recon_T = self.decoderT(tokensT*pad_mask,pad_mask)
        tokensP = self.encoderP(cond)
        rounded_tokensP, loss_roundP,indexP = self.codebook(tokensP)
        reconP = self.decoderP(rounded_tokensP)
        #reconP = self.decoderP(tokensP)
        tokensD = self.encoderD(density)
        rounded_tokensD, loss_roundD,indexD = self.codebook(tokensD)
        reconD = self.decoderD(rounded_tokensD)
        #reconD = self.decoderD(tokensD)
        #ori_input = torch.cat([gt_cart_coords.view(200,-1),torch.sign(cond)*(torch.log(torch.abs(cond)+1e-8)/19+1)], dim=-1)
        ori_input = torch.cat([rounded_tokensT,rounded_tokensP,rounded_tokensD], dim=1)
        #ori_input = torch.cat([tokensT,tokensP,tokensD.unsqueeze(1)], dim=1) + self.posi_enc
        #ori_input = torch.cat([tokensT,tokensP,tokensD.unsqueeze(1)], dim=1)
        #print(pad_mask)
        #input()
        trans_plan, dw = self.codebook.cal_TWD_from_index(indexT,indexD,indexP)

        mask = torch.zeros((ori_input.shape[0], 24+12+1,1), requires_grad=False).float().to(self.device)
        choices = torch.randint(0, 3, (ori_input.shape[0],))
        #for time efficiency:
        delta_len = ori_input.shape[0]//3
        choices[:delta_len] = 0
        choices[delta_len:2*delta_len] = 1
        choices[2*delta_len:] = 2
        
        mask[choices==0,:24] = 1
        mask[choices==1,24:-1] = 1
        mask[choices==2,-1] = 1
        #num_graphs = batch[-1].item() + 1
        num_graphs = ori_input.shape[0]
        batch_size = num_graphs
        #assert batch.shape[0] // num_graphs == self.num_node, 'args num_node is inconsistent with input data'
        time_steps = torch.randint(0, self.num_time_steps, size=(num_graphs,), device=self.device)

        sigmas_per_graph = self.sigmas.index_select(0, time_steps)
        sigmas_per_node = sigmas_per_graph.index_select(0, batch).view(-1,1)
        # gt_cart_coords = correct_cart_coords(gt_cart_coords, lengths, angles, num_atoms, batch)
        # gt_cart_coords = frac_to_cart_coords(gt_frac_coords, lengths, angles, num_atoms)
        
        #shrinked_coords = self.shrink_coords(gt_cart_coords,batch_size).to(self.device)
        cart_coords_noise = torch.randn_like(ori_input)*mask

        cart_coords_perturbed = ori_input + sigmas_per_graph.unsqueeze(dim=-1).unsqueeze(dim=-1) * cart_coords_noise
        #cart_coords_perturbed = correct_cart_coords(cart_coords_perturbed, lengths, angles, num_atoms, batch)
        #cart_coords_perturbed = cart_coords_perturbed.reshape((batch_size,-1))
        #fc_score = self.backbone(cart_coords_perturbed, mask)
        #fc_score = self.backbone(cart_coords_perturbed, mask) - self.posi_enc
        fc_score = self.backbone(cart_coords_perturbed, mask)
        diff_tokens = cart_coords_perturbed + sigmas_per_graph.unsqueeze(dim=-1).unsqueeze(dim=-1) * fc_score*mask
        recon_diffT = self.decoderT(diff_tokens[:,:24,:]*pad_mask,pad_mask)
        recon_diffP = self.decoderP(diff_tokens[:,24:-1,:])
        recon_diffD = self.decoderD(diff_tokens[:,-1,:])
        stamp = torch.mean((fc_score*mask + cart_coords_noise)**2)*mask.shape[0]*mask.shape[1]/torch.sum(mask)
        score_loss = F.mse_loss(fc_score*mask,-cart_coords_noise)
        score_loss += F.mse_loss(padded_coords*pad_mask,recon_T*pad_mask)
        score_loss += F.mse_loss(padded_coords*pad_mask,recon_diffT*pad_mask)
        score_loss += 2*F.mse_loss(cond, reconP)
        score_loss += 2*F.mse_loss(cond, recon_diffP)
        score_loss += F.mse_loss(density, reconD.squeeze(-1))
        score_loss += F.mse_loss(density, recon_diffD.squeeze(-1))
        score_loss += 0.00000005*torch.sum(ori_input**2)
        score_loss += 0.1*(loss_roundT + loss_roundP + loss_roundD)
        score_loss += dw
        # print(torch.mean(ori_input[torch.where(mask==1)[0]]))
        # print(torch.var(ori_input[torch.where(mask==1)[0]]))
        # print(stamp)
        # print(F.mse_loss(fc_score*mask,-cart_coords_noise))
        # print(ori_input[0][0])
        # print(fc_score[0][0])
        # print(cart_coords_noise[0][0])
        #input()
        # print(F.mse_loss(padded_coords*pad_mask,recon_T*pad_mask))
        # print(F.mse_loss(padded_coords*pad_mask,recon_diffT*pad_mask))
        # print(2*F.mse_loss(cond, reconP))
        # print(2*F.mse_loss(cond, recon_diffP))
        # print(F.mse_loss(density, reconD.squeeze(-1)))
        # print(F.mse_loss(density, recon_diffD.squeeze(-1)))
        #input()
        return score_loss, 0.0, 0.0

    def padding(self, gt_cart_coords, batch):
        batch_size = batch[-1].item() + 1
        padded_coords = torch.zeros((batch_size, 24, 3)).to(self.device)
        pad_mask = torch.zeros((batch_size, 24, 1)).to(self.device)
        for i in range(batch_size):
            index = torch.where(batch == i)[0]
            padded_coords[i, :index.shape[0]] = gt_cart_coords[index]
            pad_mask[i, :index.shape[0]] = 1
        return padded_coords, pad_mask
    
    def padding_tokens(self, gt_cart_coords, batch):
        batch_size = torch.max(batch).item() + 1
        padded_coords = torch.zeros((batch_size, 24, self.latent_dim)).to(self.device)
        pad_mask = torch.zeros((batch_size, 24, 1)).to(self.device)
        for i in range(batch_size):
            index = torch.where(batch == i)[0]
            padded_coords[i, :index.shape[0]] = gt_cart_coords[index]
            pad_mask[i, :index.shape[0]] = 1
        return padded_coords, pad_mask

    def process(self, data_batch, task_choices, num_langevin_steps=100, num_gen_steps=10, coord_temp=0.01, latent_dim=64):
        padded_coords, pad_mask = self.padding(data_batch.cart_coords, data_batch.batch)
        property = data_batch.y.view(-1,12)
        density = data_batch.density
        edge_index = data_batch.edge_index
        batch_size = task_choices.shape[0]
        tokens = torch.zeros((batch_size, 37, latent_dim)).float().to(self.device)
        mask = torch.zeros((batch_size,37,1)).float().to(self.device)
        freq = torch.zeros(self.codebook_num).to(self.device)
        freq_matrix = torch.zeros(self.codebook_num,self.codebook_num).to(self.device)
        for i in range(len(task_choices)):
            if task_choices[i] == 0: #coords
                tokensT = torch.randn((1,24,latent_dim)).to(self.device)
                tokensP = self.encoderP(property[i].unsqueeze(dim=0))
                tokensP,_,index0 = self.codebook(tokensP,dropout=0.0)
                freq[index0] += 1
                tokensD = self.encoderD(density[i].unsqueeze(dim=0)).unsqueeze(0).unsqueeze(0)
                tokensD,_,index1 = self.codebook(tokensD,dropout=0.0)
                #print(index)
                freq[index1] += 1
                #tokens[i] = torch.cat([tokensT,tokensP,tokensD], dim=1) + self.posi_enc
                tokens[i] = torch.cat([tokensT,tokensP,tokensD], dim=1)
                mask[i,:24] = 1
            elif task_choices[i] == 1: #property
                edges = self.get_edges(data_batch.batch, edge_index, i)
                #print(edges)
                #input()
                #tokensT = self.encoderT(padded_coords[i].unsqueeze(dim=0),edge_index=edges_index[i])
                tokensT = self.encoderT(padded_coords[i],edge_index=edges).unsqueeze(0)
                tokensT,_,index = self.codebook(tokensT,dropout=0.0)
                freq[index] += 1
                #print(index)
                tokensP = torch.randn((1,12,latent_dim)).to(self.device)
                tokensD = self.encoderD(density[i].unsqueeze(dim=0)).unsqueeze(0).unsqueeze(0)
                tokensD,_,index = self.codebook(tokensD,dropout=0.0)
                freq[index] += 1
                #tokens[i] = torch.cat([tokensT,tokensP,tokensD], dim=1) + self.posi_enc
                tokens[i] = torch.cat([tokensT,tokensP,tokensD], dim=1)
                mask[i,24:-1] = 1
            else: #density
                edges = self.get_edges(data_batch.batch, edge_index, i)
                #tokensT = self.encoderT(padded_coords[i].unsqueeze(dim=0),edge_index=edge_index[i])
                tokensT = self.encoderT(padded_coords[i],edge_index=edges).unsqueeze(0)
                tokensT,_,index0 = self.codebook(tokensT,dropout=0.0)
                freq[index0] += 1
                tokensP = self.encoderP(property[i].unsqueeze(dim=0))
                tokensP,_,index1 = self.codebook(tokensP,dropout=0.0)
                freq[index1] += 1
                for j in index0.squeeze():
                    for k in index1.squeeze():
                        freq_matrix[j,k] += 1
                #freq_matrix[index0.squeeze(),index1.squeeze()] += 1
                tokensD = torch.randn((1,1,latent_dim)).to(self.device)
                #tokens[i] = torch.cat([tokensT,tokensP,tokensD], dim=1) + self.posi_enc
                tokens[i] = torch.cat([tokensT,tokensP,tokensD], dim=1)
                mask[i,-1] = 1
        #print(freq)
        #input()
        sigmas = self.sigmas
        sigmas = torch.cat([torch.zeros([1], device=sigmas.device), sigmas])
        for t in tqdm(range(num_gen_steps, 0, -1)):
            #current_alpha = step_rate * (sigmas[t] / sigmas[1]) ** 2 #ini
            #current_alpha = step_rate
            current_alpha = 0.00000001*(sigmas[t] / sigmas[1])**2
            for _ in range(num_langevin_steps):
                scores_per_node_pos = self.backbone(tokens, mask)
                tokens += (current_alpha * scores_per_node_pos + (10 * current_alpha).sqrt() * (coord_temp * torch.randn_like(tokens))) * tokens

        # tokens -= self.posi_enc
        #tokens,_,_ = self.codebook(tokens)#useround
        coords = self.decoderT(tokens[:,:24,:]*pad_mask,pad_mask)
        property = self.decoderP(tokens[:,24:-1,:])
        density = self.decoderD(tokens[:,-1,:])
        #print(coords[0])
        #input()
        return coords, property, density, pad_mask, freq, freq_matrix
    
    def get_edges(self, batch, edge_index, sample_index):
        indices = torch.where(batch == sample_index)[0]
        first_index = indices[0].item()
        last_index = indices[-1].item()
        indices = torch.where((edge_index >= first_index) & (edge_index <= last_index))
        #print(indices)
        left_index = torch.min(indices[1]).item()
        right_index = torch.max(indices[1]).item()
        return edge_index[:,left_index:right_index] - first_index



    def get_score_norm(self, sigma):
        sigma_min, sigma_max = self.sigmas[0], self.sigmas[-1]
        sigma_index = (torch.log(sigma) - torch.log(sigma_min)) / (torch.log(sigma_max) - torch.log(sigma_min)) * (len(self.sigmas) - 1)
        sigma_index = torch.round(torch.clip(sigma_index, 0, len(self.sigmas)-1)).long()
        return self.score_norms[sigma_index]

    def predict_edge(self, cart_coords, batch, train=True):
        # cut_off_edge_index, distance_vectors, pbc_offsets = get_pbc_cutoff_graphs(cart_coords,
        #                                                                           lengths, angles,
        #                                                                           num_atoms, self.cutoff,
        #                                                                           self.max_num_neighbors)
        max_num_neighbors=self.num_node
        radius=5
        #padded_coords, pad_mask = self.padding(cart_coords, batch)
        cut_off_edge_index = radius_graph(cart_coords, radius, batch=batch, loop=False, max_num_neighbors=max_num_neighbors)
        # distance_vectors = cart_coords[cut_off_edge_index[1]] - cart_coords[cut_off_edge_index[0]]
        node_emb = self.encoderT(cart_coords, edge_index=None, batch=batch)
        #print(node_emb.shape)
        node_emb, pad_mask = self.padding_tokens(node_emb, batch)
        #print(node_emb.shape)
        #node_emb = node_emb[:,:72].view(batch_size,24,3)
        node_emb = self.fc_edge_prob(node_emb).view(-1,node_emb.shape[-1])
        #print(node_emb.shape)
        flattened_pad_mask = pad_mask.view(-1,1).squeeze()
        if train:
            node_emb = node_emb[flattened_pad_mask>0].view(batch.shape[0], -1) #correct
        else:
            node_emb = node_emb.view(batch.shape[0], -1)
        #print(node_emb[cut_off_edge_index[0]].shape)
        #print(node_emb[cut_off_edge_index[1]].shape)
        #input()
        edge_prob = self.binlin(node_emb[cut_off_edge_index[0]], node_emb[cut_off_edge_index[1]])
        edge_prob = F.sigmoid(edge_prob)
        return cut_off_edge_index, edge_prob

    def predict_pos_score(self, cart_coords, mask=None):
        return self.backbone(cart_coords,mask)



    @torch.no_grad()
    def generate(self, num_gen_steps=10, num_langevin_steps=100, coord_temp=0.01, threshold=0.6, cond=None, density=None,latent_dim=128):
        
        num_node = 17
        sigmas = self.sigmas
        sigmas = torch.cat([torch.zeros([1], device=sigmas.device), sigmas])
        tokensP = self.encoderP(cond)
        rounded_tokensP, _, _ = self.codebook(tokensP,dropout=0.0)
        tokensD = self.encoderD(density).unsqueeze(1)
        rounded_tokensD, _, _ = self.codebook(tokensD,dropout=0.0)
        tokensT = torch.randn((cond.shape[0],24,latent_dim)).to(self.device)
        mask = torch.zeros((cond.shape[0],37,1)).float().to(self.device)
        mask[:,:num_node] = 1
        mask[:,24:] = 1
        #tokens = torch.cat([tokensT,rounded_tokensP,rounded_tokensD], dim=1) + self.posi_enc*mask
        tokens = torch.cat([tokensT,rounded_tokensP,rounded_tokensD], dim=1)
        #tokens = torch.cat([tokensT,tokensP,tokensD], dim=1)
        mask[:,24:] = 0

        # num_of_samples_gen = num_atoms.shape[0]
        # batch = torch.repeat_interleave(torch.arange(num_of_samples_gen, device=num_atoms.device), self.num_node)
        # #frac_coords_init = torch.rand(size=(batch.shape[0], 3), device=lengths.device) - 0.5
        # #frac_coords_init = torch.rand(size=(angles.shape[0], 45), device=lengths.device) - 0.5
        # # frac_coords_init = 10*(torch.rand((num_of_samples_gen,self.num_node,3),device='cuda') - 0.5)
        # frac_coords_init = torch.distributions.normal.Normal(torch.zeros([self.num_node,3]),
        #                                   10*torch.ones([self.num_node,3])).sample([num_of_samples_gen]).to(self.device)
        # #cart_coords_init = frac_to_cart_coords(frac_coords_init, lengths, angles, num_atoms)
        # cart_coords_init = frac_coords_init

        # cart_coords = torch.cat([cart_coords_init.view(num_of_samples_gen,-1),cond],dim=-1)
        
        for t in tqdm(range(num_gen_steps, 0, -1)):
            #current_alpha = step_rate * (sigmas[t] / sigmas[1]) ** 2 #ini
            #current_alpha = step_rate
            current_alpha = 0.00000005*(sigmas[t] / sigmas[1])**2
            for _ in range(num_langevin_steps):
                scores_per_node_pos = self.predict_pos_score(tokens,mask)
                #print('adfjkahsdfkj')
                #print(current_alpha)
                #print(scores_per_node_pos)
                #input()
                tokens[:,:num_node] += (current_alpha * scores_per_node_pos + (10 * current_alpha).sqrt() * (coord_temp * torch.randn_like(tokens)))[:,:num_node]
                #cart_coords += current_alpha * scores_per_node_pos
                #cart_coords += current_alpha * scores_per_node_pos + (2 * current_alpha).sqrt() * (coord_temp * torch.randn_like(cart_coords))
                #cart_coords = correct_cart_coords(cart_coords, lengths, angles, num_atoms, batch)
                

        
        batch = torch.zeros((cond.shape[0],24),dtype=torch.long).to(self.device) - 1
        batch[:,:num_node] += torch.arange(cond.shape[0]).to(self.device).unsqueeze(-1) + 1
        batch = batch.view(-1)
        #print(batch[:100])
        #print(batch[-100:])
        cart_coords = self.decoderT(tokens[:,:24,:]*mask[:,:24],mask[:,:24]).view(-1,3)
        # print(cart_coords.shape)
        # print('asjkasfhkasfh')
        # print(batch.shape)
        # print('asjkasfhkasfh')
        cutoff_ind, edge_prob = self.predict_edge(cart_coords, batch, train=False)
        edge_prob = edge_prob.view(-1)
        edge_index = cutoff_ind[:, edge_prob >= threshold]
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]

        return cart_coords.view(-1,24,3)[:,:num_node], edge_index
        #return tokens[:,:num_node], torch.tensor([]).repeat(tokens.shape[0],1)
    
    def add_positional_encoding(self, latents):
        for i in range(latents.shape[0]):
            if i < 24:
                latents[i] += self.one_posi_enc(0, latents.shape[-1])
            else:
                latents[i] += self.one_posi_enc(i - 23, latents.shape[-1])
        return latents
    
    def delete_positional_encoding(self, latents):
        for i in range(latents.shape[0]):
            if i < 24:
                latents[i] -= self.one_posi_enc(0, latents.shape[-1])
            else:
                latents[i] -= self.one_posi_enc(i - 23, latents.shape[-1])
        return latents
    
    def one_posi_enc(self, index, dim):
        enc = torch.zeros(dim).to(self.device)
        #print(dim//2)
        #input()
        for i in range(dim//2):
            enc[2*i] = torch.sin(torch.tensor(index*i))
            enc[2*i+1] = torch.cos(torch.tensor(index*i))
        return enc
    
    @torch.no_grad()
    def create_posi_enc(self, dim=128,length=37):
        self.posi_enc = torch.zeros(length,dim).to(self.device)
        for i in range(length):
            if i < 24:
                self.posi_enc[i] = self.one_posi_enc(0,dim)
            else:
                self.posi_enc[i] = self.one_posi_enc(i-23,dim)
        return

