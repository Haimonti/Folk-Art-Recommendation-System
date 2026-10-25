import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OnlyLRD(nn.Module):
    def __init__(self, user_num, item_num, input_dim, llm_emb_path, num_latent_relations=8, llm_dim=384):
        super(OnlyLRD, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.num_latent_relations = num_latent_relations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Standard Embeddings
        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size).to(self.device) 
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size).to(self.device) 

        # 2. LLM Embeddings for Language Knowledge
        raw_llm_embeddings = torch.load(llm_emb_path).to(self.device)
        self.llm_embeddings = nn.Embedding.from_pretrained(raw_llm_embeddings, freeze=True)
        
        # 3. LRD Core Modules
        self.W1 = nn.Linear(llm_dim, self.hidden_size).to(self.device)
        self.relation_extractor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.num_latent_relations),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        self.latent_relation_embeddings = nn.Embedding(self.num_latent_relations, self.hidden_size).to(self.device)

        # 4. Final aggregation mapping
        self.agg_linear = nn.Linear(self.hidden_size * self.num_latent_relations, self.hidden_size).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)

    def forward(self, g, user_index, last_item_index, pos_tar=None, neg_tar=None, is_training=False):
        g = g.to(self.device)
        
        # Get target embeddings
        target_item = pos_tar if is_training else last_item_index
        target_emb = self.item_embedding(target_item)
        target_llm = self.W1(self.llm_embeddings(target_item))
        
        user_emb = self.user_embedding(user_index)
        
        # --- LRD Aggregation ---
        # Extract historical items directly from the DGL batch graph nodes
        hist_items = g.nodes['item'].data['item_id'].to(self.device)
        hist_emb = self.item_embedding(hist_items)
        hist_llm = self.W1(self.llm_embeddings(hist_items))
        
        # DGL Unbatching: Group historical items by user in the batch
        b_item_sizes = g.batch_num_nodes('item').tolist()
        hist_emb_split = torch.split(hist_emb, b_item_sizes)
        hist_llm_split = torch.split(hist_llm, b_item_sizes)
        
        m_u_list = []
        
        for i in range(len(user_index)):
            h_emb = hist_emb_split[i] # Shape: [seq_len, hidden_size]
            h_llm = hist_llm_split[i] # Shape: [seq_len, hidden_size]
            
            # Expand target for this sequence
            t_llm_exp = target_llm[i].unsqueeze(0).expand(h_llm.shape[0], -1)
            t_emb_exp = target_emb[i].unsqueeze(0).expand(h_emb.shape[0], -1)
            
            # 1. Relation Extraction: q(r | v_i, v_j)
            q_r = self.relation_extractor(torch.cat([h_llm, t_llm_exp], dim=-1)) # [seq_len, num_relations]
            
            # 2. Relation Intensity (DistMult triplet score): phi(v_i, v_j, r)
            rel_weights = self.latent_relation_embeddings.weight # [num_relations, hidden_size]
            
            s_u_r_list = []
            for r in range(self.num_latent_relations):
                r_emb = rel_weights[r].unsqueeze(0) # [1, hidden]
                
                # DistMult: v_i^T * diag(r) * v_j
                phi = torch.sum(h_emb * r_emb * t_emb_exp, dim=-1) # [seq_len]
                omega = F.softmax(phi, dim=0).unsqueeze(-1) # [seq_len, 1]
                
                # Weight historical items
                s_u_r = torch.sum(omega * h_emb, dim=0) # [hidden_size]
                s_u_r_list.append(s_u_r)
                
            # Concat all relation representations and map to hidden_size
            m_u = self.agg_linear(torch.cat(s_u_r_list, dim=-1)) 
            m_u_list.append(m_u)
            
        m_u_batch = torch.stack(m_u_list, dim=0)
        unified_embedding = user_emb + m_u_batch

        # --- Prediction & Loss Outputs ---
        if is_training:
            pos_id = self.item_embedding(pos_tar)
            neg_id = self.item_embedding(neg_tar)
            last_id = self.item_embedding(last_item_index)
            
            pos_score = torch.sum(unified_embedding * pos_id, dim=-1)
            neg_score = torch.sum(unified_embedding * neg_id, dim=-1)
            
            # Extract q_r for the auxiliary LRD reconstruction loss
            e_last = self.W1(self.llm_embeddings(last_item_index))
            e_pos = self.W1(self.llm_embeddings(pos_tar))
            q_r_loss = self.relation_extractor(torch.cat([e_last, e_pos], dim=-1))
            
            return pos_score, neg_score, q_r_loss, last_id, pos_id, neg_id, self.latent_relation_embeddings.weight
        else:
            score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0))
            if neg_tar is not None:
                neg_embedding = self.item_embedding(neg_tar)
                score_neg = torch.matmul(unified_embedding.unsqueeze(1), neg_embedding.transpose(2, 1)).squeeze(1)
                return score, score_neg
            return score, None
        


def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1).to(user_index.device)
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]

def graph_item(bg, last_index, item_embedding):
    b_item_size = bg.batch_num_nodes('item')
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]

def order_update(edges):
    dic = {}
    dic['order'] = torch.sort(edges.data['time'])[1]
    dic['re_order'] = len(edges.data['time']) - dic['order']
    return dic


def collate(data):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        #else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    user = []
    user_l = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['user'])
        user_l.append(da[1]['u_alis'])
        graph.append(da[0][0]) 
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
        #print('batch graph',da[0][0])
        #break               
    return torch.tensor(user_l).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long()

def collate_bpr(data, user_neg_data):
    user_l = []
    graph = []
    label = []
    last_item = []
    neg_item = [] # List for the negative samples

    for da in data:
        # Extract the user ID (ensure it's an integer for lookup)
        # da[1]['user'] is a tensor like tensor([5]), so we use .item()
        u_id = da[1]['user'].item()
        
        user_l.append(da[1]['u_alis'])
        graph.append(da[0][0]) 
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
        
        # --- BPR Negative Sampling ---
        # 1. Retrieve the list of valid negatives for this user
        valid_negatives = user_neg_data[u_id]
        
        # 2. Randomly sample ONE item from that list
        # We use np.random.choice for efficiency
        neg_id = np.random.choice(valid_negatives)
        
        neg_item.append(neg_id)
        # -----------------------------

    return (torch.tensor(user_l).long(), 
            dgl.batch(graph), 
            torch.tensor(label).long(), 
            torch.tensor(last_item).long(), 
            torch.tensor(neg_item).long()) # Returns the specific negative items

def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u.item()], neg_num, replace=False)
        # print('neg[i]',neg[i])
    # sys.exit("Terminating the program.")           
    return neg



def collate_test(data, user_neg):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    user_alis = []
    global_users = [] # Track the actual user IDs
    graph = []
    label = []
    last_item = []
    for da in data:
        user_alis.append(da[1]['u_alis'])
        global_users.append(da[1]['user']) # Grab the global ID
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])

    # Pass global_users to neg_generate, but return user_alis for the graph
    return (torch.tensor(user_alis).long(),
            dgl.batch(graph),
            torch.tensor(label).long(),
            torch.tensor(last_item).long(),
            torch.Tensor(neg_generate(global_users, user_neg)).long())

