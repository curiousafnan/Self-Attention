import torch
import math


class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_states):
        super().__init__()
        self.hidden = hidden_states
        self.q = torch.nn.Linear(self.hidden,self.hidden,bias=False)
        self.k = torch.nn.Linear(self.hidden,self.hidden,bias=False)
        self.v = torch.nn.Linear(self.hidden,self.hidden,bias=False)

    def forward(self,vector,mask=None):
        _,seq_len,_=vector.shape
        q = self.q(vector)
        k = self.k(vector)
        v = self.v(vector)
        k_t = torch.transpose(k,-2,-1)
        scores = (q@k_t) / math.sqrt(self.hidden)
        if mask is not None:
            mask_matrix = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(mask_matrix,float('-inf'))
        pred_q_k = torch.softmax(scores,dim=-1)
        output = pred_q_k @ v
        return output , pred_q_k
