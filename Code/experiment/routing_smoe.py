import torch
import torch.nn as nn
import torch.nn.functional as F

class SMoERouter(nn.Module):
    def __init__(self, hidden_size, num_experts, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.router_weights = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states, attention_matrix=None):
        batch_size, seq_len, _ = hidden_states.shape
        base_logits = self.router_weights(hidden_states)
        
        if attention_matrix is None:
            # Only compute attention if needed and for small seq_len
            if seq_len > 512:  # Safety for memory
                attention_matrix = torch.eye(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1, -1)
            else:
                norm_hidden = F.normalize(hidden_states, p=2, dim=-1)
                attention_matrix = torch.bmm(norm_hidden, norm_hidden.transpose(1, 2))
                attention_matrix = F.softmax(attention_matrix / self.temperature, dim=-1)
        
        smoothed_logits = torch.bmm(attention_matrix, base_logits)
        routing_probs = F.softmax(smoothed_logits, dim=-1)
        
        return routing_probs

class SMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.router = SMoERouter(hidden_size, num_experts)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_experts)
        ])

    def forward(self, hidden_states, attention_matrix=None):
        routing_probs = self.router(hidden_states, attention_matrix)
        
        # Soft routing (better gradient)
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        flat_output = torch.zeros_like(flat_hidden)
        
        for i, expert in enumerate(self.experts):
            expert_weight = routing_probs[..., i].view(-1, 1)
            expert_out = expert(flat_hidden)
            flat_output += expert_out * expert_weight
        
        return flat_output.view(batch_size, seq_len, hidden_size)