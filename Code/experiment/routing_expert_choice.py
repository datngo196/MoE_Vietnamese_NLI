import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertChoiceMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, capacity_factor=1.2):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_experts)
        ])

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        total_tokens = batch_size * seq_len
        flat_hidden = hidden_states.view(total_tokens, hidden_size)
        
        router_logits = self.router(flat_hidden)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Calculate capacity
        expert_capacity = int(total_tokens * self.capacity_factor / self.num_experts)
        expert_capacity = min(expert_capacity, total_tokens)
        
        final_output = torch.zeros_like(flat_hidden)
        
        for i, expert in enumerate(self.experts):
            expert_scores = routing_weights[:, i]
            top_scores, top_indices = torch.topk(expert_scores, k=expert_capacity, dim=0)
            
            expert_input = flat_hidden[top_indices]
            expert_output = expert(expert_input) * top_scores.unsqueeze(-1)
            
            final_output.index_add_(0, top_indices, expert_output)
        
        return final_output.view(batch_size, seq_len, hidden_size)