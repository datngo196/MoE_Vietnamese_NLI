import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekMoELayer(nn.Module):
    def __init__(self, hidden_size, num_shared_experts=2, num_routed_experts=8, 
                 num_routed_to_select=2, max_devices=4):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.num_routed_to_select = num_routed_to_select
        self.max_devices = max_devices
        
        # Shared Experts (always active)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4), 
                nn.GELU(), 
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_shared_experts)
        ])
        
        # Routed Experts
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4), 
                nn.GELU(), 
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_routed_experts)
        ])
        
        self.router = nn.Linear(hidden_size, num_routed_experts, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        final_output = torch.zeros_like(flat_hidden)
        
        # 1. Shared Experts
        for shared in self.shared_experts:
            final_output += shared(flat_hidden)
        
        # 2. Routed Experts with Top-K
        router_logits = self.router(flat_hidden)
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Noisy Top-K during training (recommended)
        if self.training:
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise
        
        top_probs, top_indices = torch.topk(routing_probs, self.num_routed_to_select, dim=-1)
        
        # Efficient routing
        for k in range(self.num_routed_to_select):
            kth_indices = top_indices[:, k]
            kth_probs = top_probs[:, k].unsqueeze(-1)
            
            for i, expert in enumerate(self.routed_experts):
                mask = (kth_indices == i)
                if mask.any():
                    expert_out = expert(flat_hidden[mask])
                    final_output[mask] += expert_out * kth_probs[mask]
        
        return final_output.view(batch_size, seq_len, hidden_size)