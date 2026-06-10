import torch
import torch.nn as nn
import torch.nn.functional as F

class MICRORouter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.num_experts = 4
        self.cognitive_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),                    # Changed from Tanh
            nn.Linear(hidden_size // 2, self.num_experts)
        )

    def forward(self, hidden_states):
        cognitive_logits = self.cognitive_classifier(hidden_states)
        return F.softmax(cognitive_logits, dim=-1)

class MICROMoELayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.router = MICRORouter(hidden_size)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(4)
        ])

    def forward(self, hidden_states):
        routing_probs = self.router(hidden_states)
        
        # Use soft routing for better gradient flow
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        
        # Weighted sum across all experts
        flat_output = torch.zeros_like(flat_hidden)
        for i, expert in enumerate(self.experts):
            expert_weight = routing_probs[..., i].view(-1, 1)
            expert_out = expert(flat_hidden)
            flat_output += expert_out * expert_weight
        
        return flat_output.view(batch_size, seq_len, hidden_size)