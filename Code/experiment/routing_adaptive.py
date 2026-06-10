import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDynamicMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, threshold=0.5, adapt_lr=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.threshold = threshold
        self.adapt_lr = adapt_lr
        
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.register_buffer("dynamic_bias", torch.zeros(num_experts))
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_experts)
        ])

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        total_tokens = flat_hidden.size(0)
        
        # Router + dynamic bias
        logits = self.router(flat_hidden) + self.dynamic_bias
        routing_probs = torch.sigmoid(logits)  # Multi-expert activation
        
        # Dynamic thresholding
        mask = routing_probs > self.threshold
        
        final_output = torch.zeros_like(flat_hidden)
        expert_load = torch.zeros(self.num_experts, device=hidden_states.device)
        
        for i, expert in enumerate(self.experts):
            expert_mask = mask[:, i]
            if expert_mask.any():
                expert_input = flat_hidden[expert_mask]
                expert_output = expert(expert_input)
                # Weighted contribution
                weights = routing_probs[expert_mask, i].unsqueeze(-1)
                final_output[expert_mask] += expert_output * weights
                
                # Weighted load for better balancing
                expert_load[i] = weights.sum()

        # Update dynamic bias (training only)
        if self.training:
            ideal_load = total_tokens / self.num_experts
            load_deviation = expert_load - ideal_load
            self.dynamic_bias.data -= self.adapt_lr * load_deviation

        return final_output.view(batch_size, seq_len, hidden_size)