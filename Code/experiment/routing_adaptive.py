import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDynamicMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, threshold=0.5, adapt_lr=0.01, expert_expansion=4.0, expert_dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.threshold = threshold
        self.adapt_lr = adapt_lr
        
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.register_buffer("dynamic_bias", torch.zeros(num_experts))
        
        intermediate_size = int(hidden_size * expert_expansion)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Dropout(expert_dropout),
                nn.Linear(intermediate_size, hidden_size),
                nn.Dropout(expert_dropout)
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
        
        # Tính toán tổng tải trọng (load) của từng chuyên gia dựa trên xác suất
        expert_load = routing_probs.sum(dim=0) 
        
        for i, expert in enumerate(self.experts):
            expert_mask = mask[:, i]
            if expert_mask.any():
                expert_input = flat_hidden[expert_mask]
                expert_output = expert(expert_input)
                # Weighted contribution
                weights = routing_probs[expert_mask, i].unsqueeze(-1)
                final_output[expert_mask] += expert_output * weights

        # Update dynamic bias (Chỉ trong lúc training)
        if self.training:
            ideal_load = total_tokens / self.num_experts
            load_deviation = expert_load - ideal_load
            self.dynamic_bias.data -= self.adapt_lr * load_deviation

        # =====================================================================
        # [NÂNG CẤP LOSS] - Squared Coefficient of Variation (CV^2) Loss
        # =====================================================================
        # Tính trung bình tải trọng của các chuyên gia
        mean_load = expert_load.mean()
        
        # Tránh lỗi chia cho 0
        if mean_load > 0:
            # CV^2 = Variance(Load) / (Mean(Load))^2
            variance_load = torch.mean((expert_load - mean_load) ** 2)
            cv_squared_loss = variance_load / (mean_load ** 2 + 1e-9)
        else:
            cv_squared_loss = torch.tensor(0.0, device=hidden_states.device)
            
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        return final_output, cv_squared_loss