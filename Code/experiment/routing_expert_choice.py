import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertChoiceMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, capacity_factor=1.2, expert_expansion=4.0, expert_dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # [NÂNG CẤP] - Động hóa hệ số mở rộng mạng và Dropout để chạy Grid Search
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
        total_tokens = batch_size * seq_len
        flat_hidden = hidden_states.view(total_tokens, hidden_size)
        
        router_logits = self.router(flat_hidden)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Cân bằng tải tự động thông qua capacity
        expert_capacity = int(total_tokens * self.capacity_factor / self.num_experts)
        expert_capacity = min(expert_capacity, total_tokens)
        
        final_output = torch.zeros_like(flat_hidden)
        
        for i, expert in enumerate(self.experts):
            expert_scores = routing_weights[:, i]
            top_scores, top_indices = torch.topk(expert_scores, k=expert_capacity, dim=0)
            
            expert_input = flat_hidden[top_indices]
            expert_output = expert(expert_input) * top_scores.unsqueeze(-1)
            
            final_output.index_add_(0, top_indices, expert_output)
            
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        # [NÂNG CẤP LOSS] - Expert Choice cân bằng hoàn hảo theo thiết kế (Balanced by design)
        # Không cần phạt loss, trả về 0.0 nhưng phải đúng định dạng tensor để backward không bị lỗi
        aux_loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=False)
        
        return final_output, aux_loss