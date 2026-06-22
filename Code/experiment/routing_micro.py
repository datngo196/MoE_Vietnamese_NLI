import torch
import torch.nn as nn
import torch.nn.functional as F

class MICRORouter(nn.Module):
    # [NÂNG CẤP] - Nhận num_experts từ bên ngoài, xóa hardcode 4
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.cognitive_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),                    
            nn.Linear(hidden_size // 2, self.num_experts)
        )

    def forward(self, hidden_states):
        cognitive_logits = self.cognitive_classifier(hidden_states)
        return F.softmax(cognitive_logits, dim=-1)

class MICROMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, expert_expansion=4.0, expert_dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.router = MICRORouter(hidden_size, num_experts)
        
        # [NÂNG CẤP] - Sử dụng expert_expansion thay vì fix cứng nhân 4
        intermediate_size = int(hidden_size * expert_expansion)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Dropout(expert_dropout),
                nn.Linear(intermediate_size, hidden_size),
                nn.Dropout(expert_dropout)
            ) for _ in range(num_experts) # Xóa hardcode range(4)
        ])

    def forward(self, hidden_states):
        routing_probs = self.router(hidden_states)
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        
        flat_output = torch.zeros_like(flat_hidden)
        for i, expert in enumerate(self.experts):
            expert_weight = routing_probs[..., i].view(-1, 1)
            expert_out = expert(flat_hidden)
            flat_output += expert_out * expert_weight
        
        final_output = flat_output.view(batch_size, seq_len, hidden_size)
        
        # =====================================================================
        # [NÂNG CẤP LOSS] - Information Maximization Entropy Loss
        # =====================================================================
        # Ép phân phối của mỗi token phải tự tin (sắc nét)
        token_entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-9), dim=-1).mean()
        
        # Ép phân phối của toàn batch phải đồng đều cho mọi chuyên gia
        mean_probs = routing_probs.view(-1, self.num_experts).mean(dim=0)
        batch_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9))
        
        # Loss = Token Entropy - Batch Entropy
        aux_loss = token_entropy - batch_entropy
        
        return final_output, aux_loss