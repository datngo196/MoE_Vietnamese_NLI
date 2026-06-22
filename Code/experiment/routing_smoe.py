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
            # Chỉ tính toán self-attention cho routing nếu seq_len nhỏ để tránh tràn VRAM
            if seq_len > 512:  
                attention_matrix = torch.eye(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1, -1)
            else:
                norm_hidden = F.normalize(hidden_states, p=2, dim=-1)
                attention_matrix = torch.bmm(norm_hidden, norm_hidden.transpose(1, 2))
                attention_matrix = F.softmax(attention_matrix / self.temperature, dim=-1)
        
        smoothed_logits = torch.bmm(attention_matrix, base_logits)
        routing_probs = F.softmax(smoothed_logits, dim=-1)
        
        return routing_probs

class SMoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, temperature=1.0, expert_expansion=4.0, expert_dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        
        # Truyền tham số động temperature vào Router
        self.router = SMoERouter(hidden_size, num_experts, temperature)
        
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

    def forward(self, hidden_states, attention_matrix=None):
        routing_probs = self.router(hidden_states, attention_matrix)
        
        # Soft routing (Dòng chảy gradient mượt hơn)
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        flat_output = torch.zeros_like(flat_hidden)
        
        for i, expert in enumerate(self.experts):
            expert_weight = routing_probs[..., i].view(-1, 1)
            expert_out = expert(flat_hidden)
            flat_output += expert_out * expert_weight
        
        final_output = flat_output.view(batch_size, seq_len, hidden_size)
        
        # =====================================================================
        # [NÂNG CẤP LOSS] - Batch-wise & Token-wise Entropy Penalty
        # =====================================================================
        # 1. Token Entropy (Minimize): Ép phân phối của 1 token phải "sắc nét", thiên về 1 vài chuyên gia
        token_entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-9), dim=-1).mean()
        
        # 2. Batch Entropy (Maximize): Ép phân phối trung bình của toàn batch phải đều cho mọi chuyên gia
        mean_probs = routing_probs.view(-1, self.num_experts).mean(dim=0)
        batch_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9))
        
        # Phạt token bị nhòe (cộng vào loss) và Thưởng batch chia đều (trừ khỏi loss)
        aux_loss = token_entropy - batch_entropy
        
        return final_output, aux_loss