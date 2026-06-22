import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekMoELayer(nn.Module):
    def __init__(self, hidden_size, num_shared_experts=2, num_routed_experts=8, 
                 num_routed_to_select=2, noise_level=0.1, 
                 expert_expansion=4.0, expert_dropout=0.1):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.num_routed_to_select = num_routed_to_select
        self.noise_level = noise_level
        
        intermediate_size = int(hidden_size * expert_expansion)
        
        # Shared Experts (Luôn luôn kích hoạt cho mọi token)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size), 
                nn.GELU(), 
                nn.Dropout(expert_dropout),
                nn.Linear(intermediate_size, hidden_size),
                nn.Dropout(expert_dropout)
            ) for _ in range(num_shared_experts)
        ])
        
        # Routed Experts (Chỉ kích hoạt theo Top-K)
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size), 
                nn.GELU(), 
                nn.Dropout(expert_dropout),
                nn.Linear(intermediate_size, hidden_size),
                nn.Dropout(expert_dropout)
            ) for _ in range(num_routed_experts)
        ])
        
        self.router = nn.Linear(hidden_size, num_routed_experts, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        total_tokens = flat_hidden.size(0)
        final_output = torch.zeros_like(flat_hidden)
        
        # 1. Chạy Shared Experts
        for shared in self.shared_experts:
            final_output += shared(flat_hidden)
        
        # 2. Chạy Routed Experts với Top-K
        router_logits = self.router(flat_hidden)
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Thêm nhiễu (Noise) khi Train để ép Router khám phá các chuyên gia mới
        if self.training and self.noise_level > 0.0:
            noise = torch.randn_like(router_logits) * self.noise_level
            router_logits = router_logits + noise
        
        top_probs, top_indices = torch.topk(routing_probs, self.num_routed_to_select, dim=-1)
        
        # Mảng đếm số lượng token đi vào từng chuyên gia (phục vụ tính Loss)
        expert_counts = torch.zeros(self.num_routed_experts, device=hidden_states.device)
        
        # Tuyến đường dữ liệu
        for k in range(self.num_routed_to_select):
            kth_indices = top_indices[:, k]
            kth_probs = top_probs[:, k].unsqueeze(-1)
            
            # Đếm token
            expert_counts.scatter_add_(0, kth_indices, torch.ones_like(kth_indices, dtype=torch.float))
            
            for i, expert in enumerate(self.routed_experts):
                mask = (kth_indices == i)
                if mask.any():
                    expert_out = expert(flat_hidden[mask])
                    final_output[mask] += expert_out * kth_probs[mask]
                    
        # [NÂNG CẤP LOSS] - Tính GShard / Switch Transformer Load Balancing Loss
        # P_i: Xác suất trung bình mà router gán cho từng chuyên gia
        mean_router_probs = routing_probs.mean(dim=0) 
        
        # f_i: Tỷ lệ token thực tế đi vào từng chuyên gia
        token_fraction = expert_counts / (total_tokens * self.num_routed_to_select)
        
        # Công thức: N * sum(P_i * f_i)
        aux_loss = self.num_routed_experts * torch.sum(mean_router_probs * token_fraction)
        
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        return final_output, aux_loss