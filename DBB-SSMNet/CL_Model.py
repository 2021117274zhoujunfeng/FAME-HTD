import torch
from torch import nn
import torch.nn.functional as F

# Unidirectional & bidirectional pyramid SSM backbones
from DPyramidSSM import Mamba, BiMamba

"""
Spectral representation network used in contrastive pre-training
and target-similarity detection.

本版本在你提供的改良版基础上，补全了“分支内双向状态空间建模”这一关键实现，
并且加入了用于消融实验的开关：
  - 仅原始分支 (main) + 双向
  - 仅原始分支 (main) + 单向
  - 双分支 (main + grad) + 单向

接口保持不变：
    input : [B, 1, Band]
    output: [B, 32]
"""


class SpectralGroupAttention(nn.Module):
    def __init__(
        self,
        band: int = 189,
        group_length: int = 20,
        channel_dim: int = 128,
        state_size: int = 128,
        device=None,
        layer: int = 1,
        dropout: float = 0.1,
        # --- 消融/结构开关 ---
        use_grad_branch: bool = True,
        main_bidirectional: bool = False,
        grad_bidirectional: bool = False,
        bi_fuse: str = "gate",  # {'gate','sum','avg'}
        bi_share_params: bool = False,
        # --- 显著性模块超参 ---
        saliency_temperature: float = 2.0,
        grad_strength: float = 1.0,
        contrast_strength: float = 1.0,
    ) -> None:
        super().__init__()

        # 为了兼容不同的 group_length，这里动态计算 stride
        stride = max(1, group_length // 4)
        self.band = int(band)
        self.group_length = int(group_length)
        self.channel_dim = int(channel_dim)
        self.state_size = int(state_size)
        self.layer = int(layer)
        self.device = device

        # 消融开关
        self.use_grad_branch = bool(use_grad_branch)
        self.main_bidirectional = bool(main_bidirectional)
        self.grad_bidirectional = bool(grad_bidirectional)
        self.bi_fuse = str(bi_fuse).lower()
        self.bi_share_params = bool(bi_share_params)

        # 0) 像元级谱形归一化（只保留形状，不关心振幅）
        self.input_norm = nn.LayerNorm(self.band)

        # 1) 1D 分组卷积：把原始谱划分为重叠的谱片段序列
        self.group_division = nn.Sequential(
            nn.Conv1d(1, self.channel_dim, self.group_length, stride=stride, padding=0),
            nn.LeakyReLU(inplace=True),
        )
        # 通道维度做 BatchNorm，保证跨 batch 的分布更稳定
        self.group_norm = nn.BatchNorm1d(self.channel_dim)

        # 分组卷积后得到的序列长度
        sequence_length = (self.band - self.group_length) // stride + 1
        self.sequence_length = int(sequence_length)

        # 2) 主分支：原始谱形序列建模
        if self.main_bidirectional:
            self.mamba_main = BiMamba(
                seq_len=self.sequence_length,
                d_model=self.channel_dim,
                state_size=self.state_size,
                layer=self.layer,
                fuse=self.bi_fuse,
                share_params=self.bi_share_params,
            )
        else:
            self.mamba_main = Mamba(
                seq_len=self.sequence_length,
                d_model=self.channel_dim,
                state_size=self.state_size,
                layer=self.layer,
            )

        # 3) 辅分支：谱梯度序列建模（可关闭，用于消融）
        if self.use_grad_branch:
            if self.grad_bidirectional:
                self.mamba_grad = BiMamba(
                    seq_len=self.sequence_length,
                    d_model=self.channel_dim,
                    state_size=self.state_size,
                    layer=self.layer,
                    fuse=self.bi_fuse,
                    share_params=self.bi_share_params,
                )
            else:
                self.mamba_grad = Mamba(
                    seq_len=self.sequence_length,
                    d_model=self.channel_dim,
                    state_size=self.state_size,
                    layer=self.layer,
                )
        else:
            self.mamba_grad = None

        # 融合后的 token 维度
        fused_dim = (2 * self.channel_dim) if self.use_grad_branch else self.channel_dim
        self.fused_dim = int(fused_dim)

        # 4) 显著性预测：输入为“对比增强”后的 token 特征
        self.saliency_mlp = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim),
            nn.SiLU(),
            nn.Linear(self.fused_dim, 1),
        )

        # 5) 最终投影头：从 fused_dim -> [B, 32]
        self.fnn = nn.Sequential(
            nn.Linear(self.fused_dim, 64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
        )

        # token 级 dropout，在 Mamba 之后、显著性之前做轻微扰动
        self.token_dropout = nn.Dropout(dropout)

        # 显著性 softmax 的温度，以及两种先验的权重（均为可学习的正值）
        self.saliency_log_temp = nn.Parameter(
            torch.log(torch.tensor(float(saliency_temperature)))
        )
        self.grad_log_strength = nn.Parameter(
            torch.log(torch.tensor(float(grad_strength)))
        )
        self.contrast_log_strength = nn.Parameter(
            torch.log(torch.tensor(float(contrast_strength)))
        )

        # 是否对输出做 L2 归一化
        self.enable_out_norm = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, 1, band]

        Returns:
            [B, 32] 光谱表征向量
        """
        B, C, L = x.shape
        if C != 1 or L != self.band:
            raise ValueError(
                f"Expected input shape [B,1,{self.band}], but got {tuple(x.shape)}"
            )

        # ----- 0) 谱向 LayerNorm -----
        x = x.squeeze(1)  # [B, band]
        x = self.input_norm(x)
        x = x.unsqueeze(1)  # [B, 1, band]

        # ----- 1) 分组光谱 token 序列 -----
        x = self.group_division(x)  # [B, Cg, L_seq]
        x = self.group_norm(x)
        x = torch.permute(x, (0, 2, 1))  # [B, L_seq, Cg]

        # ----- 2) 主分支 Mamba（原始谱形）-----
        x_main = self.mamba_main(x)  # [B, L_seq, C]

        # ----- 3) 辅分支 Mamba（谱梯度，可选）-----
        if self.use_grad_branch:
            x_diff = x[:, 1:, :] - x[:, :-1, :]  # [B, L_seq-1, C]
            x_diff = F.pad(x_diff, (0, 0, 0, 1))  # 尾部补零 -> [B, L_seq, C]
            x_grad = self.mamba_grad(x_diff)  # [B, L_seq, C]
            x_fused = torch.cat([x_main, x_grad], dim=-1)  # [B, L_seq, 2C]
        else:
            x_grad = None
            x_fused = x_main  # [B, L_seq, C]

        x_fused = self.token_dropout(x_fused)

        # ----- 4) 显著性加权池化：对比先验 +（可选）梯度先验 -----
        # (1) 相对对比能量：与该像元整体谱形差异越大，权重越高
        global_context = x_fused.mean(dim=1, keepdim=True)  # [B, 1, fused_dim]
        contrast = x_fused - global_context  # [B, L_seq, fused_dim]
        contrast_energy = (contrast**2).mean(dim=-1, keepdim=True)  # [B, L_seq, 1]

        # (2) 从对比特征中学习显著性
        saliency_logits = self.saliency_mlp(contrast)  # [B, L_seq, 1]

        # (3) 先验融合：对比能量始终启用；梯度能量仅在启用 grad 分支时启用
        contrast_strength = self.contrast_log_strength.exp()
        saliency_logits = saliency_logits + contrast_strength * contrast_energy

        if self.use_grad_branch and (x_grad is not None):
            grad_energy = (x_grad**2).mean(dim=-1, keepdim=True)  # [B, L_seq, 1]
            grad_strength = self.grad_log_strength.exp()
            saliency_logits = saliency_logits + grad_strength * grad_energy

        # (4) softmax 温度
        temp = torch.clamp(self.saliency_log_temp.exp(), min=0.5, max=5.0)
        saliency = torch.softmax(temp * saliency_logits, dim=1)  # [B, L_seq, 1]

        # (5) 显著性加权平均池化
        pooled = (x_fused * saliency).sum(dim=1)  # [B, fused_dim]

        # ----- 5) 投影头 -----
        out = self.fnn(pooled)  # [B, 32]
        if self.enable_out_norm:
            out = F.normalize(out, dim=-1)
        return out


# ----------------- 复杂度评估脚本（保持原接口） -----------------
if __name__ == "__main__":
    from torchsummary import summary
    from thop import profile

    bands = [205, 102, 189, 189]
    group_lengths = [5, 15, 30, 5]
    channels = [16, 16, 16, 16]

    for num in range(len(bands)):
        print(f"\n=== 配置 num = {num} ===")
        print(
            f"bands: {bands[num]}, group_lengths: {group_lengths[num]}, channels: {channels[num]}"
        )

        model = SpectralGroupAttention(
            band=bands[num],
            group_length=group_lengths[num],
            channel_dim=channels[num],
            state_size=16,
            device="cuda:0",
            layer=1,
            # 示例：打开主分支双向，关闭梯度分支
            use_grad_branch=False,
            main_bidirectional=True,
        ).cuda()

        input_size = (1, bands[num])
        input_tensor = torch.randn(1, *input_size).cuda()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_tensor = input_tensor.to(device)

        print("模型参数信息：")
        summary(model, input_size, device=device.type)

        flops, params = profile(model, inputs=(input_tensor,))
        print(f"参数数量: {params:,}")
        print(f"计算复杂度 (FLOPs): {flops:,}")

        param_memory = params * 4 / (1024**2)
        print(f"参数内存需求 (MB): {param_memory:.2f}")
