import torch
import torch.nn.functional as F

def cmkd_loss(student_logits, teacher_logits, labels, temperature=4.0, threshold=0.8, num_samples=10000):
    B, C, D, H, W = student_logits.shape
    total_pixels = B * D * H * W

    student_flat = student_logits.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (N, C)
    teacher_flat = teacher_logits.permute(0, 2, 3, 4, 1).reshape(-1, C)
    labels_flat = labels.reshape(-1)  # (N,)

    # 检查 labels 范围
    assert torch.all(labels_flat >= 0) and torch.all(labels_flat < C), "Label out of range"

    # 随机采样（仅在像素过多时）
    if total_pixels > num_samples:
        indices = torch.randperm(total_pixels, device=student_logits.device)[:num_samples]
        student_flat = student_flat[indices]
        teacher_flat = teacher_flat[indices]
        labels_flat = labels_flat[indices]
    else:
        indices = None

    T = temperature
    # 安全 softmax（clip 防止溢出）
    def safe_softmax(logits, dim=1):
        logits = logits - torch.max(logits, dim=dim, keepdim=True)[0]  # 数值稳定
        return F.softmax(logits, dim=dim)

    # 完整软标签（用于余弦相似度）
    student_soft = safe_softmax(student_flat / T, dim=1)
    teacher_soft = safe_softmax(teacher_flat / T, dim=1)

    # 非目标类 logits（将目标类设为极小的负数，而非 -inf）
    student_logits_nc = student_flat.clone()
    teacher_logits_nc = teacher_flat.clone()
    M = student_flat.size(0)
    # 将目标类 logits 设为 -1e9（不会溢出）
    student_logits_nc[range(M), labels_flat] = -1e9
    teacher_logits_nc[range(M), labels_flat] = -1e9

    # 非目标类概率
    student_soft_nc = safe_softmax(student_logits_nc / T, dim=1)
    teacher_soft_nc = safe_softmax(teacher_logits_nc / T, dim=1)

    # KL 散度（逐像素）
    kl_div = F.kl_div(
        (student_logits_nc / T).log_softmax(dim=1),  # 使用 log_softmax
        teacher_soft_nc,
        reduction='none'
    ).sum(dim=1)  # (M,)

    # 余弦相似度（用于选择样本）
    cos_sim = F.cosine_similarity(student_soft, teacher_soft, dim=1)
    mask = (cos_sim > threshold).float()

    # 如果 mask 全为 0，则返回 0 损失，避免 NaN
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    loss = (mask * kl_div).sum() / mask.sum()
    return loss