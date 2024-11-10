import torch

# 定义方向向量
facing_direction = torch.tensor([[-0.2424, 0.9702],
                                 [-0.1922, 0.9814],
                                 [-0.1511, 0.9885],
                                 [-0.1281, 0.9918]])

motion_direction = torch.tensor([[0.1857, 0.9826],
                                 [0.3239, 0.9461],
                                 [0.4378, 0.8991],
                                 [0.5383, 0.8427]])

# 计算两个方向向量之间的点积
dot_product = (facing_direction * motion_direction).sum(dim=1)
facing_norm = torch.norm(facing_direction, dim=1)
motion_norm = torch.norm(motion_direction, dim=1)

# 计算夹角的余弦值，并求弧度和角度
cosine_similarity = dot_product / (facing_norm * motion_norm)
angles_radians = torch.acos(cosine_similarity)
angles_degrees = angles_radians * (180 / torch.pi)  # 将弧度转换为角度

print("Angles in radians:", angles_radians)
print("Angles in degrees:", angles_degrees)
