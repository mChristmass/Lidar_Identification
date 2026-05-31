import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# def visualize_sample(depth, intensity, mask, save_path=None, title=None):
#
#     cmap = ListedColormap([
#         [0,0,0,0],
#         [1,0,0,1]
#     ])
#
#     fig, axes = plt.subplots(1,2, figsize=(10,4))
#
#     axes[0].imshow(depth, cmap="coolwarm")#"viridis")
#     axes[0].imshow(mask, cmap=cmap, alpha=0.4)
#     axes[0].set_title("Depth + Mask")
#     axes[0].axis("off")
#
#     axes[1].imshow(intensity, cmap="gray")
#     axes[1].imshow(mask, cmap=cmap, alpha=0.4)
#     axes[1].set_title("Intensity + Mask")
#     axes[1].axis("off")
#
#     if title:
#         fig.suptitle(title)
#
#     if save_path:
#         plt.savefig(save_path, dpi=150)
#
#     plt.close()


# 用于双阶段框架stage1的单通道可视化

def visualize_sample(intensity, mask, save_path=None, title=None):
    """
    针对单通道 Intensity 数据优化的可视化函数
    """
    # 定义遮罩颜色：0 为全透明，1 为红色半透明
    cmap = ListedColormap([
        [0, 0, 0, 0],  # 背景透明
        [1, 0, 0, 0.6] # 预测目标为红色，增加一点透明度观察原图细节
    ])

    # 创建 1行2列 的对比布局
    # 左图：原始强度图 | 右图：强度图 + 预测遮罩
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 1. 纯 Intensity 图
    axes[0].imshow(intensity, cmap="gray")
    axes[0].set_title("Original Intensity")
    axes[0].axis("off")

    # 2. Intensity + Mask 叠加图
    axes[1].imshow(intensity, cmap="gray")
    axes[1].imshow(mask, cmap=cmap) # 叠加红色遮罩
    axes[1].set_title("Intensity + Predicted Mask")
    axes[1].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout() # 自动调整布局，防止标题重叠

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()