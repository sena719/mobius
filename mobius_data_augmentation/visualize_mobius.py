import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from mobius_transformation import Mobius

DATA_PATH = 'data/'
N_IMAGES = 8   # 表示する画像枚数
N_TRANSFORMS = 3  # 各画像に対して何パターンの変換を見せるか

dataset = datasets.KMNIST(root=DATA_PATH, train=True, download=False, transform=None)
mobius = Mobius(rand=True, interpolation=True, dataset='kmnist', std=0.01, madmissable=False, M=3)
resize = transforms.Resize((28, 28))

# N_IMAGES行 × (1 + N_TRANSFORMS)列のグリッド
n_cols = 1 + N_TRANSFORMS
fig, axes = plt.subplots(N_IMAGES, n_cols, figsize=(n_cols * 2, N_IMAGES * 2))

axes[0, 0].set_title('Original', fontsize=11, fontweight='bold')
for k in range(N_TRANSFORMS):
    axes[0, k + 1].set_title(f'Mobius {k+1}', fontsize=11, fontweight='bold')

kmnist_labels = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']

for i in range(N_IMAGES):
    img_pil, label = dataset[i]
    img_rgb = img_pil.convert('RGB')
    img_rgb = resize(img_rgb)

    axes[i, 0].imshow(img_pil, cmap='gray')
    axes[i, 0].set_ylabel(f'{kmnist_labels[label]} ({label})', fontsize=9)
    axes[i, 0].axis('off')

    for k in range(N_TRANSFORMS):
        img_mobius = mobius(img_rgb)
        axes[i, k + 1].imshow(img_mobius)
        axes[i, k + 1].axis('off')

plt.suptitle('KMNIST: Before vs After Mobius Transformation', fontsize=13, y=1.01)
plt.tight_layout()

out_path = 'mobius_visualization.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
