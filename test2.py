import torch
causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(64).bool()

teacher_forcing_ration = 0.8

indices = torch.rand(causal_mask.size(0))
indices = indices > teacher_forcing_ration
indices[:16] = False
causal_mask = causal_mask.clone()
causal_mask[:, indices] = True

# plot
import matplotlib.pyplot as plt
import seaborn as sns
# no colorbar
sns.heatmap(causal_mask, cmap='Blues', cbar=False)
plt.xlabel('Input Sequence Step', fontsize=15)
plt.ylabel('Output Sequence Step', fontsize=15)
# same aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

# tick mark on 0, 16, ...
plt.xticks([0, 16, 32, 48, 64], [0, 16, 32, 48, 64], fontsize=12)
plt.yticks([0, 16, 32, 48, 64], [0, 16, 32, 48, 64], fontsize=12)

# draw thicker vertical lines at every 16 x ticks
for x in range(0, 96, 16):
    plt.axvline(x, color='grey', linewidth=1, linestyle='--')
    
# draw thicker horizontal lines at every 16 y ticks
for y in range(0, 96, 16):
    plt.axhline(y, color='grey', linewidth=1, linestyle='--')

plt.tight_layout()
plt.savefig('causal_mask_with_src_mask.png', dpi=300)
plt.show()