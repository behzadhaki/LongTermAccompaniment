import torch

from model.LTA import generate_memory_mask_for_K_bars_ahead_prediction

out_steps = 128
in_steps = 128
seg_len = 1
KBarsAhead = 2
mask = generate_memory_mask_for_K_bars_ahead_prediction(in_steps, out_steps, KBarsAhead, seg_len)

# plot heatmap
from matplotlib import pyplot as plt
import seaborn as sns
# create a 4x1 ratio figure
plt.figure(figsize=(20, 20))

ax = sns.heatmap(mask.transpose(0, 1).cpu().numpy(), linewidths=1, linecolor='black', clip_on=False)



for _, spine in ax.spines.items():
    spine.set_visible(True)
# remove the color bar
ax.collections[0].colorbar.remove()
# increase xy text size
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# skip every 16 x ticks
x_tick_ix = list(range(0, out_steps, 4))


plt.xticks(ticks=range(0, out_steps, 4), labels=range(0, out_steps, 4))

# skip every 4 y ticks
# format 0 as 000

y_tick_texts = ["t="+f"{i:03d}" for i in range(0, in_steps//seg_len, 4)]
plt.yticks(ticks=range(0, in_steps//seg_len, 4), labels=y_tick_texts, fontsize=28)

x_tick_texts = ["t="+f"{i:03d}" for i in range(0, out_steps, 4)]
plt.xticks(ticks=range(0, out_steps, 4), labels=x_tick_texts, fontsize=28)

# place texts in the middle of the ticks
plt.xticks(ha='left')
plt.yticks(va='bottom')

# draw thicker vertical lines at every 16 x ticks
for x in x_tick_ix:
    plt.axvline(x, color='black', linewidth=4) if x % seg_len == 0 else plt.axvline(x, color='black', linewidth=1)

# draw thicker horizontal lines at every 4 y ticks
for y in range(0, in_steps, 4):
    plt.axhline(y, color='black', linewidth=4) if (y/seg_len)%4 == 0 else plt.axhline(y, color='black', linewidth=1)

# set x and y labels
plt.xlabel('Output Step Prediction', fontsize=35)
plt.ylabel('Input Step Encodings', fontsize=35)
plt.title(f'Cross Attention Mask ({KBarsAhead} bar look-ahead)', fontsize=40)

# first 16x16 in green (border only)
ax.add_patch(plt.Rectangle((0, 0), 16, 16, fill=False, edgecolor='green', lw=4))

# transpose figure
plt.gca().invert_yaxis()

# tight layout
plt.tight_layout()

# save with high dpi
plt.savefig(f'Cross Attention Mask ({KBarsAhead} bar look-ahead).png', dpi=300)

plt.show()