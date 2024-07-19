import torch

from model.LongTermAccompanimentBeatwiseUpcomingBars import generate_memory_mask_for_K_bars_ahead_prediction

out_steps = 128
in_steps = 128
seg_len = 4
mask = generate_memory_mask_for_K_bars_ahead_prediction(in_steps, out_steps, 1, seg_len)

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
y_tick_texts = [str(i) for i in range(0, in_steps//seg_len, 4)]
plt.yticks(ticks=range(0, in_steps//seg_len, 4), labels=y_tick_texts)

x_tick_texts = ["B"+str(int(i/4))+", t="+str(i) for i in range(0, out_steps, 4)]
plt.xticks(ticks=range(0, out_steps, 4), labels=x_tick_texts, fontsize=20)

# place texts in the middle of the ticks
plt.xticks(ha='left')
plt.yticks(va='bottom')

# draw thicker vertical lines at every 16 x ticks
for x in x_tick_ix:
    plt.axvline(x, color='black', linewidth=6) if x % seg_len == 0 else plt.axvline(x, color='black', linewidth=4)

# draw thicker horizontal lines at every 4 y ticks
for y in range(0, in_steps, 4):
    plt.axhline(y, color='black', linewidth=6) if (y/seg_len)%4 == 0 else plt.axhline(y, color='black', linewidth=4)

# set x and y labels
plt.xlabel('Output Step Prediction', fontsize=45)
plt.ylabel('Input Segment Embedding\n', fontsize=45)

# transpose figure
plt.gca().invert_yaxis()

plt.show()