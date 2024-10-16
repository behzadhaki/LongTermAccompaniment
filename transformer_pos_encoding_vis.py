import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_length, d_model):
    # Initialize the positional encoding matrix with zeros
    positional_encoding = np.zeros((seq_length, d_model))

    # Get the position indices (0, 1, 2, ..., seq_length-1)
    position = np.arange(0, seq_length)[:, np.newaxis]

    # Calculate the div_term (10000^(2i/d_model)) for each dimension
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Apply sine to even indices in the array (2i)
    positional_encoding[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices in the array (2i+1)
    positional_encoding[:, 1::2] = np.cos(position * div_term)

    return positional_encoding

# Parameters
seq_length = 100  # Length of the sequence
d_model = 64  # Embedding dimension

# Get the positional encodings
pos_encoding = get_positional_encoding(seq_length, d_model)

# Plot the positional encodings as a heatmap
plt.figure(figsize=(15, 10))

# First subplot: Full positional encoding heatmap
plt.subplot(2, 1, 1)
plt.imshow(pos_encoding, cmap='inferno', aspect='auto')
cbar = plt.colorbar(label='Encoding value')
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Encoding value', fontsize=22)
cbar.set_ticks([-1, 0, 1])
plt.title('Positional Encoding Heatmap ($d=64$)', fontsize=25, loc='center')
plt.xlabel('Embedding Dimensions ($i$)', fontsize=22)
plt.ylabel('Sequence Position ($pos$)', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Second subplot: Zoomed-in heatmap for 3 specific positions
positions_to_zoom = [0, 20, 40]  # Specify the positions you want to zoom in on
zoomed_pos_encoding = pos_encoding[positions_to_zoom, :]
# add a bit of extra space between the subplots
plt.subplots_adjust(hspace=0.5)

plt.subplot(2, 1, 2)
plt.imshow(zoomed_pos_encoding, cmap='inferno', aspect='auto')
cbar = plt.colorbar(label='Encoding value')
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Encoding value', fontsize=22)
cbar.set_ticks([-1, 0, 1])

plt.xlabel('Embedding Dimensions ($i$)', fontsize=22)
plt.ylabel('Sequence Position ($pos$)', fontsize=22)
plt.yticks(ticks=np.arange(len(positions_to_zoom)), labels=positions_to_zoom, fontsize=18)
plt.xticks(fontsize=18)
plt.tight_layout()

plt.savefig('positional_encoding_heatmap.png', dpi=300)
plt.show()
