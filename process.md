## visulaization1
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(df1['p_x'], df1['p_y'], c=df1['land_type'], 
           cmap='tab10', s=1, alpha=0.5)
plt.colorbar(label='Land Type')
plt.title('Spatial Distribution of Land Types')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

plt.subplot(1, 2, 2)
for land_type in sorted(df['land_type'].unique()):
    subset = df[df['land_type'] == land_type]
    plt.scatter(subset['p_x'], subset['p_y'], 
               label=f'Class {land_type}', s=1, alpha=0.6)
plt.legend(markerscale=5)
plt.title('Spatial Distribution by Class')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

plt.tight_layout()
plt.show()

## visualization 2
band_cols = [col for col in df1.columns if col.startswith('Band_')]

# Select every 20th band (Band_1, Band_21, Band_41, ...)
selected_bands = band_cols[::20]  # Start from 0, step by 20

print(f"Total bands: {len(band_cols)}")
print(f"Selected bands: {len(selected_bands)}")
print(f"Selected band names: {selected_bands}")

corr = df1[selected_bands].corr()

print(f"\nCorrelation matrix shape: {corr.shape}")

# ============================================
# Apply masking and filtering
# ============================================
f, ax = plt.subplots(figsize=(15, 12))

# Create upper triangle mask
mask = np.triu(np.ones_like(corr, dtype=bool))

# Define thresholds
cut_off = 0.25
extreme_1 = 0.5
extreme_2 = 0.75
extreme_3 = 0.9

# Mask values below cut-off
mask |= np.abs(corr) < cut_off

# Apply mask
corr_masked = corr.copy()
corr_masked[mask] = np.nan

remove_empty_rows_and_cols = True

if remove_empty_rows_and_cols:
    wanted_cols = np.flatnonzero(np.count_nonzero(~mask, axis=1))
    wanted_rows = np.flatnonzero(np.count_nonzero(~mask, axis=0))
    corr_masked = corr_masked.iloc[wanted_cols, wanted_rows]

annot = [[f"{val:.4f}"
          + ('' if np.isnan(val) or abs(val) < extreme_1 else '\n*')
          + ('' if np.isnan(val) or abs(val) < extreme_2 else '*')
          + ('' if np.isnan(val) or abs(val) < extreme_3 else '*')
          for val in row] for row in corr_masked.to_numpy()]

heatmap = sns.heatmap(corr_masked, 
                      vmin=-1, 
                      vmax=1, 
                      annot=annot, 
                      fmt='', 
                      cmap='crest',
                      cbar_kws={'label': 'Correlation Coefficient'},
                      linewidths=0.5,
                      linecolor='white')

heatmap.set_title('Triangle Correlation Heatmap (Every 20th Band)\n|r| ≥ 0.25 only', 
                 fontdict={'fontsize': 14, 'fontweight': 'bold'}, 
                 pad=16)

# Rotate labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Add legend for asterisks
legend_text = "* = |r| ≥ 0.5\n** = |r| ≥ 0.75\n*** = |r| ≥ 0.9"
plt.text(1.15, 0.5, legend_text, 
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("CORRELATION STATISTICS")
print("="*60)

# Flatten correlation matrix (excluding diagonal)
corr_values = corr.values[np.triu_indices_from(corr.values, k=1)]

print(f"Number of correlations: {len(corr_values)}")
print(f"Mean correlation: {np.mean(corr_values):.4f}")
print(f"Median correlation: {np.median(corr_values):.4f}")
print(f"Max correlation: {np.max(corr_values):.4f}")
print(f"Min correlation: {np.min(corr_values):.4f}")

print(f"\nHigh correlations (|r| ≥ 0.9): {np.sum(np.abs(corr_values) >= 0.9)}")
print(f"Medium correlations (0.75 ≤ |r| < 0.9): {np.sum((np.abs(corr_values) >= 0.75) & (np.abs(corr_values) < 0.9))}")
print(f"Moderate correlations (0.5 ≤ |r| < 0.75): {np.sum((np.abs(corr_values) >= 0.5) & (np.abs(corr_values) < 0.75))}")
print(f"Weak correlations (0.25 ≤ |r| < 0.5): {np.sum((np.abs(corr_values) >= 0.25) & (np.abs(corr_values) < 0.5))}")
print(f"Very weak correlations (|r| < 0.25): {np.sum(np.abs(corr_values) < 0.25)}")
print("="*60)

## visualization 3
# RGB Natural Color Map
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Convert hex to RGB for visualization
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Create RGB array for natural color image
rgb_colors = df1['rgb_hex'].apply(hex_to_rgb)
rgb_array = np.array(rgb_colors.tolist()) / 255.0  # Normalize to 0-1

# Left: Natural color (RGB)
axes[0].scatter(df1['p_x'], df1['p_y'], 
               c=rgb_array, 
               s=2, 
               alpha=0.9)
axes[0].set_title('Natural Color (RGB) Image', 
                  fontsize=16, fontweight='bold')
axes[0].set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
axes[0].set_aspect('equal')

# Right: Land type classification
scatter = axes[1].scatter(df1['p_x'], df1['p_y'], 
                         c=df1['land_type'],
                         cmap='tab10',
                         s=2,
                         alpha=0.9)
axes[1].set_title('Land Type Classification', 
                  fontsize=16, fontweight='bold')
axes[1].set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
axes[1].set_aspect('equal')

# Add colorbar
cbar = plt.colorbar(scatter, ax=axes[1], 
                   ticks=sorted(df1['land_type'].unique()))
cbar.set_label('Land Type Class', fontsize=12, fontweight='bold')

plt.suptitle('Hyperspectral Image - Tyrol, Austria', 
            fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

## visualization 4
# ============================================
# METHOD 4: Professional Map with Custom Legend
# ============================================
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(18, 12))

# Define custom colors for each class
class_colors = sns.color_palette("Spectral", n_colors=n_classes)

# Plot each class with its color
for idx, land_type in enumerate(sorted(df1['land_type'].unique())):
    subset = df1[df1['land_type'] == land_type]
    ax.scatter(subset['p_x'], subset['p_y'],
              c=[class_colors[idx]],
              s=3,
              alpha=0.8,
              label=f'Class {land_type} ({len(subset):,} px)',
              edgecolors='none')

# Customize plot
ax.set_title('Hyperspectral Land Cover Classification Map\nTyrol Alpine Region, Austria',
            fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('Easting (X Coordinate in meters)', fontsize=14, fontweight='bold')
ax.set_ylabel('Northing (Y Coordinate in meters)', fontsize=14, fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

# Add legend
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
         fontsize=11, framealpha=0.9,
         title='Land Type Classes', title_fontsize=13)

# Add text annotation
textstr = f'Total Pixels: {len(df1):,}\nSpectral Bands: 218\nSpatial Coverage: Alpine Region'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
       verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

## visualization -6 
# ============================================
# METHOD 5: 3D Visualization (Optional)
# ============================================
from mpl_toolkits.mplot3d import Axes3D

# Sample data for faster plotting (optional)
sample_size = 50000
df_sample = df1.sample(n=min(sample_size, len(df1)), random_state=42)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Use first principal component as Z-axis
# (or you can use elevation if available)
scatter = ax.scatter(df_sample['p_x'], 
                    df_sample['p_y'],
                    df_sample['Band_1'],  # or use PCA component
                    c=df_sample['land_type'],
                    cmap='tab10',
                    s=5,
                    alpha=0.6)

ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
ax.set_zlabel('Band_1 Reflectance', fontsize=12, fontweight='bold')
ax.set_title('3D Spatial-Spectral Visualization', 
            fontsize=16, fontweight='bold', pad=20)

plt.colorbar(scatter, label='Land Type', shrink=0.5)
plt.tight_layout()
plt.show()

