import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot_prior_distributions(
    prior_df: pd.DataFrame, 
    min_rel_sigma: float = 0.10, 
    max_rel_sigma: float = 1.0,   
    show_plot: bool = True
):
    """
    Generates a Matplotlib figure of prior distributions.
    Returns: (fig, updated_prior_df)
    """
    # 1. Translate Confidence -> Sigma
    prior_df = prior_df.copy()
    
    prior_df['relative_sigma'] = max_rel_sigma - (
        prior_df['confidence'] * (max_rel_sigma - min_rel_sigma)
    )
    
    # Calculate absolute sigma (Standard Deviation)
    global_scale = prior_df["attr_weight"].mean()
    # Handle edge case if mean is 0
    if global_scale == 0: global_scale = 0.01
        
    prior_df['sigma'] = global_scale * prior_df['relative_sigma']

    # Avoid ultra narrow priors
    prior_df['sigma'] = np.clip(
        prior_df['sigma'],
        a_min=0.01 * global_scale,
        a_max=2.0 * global_scale
    )

    if not show_plot:
        return None, prior_df

    # 2. Filter out "Epsilon" channels for plotting only
    # We hide channels with weight < 0.001 (0.1%)
    plot_data = prior_df[prior_df['attr_weight'] > 0.001].sort_values("attr_weight", ascending=False)
    
    if len(plot_data) == 0:
        plot_data = prior_df.sort_values("attr_weight", ascending=False).head(5)

    # 3. Setup Plot (Object-Oriented for Streamlit)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_style("whitegrid")
    
    # Generate X-axis
    # We add 30% padding to the right
    x_max = plot_data['attr_weight'].max() * 1.3
    if x_max == 0: x_max = 1.0 # Safety check
        
    # INCREASED RESOLUTION: 2000 points to capture sharp spikes
    x = np.linspace(0, x_max, 2000)
    
    colors = sns.color_palette("tab10", n_colors=len(plot_data))

    # Optional: Print to console for server logs (Streamlit users won't see this)
    print(f"{'Channel':<20} | {'Weight':<8} | {'Conf':<6} | {'Sigma':<8} | {'Implied Range (±2σ)'}")
    print("-" * 85)

    max_y_density = 0

    for i, (idx, row) in enumerate(plot_data.iterrows()):
        mu = row['attr_weight']
        sigma = row['sigma']
        
        y = norm.pdf(x, loc=mu, scale=sigma)
        
        # Track max height from ALL plotted channels
        if len(y) > 0:
            max_y_density = max(max_y_density, y.max())
        
        # Plot on 'ax' instead of 'plt'
        ax.plot(x, y, label=f"{row['channel']} ({row['confidence']:.2f})", color=colors[i], linewidth=2)
        ax.fill_between(x, y, alpha=0.1, color=colors[i])
        
        lower = max(0, mu - 2*sigma)
        upper = mu + 2*sigma
        print(f"{row['channel']:<20} | {mu:.4f}   | {row['confidence']:.2f}   | {sigma:.4f}   | {lower:.3f} - {upper:.3f}")

    # 4. Smart Axis Scaling
    ax.set_title("Prior Distributions (Attribution Confidence)", fontsize=14)
    ax.set_xlabel("Attribution Weight", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    
    # Set Y-limit to fit the tallest spike + 10% padding
    if max_y_density > 0:
        ax.set_ylim(0, max_y_density * 1.1)
        
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)
    
    # Clean up layout
    plt.tight_layout()
    
    # RETURN the figure object so Streamlit can use st.pyplot(fig)
    return fig, prior_df