import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import plotly.graph_objects as go
from collections import Counter

# Transition MAtrix Heatmap
def plot_transition_heatmap(matrix, title="Transition Matrix", figsize=(10, 8), cmap="Blues"):
    """
    Plots a heatmap for a given transition matrix.

    Parameters:
    - matrix (pd.DataFrame): A pandas DataFrame representing the transition matrix
    - title (str): Title of the heatmap
    - figsize (tuple): Size of the plot
    - cmap (str): Color map for the heatmap
    """
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap)
    plt.title(title, fontsize=14)
    plt.ylabel("From Channel")
    plt.xlabel("To Channel")
    plt.tight_layout()
    plt.show()


# R Matrix Heatmap
def plot_absorbing_heatmap(matrix, title="R Matrix: Transient → Absorbing Probabilities",figsize=(8, 6), cmap="Blues"):
    """
    Plots a heatmap for the R matrix showing transient to absorbing state probabilities.

    Parameters:
    - matrix (pd.DataFrame): Transition probability matrix (R matrix)
    - title (str): Title of the heatmap
    - figsize (tuple): Size of the plot
    - cmap (str): Color map for the heatmap
    """
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap)
    plt.title(title, fontsize=14)
    plt.ylabel("From Channel")
    plt.xlabel("To Absorbing State")
    plt.tight_layout()
    plt.show()

    
# Image display
def display_image(image_path, figsize=(10, 10), hide_axis=True):
    """
    Displays an image using matplotlib.

    Parameters:
    - image_path (str): Path to the image file
    - figsize (tuple): Size of the display figure
    - hide_axis (bool): Whether to hide axis ticks and labels
    """
    img = mpimg.imread(image_path)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    if hide_axis:
        plt.axis('off')
    plt.show()


# Plot ranked channel
def plot_ranked_channels(ranked_channels, exclude_start=True, title='Ranked Channels by Expected Visits Before Absorption',palette='viridis'):
    """
    Plots a bar chart of ranked channels based on expected visits before absorption.

    Parameters:
    - ranked_channels (pd.Series): Series with channel names as index and expected visits as values
    - exclude_start (bool): Whether to drop 'Start' from the ranking
    - title (str): Title of the plot
    """
    if exclude_start:
        channel_imp = ranked_channels.drop('Start', errors='ignore')
    else:
        channel_imp = ranked_channels

    plt.figure(figsize=(10, 6))
    sns.barplot(x=channel_imp.index, y=channel_imp.values)
    plt.title(title)
    plt.xlabel('Channels')
    plt.ylabel('Expected Visits')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Plot sankey diagram
def plot_sankey_from_matrix(df_counts):
    # Extract unique labels (channels)
    labels = list(df_counts.index)

    # Prepare source, target, and value lists for Sankey
    sources = []
    targets = []
    values = []

    label_to_index = {label: i for i, label in enumerate(labels)}

    for from_label in labels:
        for to_label in labels:
            count = df_counts.loc[from_label, to_label]
            if count > 0:
                sources.append(label_to_index[from_label])
                targets.append(label_to_index[to_label])
                values.append(count)

    # Create Sankey figure
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="skyblue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="lightgreen"
        ))])

    sankey_fig.update_layout(title_text="Sankey Diagram of Channel Transitions", font_size=12)
    sankey_fig.show()


# Plot conversion probability

def plot_conversion_probabilities(b_matrix, absorbing_state='conversion', figsize=(10, 6), palette='viridis'):
    """
    Plots a horizontal bar chart of channel conversion probabilities from the B matrix.

    Parameters:
    - b_matrix (pd.DataFrame): Absorption probability matrix (B = N × R)
    - absorbing_state (str): Column name representing the absorbing state (e.g., 'Conversion')
    - figsize (tuple): Size of the plot
    - palette (str): Color palette for the bars
    """
    conversion_probs = b_matrix[absorbing_state].sort_values(ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(x=conversion_probs.values, y=conversion_probs.index, palette=palette)
    plt.title(f"Channel Conversion Probabilities (from B matrix)", fontsize=14)
    plt.xlabel("Probability of Conversion")
    plt.ylabel("Channel")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# Plot channel removal
def plot_removal_effects(effects_series, title="Removal Effects on Conversion Probability (in pp)"):
    """
    Plots the removal effects of each channel on conversion probability.

    Args:
        effects_series (pd.Series): Series of removal effects (numeric values), indexed by channel.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=effects_series.values, y=effects_series.index, palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel("Removal Effect (pp)")
    plt.ylabel("Channel")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def generate_sankey_all_labels(df_paths, journey_col="user_journey"):
    
    # Count transitions
    transitions = []
    for path in df_paths[journey_col]:
        transitions += list(zip(path[:-1], path[1:]))

    counts = Counter(transitions)

    states = list({s for t in counts.keys() for s in t})
    idx = {s:i for i,s in enumerate(states)}

    sources = [idx[s] for s,_ in counts.keys()]
    targets = [idx[t] for _,t in counts.keys()]
    values = list(counts.values())

    # Colors
    node_colors = [
        "rgba(0,200,255,0.9)" if s not in ['conversion','dropped']
        else ('rgba(0,255,0,0.9)' if s=='conversion' else 'rgba(255,0,0,0.9)')
        for s in states
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=22,
            line=dict(color="white", width=0.5),
            label=states,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            hovertemplate='%{value} transitions<extra></extra>',
            color="rgba(180,180,180,0.35)"
        )
    )])

    fig.update_layout(
        title_text="Customer Journey Sankey — Dark Mode with All Transition Counts",
        font=dict(color="white", size=13),
        paper_bgcolor="black",
        plot_bgcolor="black",
        height=750
    )

    fig.show()

    # After rendering -> access link positions for annotations
    fig_data = fig['data'][0]
    link = fig_data['link']
    node = fig_data['node']

    xs = []
    ys = []
    for i in range(len(values)):
        src = sources[i]
        tgt = targets[i]
        
        x0 = node['x'][src]
        x1 = node['x'][tgt]
        y0 = node['y'][src]
        y1 = node['y'][tgt]

        xs.append((x0 + x1) / 2 + 0.01)  
        ys.append((y0 + y1) / 2)

    # Add annotations for every count – no overlap collapse
    for i, v in enumerate(values):
        fig.add_annotation(
            x=xs[i], y=ys[i],
            text=str(v),
            showarrow=False,
            font=dict(color="yellow", size=11, family="Arial Black"),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="rgba(255,255,255,0.6)",
            borderwidth=0.5,
            borderpad=2
        )

    fig.update_traces()
    fig.show()
    return fig


def generate_sankey_pro(df_paths, journey_col="user_journey", min_threshold=0.005):
    
    # 1️⃣ Count transitions
    transitions = []
    for path in df_paths[journey_col]:
        transitions += list(zip(path[:-1], path[1:]))
    trans_counts = Counter(transitions)
    total_transitions = sum(trans_counts.values())

    # 2️⃣ Sort states by total touches (bigger flows higher vertically)
    state_traffic = Counter([src for src,_ in transitions] + [tgt for _,tgt in transitions])
    states = sorted(state_traffic.keys(), key=lambda x: state_traffic[x], reverse=True)
    state_index = {s:i for i,s in enumerate(states)}

    # 3️⃣ Build Sankey arrays
    sources, targets, values, labels, colors = [], [], [], [], []
    
    max_count = max(trans_counts.values())

    for (src,tgt), count in trans_counts.items():
        s = state_index[src]
        t = state_index[tgt]
        pct = count / total_transitions * 100

        sources.append(s)
        targets.append(t)
        values.append(count)

        # Impact-based coloring (scaled)
        impact_intensity = int((count / max_count) * 255)
        colors.append(f"rgba(0,255,{impact_intensity},0.35)")  # Green → Aqua by intensity

        # Only major flows get labels
        if pct >= min_threshold * 100:
            labels.append(f"{count:,} ({pct:.1f}%)")
        else:
            labels.append("")

    # 4️⃣ Color nodes
    node_colors = [
        "rgba(0, 204, 255, 0.9)" if s not in ['conversion','dropped']
        else ('rgba(0,255,0,0.9)' if s=='conversion'
              else 'rgba(255,0,0,0.9)')
        for s in states
    ]

    # 5️⃣ Build figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=20,
            line=dict(color="white", width=0.5),
            label=states,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=labels,
            color=colors,
            hovertemplate='Path: %{source.label} → %{target.label}<br>'
                          'Transitions: %{value:,}<br>'
                          'Flow Share: %{customdata:.2f}%<extra></extra>',
            customdata=[v/total_transitions*100 for v in values]
        )
    )])

    fig.update_layout(
        title_text="Customer Journey — Major Paths (Counts + % Share + Impact Color)",
        font=dict(color="white", size=14),
        paper_bgcolor="black",
        plot_bgcolor="black",
        height=780
    )

    fig.show()
    return fig
