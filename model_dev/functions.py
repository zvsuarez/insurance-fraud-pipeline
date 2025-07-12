
import matplotlib.pyplot as plt
from scipy.stats import skew
import seaborn as sns


def histograms(data, var, n_cols=4):
    numeric_cols = data[var].columns
    n_rows = -(-len(numeric_cols) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(data[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'{col}')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    #fig.savefig('num_hist.png', dpi=300, bbox_inches='tight')
    plt.show()



def box_plot(data, var, n_cols=4):
    numeric_cols = data[var].columns
    n_rows = -(-len(numeric_cols) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=data[col], ax=axes[i], color='lightgreen')
        axes[i].set_title(f'{col}')
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    #fig.savefig('num_box.png', dpi=300, bbox_inches='tight')
    plt.show()



def winsorize(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return lower, upper


def plot_post_boxplot(variants, feature):
    plt.figure(figsize=(8, 4))
    for i, (data, title) in enumerate(variants):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x=data)
        plt.title(title)
        plt.ylabel(feature)

    plt.tight_layout()
    plt.show()


def plot_post_hist(variants, feature):
    plt.figure(figsize=(8, 6))
    for i, (data, title) in enumerate(variants):
        plt.subplot(2, 2, i + 1)
        sns.histplot(data, kde=True, bins=30)
        plt.title(f'{title} (Skew: {skew(data):.4f})')
        plt.xlabel(feature)

    plt.tight_layout()
    plt.show()


def multicount_by_target(data, cols, target_col, target_val, hue=None, n_cols=2, figsize=(14, 6)):
    
    filtered = data[data[target_col] == target_val]
    n_rows = -(-len(cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    if hue != None:
        for i, col in enumerate(cols):
            ax = sns.countplot(data=data, x=col, ax=axes[i], hue=target_col)
            axes[i].set_title(f'Fraud vs Non-Fraud per {col}')
            #axes[i].set_title(f'{col} by {target_col} = {target_val}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
            for container in ax.containers:
                ax.bar_label(container)
    else:
        for i, col in enumerate(cols):
            ax = sns.countplot(data=filtered, x=col, ax=axes[i])
            axes[i].set_title(f'{col} by {target_col} = {target_val}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
            ax.bar_label(ax.containers[0])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()