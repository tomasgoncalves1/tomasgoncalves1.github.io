import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from minisom import MiniSom

def plot_dendrogram(model, **kwargs):
    '''
    Create linkage matrix and then plot the dendrogram
    Arguments:
    - model(HierarchicalClustering Model): hierarchical clustering model.
    - **kwargs
    Returns:
    None, but dendrogram plot is produced.
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def visualize_data_points_grid(data, scaled_data, som_model, color_variable, color_dict=plt.get_cmap('tab10').colors):
  '''
  Plots scatter data points on top of a grid that represents
  the self-organizing map. 

  Each data point can be color coded with a "target" variable and 
  we are plotting the distance map in the background.

  Arguments:
  - som_model(minisom.som): Trained self-organizing map.
  - color_variable(str): Name of the column to use in the plot.

  Returns:
  - None, but a plot is shown.
  '''

  # Subset target variable to color data points
  target = data[color_variable]

  fig, ax = plt.subplots()

  # Get weights for SOM winners
  w_x, w_y = zip(*[som_model.winner(d) for d in scaled_data])
  w_x = np.array(w_x)
  w_y = np.array(w_y)

  # Plot distance back on the background
  plt.pcolor(som_model.distance_map().T, cmap='bone_r', alpha=.2)
  plt.colorbar()

  # Iterate through every data points - add some random perturbation just
  # to avoid getting scatters on top of each other.
  for i, c in enumerate( np.unique(target) ):
      idx_target = target==c
      plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                  w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                  s=50, color=color_dict[i], label=c)

  ax.legend(bbox_to_anchor=(1.4, 1))
  plt.grid()
  plt.show()


def visualize_dimensionality_reduction(transformation, targets, cmap=plt.cm.tab10):
    targets = np.array(targets)
    unique_labels = np.unique(targets)
    colors = [cmap(i) for i in range(len(unique_labels))]

    color_dict = {label: color for label, color in zip(unique_labels, colors)}
    mapped_colors = [color_dict[label] for label in targets]

    scatter = plt.scatter(transformation[:, 0], transformation[:, 1], 
                          c=mapped_colors)

    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color_dict[label], markersize=10) 
               for label in unique_labels]
    
    plt.legend(handles, unique_labels, title='Clusters', loc=0)
    plt.show()
