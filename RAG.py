"""
===========
RAG Merging
===========
This example constructs a Region Adjacency Graph (RAG) and progressively merges
regions that are similar in color, Merging two adjacent regions produces
a new region with all the pixels from the merged regions. Regions are merged
until no highly similar region pairs remain.
"""

from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np

def _weight_mean_color(graph, src, dst, n):
    """
        Callback to handle merging nodes by recomputing mean color.
        The method expects that the mean color of `dst` is already computed.

        Parameters
        ----------
        graph : RAG(Region Adjacency Graph)
            the graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbot of `src` or `dst` or both.
        Returns:
        data: dict
        A dictionary with the `weight` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src ,dst):
    """
        Callback called before merging two nodes of a mean color distance graph.
        This method computes the mean color of 'dst'.
        Parameters
        ----------
        graph: RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
    """

    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count'])





labels = segmentation.slic(img, compactness=30, n_segments=8000)
g = graph.rag_mean_color(img, labels)
labels2 = graph.merge_hierarchical(labels, g, thresh=40, rag_copy=False, in_place_merge=True, merge_func=merge_mean_color, weight_func=_weight_mean_color)
g2 = graph.rag_mean_color(img, labels2)

out = color.label2rgb(labels2, img, kind='overlay')
out2 = color.label2rgb(labels2, img, kind='avg')

out = segmentation.mark_boundaries(out, labels2, (0,0,0))

out2 = segmentation.mark_boundaries(out2, labels2, (0,0,0))

# cmap = colors.ListedColormap(['#6599FF','#5577ee', '#ff9900'])

#
# cbar1 = plt.colorbar(graph.show_rag(labels, g, out,  edge_width=1.2), fraction=0.03)
# cbar2 = plt.colorbar(graph.show_rag(labels2, g2, out2, edge_width=1.2), fraction=0.03)


# plt.figure(figsize=(12,8))
# plt.title("USE OVERLAY")
# plt.imshow(out)




img = img.convert("RGB")

img = np.array(img)








img = img.convert("RGB")

img = np.array(img)


g = graph.rag_mean_color(img, labels)
labels2 = graph.merge_hierarchical(labels, g, thresh=40, rag_copy=False, in_place_merge=True, merge_func=merge_mean_color, weight_func=_weight_mean_color)
g2 = graph.rag_mean_color(img, labels2)

out = color.label2rgb(labels2, img, kind='overlay')
out2 = color.label2rgb(labels2, img, kind='avg')

out = segmentation.mark_boundaries(out, labels2, (0,0,0))

out2 = segmentation.mark_boundaries(out2, labels2, (0,0,0))

# cmap = colors.ListedColormap(['#6599FF','#5577ee', '#ff9900'])

#
# cbar1 = plt.colorbar(graph.show_rag(labels, g, out,  edge_width=1.2), fraction=0.03)
# cbar2 = plt.colorbar(graph.show_rag(labels2, g2, out2, edge_width=1.2), fraction=0.03)


# plt.figure(figsize=(12,8))
# plt.title("USE OVERLAY")
# plt.imshow(out)



