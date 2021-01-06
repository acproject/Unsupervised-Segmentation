from demo_modify import MyNet
import torch
import time
import os,sys
import cv2
from skimage import segmentation
from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np


model = torch.load('MyFCN_Model_CPU.pt')
# model = torch.load('MyFCN_Model_CUDA.pt')
class Args(object):
    input_image_path = 'image/56.png'  # image/coral.jpg image/tiger.jpg
    train_epoch = 1
    mod_dim1 = 64  #
    mod_dim2 = 32
    gpu_id = 0

    min_label_num = 4  # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result image.
def run():
    start_time0 = time.time()


    args = Args()
    torch.cuda.manual_seed_all(1983)
    np.random.seed(1983)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    image = cv2.imread(args.input_image_path)

    '''segmentation ML'''
    # seg_map = segmentation.felzenszwalb(image, scale=32, sigma=0.5, min_size=64)
    start = time.time()
    seg_map = segmentation.slic(image, n_segments=16000, compactness=10)
    end = time.time() - start
    print('SLIC: TimeUsed: %.2f' % end)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    '''train init'''
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-1, momentum=0.0)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image

    '''train loop'''
    start_time1 = time.time()
    model.eval()
    for batch_idx in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        '''show image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)
        cv2.imshow("seg_pt", show)
        cv2.waitKey(1)


        if len(un_label) < args.min_label_num:
            break

    '''save'''
    time0 = time.time() - start_time0
    time1 = time.time() - start_time1
    print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    cv2.imwrite("seg_%s_%ds.jpg" % (args.input_image_path[6:-4], time1), show)


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


def snic(image, seeds, compactness, nd_computation=None, image_distance=None, update_func=None):
    """
    Computes a superpixelation from a given image.

    Add a few changes to improve performance:
    - removed normalization factors form metric (-> returned distance map has to be scaled by a constant factor)
    - added a distance map
        - this allows prechecking, if another candidate is already registered to a pixel with a smaller distance
    :param image: cielab image (as python list - no numpy! use to. tolist()). Choose a corresponding nd_computations for other image formats
    :param seeds: expected number of super pixels [int] or a iterable containing seeds
    :param compactness: compactness parameter (inverse color weight)
    :param nd_computation: NdComputations instance for interpolating and computing ndim distances in the image
    :param image_distance: function (int[2], int[2],color[n], color[n]) for computing a distance metric in the image
    :param update_func: optional function (percentage: float) which can be used to monitor progress
    :return: labeled image, distance map, number of superpixels in the image
    """
    image_size = [len(image), len(image[0])]
    label_map = [[-1] * image_size[1] for _ in range(image_size[0])]
    distance_map = [[sys.float_info.max] * image_size[1] for _ in range(image_size[0])]

    if nd_computation is None:
        nd_computation = nd_computations["3"]
    nd_lerp = nd_computation.lerp

    if type(seeds) is int:
        # generate equidistant grid and flatten into list
        grid = [seed for row in compute_grid(image_size, seeds) for seed in row]

        real_number_of_pixels = len(grid)
    else:
        # assume seeds is an iterable
        grid = seeds
        real_number_of_pixels = len(seeds)

    if image_distance is None:
        image_distance = create_augmented_snic_distance(image_size, real_number_of_pixels, compactness)

    # store centroids
    centroids_pos = grid  # flatten grid
    centroids = [[pos, image[pos[0]][pos[1]], 0] for pos in centroids_pos]  # [position, color at position, #pixels]

    # create priority queue
    queue = Queue(image_size[0] * image_size[1] * 4)  # [position, color, centroid_idx]
    q_add = queue.add  # cache some functions
    q_pop = queue.pop
    # we create a priority queue and fill with the centroids itself. Since the python priority queue can not
    # handle multiple entries with the same key, we start inserting the super pixel seeds with negative values. This
    # makes sure they get processed before any other pixels. Since distances can not be negative, all new
    # pixels will have a positive value, and therefore will be handles only after all seeds have been processed.
    for k in range(real_number_of_pixels):
        init_centroid = centroids[k]

        q_len = -queue.length()
        q_add(q_len, [init_centroid[0], init_centroid[1], k])
        distance_map[init_centroid[0][0]][init_centroid[0][1]] = q_len

    # classification
    classified_pixels = 0
    # while not q_empty(): -> replaced with "try: while True:" to speed-up code (~1sec with 50k iterations)
    try:
        while True:
            # get pixel that has the currently smallest distance to a centroid
            item = q_pop()
            candidate_distance = item._key
            candidate = item.value
            candidate_pos = candidate[0]

            # test if pixel is not already labeled
            # if label_map[candidate_pos[1] * im_width + candidate_pos[0]] == -1:
            if label_map[candidate_pos[0]][candidate_pos[1]] == -1:
                centroid_idx = candidate[2]

                # label new pixel
                label_map[candidate_pos[0]][candidate_pos[1]] = centroid_idx
                #
                distance_map[candidate_pos[0]][candidate_pos[1]] = candidate_distance
                # label_map[candidate_pos[1] * im_width + candidate_pos[0]] = centroid_idx
                classified_pixels += 1

                # online update of centroid
                centroid = centroids[centroid_idx]
                num_pixels = centroid[2] + 1
                lerp_ratio = 1 / num_pixels

                # adjust centroid position
                centroid[0] = lerp2(centroid[0], candidate_pos, lerp_ratio)
                # update centroid color
                centroid[1] = nd_lerp(centroid[1], candidate[1], lerp_ratio)
                # adjust number of pixels counted towards this super pixel
                centroid[2] = num_pixels

                # add new candidates to queue
                neighbours, neighbour_num = get_4_neighbourhood_1(candidate_pos, image_size)
                for i in range(neighbour_num):
                    neighbour_pos = neighbours[i]
                    # Check if neighbour is already labeled, as these pixels would get discarded later on.
                    # We filter them here as queue insertions are expensive
                    # if label_map[neighbour_pos[1] * im_width + neighbour_pos[0]] == -1:
                    if label_map[neighbour_pos[0]][neighbour_pos[1]] == -1:
                        neighbour_color = image[neighbour_pos[0]][neighbour_pos[1]]
                        neighbour = [neighbour_pos, neighbour_color, centroid_idx]

                        distance = image_distance(neighbour_pos, centroid[0], neighbour_color, centroid[1])

                        # test if another candidate with a lower distance, is not already
                        # registered to this pixel
                        if distance_map[neighbour_pos[0]][neighbour_pos[1]] >= distance:
                            distance_map[neighbour_pos[0]][neighbour_pos[1]] = distance
                            q_add(distance, neighbour)

                # status update
                if (update_func is not None) and (classified_pixels % 10000 == 0):
                    update_func(classified_pixels)
    except IndexError:
        pass

    return label_map, distance_map, real_number_of_pixels, centroids


if __name__ == '__main__':
    run()
