import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay
import argparse

IMAGE_PATH_INPUT = "./data/images_input/"
IMAGE_PATH_OUTPUT = "./data/results/"
IMAGE_PATH_GT= "./data/images_gt/"

NUM_IMAGES = 1537

# Separate background and foreground and apply gaussian blur
def applyFilter(img, segmask, segments):
    img_blur = cv2.GaussianBlur(img,(25,25),0)
    segmask_255 = np.uint8(segmask * 255)
    cv2.imwrite(IMAGE_PATH_OUTPUT+str(i).zfill(5)+"_output_segmentation.png", segmask_255)
    cv2.imwrite(IMAGE_PATH_OUTPUT+str(i).zfill(5)+"_segmentation.png", mark_boundaries(img, segments)*255)
    segmask_inv = cv2.bitwise_not(segmask_255)
    # get foreground and background region
    foreground = cv2.bitwise_and(img, img, mask = segmask_255)
    background = cv2.bitwise_and(img_blur, img_blur, mask = segmask_inv)
    filtered_img = cv2.add(foreground,background)
    cv2.imwrite(IMAGE_PATH_OUTPUT+str(i).zfill(5)+"_filter.png", filtered_img)

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=20)
    segments_ids = np.unique(segments)
    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])
    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] 
    ranges = [0, 360, 0, 1] 
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])
    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)
    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(superpixels, fg_markings, bg_markings):
    fg_segments = np.unique(superpixels[fg_markings!=0])
    bg_segments = np.unique(superpixels[bg_markings!=0])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    # Make sure don't divide by zero
    result = []
    for h in histograms:
        if(np.count_nonzero(h) != 0):
            result.append(h / h.sum())
        else:
            result.append(h)
    return np.float32(result)

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)
    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]]
        hi = norm_hists[i]                 
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on histogram matching
            hn = norm_hists[n]             
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) 
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) 
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

def segmentAndFilter(i, addFilterStep):
    # read image
    img = cv2.imread(IMAGE_PATH_INPUT+str(i).zfill(5)+".jpg")
    if img is None:
        print("Please select an image that exists in the images_input folder, and run 'python graphcut_and_blur.py -i [ID]'")
        return 0.0

    # mark foreground and background
    img_height, img_width, img_channel = img.shape
    img_height_center = img_height//2
    img_width_center = img_width//2
    fg_markings = np.zeros((img_height, img_width))
    fg_markings[img_height_center:img_height,img_width_center-20:img_width_center+20] = 1
    bg_markings = np.zeros((img_height, img_width))
    bg_markings[0:100,0:100] = 1
    bg_markings[0:100,img_width-100:img_width] = 1

    # get image features for graph construction
    centers, colors_hists, segments, neighbors = superpixels_histograms_neighbors(img)
    fg_segments, bg_segments = find_superpixels_under_marking(segments, fg_markings, bg_markings)
    # get cumulative BG/FG histograms, before normalization
    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, colors_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, colors_hists)
    norm_hists = normalize_histograms(colors_hists)
    # graph cut
    graph_cut = do_graph_cut((fg_cumulative_hist, bg_cumulative_hist),
                             (fg_segments,        bg_segments),
                             norm_hists,
                             neighbors)
    segmask = pixels_for_segment_selection(segments, np.nonzero(graph_cut))

    # filtering step
    if addFilterStep:
        applyFilter(img, segmask, segments)

    # calculate IoU
    segmask_1 = np.uint8(segmask * 1)
    segmask_gt = cv2.imread(IMAGE_PATH_GT+str(i).zfill(5)+"_mask.jpg",0)
    segmask_gt = segmask_gt//255
    intersection = (segmask_gt + segmask_1) == 2
    union = (segmask_gt + segmask_1) >= 1
    iou = np.sum(intersection) / np.sum(union)
    print("Image %s: IoU is %f" % (i, iou))

    return iou


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--imageID", "-i", help="Input image ID to run. Example: 'python graphcut_and_blur.py -i 1'")
    parser.add_argument("--averageIoU", "-iou", help="Calculate average IoU of all 1537 images")
    args = parser.parse_args()

    # code for calculating the IoU of all images
    if args.averageIoU:
        iou_all = np.zeros((NUM_IMAGES))
        for i in range(1,NUM_IMAGES):
            iou = segmentAndFilter(i, False)
            iou_all[i-1] = iou
        print(np.mean(iou_all))
    else:
        i = 29 # default image
        if args.imageID:
            i = args.imageID
        iou = segmentAndFilter(i, True)
