from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from answer import answer


def calc_median(image_pixels_float, cluster_centers, cluster_labels):
    cluster_dict = {}
    for i in range(len(cluster_centers)):
        cluster_number = str(i)
        cluster_dict[cluster_number] = []
    for i in range(len(cluster_labels)):
        cluster_number = str(cluster_labels[i])
        pixel = image_pixels_float[i]
        cluster_dict[cluster_number].append(pixel)
    median_dict = {}
    for i in cluster_dict.keys():
        median_pixel = np.median(cluster_dict[i], axis=0)
        median_dict[i] = median_pixel
    return median_dict


def PSNR(image1, image2):
    mse = mean_squared_error(image1, image2)
    psnr = 10 * math.log10(np.max(image1) / mse)
    return psnr


def assemble_img(moravg_dict, cluster_labels, image_pixels_float):
    image_new = image_pixels_float.copy()
    for i in range(len(cluster_labels)):
        image_new[i] = moravg_dict[str(cluster_labels[i])]
    return image_new


def centers_to_avg_dict(centers):
    avg_dict = {}
    for i in range(len(centers)):
        cluster_number = str(i)
        avg_dict[cluster_number] = centers[i]
    return avg_dict


def clusterize_and_assemble(n_clusters=None):
    clf = KMeans(init='k-means++', random_state=241, n_clusters=n_clusters)
    clf.fit_predict(image_pixels_float)

    cluster_centers = clf.cluster_centers_
    cluster_labels = clf.labels_

    avg_dict = centers_to_avg_dict(cluster_centers)
    median_dict = calc_median(image_pixels_float, cluster_centers, cluster_labels)

    image_avg = assemble_img(avg_dict, cluster_labels, image_pixels_float)
    image_median = assemble_img(median_dict, cluster_labels, image_pixels_float)

    psnr_image_avg = PSNR(image_pixels_float, image_avg)
    psnr_image_median = PSNR(image_pixels_float, image_median)

    print("n: " + str(n_clusters))
    print("PSNR image_avg: " + str(psnr_image_avg))
    print("PSNR image_median: " + str(psnr_image_median))

    return psnr_image_avg, psnr_image_median


def estimate_clusters_min(image_pixels_float):
    clusters_min = None
    for i in range(1, 21):
        psnr_image_avg, psnr_image_median = clusterize_and_assemble(i)
        if psnr_image_avg >= 20 or psnr_image_median >= 20:
            clusters_min = i
            break
    return clusters_min


image = imread('parrots.jpg')
image_float = img_as_float(image)

image_pixels_float = []
for i in image_float:
    for k in i:
        image_pixels_float.append(k)

image_pixels_float = np.matrix(image_pixels_float)

clusters_min = estimate_clusters_min(image_pixels_float)
print("Clusters min:" + str(clusters_min))

answer([clusters_min], '1')
