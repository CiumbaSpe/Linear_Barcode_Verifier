import cv2
import numpy as np
from settings import *
from utils import visualize_best_cluster_mask, visualize_candidate_mask, visualize_best_cluster_bbox


class Localizer: 
    def __init__(self): 
        pass

    def localize(self, image): 

        # First apply image derivative filter on the image
        # ksize=-1 use sharr method
        # CV_32F save in floating point 32
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=-1)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=-1)

        gradient = cv2.magnitude(grad_x, grad_y)

        # abs and convert to unint8, i do not need sign, just intensity
        gradient = cv2.convertScaleAbs(gradient)

        # create a matrix of all the possible patches 
        n_x = (gradient.shape[1] - PATCH_SIZE) // STRIDE + 1
        n_y = (gradient.shape[0] - PATCH_SIZE) // STRIDE + 1
        E = np.zeros((n_y, n_x))        # energy map for each patch
        theta = np.zeros((n_y, n_x))    # main direction
        C = np.zeros((n_y, n_x))        # coerence/anisotropy of the response

        # for each patches evaluate mean 
        for i in range(n_y):
            for j in range(n_x):
                patch_gx = grad_x[i * STRIDE:(i * STRIDE)+PATCH_SIZE, j * STRIDE:(j * STRIDE)+PATCH_SIZE]
                patch_gy = grad_y[i * STRIDE:(i * STRIDE)+PATCH_SIZE, j * STRIDE:(j * STRIDE)+PATCH_SIZE]

                Jxx = np.mean(patch_gx ** 2)
                Jyy = np.mean(patch_gy ** 2)
                Jxy = np.mean(patch_gx * patch_gy)

                E[i, j] = Jxx + Jyy
                theta[i, j] = 0.5 * np.atan2(2 * Jxy, Jxx - Jyy)
                C[i, j] = np.sqrt((Jxx - Jyy)**2 + 4*(Jxy**2)) / (Jxx + Jyy + EPS)    

        # create candidate mask
        energy_thr = np.percentile(E, ENERGY_TRAHSOLD)
        candidate_mask = np.zeros((n_y, n_x), dtype=np.uint8)
        for i in range(n_y): 
            for j in range(n_x):
                if (E[i, j] > energy_thr and C[i, j] > COERENCE_TRASHOLD): 
                    candidate_mask[i, j] = 1
        
        #visualize_candidate_mask(image, candidate_mask, STRIDE, PATCH_SIZE)

        # create cluster
        visited = np.zeros((n_y, n_x), dtype=bool)
        clusters = []
        
        for i in range(n_y): 
            for j in range(n_x):
                
                if(not visited[i, j] and candidate_mask[i, j]):

                    current_cluster = []
                    current_stack = [(i, j)]
                    visited[i, j] = 1            

                    while(len(current_stack) > 0):

                        y, x = current_stack.pop()
                        current_cluster.append((y, x))

                        nodes = [(y + dy, x) for dy in (1, -1, 2, -2)] + [(y, x + dx) for dx in (1, -1, 2, -2)]
        
                        for ny, nx in nodes:
                            if ((nx >= 0 and nx < n_x) and (ny >= 0 and ny < n_y)
                                and not visited[ny, nx] and candidate_mask[ny, nx]):

                                # consider also theta, the angle difference between patch must be below a trashold
                                # that because i want to connect patches which goes in similar direction too.
                                d = np.abs(theta[y, x] - theta[ny, nx])
                                d = min(d, np.pi - d) # because theta and theta + pi are equivalent

                                if (d <= THETA_TRASHOLD):
                                    current_stack.append((ny, nx))
                                    visited[ny, nx] = 1
        
                    clusters.append(current_cluster)

        # score clusters using both size and density
        scored_clusters = []
        for cluster in clusters:
            max_y = min_y = cluster[0][0]
            max_x = min_x = cluster[0][1]
            for y, x in cluster[1:]:
                if(y > max_y): max_y = y
                if(y < min_y): min_y = y
                if(x > max_x): max_x = x 
                if(x < min_x): min_x = x

            size = len(cluster)
            bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
            density = size / bbox_area
            score = size * density

            scored_clusters.append({
                "cells": cluster,
                "size": size,
                "bbox_area": bbox_area,
                "density": density,
                "score": score,
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
            })

        if not scored_clusters:
            raise ValueError("No connected clusters found in the candidate mask.")

        best_cluster = max(scored_clusters, key=lambda cluster: cluster["score"])

        # Create best cluster mask 
        best_cluster_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for yb, xb in best_cluster["cells"]:
            y0 = yb * STRIDE
            x0 = xb * STRIDE
            y1 = y0 + PATCH_SIZE
            x1 = x0 + PATCH_SIZE
            best_cluster_mask[y0:y1, x0:x1] = 255

        #visualize_best_cluster_mask(best_cluster_mask, scale=0.5)

        # find a bbox for the best cluster mask
        contours, _ = cv2.findContours(best_cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        #visualize_best_cluster_bbox(image, box)
        

        return rect
