import cv2
from utils import visualize_image, visualize_profile
from scipy import stats
import numpy as np
from settings import *



class Verifier:
    def __init__(self):
        pass

    def verify_from_rect(self, rect, gray_image):
       
        # Rotate the original grayscale image according to the geometry
        # estimated by the localizer, then extract the rectified ROI.
        (center_x, center_y), (w, h), angle = rect
        if w < h:
            angle += 90

        long_side = max(w, h)
        short_side = min(w, h) - SHRINK

        center = (center_x, center_y)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray_image,
            M,
            (gray_image.shape[1], gray_image.shape[0]),
            borderValue=255,
        )

        # visualize_image(rotated)

        # Get the rectified ROI from the rotated grayscale image.
        x0 = int(max(0, center_x - long_side / 2))
        x1 = int(min(gray_image.shape[1], center_x + long_side / 2))
        y0 = int(max(0, center_y - short_side / 2))
        y1 = int(min(gray_image.shape[0], center_y + short_side / 2))
        roi = rotated[y0:y1, x0:x1]

        visualize_image(roi)

        # get the profile 
        profile = roi[roi.shape[0]//2, :]

        visualize_profile(profile)

        # get contrast, in percentage
        profile_pct = profile.astype(np.float32).copy() * 100.0 / 255.0
        Rmin = np.min(profile_pct)
        Rmax = np.max(profile_pct)
        contrast = Rmax - Rmin
        print(f"rmin: {Rmin}, rmax: {Rmax}, contrast: {contrast}")

        profile = profile.reshape(1, -1)
        t, profile = cv2.threshold(profile, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        profile = profile.ravel()


        # Compress in run lengths 0/1 
        runs = []
        current_value = profile[0]
        current_length = 1
        start = 0
        end = 1

        for value in profile[1:]:
            if value == current_value:
                current_length += 1    
            else:
                runs.append((current_value, current_length, start, end))
                current_value = value
                current_length = 1
                start = end
            end += 1

        runs.append((current_value, current_length, start, end))

        # print(runs)
        # print(len(profile))

        # clean runs 
        # (remove first and last)
        runs.pop(0)
        runs.pop()

        runs = [(value, start, end) for (value, j, start, end) in runs] 

        edge_contrasts = []
        
        for i in range(0, len(runs)):
            if(i + 1 != len(runs)):
                value1, start1, end1 = runs[i]
                _, start2, end2 = runs[i + 1]

                if (end1 == start2): # valid pair
                    if (value1 == 0): # bar-space
                        Rs = np.max(profile_pct[start2:end2])
                        Rb = np.min(profile_pct[start1:end1])
                    else: # space-bar
                        Rs = np.max(profile_pct[start1:end1])
                        Rb = np.min(profile_pct[start2:end2])

                    edge_contrasts.append(Rs - Rb)

        min_edge_contrast = min(edge_contrasts)
        print(f"contrast: {contrast}, min_edge_contrast: {min_edge_contrast}")

        # measures how strong the worst local edge is relative to the global contrast of the symbol
        modulation = 100 * min_edge_contrast / contrast
        print(modulation)

        # remove noise 
        #runs = [(_, j, _, _) for (_, j, _, _) in runs if j > 2]


        # # Estimate X-dimension
        
        # sorted_length = sorted([length for (_, length) in runs])
        # # take first fourth and take the mode
        # k = max(3, len(sorted_length) // 4)
        # thinner = stats.mode(sorted_length[:k]).mode

        # # thinner = min(runs, key = lambda x: x[1])

        # print(thinner)

        return roi

        
