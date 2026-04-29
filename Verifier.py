from dataclasses import dataclass

import cv2
from utils import visualize_image, visualize_profile
from scipy import stats
import numpy as np
from settings import *


@dataclass
class Run:
    value: int
    length: int
    start: int
    end: int


class Verifier:
    def __init__(self):
        pass

    def verify_profile(self, profile):

        ### --- CONTRAST METRIC ---
        # Get contrast, in percentage
        profile_pct = profile.astype(np.float32).copy() * 100.0 / 255.0
        Rmin = np.min(profile_pct)
        Rmax = np.max(profile_pct)
        contrast = Rmax - Rmin
        # --- end of the evaluation of contrast ---

        # Treshold the profile in 0/1
        profile = profile.reshape(1, -1)
        t, profile = cv2.threshold(profile, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        profile = profile.ravel()

        # Compress in run lengths 0/1 meaning (value, length, start, end) e.g. [Run(0, 3, 0, 4) Run(1, 5, 4, 6) ...]
        # length is the number of equal value found starting from "start" and ending at "end"
        runs = []
        current_value = profile[0]
        current_length = 1
        start = 0
        end = 1

        for value in profile[1:]:
            if value == current_value:
                current_length += 1    
            else:
                runs.append(Run(current_value, current_length, start, end))
                current_value = value
                current_length = 1
                start = end
            end += 1

        runs.append(Run(current_value, current_length, start, end))

        # clean runs, there might be some noise 
        # (the first and last lines might be cropped to much or inaccurate)
        runs.pop(0)
        runs.pop()

        # remove those run which lengths is less than 3 
                
        sorted_lengths = sorted([run.length for run in runs if run.length > 3])
        
        top_x = [(sorted_lengths[0], 1)]
        mode = top_x[0]
        i = 1
        while len(top_x) < 3: 
            
            if sorted_lengths[i] != top_x[-1][0]:
                top_x.append((sorted_lengths[i], 1))
            else:
                top_x[-1] = (top_x[-1][0], top_x[-1][1] + 1)
                if(top_x[-1][1] >= mode[1]):
                    mode = top_x[-1]

            i += 1

        x_dimension = mode[0]


        runs = [run for run in runs if run.length > 0.5 * x_dimension]


        # after the i removed the run i must merge back together runs
        i = 0
        while i < len(runs) -1:
            current_run = runs[i]
            next_run = runs[i + 1]

            if (current_run.end != next_run.start):
                # update lenght
                current_run.length = next_run.end - current_run.start

                # update the start and end (case value = value or multiple removal)
                if(current_run.value == next_run.value):
                    next_run.length = current_run.length
                    next_run.start = current_run.start            
                    runs.pop(i)
                else: 
                    current_run.end = next_run.start
                

            i += 1
            


        ### --- MODULATION METRIC ---

        edge_contrasts = []
        
        for i in range(0, len(runs) - 1):
            current_run = runs[i]
            next_run = runs[i + 1]

            if (current_run.end == next_run.start): # valid pair
                if (current_run.value == 0): # bar-space
                    Rs = np.max(profile_pct[next_run.start:next_run.end])
                    Rb = np.min(profile_pct[current_run.start:current_run.end])
                else: # space-bar
                    Rs = np.max(profile_pct[current_run.start:current_run.end])
                    Rb = np.min(profile_pct[next_run.start:next_run.end])

                edge_contrasts.append(Rs - Rb)

        min_edge_contrast = min(edge_contrasts)

        # measures how strong the worst local edge is relative to the global contrast of the symbol
        modulation = 100 * min_edge_contrast / contrast

        

        return contrast, modulation

    def verify_from_rect(self, rect, gray_image):
       
        # Rotate the original grayscale image according to the geometry
        # estimated by the localizer, then extract the rectified ROI.
        (center_x, center_y), (w, h), angle = rect
        if w < h:
            angle += 90

        long_side = max(w, h)
        short_side = min(w, h) - SHRINK

        # rotate the image around the center of the rectangle evaluated by the localizer
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

        contrasts = []
        modulations = []
        ys = np.linspace(0, roi.shape[0] - 1, N_PROFILES, dtype=np.int32)

        for y in ys:
            # get the profile 
            profile = roi[y, :]

            # visualize_profile(profile)

            contrast, modulation = self.verify_profile(profile)
            contrasts.append(contrast)
            modulations.append(modulation)

        return float(np.mean(contrasts)), float(np.mean(modulations))

        
