"""
This module is for computing a time-series of SA and FA responses from an optical tactile input.
Responses computed according to marker displacement from rest (SA) and change of marker displacement (RA)
Marker detection and tracking uses parameters set in json file - see ./tip_params/tipA.json
"""

import cv2
import numpy as np
import scipy.spatial.distance as ssd
import json
import os

class AfferentResponse():
    def __init__(self, display, **params):
        cv_params = cv2.SimpleBlobDetector_Params()
        cv_params.blobColor = params['blob_color']
        cv_params.minThreshold = params['min_threshold']
        cv_params.maxThreshold = params['max_threshold']
        cv_params.filterByArea = params['filter_by_area']
        cv_params.minArea = params['min_area']
        cv_params.maxArea = params['max_area']
        cv_params.filterByCircularity = params['filter_by_circularity']        
        cv_params.minCircularity = params['min_circularity']
        cv_params.filterByConvexity = params['filer_by_convexity']
        cv_params.minConvexity = params['min_convexity']
        cv_params.filterByInertia = params['filter_by_inertia']        
        cv_params.minInertiaRatio = params['min_inertia_ratio'] 

        self.params = params
        self.max_pin_dist_from_centroid = params['max_pin_dist_from_centroid']
        self.min_pin_separation = params['min_pin_seperation']
        self.max_tracking_move = params['max_tracking_move']

        self._detector = cv2.SimpleBlobDetector_create(cv_params)   
        self._display = display

    def _detect_pins(self, frame):	
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = self._detector.detect(frame)
        pins = np.array([k.pt for k in keypoints])
        return pins, keypoints

    def _select_pins(self, pins):
        centroid = np.mean(pins, axis=0)
        centroid = centroid[np.newaxis,:]
        pin_rads = np.squeeze(ssd.cdist(pins, centroid, 'euclidean'))
        pins = pins[pin_rads <= self.max_pin_dist_from_centroid,:]
        pin_dists = ssd.cdist(pins, pins, 'euclidean')
        processed = np.zeros(pins.shape[0], dtype=bool)
        selected = np.zeros(pins.shape[0], dtype=bool)
        idx = np.argmin(ssd.cdist(pins, centroid, 'euclidean'))
        selected[idx] = True
        while not np.all(processed):
            processed = processed | selected | (pin_dists[idx,:] < self.min_pin_separation)     
            pin_dists[idx, processed] = np.inf
            idx = np.argmin(pin_dists[idx,:])
            selected[idx] = True
        pins = pins[selected,:]
        pins = pins[pins[:,0].argsort(),:]
        return pins

    def _map_pins(self, pins, prev_pins):
        pin_dists = ssd.cdist(pins, prev_pins, 'euclidean')
        min_pin_idxs = np.argmin(pin_dists, axis=0)
        pins = pins[min_pin_idxs,:]
        min_pin_dists = np.min(pin_dists, axis=0)        
        rep_pin_idxs = min_pin_dists>self.max_tracking_move
        pins[rep_pin_idxs,:] = prev_pins[rep_pin_idxs,:]
        return pins

    def init_pins(self,source):
        cap = cv2.VideoCapture(source)
        try:
            for i in range(10): # reject first 10 frames - hysteresis
                ret, frame = cap.read()
            ret, frame = cap.read() # use 11th frame
            if ret:
                pins, keypoints = self._detect_pins(frame)
                if self._display==True:
                    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.imshow('sensor', im_with_keypoints)
                    cv2.waitKey(2000)
                if pins.shape[0] == 0:
                    raise RuntimeError(source + ': failed to identify any pins in frame')
                pins_init = self._select_pins(pins) 
            else:
                raise RuntimeError(source + ': failed to capture initialisation frame')
        except RuntimeError as error:
            print(error)
            pins_init = np.array([]) # return empty array if unable to collect frame e.g. missing or corrupted file
        cap.release()
        cv2.destroyAllWindows
        return pins_init

    def firing(self, source, pins_init_0):
        pins_init = self.init_pins(source)
        firingSA_dict = {}
        firingRA_dict = {}
        if pins_init.size == 0:
            pins_init_mapped = np.array([])
            n_frames = 1
        else:
            pins_init_mapped = self._map_pins(pins_init, pins_init_0) # map initial pin positions to first tap  
            firingSA_nSub1 = np.zeros(len(pins_init_mapped))
            img = np.zeros((480,1920,3), np.uint8)
            cap = cv2.VideoCapture(source)
            pins_nSub1 = pins_init_mapped
            n_frames = 0
            for i in range(10): # reject first 10 frames - hysteresis
                ret, frame = cap.read()
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    n_frames += 1
                    pins_n, keypoints = self._detect_pins(frame)            
                    pins_nSub1 = self._map_pins(pins_n, pins_nSub1)

                    # calc. SA response by euclidian distance from initial pin position
                    firingSA_n = np.diagonal(ssd.cdist(pins_nSub1, pins_init_mapped, 'euclidean'))
                    firingSA_n.setflags(write=1)                        

                    # calc. RA response by diff of SA response
                    firingRA_n = abs(firingSA_n - firingSA_nSub1)
                    firingSA_nSub1 = firingSA_n
                    firingSA_dict["{}".format(n_frames-1)] = firingSA_n
                    firingRA_dict["{}".format(n_frames-1)] = firingRA_n
                    
                    # optional show figure
                    if self._display == True:
                        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        img[0:480,1280:1920,:] = im_with_keypoints
                        for i in range(len(pins_nSub1)):
                            # draw SA response
                            cv2.circle(img,(int(pins_init_mapped[i,0]),int(pins_init_mapped[i,1])),10,(0,int((firingSA_n[i]/20)*50),0),-1)
                            cv2.circle(img,(int(pins_init_mapped[i,0]),int(pins_init_mapped[i,1])),10,(0,50,0),0)
                            # draw RA response
                            cv2.circle(img,(int(pins_init_mapped[i,0])+640,int(pins_init_mapped[i,1])),10,(0,0,int(firingRA_n[i]*50)),-1)
                            cv2.circle(img,(int(pins_init_mapped[i,0])+640,int(pins_init_mapped[i,1])),10,(0,0,100),0)
                        cv2.imshow('sensor', img)
                        if cv2.waitKey(1) & 0xFF == ord('q'): break # pres 'q' to quit
                else:
                    break
            cap.release()
            cv2.destroyAllWindows
        return(firingSA_dict, firingRA_dict, pins_init_mapped, n_frames)

# example workflow for extracting SA & FA responses
if __name__ == "__main__":
    
    # retrieve tip parameters
    with open(os.path.dirname(__file__)+"/tip_params/tipA.json", "r") as read_file:
        params = json.load(read_file)
    # instantiate afferent object 
    afferents = AfferentResponse(True, **params)

    # get initiral pin positions for first data-point in the series
    pins_init = afferents.init_pins('path-to-video-file-of-first-data-point-in-series')
    # compute SA & FA response for specific data-point
    firing = afferents.accumulate_firing_press('path-to-video-file-of-specific-data-point-in-series', pins_init)
