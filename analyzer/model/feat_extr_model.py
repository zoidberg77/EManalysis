import os, sys
import numpy as np

from analyzer.model.utils.extracting import compute_regions, compute_intentsity, compute_dist_graph

class FeatureExtractor():
    '''
    Using this model to build up your feature matrix that will be clustered.
    :param emvol & gtvol: (np.array) Both are the data volumes.
    :param dprc: (string) data processing mode that sets how your data should be threated down the pipe.
                This is important as you might face memory problems loading the whole dataset into your RAM. Distinguish between two setups:
                - 'full': This enables reading the whole stack at once. Or at least the 'chunk_size' you set.
                - 'iter': This iterates over each slice/image and extracts information one by one. This might help you to process the whole dataset without running into memory error.
    '''
    def __init__(self, emvol, gtvol, dprc='full', mode='3d', fpath=os.path.join(os.getcwd(), 'features/')):
        self.emvol = emvol
		self.gtvol = gtvol
        #self.fns =
        self.dprc = dprc
        self.mode = mode
        self.fpath = fpath

    def get_seg_size(self, fns, save=True):
        '''
        Extract the size of each mitochondria segment.
        :param save: (bool) if true you save the extracted features to a .json file and store it for further clustering.
        '''
        result_dict = compute_regions(self.gtvol, fns=self.fns, dprc=self.dprc, mode=self.mode)

        if save:
    		with open(os.path.join(self.fpath, 'sizef.json'), 'w') as f:
    	    	json.dump(result_dict, f)
                f.close()
