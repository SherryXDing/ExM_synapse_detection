#!/bin/bash
# Script to run the synapse detection pipeline
# Case 1: 
# Running on large sequence of 2D tif images (the stitched output data)
# Args: a folder of 2D tif image slides
#       (optional) output direcroty
#       (optional) a folder of 2D tif mask slices
#       (optional) a threshold to remove small pieces  
# Output: a hdf5 image or 2D tif slices with detected synapses
#         csv files indicating synapses location, size, and number of voxels    
# Case 2: 
# Running on a 3D tif image crop 
# Args: one 3D tif image
#       (optional) output directory
#       (optional) a folder of 3D masks
#       (optional) a threshold to remove small pieces
# Output: 3D tif images with deteced synapses
#         a csv file indicating location, size, and number of voxels
#         (optional) individual masks for each synapse 


######## Main ########
SCRIPT_DIR=/groups/scicompsoft/home/dingx/Documents/ExM/unet_pipeline
INPUT_DIR=""
OUTPUT_DIR=""
MASK_DIR=""