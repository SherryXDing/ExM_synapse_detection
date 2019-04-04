#!/bin/bash
# Script to run the synapse detection pipeline on 
# Case 1: 
# Running on large sequence of 2D tif images (the stitched output data)
# Args: a folder of 2D tif image slides
#       (optional) output direcroty
#       (optional) a folder of 2D tif mask slices
#       (optional) a threshold to remove small pieces  
# Output: a hdf5 image or 2D tif slices with detected synapses
#         csv files indicating synapses location, size, and number of voxels    
# Case 2: 
# Running on 3D tif image crop 
# Args: a foler of 3D tif images
#       (optional) output directory
#       (optional) a folder of 3D masks
#       (optional) a threshold to remove small pieces
# Output: 3D tif images with deteced synapses
#         a csv file indicating location, size, and number of voxels
#         (optional) individual masks for each synapse 


# A function to print the usage of this manuscript
usage()
{
    echo "Usage for 2D tiff slices:"
    echo "bash Pipeline.sh -2D -i <input_data_directory> -o <output_result_directory> -m <input_mask_directory> -t <threshold_to_remove_small_piece> -s"
    echo "-m <input_mask_directory>, -t <threshold_to_remove_small_piece>, and -s (hfd5 result to tiff) are optional"
    echo "Usage for 3D tiff images:"
    echo "bash Pipeline.sh -3D -i <input_data_directory> -o <output_result_directory> -m <input_mask_directory> -t <threshold_to_remove_small_piece> -s"
    echo "-o <output_result_directory>, -m <input_mask_directory>, -t <threshold_to_remove_small_piece>, and -s (output individual masks) are optional"
}

source activate tf 

######## Main ########
# Directory of the script
SCRIPT_DIR=/groups/scicompsoft/home/dingx/Documents/ExM/unet_pipeline_v1
# Directory of the input data
INPUT_DIR=""
# Directory for the output result
OUTPUT_DIR=""
# Directory of the mask
MASK_DIR=""
# Threshold to remove small pieces
THRESHOLD=10
# Hdf5 result to tiff (only for Case 1)
TO_TIFF=false
# If output individual masks (only for Case 2)
INDIVIDUAL_MASK=0

######## Case 1
if [[ $1 == "-2D" ]]; then
    shift
    while [[ $1 != "" ]]; do 
        case $1 in
            -i)
                INPUT_DIR=$2
                shift 2
                ;;
            -o)
                OUTPUT_DIR=$2
                shift 2
                ;;
            -m)
                MASK_DIR=$2
                shift 2
                ;;
            -t)
                THRESHOLD=$2
                shift 2
                ;;
            -s)
                TO_TIFF=true
                shift
                ;;
        esac
    done 
    if [[ ( $INPUT_DIR == "" ) || ( $OUTPUT_DIR == "" ) ]]; then # Error if there is no input or output directory
        echo "ERROR! Please provide input and output directory."
        usage
        exit 1
    elif [[ `ls $INPUT_DIR/*.tif | wc -l` == 0 ]]; then # Error if input image does not exist
        echo "ERROR! Input tif image does not exist."
        usage
        exit 1
    fi
    # Create output directory if not exist
    mkdir -p $OUTPUT_DIR
    # Tiff to hdf5 for image slices, output slices_to_volume.h5 file into $OUTPUT_DIR
    bsub -n 2 -o $OUTPUT_DIR/img_tif2hdf.log "python $SCRIPT_DIR/tif_to_h5.py -i $INPUT_DIR -o $OUTPUT_DIR" 
    wait
    # If mask folder is provided, output slices_to_volume.h5 file into $MASK_DIR
    if [[ $MASK_DIR != "" ]]; then
        if [[ `ls $INPUT_DIR/*.tif | wc -l` != 0 ]]; then 
            bsub -n 2 -o $OUTPUT_DIR/mask_tif2hdf.log "python $SCRIPT_DIR/tif_to_h5.py -i $MASK_DIR"
            wait
        else
            echo "ERROR! Mask tif image does not exist."
            usage
            exit 1
        fi
    fi
    # Get the dimension of image
    A_IMG=`ls $INPUT_DIR/*.tif | head -n 1`
    WIDTH=`identify $A_IMG | cut -d ' ' -f 3 | cut -d 'x' -f 1`
    HEIGHT=`identify $A_IMG | cut -d ' ' -f 3 | cut -d 'x' -f 2`
    SLICE=`ls $INPUT_DIR/*.tif | wc -l`
    # Number of loops in x-dimension
    if [[ $(( WIDTH%1000 )) < 500 ]]; then
        NUM_ROW=$(( WIDTH/1000 ))
    else
        NUM_ROW=$(( WITDH/1000+1 ))
    fi
    # Number of loops in y-dimension
    if [[ $(( HEIGHT%1000 )) < 500 ]]; then
        NUM_COL=$(( HEIGHT/1000 ))
    else
        NUM_COL=$(( HEIGHT/1000+1 ))
    fi
    # Number of loops in z-dimension
    if [[ $(( SLICE%1000 )) < 500 ]]; then
        NUM_VOL=$(( SLICE/1000 ))
    else
        NUM_VOL=$(( SLICE/1000+1 ))
    fi
    # Loop to process the whole image
    for (( ROW=0; ROW<$NUM_ROW; ROW++ )); do
        for (( COL=0; COL<$NUM_COL; COL++ )); do
            for (( VOL=0; VOL<$NUM_VOL; VOL++ )); do
                # Determine the location
                MIN_ROW=$(( ROW*1000 ))
                MIN_COL=$(( COL*1000 ))
                MIN_VOL=$(( VOL*1000 ))
                if [[ $ROW == $(( NUM_ROW-1 )) ]]; then
                    MAX_ROW=$(( WIDTH-1 ))
                else
                    MAX_ROW=$(( ROW*1000+999 ))
                fi
                if [[ $COL == $(( NUM_COL-1 )) ]]; then
                    MAX_COL=$(( HEIGHT-1 ))
                else
                    MAX_COL=$(( COL*1000+999 ))
                fi
                if [[ $VOL == $(( NUM_VOL-1 )) ]]; then
                    MAX_VOL=$(( SLICE-1 ))
                else
                    MAX_COL=$(( VOL*1000+999 ))
                fi
                # Submit GPU jobs
                PIDS=""  # Store job IDs
                if [[ $MASK_DIR != "" ]]; then
                    bsub -n 2 -gpu "num=1" -q gpu_any -o $OUTPUT_DIR/main.log \
                    "python $SCRIPT_DIR/main_2d.py -i $OUTPUT_DIR/slices_to_volume.h5 -l $MIN_ROW,$MIN_COL,$MIN_VOL,$MAX_ROW,$MAX_COL,$MAX_VOL -m $MASK_DIR/slices_to_volume.h5 -t $THRESHOLD"
                    PIDS+="$!"
                else
                    bsub -n 2 -gpu "num=1" -q gpu_any -o $OUTPUT_DIR/main.log \
                    "python $SCRIPT_DIR/main_2d.py -i $OUTPUT_DIR/slices_to_volume.h5 -l $MIN_ROW,$MIN_COL,$MIN_VOL,$MAX_ROW,$MAX_COL,$MAX_VOL -t $THRESHOLD"
                    PIDS+="$!"
                fi
            done
        done
    done
    if [[ $TO_TIFF == "true" ]]; then
        wait $PIDS  # Wait until the whole hdf5 image is processed
        bsub -n 2 -o $OUTPUT_DIR/result_hdf2tif.log "python $SCRIPT_DIR/h5_to_tif.py -i $OUTPUT_DIR/slices_to_volume.h5" 
    fi

######## Case 2
elif [[ $1 == "-3D" ]]; then
    shift
    while [[ $1 != "" ]]; do 
        case $1 in
            -i)
                INPUT_DIR=$2
                shift 2
                ;;
            -o)
                OUTPUT_DIR=$2
                shift 2
                ;;
            -m)
                MASK_DIR=$2
                shift 2
                ;;
            -t)
                THRESHOLD=$2
                shift 2
                ;;
            -s)
                INDIVIDUAL_MASK=1
                shift
                ;;
        esac
    done
    if [[ $INPUT_DIR == "" ]]; then # Error if there is no input directory
        echo "ERROR! Please provide input directory."
        usage
        exit 1
    elif [[ `ls $INPUT_DIR/*.tif | wc -l` == 0 ]]; then # Error if input image does not exist
        echo "ERROR! Input tif image does not exist."
        usage
        exit 1
    fi
    if [[ $OUTPUT_DIR == "" ]]; then # If no user specific output directory
        OUTPUT_DIR=$INPUT_DIR
    fi
    if [[ $MASK_DIR != "" ]]; then 
        if [[ `ls $MASK_DIR/*.tif | wc -l` == 0 ]]; then 
            echo "ERROR! Mask image does not exist."
            usage
            exit 1
        fi
    fi
    # Create output directory if not exist
    mkdir -p $OUTPUT_DIR
    # Submit GPU jobs
    for IMG in `ls $INPUT_DIR/*.tif`; do
        if [[ $MASK_DIR != "" ]]; then
            for MASK in `ls $MASK_DIR/*.tif`; do
                bsub -n 2 -gpu "num=1" -q gpu_any -o $OUTPUT_DIR/${MASK}_${IMG}.log \
                "python $SCRIPT_DIR/main_3d.py -i $IMG -o $OUTPUT_DIR -m $MASK -t $THRESHOLD -s $INDIVIDUAL_MASK"
            done
        else
            bsub -n 2 -gpu "num=1" -q gpu_any -o $OUTPUT_DIR/${MASK}_${IMG}.log \
            "python $SCRIPT_DIR/main_3d.py -i $IMG -o $OUTPUT_DIR -t $THRESHOLD -s $INDIVIDUAL_MASK"
        fi
    done 

# -2D or -3D has to be provided as the first parameter
else
    echo "ERROR! Please provide whether it is 2D data or 3D data!"
    usage
    exit 1
fi