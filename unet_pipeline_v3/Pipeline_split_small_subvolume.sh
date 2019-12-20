#!/bin/bash
# Enter-point code to run unet or postprocessing separately
# Case 1:
# Running U-net
# Args: a folder of 2D tif image slides
#       an output folder
# Output: a hdf5 image or 2D tif slices with detected synapses by unet
# Usage: bash Pipeline_split.sh -unet -i <input_tiff_directory> -o <output_result_directory>
# Case 2:
# Running post-processing
# Args: a folder of hdf5 image (only one unet result in hdf5 format in that folder!)
#       a folder of 2D tif mask slices
#       (optional) an output folder (it's not necessary but recommend to provide)
#       (optional) a threshold to remove small pieces 
#       (optional) a threshold to remove the object if it falls in the mask less than percentage.
#                  If this is not provided, objects will be excluded if its center falls out of the mask.
#       (optional) a parameter indicating whether writing hdf5 result back to tiff slices
# Output: a hdf5 image or 2D tif slices with postprocessed synapses
#         csv files indicating synapses location, size, and number of voxels
# Usage: bash Pipeline_split.sh -post -i <input_hdf5_directory> -m <mask_tiff_directory> -o <output_result_directory> -t <number_of_voxels_threshold_to_remove_small_piece> -p <mask_overlap_percentage_threshold_to_remove_object> -s
# Written by Sherry Ding @ SciCompSoft


# A function to print the usage of this manuscript
usage()
{
    echo "Usage for running unet on 2D tiff slices:"
    echo "bash Pipeline_split_small_subvolume.sh -unet -i <input_tiff_directory> -o <output_result_directory>"
    echo "Usage for running post-processing on unet results:"
    echo "bash Pipeline_split_small_subvolume.sh -post -i <input_hdf5_directory> -m <mask_tiff_directory> -o <output_result_directory> -t <number_of_voxels_threshold_to_remove_small_piece> -p <mask_overlap_percentage_threshold_to_remove_object> -s"
    echo "-o <output_result_directory>, -t <number_of_voxels_threshold_to_remove_small_piece>, -p <mask_overlap_percentage_threshold_to_remove_object> and -s (hfd5 result to tiff) are optional"
}


######## Main ########
# Directory of the script, change if you move the singularity image
SCRIPT_DIR=/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN
# Directory of the input data
INPUT_DIR=""
# Directory for the output result
OUTPUT_DIR=""
# Directory of the mask
MASK_DIR=""
# Number of voxels threshold to remove small pieces
THRESHOLD=10
# Mask overlap percentage threshold to remove objects
PERCENTAGE=1.0
# Hdf5 result to tiff
TO_TIFF=false
# A random number to distinguish each call of this script
RANDIDX=$RANDOM

######## Case 1
if [[ $1 == "-unet" ]]; then
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
    bsub -J "tiftohdf${RANDIDX}_img" -n 3 -o $OUTPUT_DIR/img_tif2hdf.log \
    "singularity run -B /groups/dickson/dicksonlab/ -B /nrs/dickson/ $SCRIPT_DIR/singularity_2D_new_postprocess.simg tif_to_h5.py -i $INPUT_DIR -o $OUTPUT_DIR"
    # Get the dimension of image
    A_IMG=`ls $INPUT_DIR/*.tif | head -n 1`
    HEIGHT=`identify $A_IMG | cut -d ' ' -f 3 | cut -d 'x' -f 1`  
    WIDTH=`identify $A_IMG | cut -d ' ' -f 3 | cut -d 'x' -f 2`  
    SLICE=`ls $INPUT_DIR/*.tif | wc -l`
    # Number of loops in x-dimension
    if [[ $(( WIDTH%1000 ))>=500 || $(( WIDTH/1000 ))==0 ]]; then
        NUM_ROW=$(( WIDTH/1000+1 ))
    else
        NUM_ROW=$(( WIDTH/1000 ))
    fi
    # Number of loops in y-dimension
    if [[ $(( HEIGHT%1000 ))>=500 || $(( HEIGHT/1000 ))==0 ]]; then
        NUM_COL=$(( HEIGHT/1000+1 ))
    else
        NUM_COL=$(( HEIGHT/1000 ))
    fi
    # Number of loops in z-dimension
    if [[ $(( SLICE%1000 ))>=500 || $(( SLICE/1000 ))==0 ]]; then
        NUM_VOL=$(( SLICE/1000+1 ))
    else
        NUM_VOL=$(( SLICE/1000 ))
    fi
    # Loop to apply 3D U-Net to the whole image
    IDX=0
    for (( ROW=0; ROW<$NUM_ROW; ROW++ )); do
        for (( COL=0; COL<$NUM_COL; COL++ )); do
            for (( VOL=0; VOL<$NUM_VOL; VOL++ )); do
                MIN_ROW=$(( ROW*1000 ))
                MIN_COL=$(( COL*1000 ))
                MIN_VOL=$(( VOL*1000 )) 
                if [[ $ROW == $(( NUM_ROW-1 )) ]]; then
                    MAX_ROW=$WIDTH
                else
                    MAX_ROW=$(( ROW*1000+1000 ))
                fi
                if [[ $COL == $(( NUM_COL-1 )) ]]; then
                    MAX_COL=$HEIGHT
                else
                    MAX_COL=$(( COL*1000+1000 ))
                fi
                if [[ $VOL == $(( NUM_VOL-1 )) ]]; then
                    MAX_VOL=$SLICE
                else
                    MAX_VOL=$(( VOL*1000+1000 ))
                fi
                # Submit GPU jobs
                ((IDX++))
                bsub -w "done("tiftohdf${RANDIDX}_img")" -J "unet${RANDIDX}_$IDX" -n 4 -gpu "num=1" -q gpu_rtx -o $OUTPUT_DIR/unet_${MIN_ROW}_${MIN_COL}_${MIN_VOL}.log \
                "singularity run --nv -B /misc/local/matlab-2018b/ -B /groups/dickson/dicksonlab/ -B /nrs/dickson/ $SCRIPT_DIR/singularity_2D_new_postprocess.simg unet_gpu.py -i $OUTPUT_DIR/slices_to_volume.h5 -l $MIN_ROW,$MIN_COL,$MIN_VOL,$MAX_ROW,$MAX_COL,$MAX_VOL"
            done
        done
    done

######## Case 2
elif [[ $1 == "-post" ]]; then
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
            -p)
                PERCENTAGE=$2
                shift 2
                ;;
            -s)
                TO_TIFF=true
                shift
                ;;
        esac
    done
    if [[ `ls $INPUT_DIR/*.h5 | wc -l` != 1 ]]; then # Error if number of h5 file in input directory is not 1
        echo "ERROR! Only one .h5 file is allowed in the input directory."
        usage
        exit 1
    fi
    # Create output directory if not exist
    if [[ ( $OUTPUT_DIR == "" ) || ( $OUTPUT_DIR == $INPUT_DIR ) ]]; then 
        OUTPUT_DIR=$INPUT_DIR/RESULT_$RANDIDX
    fi
    mkdir -p $OUTPUT_DIR
    # Copy input hdf5 file (unet result) into output directory
    bsub -J "cp_input${RANDIDX}" -n 32 -o /dev/null "cp $INPUT_DIR/*.h5 $OUTPUT_DIR"
    if [[ ( $MASK_DIR == "" ) || ( `ls $MASK_DIR/*.tif | wc -l` == 0 ) ]]; then # Error if there is no mask
        echo "ERROR! Please provide mask."
        usage
        exit 1
    fi
    mkdir -p $OUTPUT_DIR/MASK
    # Tiff to hdf5 for mask slices, output slices_to_volume.h5 file into $OUTPUT_DIR/MASK
    bsub -w "done("cp_input${RANDIDX}")" -J "tiftohdf${RANDIDX}_mask" -n 3 -o $OUTPUT_DIR/mask_tif2hdf.log \
    "singularity run -B /groups/dickson/dicksonlab/ -B /nrs/dickson/ $SCRIPT_DIR/singularity_2D_new_postprocess.simg tif_to_h5.py -i $MASK_DIR -o $OUTPUT_DIR/MASK"
    # Get the dimension of image
    A_IMG=`ls $MASK_DIR/*.tif | head -n 1`
    HEIGHT=`identify $A_IMG | cut -d ' ' -f 3 | cut -d 'x' -f 1`  
    WIDTH=`identify $A_IMG | cut -d ' ' -f 3 | cut -d 'x' -f 2`  
    SLICE=`ls $MASK_DIR/*.tif | wc -l`
    # Number of loops in x-dimension
    if [[ $(( WIDTH%500 ))>=200 || $(( WIDTH/500 ))==0 ]]; then
        NUM_ROW=$(( WIDTH/500+1 ))
    else
        NUM_ROW=$(( WIDTH/500 ))
    fi
    # Number of loops in y-dimension
    if [[ $(( HEIGHT%500 ))>=200 || $(( HEIGHT/500 ))==0 ]]; then
        NUM_COL=$(( HEIGHT/500+1 ))
    else
        NUM_COL=$(( HEIGHT/500 ))
    fi
    # Number of loops in z-dimension
    if [[ $(( SLICE%500 ))>=200 || $(( SLICE/500 ))==0 ]]; then
        NUM_VOL=$(( SLICE/500+1 ))
    else
        NUM_VOL=$(( SLICE/500 ))
    fi
    # Loop to run post-processing on the whole image
    INPUT_IMG=`ls $OUTPUT_DIR/*.h5`
    IDX=0
    for (( ROW=0; ROW<$NUM_ROW; ROW++ )); do
        for (( COL=0; COL<$NUM_COL; COL++ )); do
            for (( VOL=0; VOL<$NUM_VOL; VOL++ )); do
                MIN_ROW=$(( ROW*500 ))
                MIN_COL=$(( COL*500 ))
                MIN_VOL=$(( VOL*500 )) 
                if [[ $ROW == $(( NUM_ROW-1 )) ]]; then
                    MAX_ROW=$WIDTH
                else
                    MAX_ROW=$(( ROW*500+500 ))
                fi
                if [[ $COL == $(( NUM_COL-1 )) ]]; then
                    MAX_COL=$HEIGHT
                else
                    MAX_COL=$(( COL*500+500 ))
                fi
                if [[ $VOL == $(( NUM_VOL-1 )) ]]; then
                    MAX_VOL=$SLICE
                else
                    MAX_VOL=$(( VOL*500+500 ))
                fi
                # Submit CPU jobs
                ((IDX++))
                bsub -w "done("tiftohdf${RANDIDX}_mask")" -J "post${RANDIDX}_$IDX" -n 3 -o $OUTPUT_DIR/post_${MIN_ROW}_${MIN_COL}_${MIN_VOL}.log \
                "singularity run -B /misc/local/matlab-2018b/ -B /groups/dickson/dicksonlab/ -B /nrs/dickson/ $SCRIPT_DIR/singularity_2D_new_postprocess.simg postprocess_cpu.py -i $INPUT_IMG -l $MIN_ROW,$MIN_COL,$MIN_VOL,$MAX_ROW,$MAX_COL,$MAX_VOL -m $OUTPUT_DIR/MASK/slices_to_volume.h5 -t $THRESHOLD -p $PERCENTAGE"
            done
        done
    done
    if [[ $TO_TIFF == "true" ]]; then
        bsub -w "done("post${RANDIDX}_*")" -J "hdftotif${RANDIDX}" -n 3 -o $OUTPUT_DIR/result_hdf2tif.log \
        "singularity run -B /groups/dickson/dicksonlab/ -B /nrs/dickson/ $SCRIPT_DIR/singularity_2D_new_postprocess.simg h5_to_tif.py -i $INPUT_IMG -o $OUTPUT_DIR/tif_results"
    fi

# -unet or -post has to be provided as the first parameter
else
    echo "ERROR! Please provide whether running unet or post-processing!"
    usage
    exit 1
fi