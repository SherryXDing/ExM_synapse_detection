import sys, getopt
from classification import *
import matlab.engine
from skimage.measure import label, regionprops
import numpy as np 
import time
import csv
import os


def remove_small_piece(out_path, img_file_name, threshold=10, individual_outpath=None):
    """
    remove blobs that have less than N voxels
    save final result to disk, output a .csv file indicating the location and size of each synapses
    (optional) output a folder of individual synapse masks
    Args:
    out_path: output directory
    img_file_name: tif image file for processing
    threshold: threshold to remove small blobs (default=10)
    individual_outpath: (optional) output directory of individual synapse masks (default=None)
    """

    print("Removing small blobs and save results to disk...")
    img = tif_read(img_file_name)
    img[img!=0] = 1
    label_img = label(img, neighbors=8)
    regionprop_img = regionprops(label_img)
    idx = 0
    csv_name = os.path.splitext(os.path.basename(img_file_name))[0]
    with open(out_path+'/stats_'+csv_name+'.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([' ID ', ' Num vxl ', ' centroid ', ' bbox row ', ' bbox col ', ' bbox vol '])

    for props in regionprop_img:
        num_voxel = props.area
        if num_voxel < threshold:  
            img[label_img==props.label] = 0
        else:
            idx += 1
            min_row, min_col, min_vol, max_row, max_col, max_vol = props.bbox
            bbox_row = (int(min_row), int(max_row))
            bbox_col = (int(min_col), int(max_col))
            bbox_vol = (int(min_vol), int(max_vol))
            center_row, center_col, center_vol = props.centroid
            center = (int(center_row), int(center_col), int(center_vol))
            csv_row = [str(idx), str(num_voxel), str(center), str(bbox_row), str(bbox_col), str(bbox_vol)]
            with open(out_path+'/stats_'+csv_name+'.csv', 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(csv_row)    
            if individual_outpath is not None:
                curr_region = np.zeros(img.shape, dtype=img.dtype)
                curr_region[label_img==props.label] = 255
                tif_write(curr_region, individual_outpath+'/'+str(idx)+'.tif')
        
    img[img!=0] = 255
    tif_write(img, img_file_name)
    return None 


def main(argv):
    """
    Main function
    """
    out_path = None
    img_file = None
    mask_file = None
    threshold = 10
    separate_mask = 0
    try:
        options, remainder = getopt.getopt(argv, "i:o:m:t:s:", ["img_file=","out_path=","mask_file=","threshold=","separate_mask="])
    except:
        print("ERROR:", sys.exc_info()[0]) 
        print("Usage: main_3d.py -i <image_file> -o <output_directory> -m <mask_file> -t <threshold> -s <output_individual_masks>")
        sys.exit(1)
    
    for opt, arg in options:
        if opt in ('-i', '--img_file'):
            img_file = arg   
        elif opt in ('-o', '--out_path'):
            out_path = arg
        elif opt in ('-m', '--mask_file'):
            mask_file = arg
        elif opt in ('-t', '--threshold'):
            threshold = int(arg)
        elif opt in ('-s', '--separate_mask'):
            separate_mask = int(arg)
    
    try:
        img = tif_read(img_file)
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        img_path = os.path.dirname(img_file)
    except:
        print("ERROR:", sys.exc_info()[0])
        sys.exit(1)

    if out_path is None:
        out_path = img_path
    elif not os.path.exists(out_path):
        os.mkdir(out_path)

    if mask_file is not None:
        try:
            mask = tif_read(mask_file)
            mask_name = os.path.splitext(os.path.basename(mask_file))[0]
        except:
            print("ERROR:", sys.exc_info()[0])
            sys.exit(1)    
    else:
        mask = None

    if separate_mask:
        if mask is not None:
            individual_outpath = out_path+'/'+mask_name+'_'+img_name+'_individual_masks'
        else:
            individual_outpath = out_path+'/'+img_name+'_individual_masks'
        if not os.path.exists(individual_outpath):
            os.mkdir(individual_outpath)
    else:
        individual_outpath = None

    start = time.time()
    print('#############################')
    img = unet_test(img=img, mask=mask)
    if mask is not None:       
        out_img_name = out_path+'/processed_'+mask_name+'_'+img_name+'.tif'
    else:
        out_img_name = out_path+'/processed_'+img_name+'.tif'
    tif_write(img, out_img_name)
    eng = matlab.engine.start_matlab()
    flag = eng.closing_watershed(out_img_name)
    eng.quit()
    remove_small_piece(out_path=out_path, img_file_name=out_img_name, threshold=threshold, individual_outpath=individual_outpath)
    end = time.time()
    print("DONE! Running time is {} seconds".format(end-start))


if __name__ == "__main__":
    main(sys.argv[1:])