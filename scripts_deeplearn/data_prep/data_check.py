import nrrd 
import os
from skimage.measure import label, regionprops
import numpy as np 
from scipy.ndimage.morphology import grey_closing


def check_conn_comp(file_path):
    """
    check number of objects(synapses) in each mask image
    file_path: directory of the mask floder
    return a dictionary {file_name: num_parts}
    """
    assert os.path.exists(file_path), \
        "Folder does not exist!"
    
    num_comp = {}
    all_files = os.listdir(file_path)
    for curr_file in all_files:
        data, head = nrrd.read(file_path+curr_file)
        data[data!=0] = 1
        labels = label(data, neighbors=8)
        num_comp.update({curr_file: np.amax(labels)})
    
    return num_comp


def merge_remove_comp(img_name):
    """
    remove components that have less than 10 voxels
    greyscale closing (merge) components that have more than 10 voxels
    """
    data, head = nrrd.read(img_name)
    new_data = np.zeros(data.shape, dtype=data.dtype)
    new_data[:,:,:] = data[:,:,:]
    data[data!=0] = 1
    data_label = label(data, neighbors=8)
    region_prop = regionprops(data_label)

    # remove components that have less than 10 voxels
    total_region = len(region_prop)
    for i in range(len(region_prop)):
        if region_prop[i].area < 10:
            new_data[data_label==region_prop[i].label] = 0
            total_region -= 1

    # merge big separated components
    if total_region > 1:
        sz = 2
        num_region = total_region
        while num_region > 1:
            sz += 2
            new_data_cp = np.zeros(new_data.shape, dtype=new_data.dtype)
            new_data_cp[:,:,:] = new_data[:,:,:]
            new_data_cp = grey_closing(new_data_cp, size=(sz,sz,sz))
            new_data_cp[new_data_cp!=0] = 1
            new_data_label = label(new_data_cp, neighbors=8)
            num_region = np.amax(new_data_label)
        if sz > 6:
            print("Double check "+img_name)
        new_data = grey_closing(new_data, size=(sz,sz,sz))

    nrrd.write(img_name+".nrrd", new_data)
    return None
        

def check_duplication(file_path):
    """
    check if there is duplicated masks in a folder
    """
    assert os.path.exists(file_path), \
        "Folder does not exist!"

    all_files = os.listdir(file_path)
    data, head = nrrd.read(file_path+all_files[0])
    data_all = np.zeros(data.shape, dtype=data.dtype)
    for curr_file in all_files:
        data, head = nrrd.read(file_path+curr_file)
        data[data!=0] = 1
        data_all += data
    
    dup_file_all = []
    if np.amax(data_all) > 1:
        print("There is duplicated masks in this floder!")
        data_all[data_all==1] = 0
        for curr_file in all_files:
            data, head = nrrd.read(file_path+curr_file)
            if np.amax(data*data_all) > 0:
                dup_file_all.append(curr_file)
        for i in dup_file_all:
            print(i)
    else:
        print("No duplicated masks.")
    
    return dup_file_all


def list_duplication_pairs(file_path, file_list):
    """
    print pairs of duplicated/overlapped masks
    file_path: directory of the mask floder 
    file_lsit: a list of duplicated/overlapped mask/image file names
    """
    assert len(file_list) != 0, \
        print("File list is empty, there is no duplicated masks!")

    while file_list:
        img, head = nrrd.read(file_path+file_list[0])
        img[img!=0] = 1
        dup_list = []
        for i in range(1, len(file_list)):
            curr_img, head = nrrd.read(file_path+file_list[i])
            curr_img[curr_img!=0] = 1
            if np.amax(img*curr_img) > 0:
                dup_list.append(file_list[i])

        print("{} images are duplicated/overlapped".format(len(dup_list)+1))
        print(file_list[0])
        for j in range(len(dup_list)):
            print(dup_list[j])
            file_list.remove(dup_list[j])        
        file_list.remove(file_list[0])

    return None


def check_syn_size(file_path):
    """
    check synapse size (number of voxels and bounding box size in x,y,z direction)
    return syn_size: a dictionary with file name as key and [NmVx, (x,y,z)] as value
    """
    assert os.path.exists(file_path), \
        "Folder does not exist!"

    syn_size = {}
    all_files = os.listdir(file_path)
    for curr_file in all_files:
        data, head = nrrd.read(file_path+curr_file)
        data[data!=0] = 1
        labels = label(data, neighbors=8)
        region_prop = regionprops(labels)
        for i in range(len(region_prop)):
            num_voxel = region_prop[i].area
            min_row, min_col, min_slice, max_row, max_col, max_slice = region_prop[i].bbox
            box_size = (max_row-min_row, max_col-min_col, max_slice-min_slice)
            syn_size.update({curr_file: [num_voxel, box_size]})
        
    return syn_size


def check_junk_size(file_name):
    """
    check junk size (number of voxels and bounding box size in x,y,z direction)
    return junk_size: a list in [NmVx, (x,y,z)]
    """
    assert os.path.exists(file_name), \
        "File does not exist!"

    junk_size = {}
    data, head = nrrd.read(file_name)
    labels = label(data, neighbors=8)
    region_prop = regionprops(labels)
    for i in range(len(region_prop)):
        num_voxel = region_prop[i].area
        min_row, min_col, min_slice, max_row, max_col, max_slice = region_prop[i].bbox
        box_size = (max_row-min_row, max_col-min_col, max_slice-min_slice)
        junk_size.update({"junk_"+str(i):[num_voxel, box_size]})
    
    return junk_size


def xyz_size(size_dict):
    """
    synapse/junk size in x,y,z axis, and number of voxels
    size_dict: a dictionary includes {"file name": [num_voxel, (x_size, y_size, z_size)]}
    rerurn a list in [num_voxel, x_size, y_size, z_size]
    """
    size_list = []
    for val in size_dict.values():
        num_voxel = val[0]
        size_3d = val[1]
        size = [num_voxel, size_3d[0], size_3d[1], size_3d[2]]
        size_list.append(size)

    return np.asarray(size_list)


def min_max_size(syn_size):
    """
    find the smallest and largest synapse/junk size in number of voxels and bounding box
    syn_size: a dictionary with number of voxels and bounding box size {"syn_name.nrrd":[num_voxel,(rows,cols,slices)]}
    """
    min_voxel = 125000000
    min_row = 500
    min_col = 500
    min_slice = 500
    max_voxel = 0
    max_row = 0
    max_col = 0
    max_slice = 0
    for val in syn_size.values():
        num_voxel = val[0]
        if num_voxel < min_voxel:
            min_voxel = num_voxel
        elif num_voxel > max_voxel:
            max_voxel = num_voxel

        box_size = val[1]
        if box_size[0] < min_row:
            min_row = box_size[0]
        elif box_size[0] > max_row:
            max_row = box_size[0]
        if box_size[1] < min_col:
            min_col = box_size[1]
        elif box_size[1] > max_col:
            max_col = box_size[1]
        if box_size[2] < min_slice:
            min_slice = box_size[2]
        elif box_size[2] > max_slice:
            max_slice = box_size[2]
    min_size = [min_voxel, (min_row, min_col, min_slice)]
    max_size = [max_voxel, (max_row, max_col, max_slice)]

    return min_size, max_size


def list_large_synapse(syn_size):
    """
    List synapse file names that have more than 50 voxels in any axis
    syn_size: a dictionary with file name as key and [NmVx, (x,y,z)] as value
    """
    large_syn = {}
    for key, val in syn_size.items():
        val_size = val[1]
        if val_size[0]>50 or val_size[1]>50 or val_size[2]>50:
            large_syn.update({key:val})
            print(key)
    return large_syn
        

def main():
    folder_name = input("Enter folder name (end with '/'): ")
    while not os.path.exists(folder_name):
        print("...No such folder, try again...")
        folder_name = input("Enter folder name (end with '/'): ")
    
    # check duplication
    print("Check duplication...")
    dup_file_all = check_duplication(folder_name)

    if not dup_file_all:  # if no duplicated files, check connectivity in each mask
        print("Check connectivity in each mask...")
        dict_comp = check_conn_comp(folder_name)
        for key, val in dict_comp.items():
            if val != 1:
                if val != 0:
                    print("Processing " + key + ": " + str(val))
                    merge_remove_comp(folder_name+key)
                else:
                    print("No object, please double check "+ key)
    else:  # if there is duplicated files, list the pairs
        print("Duplicated masks are:")
        list_duplication_pairs(folder_name, dup_file_all)


if __name__ == "__main__":
    main()