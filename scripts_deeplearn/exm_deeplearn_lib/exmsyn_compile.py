import os
import numpy as np 
from exm_deeplearn_lib.exmsyn_data import tif_read, augment_small_sample, augment_random
from scipy.ndimage import zoom
from random import shuffle, sample
import nrrd  


def _get_data_small_sample(img_mask_name, neg_mask_val=False, is_zoom=1.0):
    """
    for DeepMask network
    get an image and corresponding mask
    do augmentation (for small sample size type augmentation)
    change the data shape to 4D, data type to float
    img_mask_name: a tuple includes the name of an image (first str) and mask (second str)
    neg_mask_val: if set the mask value to -1 and 1 instead of 0 and 1
    is_zoom: scale of zooming the mask (1.0: no zoom)
    return a list of 4D images and a list of 4D masks
    """
    img, head =nrrd.read(img_mask_name[0])  # img = tif_read(img_mask_name[0])
    mask, head = nrrd.read(img_mask_name[1])  # mask = tif_read(img_mask_name[1])
    aug_img, aug_mask = augment_small_sample(img, mask)

    img_all=[]
    mask_all=[]
    for i in range(len(aug_img)):
        curr_img = aug_img[i]
        curr_img = np.float32(curr_img)
        img_4d = np.zeros((curr_img.shape[0],curr_img.shape[1],curr_img.shape[2],1), dtype=curr_img.dtype)
        img_4d[:,:,:,0] = curr_img            
        img_all.append(img_4d)

    # mask values are changed to 1 and 0(or -1), and zoom into a scale of the original size
    for m in range(len(aug_mask)):
        curr_mask = aug_mask[m]
        curr_mask = zoom(curr_mask, is_zoom)
        curr_mask = np.float32(curr_mask)
        curr_mask[curr_mask!=0]=1
        if neg_mask_val:
            curr_mask[curr_mask==0]=-1
        mask_all.append(curr_mask)

    return img_all, mask_all


def _get_data(img_mask_names, neg_mask_val=False, is_zoom=1.0):
    """
    for DeepMask network
    do random augmentation on a list of images and a list of corresponding masks
    change the data shape to 4D, data type to float
    img_mask_names: a list of tuples of str, each tuple includes the name of an image (first str) and mask (second str)
    neg_mask_val: if set the mask value to -1 and 1 instead of 0 and 1
    is_zoom: scale of zooming the mask (1.0: no zoom)
    return a list of 4D images and a list of 4D masks
    """
    imgs = []
    masks = []
    for i in range(len(img_mask_names)):
        curr_name = img_mask_names[i]
        curr_img, head = nrrd.read(curr_name[0])  # curr_img = tif_read(curr_name[0])
        curr_mask, head = nrrd.read(curr_name[1])  # curr_mask = tif_read(curr_name[1])
        imgs.append(curr_img)
        masks.append(curr_mask)
    # data augmentation
    aug_img, aug_mask = augment_random(imgs, masks)
    mu_img = np.float32(sum(aug_img))/len(aug_img) # mean img
    std_img = np.sqrt(sum((np.float32(x) - mu_img)**2 for x in aug_img) / len(aug_img))  # std img

    img_all = []
    mask_all = []
    for i in range(len(aug_img)):
        curr_img = aug_img[i]
        curr_img = np.float32(curr_img)
        curr_img = (curr_img - mu_img) / std_img  # normalization
        img_4d = np.zeros((curr_img.shape[0],curr_img.shape[1],curr_img.shape[2],1), dtype=curr_img.dtype)
        img_4d[:,:,:,0] = curr_img            
        img_all.append(img_4d)

    # mask values are changed to 1 and 0(or -1), and zoom into a scale of the original size
    for m in range(len(aug_mask)):
        curr_mask = aug_mask[m]
        curr_mask = zoom(curr_mask, is_zoom)
        curr_mask = np.float32(curr_mask)
        curr_mask[curr_mask!=0]=1
        if neg_mask_val:
            curr_mask[curr_mask==0]=-1
        mask_all.append(curr_mask)

    return img_all, mask_all


def get_file_names(file_path):
    """
    for DeepMask network
    generate list of tuples of image and mask names(str) from a list of folder names that stores images and masks
    each tuple includes the name of an image (first str) and mask (second str)
    """
    sample_names = []

    for i in range(len(file_path)):
        assert os.path.exists(file_path[i]), \
            "Folder of samples does not exist!"
    
        num_samples = len([name for name in os.listdir(file_path[i])])
        assert num_samples % 2 ==0, \
            "Number of images and masks does not match!"
        num_samples = int(num_samples/2)  # Number of samples

        for j in range(num_samples):
            img_name = file_path[i] + 'img_' + str(j) + '.nrrd'
            mask_name = file_path[i] + 'mask_' + str(j) + '.nrrd'
            img_mask_name = (img_name, mask_name)
            sample_names.append(img_mask_name)

    return sample_names


def prepare_data(img_mask_names, vali_pct=0.2, neg_mask_val=False, is_zoom=1.0):
    """
    for DeepMask network
    prepare input image and output mask for all images and masks, generate training set and validation set 
    img_mask_names: list of tuples of str, each tuple includes the name of an image (first str) and mask (second str)
    vali_pct: percentage of total data that are used as validation data
    neg_mask_val: if set the mask value to -1 and 1 instead of 0 and 1
    is_zoom: scale of zooming the mask (1.0 means no zooming)
    return training and validation dataset in 5D size as (num_data,:,:,:,channel=1)
    """
    num_data = len(img_mask_names)
    train_img = []
    train_mask = []
    test_img = []
    test_mask = []

    if vali_pct != 0:
        idx = np.random.permutation(num_data)
        test_idx = idx[:int(round(num_data*vali_pct))]
        train_idx = idx[int(round(num_data*vali_pct)):]

        for i in range(len(train_idx)):
            curr_name = img_mask_names[train_idx[i]]
            img, mask = _get_data_small_sample(curr_name, neg_mask_val, is_zoom)
            train_img.extend(img)
            train_mask.extend(mask)
    
        for j in range(len(test_idx)):
            curr_name = img_mask_names[test_idx[j]]
            img, mask = _get_data_small_sample(curr_name, neg_mask_val, is_zoom)
            test_img.extend(img)
            test_mask.extend(mask)

    else:
        for i in range(len(img_mask_names)):
            curr_name = img_mask_names[i]
            img, mask = _get_data_small_sample(curr_name, neg_mask_val, is_zoom)
            train_img.extend(img)
            train_mask.extend(mask)

    return np.asarray(train_img), np.asarray(train_mask), np.asarray(test_img), np.asarray(test_mask)


def gen_deepmask_batch(img_mask_names, labels, batch_sz=32, neg_mask_val=False, is_zoom=1.0):
    """
    a generator that yields training batches for deepmask network (feed keras fit_generator), no validation set
    img_mask_names: list of tuples of images and mask names (first str: image, second str: mask)
    labels: list of labels corresponding to each tuple in img_mask_names
    batch_sz: batch size
    neg_mask_val: if set the mask value to -1 and 1 instead of 0 and 1
    is_zoom: scale of zooming the mask (1.0 means no zooming)
    yield training images and masks in 5D size as (num_data,:,:,:,channel=1), and training scores in a 1D array
    """
    assert len(img_mask_names) == len(labels), \
        "Number of samples and labels are not match!"

    combined_names_labels = list(zip(img_mask_names,labels))
    shuffle(combined_names_labels)  # shuffle img_mask_names and corresponding labels
    img_mask_names, labels = zip(*combined_names_labels)
    img_mask_names = list(img_mask_names)
    labels = list(labels)
    start_id = 0  # starting sample id

    while True:
        if start_id + batch_sz >= len(img_mask_names):
            batch_img_mask_names = img_mask_names[start_id:]
            batch_labels = labels[start_id:]
            start_id = 0
        else:
            batch_img_mask_names = img_mask_names[start_id:start_id+batch_sz]
            batch_labels = labels[start_id:start_id+batch_sz]
            start_id += batch_sz
        
        train_img, train_mask = _get_data(batch_img_mask_names, neg_mask_val, is_zoom)
       
        train_img = np.asarray(train_img)
        train_mask = np.asarray(train_mask)
        train_score = np.float32(batch_labels)

        yield ({'in':train_img}, {'seg_out':train_mask, 'score_out':train_score})


def gen_deepmask_batch_general(pos_img_mask_names, neg_img_mask_names, batch_sz=32, neg_mask_val=False, is_zoom=1.0):
    """
    a generator that yields training batches for deepmask network (feed keras fit_generator), no validation set
    a general version suitable for unbalanced training samples
    pos_img_mask_names, neg_img_mask_names: list of tuples of images and mask names (1st str: image, 2nd str: mask) for positive and negative samples
    batch_sz: batch size
    neg_mask_val: if set the mask value to -1 and 1 instead of 0 and 1
    is_zoom: scale of zooming the mask (1.0 means no zooming)
    yield training images and masks in 5D size as (num_data,:,:,:,channel=1), and training scores in a 1D array
    """

    while True:
        pos_names = sample(pos_img_mask_names, k=int(batch_sz/2))
        pos_labels = [1] * len(pos_names)
        neg_names = sample(neg_img_mask_names, k=int(batch_sz/2))
        neg_labels = [0] * len(neg_names)

        batch_img_mask_names = pos_names + neg_names
        batch_labels = pos_labels + neg_labels
        combined_names_labels = list(zip(batch_img_mask_names, batch_labels))
        shuffle(combined_names_labels)  # shuffle batch samples
        batch_img_mask_names, batch_labels = zip(*combined_names_labels)
        batch_img_mask_names = list(batch_img_mask_names)
        batch_labels = list(batch_labels)
        
        train_img, train_mask = _get_data(batch_img_mask_names, neg_mask_val, is_zoom)
       
        train_img = np.asarray(train_img)
        train_mask = np.asarray(train_mask)
        train_score = np.float32(batch_labels)

        yield ({'in':train_img}, {'seg_out':train_mask, 'score_out':train_score})


def gen_vgg_batch(pos_img_names, neg_img_names, batch_sz=32):
    """
    a generator that yields training batches for vgg network (feed keras fit_generator)
    a general version suitable for unbalanced training samples, classifying between postive samples and negative samples
    pos_img_names, neg_img_names: list of images names for positive and negative samples
    batch_sz: batch size
    yield training images 5D size as (num_data,:,:,:,channel=1), and training scores in a 1D array
    """
    while True:
        pos_names = sample(pos_img_names, k=int(batch_sz/2))
        pos_labels = [1] * len(pos_names)
        neg_names = sample(neg_img_names, k=batch_sz-int(batch_sz/2))
        neg_labels = [0] * len(neg_names)

        batch_img_names = pos_names + neg_names
        batch_labels = pos_labels + neg_labels
        combined_names_labels = list(zip(batch_img_names, batch_labels))
        shuffle(combined_names_labels)  # shuffle batch samples
        batch_img_names, batch_labels = zip(*combined_names_labels)
        batch_img_names = list(batch_img_names)
        batch_labels = list(batch_labels)

        batch_img = []
        x_flip = np.random.randint(2, size=batch_sz)
        z_flip = np.random.randint(2, size=batch_sz)
        rot_angle = np.random.randint(4, size=batch_sz)
        
        for i in range(len(batch_img_names)):
            curr_img, head = nrrd.read(batch_img_names[i])
            # data augmentation
            if x_flip[i]:
                curr_img = np.flip(curr_img, axis=0)
            if z_flip[i]:
                curr_img = np.flip(curr_img, axis=2)
            if rot_angle[i]:
                curr_img = np.rot90(curr_img, rot_angle[i])

            curr_img = np.float32(curr_img)
            batch_img.append(curr_img)
        
        mu_img = sum(batch_img)/len(batch_img) # mean img
        std_img = np.sqrt(sum((x - mu_img)**2 for x in batch_img) / len(batch_img))  # std img

        imgs_final = np.zeros((batch_sz, mu_img.shape[0], mu_img.shape[1], mu_img.shape[2],1), dtype=mu_img.dtype)
        for i in range(len(batch_img)):
            curr_img = batch_img[i]
            # curr_img = (curr_img - mu_img) / np.maximum(std_img,1)  # normalization
            imgs_final[i,:,:,:,0] = curr_img
        
        batch_labels = np.asarray(batch_labels)

        yield imgs_final, batch_labels


def gen_unet_batch_v1(img_mask_names, crop_sz=(64,64,64), mask_sz=(24,24,24), batch_sz=32):
    """
    a generator that yields training batches for 3D-Unet network (randomly cropped images and corresponding masks)
    img_mask_names: list of tuples of images and mask names (1st str: image, 2nd str: mask)
    crop_sz: size in (x,y,z) of random cropping
    mask_sz: size of cropped mask in (x,y,z)
    batch_sz: batch size
    yield training images and masks in  5D size as (num_data,:,:,:,channel=1)
    """
    imgs = []
    masks = []
    # read all images and masks into lists 
    for i in range(len(img_mask_names)):
        curr_name = img_mask_names[i]
        curr_img, head = nrrd.read(curr_name[0])
        curr_mask, head = nrrd.read(curr_name[1])
        assert curr_img.shape == curr_mask.shape, "Image and mask size do not match!"
        
        curr_img = np.float32(curr_img)
        curr_img = (curr_img - curr_img.mean()) / curr_img.std()  # normalize image
        imgs.append(curr_img)
        curr_mask = np.float32(curr_mask)
        masks.append(curr_mask)

    batch_img = np.zeros((batch_sz, crop_sz[0], crop_sz[1], crop_sz[2], 1), dtype='float32')
    batch_mask = np.zeros((batch_sz, mask_sz[0], mask_sz[1], mask_sz[2], 1), dtype='float32')    
    
    while True:
        # randomly crop an image from imgs list
        idx = np.random.randint(0, len(imgs))
        img_for_crop = imgs[idx]
        mask_for_crop = masks[idx]
        num_crop = 0
        while num_crop < batch_sz:
            x = np.random.randint(0, img_for_crop.shape[0]-crop_sz[0])
            y = np.random.randint(0, img_for_crop.shape[1]-crop_sz[1])
            z = np.random.randint(0, img_for_crop.shape[2]-crop_sz[2])
            cropped_img = img_for_crop[x:x+crop_sz[0], y:y+crop_sz[1], z:z+crop_sz[2]]
            cropped_mask = mask_for_crop[x:x+crop_sz[0], y:y+crop_sz[1], z:z+crop_sz[2]]
            shrink_sz = (int((crop_sz[0]-mask_sz[0])/2), int((crop_sz[1]-mask_sz[1])/2), int((crop_sz[2]-mask_sz[2])/2))
            cropped_mask = cropped_mask[shrink_sz[0]:crop_sz[0]-shrink_sz[0], shrink_sz[1]:crop_sz[1]-shrink_sz[1], shrink_sz[2]:crop_sz[2]-shrink_sz[2]]
            # if include the random crop in training
            is_include = False
            num_syn_vxl = len(cropped_mask[cropped_mask==1])
            accept_prob = np.random.random()
            if num_syn_vxl > 2000 or accept_prob > 0.95:
                is_include = True
            elif 1000 < num_syn_vxl <= 2000 and accept_prob > 0.5:
                is_include = True
            elif 500 < num_syn_vxl <= 1000 and accept_prob > 0.75:
                is_include = True
            elif 0 < num_syn_vxl <=500 and accept_prob > 0.85:
                is_include = True
            
            # include the crop
            if is_include:
                batch_img[num_crop,:,:,:,0] = cropped_img
                batch_mask[num_crop,:,:,:,0] = cropped_mask
                num_crop += 1
        
        # data augmentation
        x_flip = np.random.randint(2, size=batch_sz)
        z_flip = np.random.randint(2, size=batch_sz)
        rot_angle = np.random.randint(4, size=batch_sz)
        for j in range(batch_sz):
            if x_flip[j]:
                batch_img[j,:,:,:,0] = np.flip(batch_img[j,:,:,:,0], axis=0)
                batch_mask[j,:,:,:,0] = np.flip(batch_mask[j,:,:,:,0], axis=0)
            if z_flip[j]:
                batch_img[j,:,:,:,0] = np.flip(batch_img[j,:,:,:,0], axis=2)
                batch_mask[j,:,:,:,0] = np.flip(batch_mask[j,:,:,:,0], axis=2)
            if rot_angle[j]:
                batch_img[j,:,:,:,0] = np.rot90(batch_img[j,:,:,:,0], rot_angle[j], axes=(0,1))
                batch_mask[j,:,:,:,0] = np.rot90(batch_mask[j,:,:,:,0], rot_angle[j], axes=(0,1))

        yield batch_img, batch_mask


def gen_unet_batch_v2(img_mask_junk_names, crop_sz=(64,64,64), mask_sz=(24,24,24), batch_sz=32):
    """
    a generator that yields training batches for 3D-Unet network (randomly cropped images and corresponding masks)
    img_mask_junk_names: list of tuples of images and traiing mask names (1st str: image, 2nd str: traiing mask 0-1, 3nd str: junk mask 0-255)
    crop_sz: size in (x,y,z) of random cropping
    mask_sz: size of cropped mask in (x,y,z)
    batch_sz: batch size
    yield training images and masks in  5D size as (num_data,:,:,:,channel=1)
    """
    imgs = []
    masks = []
    junks = []
    # read all images and masks into lists 
    for i in range(len(img_mask_junk_names)):
        curr_name = img_mask_junk_names[i]
        curr_img, head = nrrd.read(curr_name[0])
        curr_mask, head = nrrd.read(curr_name[1])
        curr_junk, head = nrrd.read(curr_name[2])
        assert curr_img.shape == curr_mask.shape, "Image and training mask size do not match!"
        assert curr_img.shape == curr_junk.shape, "Image and junk mask size do not match!"
        
        curr_img = np.float32(curr_img)
        curr_img = (curr_img - curr_img.mean()) / curr_img.std()  # normalize image
        imgs.append(curr_img)
        curr_mask = np.float32(curr_mask)
        masks.append(curr_mask)
        junks.append(curr_junk)    
    
    while True:
        batch_img = np.zeros((batch_sz, crop_sz[0], crop_sz[1], crop_sz[2], 1), dtype='float32')
        batch_mask = np.zeros((batch_sz, mask_sz[0], mask_sz[1], mask_sz[2], 1), dtype='float32')
        
        # randomly crop an image from imgs list
        idx = np.random.randint(0, len(imgs))
        img_for_crop = imgs[idx]
        mask_for_crop = masks[idx]
        junk_for_crop = junks[idx]  # only used for including enough junk crops
        num_crop = 0
        while num_crop < batch_sz:
            x = np.random.randint(0, img_for_crop.shape[0]-crop_sz[0])
            y = np.random.randint(0, img_for_crop.shape[1]-crop_sz[1])
            z = np.random.randint(0, img_for_crop.shape[2]-crop_sz[2])
            cropped_img = img_for_crop[x:x+crop_sz[0], y:y+crop_sz[1], z:z+crop_sz[2]]
            cropped_mask = mask_for_crop[x:x+crop_sz[0], y:y+crop_sz[1], z:z+crop_sz[2]]
            cropped_junk = junk_for_crop[x:x+crop_sz[0], y:y+crop_sz[1], z:z+crop_sz[2]]
            shrink_sz = (int((crop_sz[0]-mask_sz[0])/2), int((crop_sz[1]-mask_sz[1])/2), int((crop_sz[2]-mask_sz[2])/2))
            cropped_mask = cropped_mask[shrink_sz[0]:crop_sz[0]-shrink_sz[0], shrink_sz[1]:crop_sz[1]-shrink_sz[1], shrink_sz[2]:crop_sz[2]-shrink_sz[2]]
            cropped_junk = cropped_junk[shrink_sz[0]:crop_sz[0]-shrink_sz[0], shrink_sz[1]:crop_sz[1]-shrink_sz[1], shrink_sz[2]:crop_sz[2]-shrink_sz[2]]
            # if include the random crop in training
            is_include = False
            num_syn_vxl = len(cropped_mask[cropped_mask==1])
            num_junk_vxl = len(cropped_junk[cropped_junk==255])
            accept_prob = np.random.random()
            if num_syn_vxl > 500 or num_junk_vxl > 100 or accept_prob > 0.98:
                is_include = True
            elif (0 < num_syn_vxl <= 500 or 0 < num_junk_vxl <= 100) and accept_prob > 0.5:
                is_include = True
            
            # include the crop
            if is_include:
                batch_img[num_crop,:,:,:,0] = cropped_img
                batch_mask[num_crop,:,:,:,0] = cropped_mask
                num_crop += 1
        
        # data augmentation
        x_flip = np.random.randint(2, size=batch_sz)
        z_flip = np.random.randint(2, size=batch_sz)
        rot_angle = np.random.randint(4, size=batch_sz)
        for j in range(batch_sz):
            if x_flip[j]:
                batch_img[j,:,:,:,0] = np.flip(batch_img[j,:,:,:,0], axis=0)
                batch_mask[j,:,:,:,0] = np.flip(batch_mask[j,:,:,:,0], axis=0)
            if z_flip[j]:
                batch_img[j,:,:,:,0] = np.flip(batch_img[j,:,:,:,0], axis=2)
                batch_mask[j,:,:,:,0] = np.flip(batch_mask[j,:,:,:,0], axis=2)
            if rot_angle[j]:
                batch_img[j,:,:,:,0] = np.rot90(batch_img[j,:,:,:,0], rot_angle[j], axes=(0,1))
                batch_mask[j,:,:,:,0] = np.rot90(batch_mask[j,:,:,:,0], rot_angle[j], axes=(0,1))
        
        yield batch_img, batch_mask


def prepare_validation_set(img_mask_names, neg_mask_val=False, is_zoom=1.0):
    """
    prepare validation dataset for the deepmask network
    img_mask_names: list of tuples of str, each tuple includes the name of an image (first str) and mask (second str)
    neg_mask_val: if set the mask value to -1 and 1 instead of 0 and 1
    is_zoom: scale of zooming the mask (1.0 means no zooming)
    return validation images in 5D array and masks in 4D array
    """
    imgs = []
    masks = []
    for i in range(len(img_mask_names)):
        curr_name = img_mask_names[i]
        curr_img, head = nrrd.read(curr_name[0])  # curr_img = tif_read(curr_name[0])
        curr_mask, head = nrrd.read(curr_name[1])  # curr_mask = tif_read(curr_name[1])
        imgs.append(curr_img)
        masks.append(curr_mask)
    mu_img = np.float32(sum(imgs))/len(imgs) # mean img
    std_img = np.sqrt(sum((np.float32(x) - mu_img)**2 for x in imgs) / len(imgs))  # std img

    img_all = []
    mask_all = []
    for i in range(len(imgs)):
        curr_img = imgs[i]
        curr_img = np.float32(curr_img)
        curr_img = (curr_img - mu_img) / std_img  # normalization
        img_4d = np.zeros((curr_img.shape[0],curr_img.shape[1],curr_img.shape[2],1), dtype=curr_img.dtype)
        img_4d[:,:,:,0] = curr_img            
        img_all.append(img_4d)

    # mask values are changed to 1 and 0(or -1), and zoom into a scale of the original size
    for m in range(len(masks)):
        curr_mask = masks[m]
        curr_mask = zoom(curr_mask, is_zoom)
        curr_mask = np.float32(curr_mask)
        curr_mask[curr_mask!=0]=1
        if neg_mask_val:
            curr_mask[curr_mask==0]=-1
        mask_all.append(curr_mask)

    img_all = np.asarray(img_all)
    mask_all = np.asarray(mask_all)
    
    return img_all, mask_all


def prepare_vgg_validation_set(pos_img_names, neg_img_names):
    """
    prepare validation dataset (with augmentation) for the vgg network
    pos_img_names, neg_img_names: image file names for positive and negative samples
    return images and labels
    """
    vali_names = pos_img_names + neg_img_names
    vali_labels = [1]*len(pos_img_names) + [0]*len(neg_img_names)
    combined_names_labels = list(zip(vali_names, vali_labels))
    shuffle(combined_names_labels)
    vali_names, vali_labels = zip(*combined_names_labels)
    vali_names = list(vali_names)
    vali_labels = list(vali_labels)

    curr_img, head = nrrd.read(vali_names[0])
    vali_img = np.zeros((len(vali_names), curr_img.shape[0], curr_img.shape[1], curr_img.shape[2], 1), dtype=curr_img.dtype)
    for i in range(len(vali_names)):
        curr_img, head = nrrd.read(vali_names[i])
        vali_img[i,:,:,:,0] = curr_img
    
    # data augmentation
    x_flip = np.random.randint(2, size=vali_img.shape[0])
    z_flip = np.random.randint(2, size=vali_img.shape[0])
    rot_angle = np.random.randint(4, size=vali_img.shape[0])
    for j in range(vali_img.shape[0]):
        if x_flip[j]:
            vali_img[j,:,:,:,0] = np.flip(vali_img[j,:,:,:,0], axis=0)
        if z_flip[j]:
            vali_img[j,:,:,:,0] = np.flip(vali_img[j,:,:,:,0], axis=2)
        if rot_angle[j]:
            vali_img[j,:,:,:,0] = np.rot90(vali_img[j,:,:,:,0], rot_angle[j], axes=(0,1))
    
    vali_img = np.float32(vali_img)
    vali_labels = np.asarray(vali_labels)
    return vali_img, vali_labels