from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, UpSampling3D, Cropping3D
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Reshape
from keras.layers import concatenate
from keras.models import Model


def _conv_block(input_tensor, n_filters, kernel_size=(3, 3, 3), batch_normalization=True, padding='same', use_bias=False):
    """
    3D convolutional layer (+ batch normalization) followed by ReLu activation
    """
    layer = Conv3D(filters=n_filters, kernel_size=kernel_size, padding=padding, use_bias=use_bias)(input_tensor)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    return layer


def vgg_like_v1(input_shape, base_filters=32):
    """ 
    a VGG structure model
    input_shape: input size in (x,y,z)
    """
    input_shape = input_shape + (1,)  # one channel
    inputs = Input(shape=input_shape)  # 48x48x48

    conv1 = _conv_block(input_tensor=inputs, n_filters=base_filters)  
    conv1 = _conv_block(input_tensor=conv1, n_filters=base_filters)  
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)  # 24x24x24

    conv2 = _conv_block(input_tensor=pool1, n_filters=2*base_filters) 
    conv2 = _conv_block(input_tensor=conv2, n_filters=2*base_filters)  
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)  # 12x12x12

    conv3 = _conv_block(input_tensor=pool2, n_filters=4*base_filters)
    conv3 = _conv_block(input_tensor=conv3, n_filters=4*base_filters)
    conv3 = _conv_block(input_tensor=conv3, n_filters=4*base_filters)
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)  # 6x6x6

    full1 = Flatten()(pool3)
    full1 = Dense(units=4*base_filters, activation='relu')(full1)
    full1 = Dropout(0.5)(full1)

    full2 = Dense(units=4*base_filters, activation='relu')(full1)
    full2 = Dropout(0.5)(full2)

    predictions = Dense(units=1, activation='sigmoid')(full2)

    model = Model(inputs=inputs, outputs=predictions)

    return model


def vgg_like_v2(input_shape, base_filters=32):
    """ 
    a VGG structure model, used this one in final object-wise detection
    input_shape: input size in (x,y,z)
    """
    input_shape = input_shape + (1,)  # one channel
    inputs = Input(shape=input_shape)  # 48x48x48

    conv1 = _conv_block(input_tensor=inputs, n_filters=base_filters, padding='valid')  # 46x46x46
    conv1 = _conv_block(input_tensor=conv1, n_filters=base_filters, padding='valid')  # 44x44x44
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)  # 22x22x22

    conv2 = _conv_block(input_tensor=pool1, n_filters=2*base_filters, padding='valid')  # 20x20x20
    conv2 = _conv_block(input_tensor=conv2, n_filters=2*base_filters, padding='valid')  # 18x18x18
    conv2 = _conv_block(input_tensor=conv2, n_filters=2*base_filters, padding='valid')  # 16x16x16
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)  # 8x8x8

    full1 = Flatten()(pool2)
    full1 = Dense(units=4*base_filters, activation='relu')(full1)
    full1 = Dropout(0.5)(full1)

    full2 = Dense(units=4*base_filters, activation='relu')(full1)
    full2 = Dropout(0.5)(full2)

    predictions = Dense(units=1, activation='sigmoid')(full2)

    model = Model(inputs=inputs, outputs=predictions)

    return model


def vgg_like_v3(input_shape, base_filters=32):
    """ 
    a VGG structure model
    input_shape: input size in (x,y,z)
    """
    input_shape = input_shape + (1,)  # one channel
    inputs = Input(shape=input_shape)  # 48x48x48

    conv1 = _conv_block(input_tensor=inputs, n_filters=base_filters, padding='valid')  # 46x46x46
    conv1 = _conv_block(input_tensor=conv1, n_filters=base_filters, padding='valid')  # 44x44x44
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)  # 22x22x22

    conv2 = _conv_block(input_tensor=pool1, n_filters=2*base_filters, padding='valid')  # 20x20x20
    conv2 = _conv_block(input_tensor=conv2, n_filters=2*base_filters, padding='valid')  # 18x18x18
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)  # 9x9x9

    full1 = Flatten()(pool2)
    full1 = Dense(units=4*base_filters, activation='relu')(full1)
    full1 = Dropout(0.5)(full1)

    full2 = Dense(units=4*base_filters, activation='relu')(full1)
    full2 = Dropout(0.5)(full2)

    predictions = Dense(units=1, activation='sigmoid')(full2)

    model = Model(inputs=inputs, outputs=predictions)

    return model


def unet_like(input_shape, base_filters=32):
    """
    a U-net structure model
    input_shape: input size in (x,y,z)
    """
    input_shape = input_shape + (1,)
    inputs = Input(shape=input_shape)  # 64x64x64

    # down-sampling
    down1 = _conv_block(input_tensor=inputs, n_filters=base_filters, padding='valid')  # 62x62x62
    down1 = _conv_block(input_tensor=down1, n_filters=base_filters, padding='valid')  # 60x60x60
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(down1)  # 30x30x30

    down2 = _conv_block(input_tensor=pool1, n_filters=2*base_filters, padding='valid')  # 28x28x28
    down2 = _conv_block(input_tensor=down2, n_filters=2*base_filters, padding='valid')  # 26x26x26
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(down2)  # 13x13x13

    center = _conv_block(input_tensor=pool2, n_filters=4*base_filters, padding='valid')  # 11x11x11
    center = _conv_block(input_tensor=center, n_filters=4*base_filters, padding='valid')  # 9x9x9

    # up-sampling
    up2 = concatenate([Cropping3D(((4, 4), (4, 4), (4, 4)))(down2), UpSampling3D(size=(2, 2, 2))(center)])  # 18x18x18
    up2 = _conv_block(input_tensor=up2, n_filters=2*base_filters, padding='valid')  # 16x16x16
    up2 = _conv_block(input_tensor=up2, n_filters=2*base_filters, padding='valid')  # 14x14x14

    up1 = concatenate([Cropping3D(((16, 16), (16, 16), (16, 16)))(down1), UpSampling3D(size=(2, 2, 2))(up2)])  # 28x28x28
    up1 = _conv_block(input_tensor=up1, n_filters=base_filters, padding='valid')  # 26x26x26
    up1 = _conv_block(input_tensor=up1, n_filters=base_filters, padding='valid')  # 24x24x24

    predictions = Conv3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid', use_bias=False)(up1)  # 24x24x24

    model = Model(inputs=inputs, outputs=predictions)

    return model


def deepmask_like(input_shape, base_filters=32):
    """
    a DeepMask structure model
    input_shape: input size in (x,y,z)
    base_filters: base filters in VGG
    """
    input_shape = input_shape + (1,)
    inputs = Input(shape=input_shape, name='in')  # 64x64x64

    # VGG
    conv1 = _conv_block(input_tensor=inputs, n_filters=base_filters) 
    conv1 = _conv_block(input_tensor=conv1, n_filters=base_filters) 
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)  # 32x32x32

    conv2 = _conv_block(input_tensor=pool1, n_filters=2*base_filters)  
    conv2 = _conv_block(input_tensor=conv2, n_filters=2*base_filters)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)  # 16x16x16

    conv3 = _conv_block(input_tensor=pool2, n_filters=4*base_filters)
    conv3 = _conv_block(input_tensor=conv3, n_filters=4*base_filters)
    conv3 = _conv_block(input_tensor=conv3, n_filters=4*base_filters)  # 16x16x16 shared layer

    # Segmentation prediction
    seg_predict = _conv_block(input_tensor=conv3, n_filters=4*base_filters, kernel_size=(1, 1, 1))  # 16x16x16
    seg_predict = Flatten()(seg_predict)
    seg_predict = Dense(units=4*base_filters)(seg_predict)
    # new output size (1/4 of the original per dimension)
    new_size = int(input_shape[0]/4), int(input_shape[1]/4), int(input_shape[2]/4)  # 16x16x16
    seg_predict = Dense(units=new_size[0]*new_size[1]*new_size[2], activation='sigmoid')(seg_predict)  # orig: no activation
    seg_predict = Reshape(target_shape=(new_size[0], new_size[1], new_size[2]), name='seg_out')(seg_predict)

    # Score prediction
    score_predict = MaxPooling3D(pool_size=(2, 2, 2))(conv3)  # 8x8x8
    score_predict = Flatten()(score_predict)
    score_predict = Dense(units=4*base_filters, activation='relu')(score_predict)
    score_predict = Dropout(0.5)(score_predict)
    score_predict = Dense(units=8*base_filters, activation='relu')(score_predict)
    score_predict = Dropout(0.5)(score_predict)
    score_predict = Dense(units=1, activation='sigmoid', name='score_out')(score_predict)  # orig: no activation

    model = Model(inputs=inputs, outputs=[seg_predict, score_predict])

    return model