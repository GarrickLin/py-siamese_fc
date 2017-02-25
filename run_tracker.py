import mxnet as mx
import numpy as np
import CustomMxOp
import cv2
from minpy.core import Function
import os
import glob
from utils import GetRectange

adjust_f = 0.0010
adjust_b = -2.1484

def imshow(img, winname="display", wk=0):
    show = img
    if show.dtype == np.float32:
        show = img.astype(np.uint8)
    #print show.dtype
    cv2.imshow(winname, show)
    cv2.waitKey(wk)
    
def avoid_empty_position(r_max, c_max, params):
    if r_max is None:
        r_max = np.ceil(params['scoreSize']/2.)
    if c_max is None:
        c_max = np.ceil(params['scoreSize']/2.)
    return (r_max, c_max)

def cross_correlation_factory(data_shape, kernel_shape):
    batch, num_filter, y, x = kernel_shape
    net = mx.sym.Variable('x')
    net = mx.sym.Convolution(net, name='conv', kernel=(y, x), num_filter=1, no_bias=True)
    conv = Function(net, input_shapes={'x': data_shape})
    return conv

def cross_correlation(data, kernel):
    batch, num_filter, y, x = kernel.shape
    net = mx.sym.Variable('x')
    net = mx.sym.Convolution(net, name='conv', kernel=(y, x), num_filter=1, no_bias=True)
    conv = Function(net, input_shapes={'x': data.shape})    
    #print conv._param_shapes
    res = conv(x=data, conv_weight=kernel)
    return res
    
def tracker_eval(net_x, s_x, z_features, x_crops, targetPosition, window, p, Conv=None):
    """
    runs a  forward pass of the search-region branch of the pre-trained Fully-Convolutional Siamese,
    reusing the feature of the exemplar z computed at the first frame.
    """
    # forward pass, using the pyramid of scaled crops as a "batch"
    x_crops = adjust_data(x_crops)
    data_iter = mx.io.NDArrayIter(x_crops)
    #net_x.bind(data_shapes=data_iter.provide_data, for_training=False)
    x_features = net_x.predict(data_iter)
    if Conv is None:
        Conv = cross_correlation_factory(x_features.shape, z_features.shape)    
    responseMaps = Conv(x=x_features, conv_weight=z_features).asnumpy()
    responseMaps = responseMaps * adjust_f + adjust_b
    upsz = p['scoreSize'] * p['responseUp']
    #responseMapsUP = np.zeros((upsz, upsz, p['numScale']), dtype=np.float32)
    responseMapsUP = []
    # Choose the scale whose response map has the highest peak
    if p['numScale'] > 1:
        currentScaleID = int(p['numScale']/2)
        bestScale = currentScaleID
        bestPeak = -float('Inf')
        for s in range(p['numScale']):
            if p['responseUp'] > 1:
                # upsample to improve accuracy
                responseMapsUP.append(cv2.resize(responseMaps[s,0,:,:,], (upsz, upsz), interpolation=cv2.INTER_CUBIC))
            else:
                responseMapsUP.append(responseMaps[s,0,:,:,])
            thisResponse = responseMapsUP[-1]
            # penalize change of scale
            if s != currentScaleID:
                thisResponse = thisResponse * p['scalePenalty']
            thisPeak = np.max(thisResponse)
            if thisPeak > bestPeak:
                bestPeak = thisPeak
                bestScale = s
        responseMap = responseMapsUP[bestScale]
    else:
        #responseMap = responseMapsUP
        responseMap = cv2.resize(responseMaps[0,0,:,:,], (upsz, upsz), interpolation=cv2.INTER_CUBIC)
        bestScale = 0
    
    # make the response map sum to 1
    responseMap = responseMap - np.min(responseMap)
    responseMap = responseMap / np.sum(responseMap)
    # apply windowing
    responseMap = (1-p['wInfluence'])*responseMap + p['wInfluence']*window
    r_max, c_max = np.unravel_index(responseMap.argmax(), responseMap.shape)
    #r_max, c_max = avoid_empty_position(r_max, c_max, p)
    p_corr = np.array((r_max, c_max))
    # Convert to crop-relative coordinates to frame coordinates
    # displacement from the center in instance final representation ...
    disp_instanceFinal = p_corr - int(p['scoreSize']*p['responseUp']/2)
    #  ... in instance input ...
    disp_instanceInput = disp_instanceFinal * p['totalStride'] / p['responseUp']
    # ... in instance original crop (in frame coordinates)
    disp_instanceFrame = disp_instanceInput * s_x / p['instanceSize']
    # position within frame in frame coordinates
    newTargetPosition = targetPosition + disp_instanceFrame
    
    return newTargetPosition, bestScale

def config_params():
    p = {}
    # These are the default hyper-params for SiamFC-3S
    # The ones for SiamFC (5 scales) are in params-5s.txt
    p['numScale'] = 3
    p['scaleStep'] = 1.0375
    p['scalePenalty'] = 0.9745
    p['scaleLR'] = 0.59 # damping factor for scale update
    p['responseUp'] = 16 # upsampling the small 17x17 response helps with the accuracy
    p['windowing'] = 'cosine' # to penalize large displacements
    p['wInfluence'] = 0.176 # windowing influence (in convex sum)
    p['net_base_path'] = 'model/'
    p['net'] = 'mxmodel_bgr'
    # execution, visualization, benchmark
    p['seq_base_path'] = 'images/demo-sequences/'
    p['video'] = 'vot15_bag'
    p['visualization'] = False
    p['gpus'] = 0
    p['bbox_output'] = False
    p['fout'] = -1
    # Params from the network architecture, have to be consistent with the training
    p['exemplarSize'] = 127
    p['instanceSize'] = 255
    p['scoreSize'] = 17
    p['totalStride'] = 8
    p['contextAmount'] = 0.5
    p['subMean'] = False
    
    return p

def get_axis_aligned_BB(region):
    """
    computes axis-aligned bbox with same area as the rotated one (REGION)
    """
    region = np.array(region)
    nv = region.size
    assert (nv==8 or nv==4)
    if nv==8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = np.min(region[0::2])
        x2 = np.max(region[0::2])
        y1 = np.min(region[1::2])
        y2 = np.max(region[1::2])
        A1 = np.linalg.norm(region[0:2]-region[2:4]) * np.linalg.norm(region[2:4]-region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1/A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[3]
        h = region[4]
        cx = x + w / 2
        cy = y + h / 2
    return (cx-1, cy-1, w, h)

def frame_generator(vpath, mode):
    if mode == "images":
        def frames():
            for img in glob.glob(os.path.join(vpath, "*.jpg")):
                yield cv2.imread(img).astype(np.float32)
        return frames()
    elif mode == "video" or mode == "camera":
        def frames():
            cap = cv2.VideoCapture(vpath)
            while 1:
                ret, frame = cap.read()
                if ret:
                    yield frame.astype(np.float32)
                else:
                    break
        return frames()      

def load_video_info(base_path, video):
    # full path to the video's files
    video_path = os.path.join(base_path, video, "imgs/")
    # load ground truth from text file
    ground_truth_path = os.path.join(base_path, video, "groundtruth.txt")
    ground_truth = open(ground_truth_path)
    raw1 = ground_truth.readline()
    #print "raw1", raw1
    region = map(float, raw1.strip().split(","))
    cx, cy, w, h = get_axis_aligned_BB(region)
    pos = (cy, cx)
    target_sz = (h, w)
    
    return frame_generator(video_path, mode="images"), np.array(pos), np.array(target_sz)

def load_camera(device):
    cap = cv2.VideoCapture(device)
    rector = GetRectange()
    while 1:
        ready, frame = cap.read()
        if not ready:
            print "device", device, "is not ready"
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key != -1:
            break
    pos, target_sz = rector.getRect(frame)
    def frames():
        while 1:
            ret, frame = cap.read()
            if ret:
                yield frame.astype(np.float32)        
            else:
                print "device", device, "is not ready"
    return frames(), frame, np.array(pos), np.array(target_sz)

def get_subwindow_tracking(im, pos, model_sz, original_sz, avgChans):
    """
    Obtain image sub-window, padding with avg channel if area goes outside of border
    """
    if original_sz is None:
        original_sz = model_sz
    sz = original_sz
    im_sz = im.shape
    # make sure the size is not too small
    assert min(im_sz[:2]) > 2, "the size is too small"
    c = (np.array(sz) + 1) / 2
    
    # check out-of-bounds coordinates, and set them to black
    context_xmin = round(pos[1] - c[1])
    context_xmax = context_xmin + sz[1] - 1
    context_ymin = round(pos[0] - c[0])
    context_ymax = context_ymin + sz[0] - 1
    left_pad = max(0, int(-context_xmin))
    top_pad = max(0, int(-context_ymin))
    right_pad = max(0, int(context_xmax - im_sz[1] + 1))
    bottom_pad = max(0, int(context_ymax - im_sz[0] + 1))
    
    context_xmin = int(context_xmin + left_pad)
    context_xmax = int(context_xmax + left_pad)
    context_ymin = int(context_ymin + top_pad)
    context_ymax = int(context_ymax + top_pad)
    
    if top_pad or left_pad or bottom_pad or right_pad:
        b = np.pad(im[:,:,0], ((top_pad,bottom_pad),(left_pad,right_pad)), mode='constant', constant_values=avgChans[0])
        g = np.pad(im[:,:,1], ((top_pad,bottom_pad),(left_pad,right_pad)), mode='constant', constant_values=avgChans[1])
        r = np.pad(im[:,:,2], ((top_pad,bottom_pad),(left_pad,right_pad)), mode='constant', constant_values=avgChans[2])    
        im = cv2.merge((b,g,r))
        #imshow(im)
        
    im_patch_original = im[context_ymin:context_ymax+1, context_xmin:context_xmax+1, :]
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, model_sz)
    else:
        im_patch = im_patch_original
    
    return im_patch, im_patch_original        

def adjust_data(data):
    """
        adjust the input from (h, w, c) to ( 1, c, h, w) for network input
    
    Parameters:
    ----------
        in_data: numpy array of shape (h, w, c) or (n, h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w) or (n, c, h, w)
    """    
    if data.dtype is not np.dtype('float32'):
        data = data.astype(np.float32)
        print "convert to float32"
    
    if len(data.shape) < 4:
        data = np.expand_dims(data, axis=0)
    data = np.moveaxis(data, -1, -3)
    
    return data

def make_scale_pyramid(im, targetPosition, in_side_scaled, out_side, avgChans, stats, p):
    """
    computes a pyramid of re-scaled copies of the target (centered on TARGETPOSITION)
    and resizes them to OUT_SIDE. If crops exceed image boundaries they are padded with AVGCHANS.
    
    """
    in_side_scaled = np.round(in_side_scaled)
    max_target_side = int(round(in_side_scaled[-1]))
    min_target_side = int(round(in_side_scaled[0]))
    beta = out_side / float(min_target_side)
    # size_in_search_area = beta * size_in_image
    # e.g. out_side = beta * min_target_side
    search_side = int(round(beta * max_target_side))
    search_region, _ = get_subwindow_tracking(im, targetPosition, (search_side, search_side), (max_target_side, max_target_side), avgChans)
    if p['subMean']:
        pass
    assert round(beta*min_target_side) == int(out_side)
    
    tmp_list = []
    tmp_pos = ((search_side-1)/2., (search_side-1)/2.)
    for s in range(p['numScale']):
        target_side = round(beta * in_side_scaled[s])    
        tmp_region, _ = get_subwindow_tracking(search_region, tmp_pos, (out_side, out_side), (target_side, target_side), avgChans)
        tmp_list.append(tmp_region)
        
    pyramid = np.stack(tmp_list)
    
    return pyramid

def tracker(demo=True):
    p = config_params()
    # Load two copies of the pre-trained network
    net_z = mx.mod.Module.load(p['net_base_path']+p['net'], 1, context=mx.gpu(0))
    data_iter = mx.io.NDArrayIter(data=np.zeros((1,3,p['exemplarSize'],p['exemplarSize'])))
    net_z.bind(data_shapes=data_iter.provide_data, for_training=False)    
    net_x = mx.mod.Module.load(p['net_base_path']+p['net'], 1, context=mx.gpu(0))
    data_iter = mx.io.NDArrayIter(data=np.zeros((3,3,p['instanceSize'],p['instanceSize'])))
    net_x.bind(data_shapes=data_iter.provide_data, for_training=False)    
    Conv = cross_correlation_factory((3,256,22,22), (1,256,6,6))
    
    if demo:
        imgFiles, targetPosition, targetSize = load_video_info(p['seq_base_path'], p['video'])
        im = imgFiles.next()
    else:
        imgFiles, im, targetPosition, targetSize = load_camera(0)
    
    wc_z = targetSize[1] + p['contextAmount']*np.sum(targetSize)
    hc_z = targetSize[0] + p['contextAmount']*np.sum(targetSize)
    s_z = np.sqrt(wc_z*hc_z)
    scale_z = p['exemplarSize'] / s_z
    
    
    d_search = (p['instanceSize'] - p['exemplarSize']) / 2
    pad = d_search / scale_z
    s_x = s_z + 2*pad
    
    # arbitrary scale saturation
    min_s_x = 0.2*s_x
    max_s_x = 5*s_x
    
    winsz = p['scoreSize'] * p['responseUp']
    if p['windowing'] == 'cosine':
        hann = np.hanning(winsz).reshape(winsz, 1)
        window = hann.dot(hann.T)
    elif p['windowing'] == 'uniform':
        window = np.ones((winsz, winsz), dtype=float32)
        
    # make the window sum 1
    window = window / np.sum(window)
    scales = np.array([p['scaleStep'] ** i for i in range(int(np.ceil(p['numScale']/2.)-p['numScale']), int(np.floor(p['numScale']/2)+1))])
    
    # prepare for first frame    
    # get avg for padding
    avgChans = np.mean(im, axis=(0,1))            
    # initialize the exemplar
    z_crop, _ = get_subwindow_tracking(im, targetPosition, (p['exemplarSize'],p['exemplarSize']), (round(s_z), round(s_z)), avgChans)
    #imshow(z_crop)
    if p['subMean']:
        pass
    # evaluate the offline-trained network for exemplar z features
    data_iter = mx.io.NDArrayIter(adjust_data(z_crop))
    z_features = net_z.predict(data_iter)          

    for i, im in enumerate(imgFiles):
        scaledInstance = s_x * scales
        scaledTarget = np.array([ targetSize*scale for scale in scales ])
        # extract scaled crops for search region x at previous target position
        x_crops = make_scale_pyramid(im, targetPosition, scaledInstance, p['instanceSize'], avgChans, None, p)
        # evaluate the offline-trained network for exemplar x features
        newTargetPosition, newScale = tracker_eval(net_x, round(s_x), z_features, x_crops, targetPosition, window, p, Conv)
        targetPosition = newTargetPosition
        # scale damping and saturation
        s_x = max(min_s_x, min(max_s_x, (1-p['scaleLR'])*s_x + p['scaleLR']*scaledInstance[newScale]))
        targetSize = (1-p['scaleLR'])*targetSize + p['scaleLR']*scaledTarget[newScale]            
 
        rectPosition = targetPosition - targetSize / 2.
        tl = tuple(np.round(rectPosition).astype(int)[::-1])
        br = tuple(np.round(rectPosition+targetSize).astype(int)[::-1])
        im_draw = im.astype(np.uint8)
        cv2.rectangle(im_draw, tl, br, (0, 255, 255), thickness=3)
        cv2.imshow("tracking", im_draw)
        cv2.waitKey(1)
        
    
if __name__ == "__main__":
    tracker()
    
    