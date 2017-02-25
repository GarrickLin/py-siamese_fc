import numpy as np
import scipy.io as sio
import mxnet as mx
import cPickle as pickle
import CustomMxOp

def move_weight_axis(x):
    return np.moveaxis(np.moveaxis(x, 2, 0), 3, 0)

def load_model_from_matlab(mat_model_path, raw_model_path):
    mdata = sio.loadmat(mat_model_path)
    mdata = mdata['model'][0]
    n_params = len(mdata)
    print n_params

    need_move_axis = ['conv1f', 'conv2f', 'conv3f', 'conv4f', 'conv5f']
    
    model_dict = {}
    
    for param in mdata:
        name = param[0][0]
        data = param[1]
        if name in need_move_axis:
            data = move_weight_axis(data)
        elif not name.endswith('x'):
            data = data.flatten()
        
        print name, data.shape
        
        model_dict[name] = data
    
    pickle.dump(model_dict, open(raw_model_path, mode='wb'), protocol=pickle.HIGHEST_PROTOCOL)  
    
def get_sym_siamese_fc():
    # conv1
    data = mx.sym.Variable('data')
    net = mx.sym.Convolution(data, name='conv1', kernel=(11,11), num_filter=96, stride=(2,2))
    net = mx.sym.Custom(net, name='bn1', op_type='custbatchnorm')
    net = mx.sym.Activation(net, name='relu1', act_type='relu')   
    net = mx.sym.Pooling(net, name='pool1', kernel=(3,3), pool_type='max', stride=(2,2))
    
    # conv2
    nets = mx.sym.SliceChannel(net, num_outputs=2, name="sliced1")
    net1 = mx.sym.Convolution(nets[0], name='conv21', kernel=(5,5), num_filter=128, stride=(1,1))
    net2 = mx.sym.Convolution(nets[1], name='conv22', kernel=(5,5), num_filter=128, stride=(1,1))
    net = mx.sym.Concat(net1, net2, name="conv2")
    net = mx.sym.Custom(net, name='bn2', op_type='custbatchnorm')
    net = mx.sym.Activation(net, name='relu2', act_type='relu')
    net = mx.sym.Pooling(net, name='pool2', kernel=(3,3), pool_type='max', stride=(2,2))
    
    # conv3
    net = mx.sym.Convolution(net, name='conv3', kernel=(3,3), num_filter=384, stride=(1,1))
    net = mx.sym.Custom(net, name='bn3', op_type='custbatchnorm')
    net = mx.sym.Activation(net, name='relu3', act_type='relu')    
    
    # conv4
    nets = mx.sym.SliceChannel(net, num_outputs=2, name="sliced2")
    net1 = mx.sym.Convolution(nets[0], name='conv41', kernel=(3,3), num_filter=192, stride=(1,1))
    net2 = mx.sym.Convolution(nets[1], name='conv42', kernel=(3,3), num_filter=192, stride=(1,1))
    net = mx.sym.Concat(net1, net2, name="conv4")
    net = mx.sym.Custom(net, name='bn4', op_type='custbatchnorm')
    net = mx.sym.Activation(net, name='relu4', act_type='relu')    
    
    # conv5
    nets = mx.sym.SliceChannel(net, num_outputs=2, name="sliced3")
    net1 = mx.sym.Convolution(nets[0], name='conv51', kernel=(3,3), num_filter=128, stride=(1,1))
    net2 = mx.sym.Convolution(nets[1], name='conv52', kernel=(3,3), num_filter=128, stride=(1,1))
    net = mx.sym.Concat(net1, net2, name="conv5")
    
    return net

def gen_mx_model(raw_model_path, mx_model_path, mode="rgb"):
    net = get_sym_siamese_fc()        
    
    model = mx.mod.Module(net)
    data_iter = mx.io.NDArrayIter(data=np.zeros((1,3,127,127)))
    model.bind(data_shapes=data_iter.provide_data)
    
    raw_model = pickle.load(open(raw_model_path, "rb"))    

    if mode == "bgr":
        #print "conv1 shape", raw_model['conv1f'].shape
        #print raw_model['conv1f'][0,0,0,0], raw_model['conv1f'][0,2,0,0]
        # swap channels
        raw_model['conv1f'][:, 0, :, :], raw_model['conv1f'][:, 2, :, :] = raw_model['conv1f'][:, 2, :, :], raw_model['conv1f'][:, 0, :, :].copy()                
        #print raw_model['conv1f'][0,0,0,0], raw_model['conv1f'][0,2,0,0]
        
    arg_params = {
        "conv1_weight": mx.nd.array(raw_model['conv1f']),
        "conv1_bias":  mx.nd.array(raw_model['conv1b']),
        "bn1_beta":  mx.nd.array(raw_model['bn1b']),
        "bn1_gamma":  mx.nd.array(raw_model['bn1m']),
        "bn1_moving_mean": mx.nd.array(raw_model['bn1x'][:,0]),
        "bn1_moving_sigma" : mx.nd.array(raw_model['bn1x'][:,1]),    
        
        "conv21_weight": mx.nd.array(raw_model['conv2f'][:128]),
        "conv21_bias": mx.nd.array(raw_model['conv2b'][:128]),
        "conv22_weight": mx.nd.array(raw_model['conv2f'][128:]),
        "conv22_bias": mx.nd.array(raw_model['conv2b'][128:]),   
        "bn2_beta":  mx.nd.array(raw_model['bn2b']),
        "bn2_gamma":  mx.nd.array(raw_model['bn2m']),
        "bn2_moving_mean": mx.nd.array(raw_model['bn2x'][:,0]),
        "bn2_moving_sigma" : mx.nd.array(raw_model['bn2x'][:,1]),    
        
        "conv3_weight": mx.nd.array(raw_model['conv3f']),
        "conv3_bias":  mx.nd.array(raw_model['conv3b']),
        "bn3_beta":  mx.nd.array(raw_model['bn3b']),
        "bn3_gamma":  mx.nd.array(raw_model['bn3m']),
        "bn3_moving_mean": mx.nd.array(raw_model['bn3x'][:,0]),
        "bn3_moving_sigma" : mx.nd.array(raw_model['bn3x'][:,1]),
        
        "conv41_weight": mx.nd.array(raw_model['conv4f'][:192]),
        "conv41_bias": mx.nd.array(raw_model['conv4b'][:192]),
        "conv42_weight": mx.nd.array(raw_model['conv4f'][192:]),
        "conv42_bias": mx.nd.array(raw_model['conv4b'][192:]),   
        "bn4_beta":  mx.nd.array(raw_model['bn4b']),
        "bn4_gamma":  mx.nd.array(raw_model['bn4m']),
        "bn4_moving_mean": mx.nd.array(raw_model['bn4x'][:,0]),
        "bn4_moving_sigma" : mx.nd.array(raw_model['bn4x'][:,1]),      
        
        "conv51_weight": mx.nd.array(raw_model['conv5f'][:128]),
        "conv51_bias": mx.nd.array(raw_model['conv5b'][:128]),
        "conv52_weight": mx.nd.array(raw_model['conv5f'][128:]),
        "conv52_bias": mx.nd.array(raw_model['conv5b'][128:]),         
        }

    mx_model_path = mx_model_path + "_" + mode
    model.init_params(arg_params=arg_params)
        
    #out_params = model.get_params()
    #print out_params
    
    model.save_checkpoint(prefix=mx_model_path, epoch=1)   
    

def adjust_data(data):
    """
        adjust the input from (h, w, c) to ( 1, c, h, w) for network input

    Parameters:
    ----------
        in_data: numpy array of shape (h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w)
    """
    
    if data.dtype is not np.dtype('float32'):
        data = data.astype(np.float32)
        print "convert to float32"
    
    data = np.expand_dims(data, axis=0)
    data = np.moveaxis(data, 3, 1)
    
    return data

def test_bgr_model():
    mx_model_path = "model/mxmodel_bgr"
    model = mx.model.FeedForward.load(mx_model_path, 1, ctx=mx.cpu(0))
    import cv2
    img = cv2.imread("images/z_crop.jpg")
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = adjust_data(img)
    res = model.predict(img)
    print res.shape
    print res[0][0]

def test_model(mx_model_path, mode="rgb"):
    mx_model_path = mx_model_path + "_" + mode
    import time
    model = mx.model.FeedForward.load(mx_model_path, 1, ctx=mx.cpu(0))

    z_crop = sio.loadmat("data/z_crop.mat")["z_crop"]
    
    #import cv2
    #draw = z_crop.copy().astype(np.uint8)
    #draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    #cv2.imshow("show", draw)
    #cv2.waitKey(0)
    
    z_crop = np.expand_dims(z_crop, axis=0)
    z_crop = np.moveaxis(z_crop, 3, 1)    
    
    print "z_crop", z_crop.shape
    time0 = time.time()
    res = model.predict(z_crop)
    print "time used", time.time() - time0   
    
    print res[0][0]
    
if __name__ == "__main__":
    mat_model_path = "model/model.mat"
    raw_model_path = "model/model_dict.pkl"
    mx_model_path = "model/mxmodel"
    
    #load_model_from_matlab(mat_model_path, raw_model_path)
    #gen_mx_model(raw_model_path, mx_model_path, mode="rgb")
    #test_model(mx_model_path, "rgb")
    
    gen_mx_model(raw_model_path, mx_model_path, mode="bgr")
    #test_bgr_model()

