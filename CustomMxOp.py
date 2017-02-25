import mxnet as mx
import minpy.numpy as np

class CustBatchNorm(mx.operator.CustomOp):    
    def forward(self, is_train, req, in_data, out_data, aux):
        #mx.nd.add(lhs, rhs)
        x = mx.nd.SwapAxis(in_data[0], 1, 3) # data         
        gamma = in_data[1] # gamma
        beta = in_data[2] # beta
        moving_mean = in_data[3] # mean
        moving_sigma = in_data[4] # sigma
        x_hat = (x - moving_mean) / (moving_sigma + 1e-9)
        out = gamma * x_hat + beta
        out = mx.nd.SwapAxis(out, 1, 3)
        self.assign(out_data[0], req[0], out)
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError
    
@mx.operator.register("custbatchnorm")
class CustBatchNormProp(mx.operator.CustomOpProp):
    def __init__(self, need_top_grad=False):
        super(CustBatchNormProp, self).__init__(need_top_grad)
        
    def list_arguments(self):
        return ['data', 'gamma', 'beta', 'moving_mean', 'moving_sigma']
    
    def list_outputs(self):
        return ['output']
    
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        other_shape = (in_shape[0][1],)
        output_shape = in_shape[0]
        return [data_shape, other_shape, other_shape, other_shape, other_shape], [output_shape], []
    
    def create_operator(self, ctx, in_shapes, in_dtypes):
        return CustBatchNorm()