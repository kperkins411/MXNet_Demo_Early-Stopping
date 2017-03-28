import mxnet as mx
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

#some constants
batch_size =32
numb_bin_digits = 10
numb_softmax_outputs = 2**numb_bin_digits
numb_Training_samples = numb_softmax_outputs*8
numb_Validation_samples = numb_softmax_outputs *2
num_epoch = 100

def getBinaryArrayAndLabels(numberrows, numbBinDigits):
    '''
    Generate numberrows that will repeat all possible values of numbBinDigits (for instance 4 digits generate 16 vals ->0000 to 1111)
    :param numberrows: how many rows of data
    :param numbBinDigits: how many binary digits
    :return: a train array shape(numberrows, numbBinDigits) , and a label array(shape(numberrows,)
    '''
    largestBinNumb = 2**numbBinDigits
    binStringStartsHere=2
    label =[row%largestBinNumb for row in range(numberrows)]
    train =  [[ int(bin(row%largestBinNumb)[binStringStartsHere:].zfill(numbBinDigits)[col]) for col in range(0,numbBinDigits)] for row in range(0,numberrows)]
    return np.asarray(train), np.asarray(label)

#create our model
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=numb_softmax_outputs*3)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = numb_softmax_outputs*2)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=numb_softmax_outputs)
mlp = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

#get train iterators
train_val,train_lab= getBinaryArrayAndLabels(numberrows=numb_Training_samples, numbBinDigits=numb_bin_digits)
itr_train=mx.io.NDArrayIter(train_val,train_lab,batch_size=batch_size,shuffle='True')

#get eval iterators
eval_val,eval_lab =getBinaryArrayAndLabels(numberrows=numb_Validation_samples, numbBinDigits=numb_bin_digits)
itr_val = mx.io.NDArrayIter(eval_val,eval_lab,batch_size=batch_size)

def epoc_end_callback_kp(epoch, symbol, arg_params, aux_params,epoch_train_eval_metrics):
    '''
    early stopping
    a function passed to fit(..) thats called at end of every epoch
    epoch_train_eval_metrics added to track training and validation accuracies both here and in
    mxnets base_module.py fit function.  If this function returns false, fit will stop training.
    Note: fit has been modified for this to work
    :param epoch:
    :param symbol:
    :param arg_params:
    :param aux_params:
    :param epoch_train_eval_metrics:
    :return:
    '''
    EVAL_ACC = 1
    for key in epoch_train_eval_metrics.keys():
        for epoch in epoch_train_eval_metrics[key].keys():
            len1 = len(epoch_train_eval_metrics[key])-1
            retval =  epoch_train_eval_metrics[key][len1][EVAL_ACC] < 1.0

            #TODO save model if retval == false

            return retval

#model
model = mx.mod.Module(context = mx.gpu(), symbol = mlp)

#train
model.fit(train_data=itr_train, eval_data = itr_val,eval_metric='acc', batch_end_callback =mx.callback.Speedometer(batch_size=batch_size,frequent=100), epoch_end_callback=epoc_end_callback_kp,optimizer='sgd',optimizer_params={'learning_rate':0.1,'momentum': 0.9},num_epoch=num_epoch)
