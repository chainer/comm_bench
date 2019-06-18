import chainer
import chainermn
import numpy as np
from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50

def setup_comm(name):
    kwargs = {}
    if name == 'pure_nccl_fp16':
        name = 'pure_nccl'
        kwargs['allreduce_grad_dtype'] = np.float16
        kwargs['batched_copy'] = True
    comm = chainermn.create_communicator(name, **kwargs)
    return comm

def setup_model(model_name, label_num):
    model_cfgs = {
        'resnet50': {'class': ResNet50, 'score_layer_name': 'fc6',
                     'kwargs': {'arch': 'fb'}},
        'resnet101': {'class': ResNet101, 'score_layer_name': 'fc6',
                      'kwargs': {'arch': 'fb'}},
        'resnet152': {'class': ResNet152, 'score_layer_name': 'fc6',
                      'kwargs': {'arch': 'fb'}}
    }
    assert model_name in model_cfgs.keys()
    model_cfg = model_cfgs[model_name]
    extractor = model_cfg['class'](
        n_class=label_num, **model_cfg['kwargs'])
    extractor.pick = model_cfg['score_layer_name']
    model = chainer.links.Classifier(extractor)
    model.cleargrads()

    return model

def update_once(model):
    opt =  chainer.optimizers.MomentumSGD()
    opt.setup(model)
    import cupy as cp
    imgs = cp.ndarray((1, 3, 224, 224), dtype=np.float32)
    labels = cp.ndarray((1,), dtype=np.int32)
    labels += 1
    model(imgs, labels)
    opt.update()
