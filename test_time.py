# -*- coding:utf-8 -*-
import mxnet as mx
import time

ctx = mx.gpu()
x = mx.nd.random_normal(shape=(1,3,224,224), ctx=ctx)
data = mx.io.DataBatch([x])
sym, args, aux = mx.model.load_checkpoint('accnn/final_0_3x', 1)
mod = mx.mod.Module(symbol=sym, data_names=['data'], label_names=None, context=ctx)
mod.bind(data_shapes=[('data',(1,3,224,224))], for_training=False)
# mod = mx.mod.Module(symbol=sym, context=ctx,
#                     data_names=['data'], label_names=['xiaolei_label', 'dalei_label'])
# mod.bind(data_shapes=[('data', (1, 3, 224, 224))],
#          label_shapes=[('xiaolei_label', (1,)), ('dalei_label', (1,))],
#          for_training=False)
mod.set_params(arg_params=args, aux_params=aux, allow_missing=True)
begin = time.time()
print("begin")
for i in range(1):
    res = mod.forward(data)
    res = mod.get_outputs()[0]
    cls = mx.nd.argmax(res, axis=1).asscalar()
    print(cls)
print("end")
print(time.time()-begin)




