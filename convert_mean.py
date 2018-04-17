# Ref: https://github.com/BVLC/caffe/issues/290#issuecomment-62846228
# Modified by WildCat
import caffe
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: python convert_protomean.py proto.mean out.npy")

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('./original-caffe-models/DB_train_w32_5.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
np.save('mean.npy', out)