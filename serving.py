# coding:utf-8 
'''
created on 2018/7/18

@author:Dxq
'''
import numpy
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
import numpy as np

tf.app.flags.DEFINE_string("host", "0.0.0.0", "TensorFlow Serving server ip")
tf.app.flags.DEFINE_integer("port", 9000, "TensorFlow Serving server port")
tf.app.flags.DEFINE_string("model_name", "dxq", "The model name")
tf.app.flags.DEFINE_integer("model_version", -1, "The model version")
tf.app.flags.DEFINE_string("signature_name", "", "The model signature name")
tf.app.flags.DEFINE_float("request_timeout", 100.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS

# Create gRPC client
channel = implementations.insecure_channel(FLAGS.host, FLAGS.port)
# print(channel)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
# print(stub)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'dxq'
trx = np.zeros(shape=[1, 224, 224, 3])
# trx = np.arange(100, step=1, dtype=np.float32)
# print(trx.shape)
# trx = np.reshape(trx, [100, 1])
request.inputs['x'].CopyFrom(tf.contrib.util.make_tensor_proto(trx, shape=[1, 224, 224, 3]))
# print(request)
result = stub.Predict(request, FLAGS.request_timeout)
# print(result)
# def main():
#     # Generate inference data
#     keys = numpy.asarray([1, 2, 3, 4])
#     keys_tensor_proto = tf.contrib.util.make_tensor_proto(keys, dtype=tf.int32)
#     features = numpy.asarray(
#         [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [9, 8, 7, 6, 5, 4, 3, 2, 1], [9, 9, 9, 9, 9, 9, 9, 9, 9]])
#     features_tensor_proto = tf.contrib.util.make_tensor_proto(
#         features, dtype=tf.float32)
#
#     # request.model_spec.name = FLAGS.model_name
#     # if FLAGS.model_version > 0:
#     #     request.model_spec.version.value = FLAGS.model_version
#     # if FLAGS.signature_name != "":
#     #     request.model_spec.signature_name = FLAGS.signature_name
#     # request.inputs["keys"].CopyFrom(keys_tensor_proto)
#     # request.inputs["features"].CopyFrom(features_tensor_proto)
#     #
#     # Send request
#     # result = stub.Predict(request, FLAGS.request_timeout)
#     # print(result)
#
#
# if __name__ == "__main__":
#     main()
