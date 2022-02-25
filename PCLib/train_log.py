import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow_graphics.nn.loss import chamfer_distance
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots


def get_optimizer(lr):
  return tf.keras.optimizers.Adam(lr)
  
def get_callbacks(name, logdir):
  return [
    #tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]
  
  
def compile_and_fit(model, name, pc_data, logdir, lr = 0.001,batch_size=64, optimizer=None, max_epochs=100):
  if optimizer is None:
    optimizer = get_optimizer(lr)
  model.compile(optimizer=optimizer, 
                loss=chamfer_distance.evaluate)
  model.summary()

  n_train = pc_data.shape[0]
  steps_per_epoch = n_train//batch_size

  history = model.fit(
      pc_data,
      pc_data,
      steps_per_epoch = steps_per_epoch,
      epochs=max_epochs,
      validation_split=0.1,
      callbacks=get_callbacks(name, logdir),
      verbose=1)
  return history
