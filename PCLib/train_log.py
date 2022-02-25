import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow_graphics.nn.loss import chamfer_distance


def get_optimizer(lr):
  return tf.keras.optimizers.Adam(lr)
  
def get_callbacks(name, logdir):
  return [
    #tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]
  
  
def compile_and_fit(model, name, dataset, logdir, lr = 0.001,batch_size=64, optimizer=None, max_epochs=100):
  if optimizer is None:
    optimizer = get_optimizer(lr)
    
  model.compile(optimizer=optimizer, 
                loss=chamfer_distance.evaluate)
  model.summary()

  history = model.fit(
      dataset,
      batch_size=batch_size
      epochs=max_epochs,
      validation_split=0.1,
      callbacks=get_callbacks(name, logdir),
      verbose=1)
  return history
