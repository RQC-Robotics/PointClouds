class PCEncoder(Model):
  def __init__(self, latent_dim, num_points, dims=(3, 64, 128)):
    super(PCEncoder, self).__init__()
    self.latent_dim = latent_dim
    self.num_points = num_points
    self.c1 = layers.Conv1D(64,  1, activation='relu', input_shape=(num_points, dims[0]))
    self.c2 = layers.Conv1D(128, 1, activation='relu', input_shape=(num_points, dims[1]))
    self.c3 = layers.Conv1D(latent_size, 1, activation='relu', input_shape=(num_points, dims[2]))

    self.bn1 = layers.BatchNormalization()
    self.bn2 = layers.BatchNormalization()
    self.bn3 = layers.BatchNormalization()

    self.mp = tf.keras.layers.MaxPooling1D(pool_size=num_points, strides=1)

    self.sequential = tf.keras.Sequential([self.c1, self.bn1, 
                                        self.c2, self.bn2, 
                                        self.c3, self.bn3, 
                                        self.mp, 
                                        layers.Flatten(input_shape=(1, latent_size))])
    
  def call(self, x):
      return self.sequential(x)

class PCDecoder(Model):
  def __init__(self, latent_dim, num_points, dims=(128, 256, 3)):
    super(PCDecoder, self).__init__()
    self.latent_dim = latent_dim
    self.num_points = num_points

    self.l1 = layers.Dense(dims[0], activation='relu')
    self.l2 = layers.Dense(dims[1], activation='relu')
    self.l3 = layers.Dense(num_points*dims[2], activation='relu')

    self.bn1 = layers.BatchNormalization()
    self.bn2 = layers.BatchNormalization()
    self.bn3 = layers.BatchNormalization()

    self.sequential = tf.keras.Sequential([self.l1, self.bn1, 
                                      self.l2, self.bn2, 
                                      self.l3, self.bn3, 
                                      layers.Reshape((num_points, dims[2]))])
    
  def call(self, x):
      return self.sequential(x)


class AutoEncoder(Model):
  def __init__(self, latent_dim, num_points, enc_dims=(3, 64, 128), dec_dims=(128, 256, 3)):
    super(AutoEncoder, self).__init__()
    self.latent_dim = latent_dim
    self.num_points = num_points
    self.encoder = PCEncoder(self.latent_dim, self.num_points, dims=enc_dims)
    self.decoder = PCDecoder(self.latent_dim, self.num_points, dims=dec_dims)

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
