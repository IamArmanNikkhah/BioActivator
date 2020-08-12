import tensorflow as tf

class bioActivatorV2(tf.keras.layers.Layer):
    def __init__(self, batch_size , **kwargs):
        super(bioActivatorV2, self).__init__(**kwargs)
        self.action_potential = 5
        self.threshold = 0
        self.initial_potential = -2.0
        self.batch_size = batch_size
        ##########################################
        self.constrain_receptor = tf.keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)
        self.constrain_frequency = tf.keras.constraints.NonNeg()
        self.receptor_initilizer = tf.keras.initializers.RandomUniform(minval=-1., maxval=1.)
        self.frequency_initializer = tf.keras.initializers.Ones()
        self.potential_initilizer = tf.keras.initializers.Constant(-2.)
        
    
    def build(self,batch_input_shape):
        input_shape_old = batch_input_shape
        input_shape = batch_input_shape[1:]
        ##############################
        self.potential = self.add_weight(shape=[self.batch_size]+input_shape, initializer=self.potential_initilizer, trainable=False, name='potential')
        self.frequency = self.add_weight(shape=input_shape, initializer= self.frequency_initializer, trainable=True, constraint=self.constrain_frequency, dtype='float32', name='frequency')
        self.receptor = self.add_weight(shape=input_shape, initializer= self.receptor_initilizer, trainable=True, constraint=self.constrain_receptor, dtype='float32', name='receptor')
        ##############################
        self.out = self.potential
        self.temp = self.potential
        ##############################
        super().build(batch_input_shape)
    
    @tf.function(autograph=True)    
    def call(self, inputs):
        self.temp.assign(self.potential + inputs, name='new_potential')
        #self.potential = self.temp
        self.potential.assign(self.temp)
        self.potential = tf.where(self.potential > 0, self.initial_potential, self.potential)
        self.out = tf.where(self.out > 0, 1.0, 0.0)
        freq = tf.math.multiply(self.frequency , self.action_potential)
        self.out = tf.math.multiply(self.out, freq)
        self.out = tf.math.multiply(self.out, self.receptor)
        return self.out
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list())
        
 class myModel(tf.keras.Model):
    def __init__(self,  batch_number, **kwargs):
        super().__init__(**kwargs)
        self.bio1 = bioActivatorV2(batch_number)
        self.conv2 = tf.keras.layers.Conv2D(32, 3, activation=None)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.drop1 = tf.keras.layers.Dropout(0.25)
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.drop2 = tf.keras.layers.Dropout(0.5)
        self.final = tf.keras.layers.Dense(10, activation='softmax')
    
    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation=None, input_shape=(28,28,1), data_format = 'channels_last')
        super().build(input_shape)

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.bio1(out)
        out = self.conv2(out)
        out = self.pool1(out)
        out = self.drop1(out)
        out = self.flat(out)
        out = self.dense1(out)
        out = self.drop2(out)
        out = self.final(out)
        return out
