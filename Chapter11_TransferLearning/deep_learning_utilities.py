import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
def display_random_imgs(ds):
    """

    """
    rnd_idx = np.unique(np.random.randint(ds.shape[0], size=(3)))
    rnd_labels = ds.loc[rnd_idx]['label']
    rnd_features = ds.loc[rnd_idx][ds.columns[1:]]
    for i in range(3):
        digit = np.array(rnd_features)[i].reshape(28,28)
        #plt.figure(figsize=(3,3))
        plt.subplot(2,2,i+1)
        print('label {0}'.format(labels_dict[rnd_labels.iloc[i]]))
        plt.imshow(digit, alpha=None, cmap=plt.cm.get_cmap('Greys'))

def get_uncompiled_model(nn_type=1, depth=2, an_per_layer=[64,64]):
    """ nn_type: 1-Multilayer Perceptron
        depth: Number of layers
    """
    if (nn_type==1):
        inputs = keras.Input(shape=(784,), name='input_imgs')
        #for layer_num in range(depth):
        #    layer_name = 'dense'+str(layer_num)
        x = layers.Dense(64, activation='relu',name='dense1')(inputs)
        x = layers.Dense(64,  activation='relu',name='dense2')(x)
        outputs = layers.Dense(10, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

    if (nn_type==2):
        inputs = keras.layers.Input(shape=(28,28,1))
        x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='valid')(inputs)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.Conv2D(64, (3,3), activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model( model_type=1,optimizer_inst=optimizers.Adam(learning_rate=1e-3),
                        loss_inst=losses.SparseCategoricalCrossentropy(from_logits=True),
                        desired_metrics=['sparse_categorical_accuracy']):
    """
    """
    if (model_type==1):
        model = get_uncompiled_model(nn_type=1)
        model.compile(optimizer=optimizer_inst, loss=loss_inst, metrics=desired_metrics)
    if (model_type==2):
        model = get_uncompiled_model(nn_type=2)
        model.compile(optimizer=optimizer_inst, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_optimizer_instance():
    """
        1: SGD() With or without momentum
        2: RMSProp - Gradient-based optimization technique proposed by Geoffry Hinton
        3: Adam(learning_rate)
    """
    pass

def get_loss_func_instance():
    """
        1: MeanSquaredError()
        2: KLDivergence()
        3: CosineSimilary()
        4: SparseCategoricalCrossentropy()
    """
    pass

def get_metrics():
    """
        1: accuracy
        2: AUC
        3: Recall
    """
    pass

def tfdata_generator(features, labels, is_training, batch_size=128):
    def map_fn(features, labels):
        x = tf.reshape(tf.cast(features, tf.float32),(28,28,1))
        y = tf.reshape(tf.cast(labels, tf.uint8), (10,))
        return x,y

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if is_training:
        dataset = dataset.shuffle(1000)
    dataset = dataset.map(map_fn)
    dataset = dataset.batch(64)
    dataset = dataset.repeat()
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def get_toy_dataset(source=1):
    """
    """
    #Load Dtaset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    y_train = to_categorical(y_train)
    x_train.reshape(x_train.shape[0], x_train.shape[1])
    training_set = tfdata_generator(x_train, y_train, is_training=True)

    return training_set

def training_pipeline():
    training_set = get_toy_dataset()
    model = get_compiled_model(model_type=2)
    model.fit(training_set, steps_per_epoch=50000//128, epochs=1, verbose=1 )

training_pipeline()

#comment
