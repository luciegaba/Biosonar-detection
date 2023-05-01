from tensorflow.keras import layers, Model, optimizers

class OdontoceteCNNClassifier():
    """
    A custom CNN-based classifier for detecting odontocete clicks in spectrograms.
    
    Attributes:
        input_shape (tuple): The shape of the input spectrogram images.
    """

    def __init__(self, input_shape=(150, 150, 3)):
        super(OdontoceteCNNClassifier, self).__init__()
        self.input_shape = input_shape
        self.build_model()

    def build_model(self):
        """
        Builds the CNN model architecture.
        """
        # Input layer
        input_layer = layers.Input(shape=self.input_shape)

        # ConvBlock 1
        conv1 = layers.Conv2D(64, kernel_size=3, padding='same')(input_layer)
        bn1 = layers.BatchNormalization()(conv1)
        relu1 = layers.ReLU()(bn1)
        pool1 = layers.MaxPooling2D(pool_size=2)(relu1)

        # ConvBlock 2
        conv2 = layers.Conv2D(128, kernel_size=3, padding='same')(pool1)
        bn2 = layers.BatchNormalization()(conv2)
        relu2 = layers.ReLU()(bn2)
        pool2 = layers.MaxPooling2D(pool_size=2)(relu2)

        # ConvBlock 3
        conv3 = layers.Conv2D(256, kernel_size=3, padding='same')(pool2)
        bn3 = layers.BatchNormalization()(conv3)
        relu3 = layers.ReLU()(bn3)
        pool3 = layers.MaxPooling2D(pool_size=2)(relu3)

        # ConvBlock 4
        conv4 = layers.Conv2D(512, kernel_size=3, padding='same')(pool3)
        bn4 = layers.BatchNormalization()(conv4)
        relu4 = layers.ReLU()(bn4)

        # Global Pooling and Fully connected layers
        gap = layers.GlobalAveragePooling2D()(relu4)
        fc1 = layers.Dense(128, activation='relu')(gap)
        output_layer = layers.Dense(1, activation='sigmoid')(fc1)

        # Create model
        self.model = Model(inputs=input_layer, outputs=output_layer)

    def compile(self, loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy']):
        """
        Compiles the model with the specified loss function, optimizer, and metric.
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def summary(self):
        """
        Prints a summary of the model architecture.
        """
        self.model.summary()
        
    def predict(self):
        """
        Predict for inference
        """