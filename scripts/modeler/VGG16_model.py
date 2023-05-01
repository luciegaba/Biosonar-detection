import tensorflow.keras as keras

class VGG16Classifier():
    """
    A custom classifier using VGG16 architecture for image classification.
    
    Attributes:
        input_shape_ (tuple): The shape of the input images.
        num_classes (int): The number of classes to predict.
        weights_ (str): Pre-trained model used in VGG16. By default, VGG16 is trained directly in our data 
    """

    def __init__(self, input_shape=(150, 150, 3), num_classes=1, weights=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
        self.build_model()

    def build_model(self):
        """
        Builds the VGG16 model architecture for classification.
        """
        vgg16 = keras.applications.VGG16(include_top=False, input_shape=self.input_shape, weights=self.weights, pooling='avg')

        # Freeze the weights of the pre-trained layers
        for layer in vgg16.layers:
            layer.trainable = False

        # Add a dense layer for binary classification
        output_layer = keras.layers.Dense(self.num_classes, activation='sigmoid')(vgg16.output)

        # Create model
        self.model = keras.Model(inputs=vgg16.input, outputs=output_layer)

    def compile(self, loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy', keras.metrics.AUC(name='auc')]):
        """
        Compiles the model with the specified loss function, optimizer, and metric.
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def summary(self):
        """
        Prints a summary of the model architecture.
        """
        self.model.summary()
