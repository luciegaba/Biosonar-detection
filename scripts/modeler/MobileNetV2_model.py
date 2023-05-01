import tensorflow.keras as keras

class MobileNetV2Classifier():
    """
    A custom classifier using MobileNetV2 architecture for image classification.
    
    Attributes:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of classes to predict.
        weights (str): Pre-trained model used in MobileNetV2. By default, MobileNetV2 is trained directly on our data 
    """

    def __init__(self, input_shape=(150, 150, 3), num_classes=1, weights=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
        self.build_model()

    def build_model(self):
        """
        Builds the MobileNetV2 model architecture for classification.
        """
        mobilenet_v2 = keras.applications.MobileNetV2(include_top=False, input_shape=self.input_shape, weights=self.weights)

        # Add custom layers for the new classification task
        x = mobilenet_v2.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        output_layer = keras.layers.Dense(self.num_classes, activation='sigmoid')(x)

        # Construct the new model with the custom layers
        self.model = keras.Model(inputs=mobilenet_v2.input, outputs=output_layer)

        # Freeze the MobileNetV2 layers (use the pre-trained weights)
        for layer in mobilenet_v2.layers:
            layer.trainable = False

    def compile(self, loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy', keras.metrics.AUC(name='auc')]):
        """
        Compiles the model with the specified loss function, optimizer, and metric.
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def summary(self):
        """
        Prints a summary of the model architecture.
        """
        self.model.summary()


    def fit(self, train_generator, valid_generator, steps_per_epoch, validation_steps, epochs=20):
        """
        Fits the model using the provided train and validation data generators.

        Args:
            train_generator (ImageDataGenerator): The training data generator.
            valid_generator (ImageDataGenerator): The validation data generator.
            steps_per_epoch (int): The number of steps per epoch.
            validation_steps (int): The number of validation steps.
            epochs (int): The number of epochs to train the model.
        """
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=valid_generator,
            validation_steps=validation_steps)
