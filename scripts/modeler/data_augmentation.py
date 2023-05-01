from keras.preprocessing.image import ImageDataGenerator


class DataAugmentation:
    """
    Data augmentation class for image data.
    """

    def __init__(self, rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True):
        """
        Initialize the DataAugmentation class with the given augmentation parameters.
        
        Args:
            rescale (float): Rescale factor for the images.
            shear_range (float): Shear range for the images.
            zoom_range (float): Zoom range for the images.
            horizontal_flip (bool): Whether to apply horizontal flipping for the images.
        """
        self.train_datagen = ImageDataGenerator(
            rescale=rescale,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip)

        self.valid_datagen = ImageDataGenerator(rescale=rescale)

    def get_train_generator(self, X_train, y_train, batch_size=32):
        """
        Returns a train generator with the provided training data and labels.
        
        Args:
            X_train (numpy array): The training data.
            y_train (numpy array): The training labels.
            batch_size (int): The batch size for the generator.

        Returns:
            train_generator (ImageDataGenerator): The training data generator.
        """
        return self.train_datagen.flow(X_train, y_train, batch_size=batch_size)

    def get_valid_generator(self, X_valid, y_valid, batch_size=32):
        """
        Returns a validation generator with the provided validation data and labels.
        
        Args:
            X_valid (numpy array): The validation data.
            y_valid (numpy array): The validation labels.
            batch_size (int): The batch size for the generator.

        Returns:
            valid_generator (ImageDataGenerator): The validation data generator.
        """
        return self.valid_datagen.flow(X_valid, y_valid, batch_size=batch_size)
