from tensorflow import keras

def define_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(512, kernel_size=(2,2), input_shape=(32, 32, 3), activation='relu', padding='same'),
        keras.layers.AvgPool2D((2,2)),
        #after passing through the above pooling layer the image shape will be 16x16x3
        keras.layers.Conv2D(512, kernel_size=(2,2), activation='relu', padding='same'),
        keras.layers.AvgPool2D((2,2)),
        #8x8x3
        keras.layers.Conv2D(256, kernel_size=(2,2), activation='relu', padding='same'),
        keras.layers.AvgPool2D((2,2)),
        #4x4
        keras.layers.Conv2D(256, kernel_size=(2,2), activation='relu', padding='same'),
        keras.layers.AvgPool2D((2,2)),
        #2x2x3
                          
        keras.layers.Flatten(),

        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
        ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model