import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import cv2

rows, cols, ch = 64, 64, 3

TARGET_SIZE = (64, 64)


def reduce_brightness(image):
    # convert to HSV so that it's easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # Randomly generate the brightness reduction factor and add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image2 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image2


def preprocess_image(image):
    #Crop image
    image1 = image[55:135, :, :]
    
    #Resize image
    image2 = cv2.resize(image1, TARGET_SIZE)

    #Normalize image
    image2 = image2.astype(np.float32)
    image2 = image2/255.0 - 0.5
    return image2


def get_augmented_row(row):
    steering = row['steering']

    # randomly choose the camera to take the image from
    camera = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left anf right cameras
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    image = load_img("data/" + row[camera].strip())
    image = img_to_array(image)

    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # flip the image and reverse the steering angle
        steering = -1*steering
        image = cv2.flip(image, 1)

    # Apply brightness augmentation
    image = reduce_brightness(image)

    # Crop, resize and normalize the image
    image = preprocess_image(image)
    return image, steering


def get_data_generator(data_frame, batch_size=32):
    N = data_frame.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a batch from the dataframe and generate augmented data for each row in the batch
        for index, row in data_frame.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = get_augmented_row(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch


def get_model():
    model = Sequential()
    # model.add(Lambda(preprocess_batch, input_shape=(160, 320, 3), output_shape=(64, 64, 3)))

    # layer 1 output shape is 32x32x32
    model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    # layer 2 output shape is 15x15x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # layer 3 output shape is 12x12x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))

    # Flatten the output
    model.add(Flatten())

    # layer 4
    model.add(Dense(1024))
    model.add(Dropout(.3))
    model.add(ELU())

    # layer 5
    model.add(Dense(512))
    model.add(ELU())

    # Finally a single output, since this is a regression problem
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

if __name__ == "__main__":
    BATCH_SIZE = 32

    data_frame = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3])

    # shuffle the data
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    # 80-20 training validation split
    training_split = 0.8

    num_rows_training = int(data_frame.shape[0]*training_split)
    num_rows_validation = int(data_frame.shape[0]*(1-training_split))

    training_data = data_frame.loc[0:num_rows_training-1]
    validation_data = data_frame.loc[num_rows_training:]

    # release the main data_frame from memory
    data_frame = None

    training_generator = get_data_generator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = get_data_generator(validation_data, batch_size=BATCH_SIZE)

    model = get_model()

    steps_per_epoch = 4000 
    
    validation_steps = 1000 

    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        steps_per_epoch=steps_per_epoch, nb_epoch=3, validation_steps=validation_steps)

    print("Saving model weights")
    model.save_weights('model.h5')
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())