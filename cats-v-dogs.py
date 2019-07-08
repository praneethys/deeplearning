# Additional packages to be installed
# pip install pillow
# conda install python.app --> use pythonw to run this script

import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# Global variables
DOWNLOAD_ZIP      = ""
CAT_SOURCE_DIR    = ""
TRAINING_DIR      = ""
TESTING_DIR       = ""
TRAINING_CATS_DIR = ""
TESTING_CATS_DIR  = ""
DOG_SOURCE_DIR    = ""
TRAINING_DOGS_DIR = ""
TESTING_DOGS_DIR  = ""
SPLIT_SIZE        = 0.9

def parse_args():
  """
  Parse commandline arguments
  """
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument("-a", "--augment_img", help='Perform image augmentation on training set', action='store_true', default=False)
  parser.add_argument("-d", "--setup_dir", help='Root dir where dataset is downloaded', action='store', type=str, default="/tmp")
  parser.add_argument("-s", "--setup_dataset", help='Setup dataset', action='store_true', default=False)

  return parser.parse_args()

def set_dir_paths(rootdir):
  """
  Setup directory paths for training and testing datasets
  """
  global DOWNLOAD_ZIP
  global CAT_SOURCE_DIR
  global TRAINING_DIR
  global TESTING_DIR
  global TRAINING_CATS_DIR
  global TESTING_CATS_DIR
  global DOG_SOURCE_DIR
  global TRAINING_DOGS_DIR
  global TESTING_DOGS_DIR

  DOWNLOAD_ZIP      = os.path.join(rootdir, "cats_and_dogs.zip")
  CAT_SOURCE_DIR    = os.path.join(rootdir, "PetImages/Cat")
  TRAINING_DIR      = os.path.join(rootdir, "cats-v-dogs/training")
  TESTING_DIR       = os.path.join(rootdir, "cats-v-dogs/testing")
  TRAINING_CATS_DIR = os.path.join(TRAINING_DIR, "cats")
  TESTING_CATS_DIR  = os.path.join(TESTING_DIR, "cats")
  DOG_SOURCE_DIR    = os.path.join(rootdir, "PetImages/Dog")
  TRAINING_DOGS_DIR = os.path.join(TRAINING_DIR, "dogs")
  TESTING_DOGS_DIR  = os.path.join(TESTING_DIR, "dogs")

def download_dataset():
  """
  Download full cats vs dogs dataset
  """
  os.system("python -m wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip -o" + DOWNLOAD_ZIP)

def extract_dataset():
  """
  Extract dataset
  """
  local_zip = DOWNLOAD_ZIP
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall(DIR)
  zip_ref.close()

def create_dirs():
  """
  Create directories for training and testing datasets
  """
  try:
    main_dir = os.path.join(DIR, "cats-v-dogs")
    os.mkdir(main_dir)
    for dir_ in ["training", "testing"]:
      path_dir = os.path.join(main_dir, dir_)
      os.mkdir(path_dir)
      for subdir_ in ["cats", "dogs"]:
        path_subdir = os.path.join(path_dir, subdir_)
        os.mkdir(path_subdir)
          
  except OSError as error:
    print(error)

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
  """ 
  Split source data into training and testing datasets 
  """
  all_files = os.listdir(SOURCE)
  num_training = int(float(len((all_files)) * SPLIT_SIZE))

  shuffled_files = random.sample(all_files, len(all_files))
  training_files = shuffled_files[:num_training-1]
  testing_files  = shuffled_files[num_training:]
  
  for _,file_ in enumerate(training_files):
    src_path = os.path.join(SOURCE, file_)
    dst_path = os.path.join(TRAINING, file_)
    if os.path.getsize(src_path):
      copyfile(src_path, dst_path)
    else:
      print(file_, "is zero length, so ignoring")

  for _,file_ in enumerate(testing_files):
    src_path = os.path.join(SOURCE, file_)
    dst_path = os.path.join(TESTING, file_)
    if os.path.getsize(src_path):
      copyfile(src_path, dst_path)
    else:
      print(file_, "is zero length, so ignoring")

def nn():
  """
  Define Convolutional Neural Network
  """
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Dropout(0.5), # Randomly set a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
  
  return model

def create_training_generator(augment_img):
  """
  Create training image data generator
  """
  if augment_img:
    print("Image augmentation is enabled for training dataset")
    train_datagen = ImageDataGenerator( rotation_range=40,      # in degrees (0-180)
                                      width_shift_range=0.2,  # fraction of total width within which to randomly translate images
                                      height_shift_range=0.2, # fraction of total height within which to randomly translate images
                                      shear_range=0.2,        # shear intensity
                                      zoom_range=0.2,         # [1-zoom_range, 1+zoom_range]
                                      horizontal_flip=True,   # randomly flip image horizontally
                                      fill_mode='nearest',    # "constant", "nearest", "reflect" or "wrap"
                                      rescale=1./255.         # rescale factor after applying all other transformations
                                      )
  else:
    train_datagen = ImageDataGenerator(rescale=1./255.)

  train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                      target_size=(150,150),
                                                      batch_size=100,
                                                      class_mode='binary')
  return train_generator

def create_validation_generator():
  """
  Create validation image data generator with image augmentation
  """
  VALIDATION_DIR = TESTING_DIR
  validation_datagen = ImageDataGenerator(rescale=1./255.)
  validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                                target_size=(150,150),
                                                                batch_size=100,
                                                                class_mode='binary')

  return validation_generator

def train_nn(model, train_generator, validation_generator):
  """
  Train the Convolutional Neural Network
  """
  myCallback = MyCallback()
  history = model.fit_generator(train_generator,
                              steps_per_epoch=100,  # no. of images = batch_size * steps
                              epochs=1,
                              validation_data=validation_generator,
                              validation_steps=50,  # no. of images = batch_size * steps
                              verbose=1,
                              callbacks=[myCallback]
                              )
  
  return history

def plot_loss_acc(history):
  """
  Plot loss and accuracy for training vs validation datasets
  """
  import matplotlib.image  as mpimg
  import matplotlib.pyplot as plt

  #-----------------------------------------------------------
  # Retrieve a list of list results on training and test data
  # sets for each training epoch
  #-----------------------------------------------------------
  acc=history.history['acc']
  val_acc=history.history['val_acc']
  loss=history.history['loss']
  val_loss=history.history['val_loss']

  epochs=range(len(acc)) # Get number of epochs

  #------------------------------------------------
  # Plot training and validation accuracy per epoch
  #------------------------------------------------
  plt.plot(epochs, acc, 'r', "Training Accuracy")
  plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title('Training and validation accuracy')
  plt.legend()

  # Create a new figure
  plt.figure()

  #------------------------------------------------
  # Plot training and validation loss per epoch
  #------------------------------------------------
  plt.plot(epochs, loss, 'r', "Training Loss")
  plt.plot(epochs, val_loss, 'b', "Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()

class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, log={}):
    if log.get('acc') > 0.95:
      print("\nReached 95% accuracy, cancelling training\n")
      self.model.stop_training = False

def main():
  args = parse_args()
  set_dir_paths(args.setup_dir)

  if args.setup_dataset:
    download_dataset()
    extract_dataset()
    create_dirs()

    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, SPLIT_SIZE)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, SPLIT_SIZE)

  model                = nn()
  train_generator      = create_training_generator(args.augment_img)
  validation_generator = create_validation_generator()
  history              = train_nn(model, train_generator, validation_generator)

  plot_loss_acc(history)

if __name__ == "__main__":
  main()
