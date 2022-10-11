import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
import numpy as np

class DNN_model:
    '''
    This class is used for making your own deep neural network and your end goal is to predict the category of the input data.

    A function is made here to automate the saving of model and it also saves the weights and biases in an excel file.

    If you want to fully customize your tensorflow model please refer to https://www.tensorflow.org/tutorials/quickstart/beginner as a guide on making a DNN model
    '''
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def make_model(train_dataset, train_labels, validation_data=None, num_of_inputs=3, num_of_outputs=11, num_of_hidden_layers=5, num_of_neurons_of_hidden_layers=15, epochs=100, batch_size=1000, save_freq=100, saved_models_path="../trained_models"):
        '''
        This function can be called if you want to create a fresh new model of color classifier

        Validation data should be in a tuple.

        Example: 

        validation_data=(test_dataset, test_labels)
        '''
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(num_of_inputs,)))
        for i in range(num_of_hidden_layers):
            model.add(tf.keras.layers.Dense(num_of_neurons_of_hidden_layers, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
        model.add(tf.keras.layers.Dense(num_of_outputs))
        model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        model.build()

        from datetime import datetime
        if os.listdir(f"{saved_models_path}"):
            # Makes the model folder
            last_model_num = os.listdir(f"{saved_models_path}")[-1][-1]
            os.mkdir(f"{saved_models_path}/model{int(last_model_num) + 1}")   
            # Makes a readme file that contains information on when was this created
            last_model = os.listdir(f"{saved_models_path}")[-1]
            with open(f"{saved_models_path}/{last_model}/readme.txt", "w+") as f:
                f.write(f"Model created on: {str(datetime.now())}")

        # If there are no models on path
        elif not os.listdir(f"{saved_models_path}"):
            # Makes the model folder
            os.mkdir(f"{saved_models_path}/model1")
            # Makes a readme file that contains information on when was this created
            last_model = os.listdir(f"{saved_models_path}")[-1]
            with open(f"{saved_models_path}/model1/readme.txt", "w+") as f:
                f.write(f"Model created on: {str(datetime.now())}")

        # Trains the newly created model
        from tensorflow.keras.callbacks import CSVLogger
        os.makedirs(f"{saved_models_path}/{last_model}/saved_per_train/train1")
        checkpoint_path = f"{saved_models_path}/{last_model}/saved_per_train/train1" + "/Epoch{epoch:02d}_loss{loss:.2f}"
        csv_logger = CSVLogger(f"{saved_models_path}/{last_model}/saved_per_train/train1/logs.csv", separator=',', append=False)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        verbose=1,
                                                        monitor='accuracy',
                                                        save_freq=save_freq) # if save_freq='epochs' it saves the model per epoch
                                                                        # if save_freq=int_type it saves the model per <int_type> of batches
        # Train the model with the new callback
        model.fit(train_dataset, 
                train_labels,  
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=[cp_callback, csv_logger], # Pass callback to training
                shuffle=True)  

        # This may generate warnings related to saving the state of the optimizer.
        # These warnings (and similar warnings throughout this notebook)
        # are in place to discourage outdated usage, and can be ignored.

        DNN_model.make_excel(train_path=f"{saved_models_path}/{last_model}/saved_per_train/train1")

    def load_model(self, model_path):
        '''Function for loading a model into the object'''
        self.model_path = model_path
        self.train_path = f"{self.model_path}/saved_per_train" # The train path of the model
        self.least_loss_checkpoint_path = self.return_path() # The file path of the least loss checkpoint
        self.loaded_checkpoint = tf.keras.models.load_model(self.least_loss_checkpoint_path) # Loads the least loss checkpoint in a variable


    def train_model(self, train_dataset, train_labels, validation_data=None, epochs=100, batch_size=1000, save_freq=100, save_param_excel=False, custom_checkpoint=None):
        '''
        Trains the loaded model and then saves it in a new train folder.
        '''
        if not custom_checkpoint: # Checks if user specified a custom checkpoint path to begin
            self.least_loss_checkpoint_path = self.return_path() # The file path of the least loss tensor model
            self.loaded_checkpoint = tf.keras.models.load_model(self.least_loss_checkpoint_path) # Loads the least loss tensor model in a variable
        elif custom_checkpoint:
            self.loaded_checkpoint = tf.keras.models.load_model(custom_checkpoint)

        # This part makes a new folder for the new train batch
        self.last_train_num = os.listdir(f"{self.train_path}")[-1][-1] # Gets the number of the latest train folder
        os.makedirs(f"{self.train_path}/train{int(self.last_train_num) + 1}") # Makes a new folder for the current train batch
        self.last_train_folder = f"{self.train_path}/train{int(self.last_train_num) + 1}" # The relative path of the current train batch folder
 
        # Create a callback that saves the model's weights
        from tensorflow.keras.callbacks import CSVLogger
        checkpoint_path = f"{self.last_train_folder}" + "/Epoch{epoch:02d}_loss{loss:.2f}" # Makes a folder and saves the model for every batch of epochs. Save frequency depends on save_freq parameter
        csv_logger = CSVLogger(f"{self.last_train_folder}/logs.csv", separator=',', append=False)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=1,
                                                 monitor='accuracy',
                                                 save_freq=save_freq) # if save_freq='epochs' it saves the model per epoch
                                                                      # if save_freq=int_type it saves the model per <int_type> of batches

        self.loaded_checkpoint.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
        # Train the model with the new callback
        self.loaded_checkpoint.fit(train_dataset, 
                train_labels,  
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=[cp_callback, csv_logger], # Pass callback to training
                shuffle=True) 
        # This may generate warnings related to saving the state of the optimizer.
        # These warnings (and similar warnings throughout this notebook)
        # are in place to discourage outdated usage, and can be ignored.    
        if save_param_excel:
            DNN_model.make_excel(train_path=f"{self.last_train_folder}")

    @staticmethod
    def make_excel(train_path):
        '''
        This function saves the weights and biases each of the model of the train in a excel file
        Defaults to the latest train if specific_train is not specified
        '''
        import xlsxwriter
        last_trained_checkpoint_list = os.listdir(f"{train_path}")
        try:
            last_trained_checkpoint_list.remove('logs.csv')
        except:
            pass
            
        for model in last_trained_checkpoint_list:
            loaded_model = tf.keras.models.load_model(f"{train_path}/{model}")

            relative_row_idx = 0 # this is for writing bias
            row_idx = 0
            max_row_idx = 0
            column_idx = 0

            workbook = xlsxwriter.Workbook(f"{train_path}/{model}/saved_weights_biases.xlsx")
            worksheet = workbook.add_worksheet()

            for layer in loaded_model.layers:
                for row_weights in layer.get_weights()[0].T: # the reason I transposed the matrix because tensorflow makes weights in transposed position of matrix
                    for weights in row_weights:
                        worksheet.write(row_idx, column_idx, weights)
                        column_idx += 1

                    column_idx += 1
                    worksheet.write(row_idx, column_idx, layer.get_weights()[1].T[relative_row_idx]) # the reason I transposed the matrix because tensorflow makes weights in transposed position of matrix
                    relative_row_idx += 1
                    row_idx += 1
                    column_idx = 0

                relative_row_idx = 0
                row_idx += 2
            workbook.close()

    def return_path(self, custom_train=None):
        '''
        Defaults to returning the latest train with the least loss model relative file path
        '''
        last_trained = os.listdir(self.train_path)[-1] # This gives the latest trained
        checkpoint_list = os.listdir(f"{self.train_path}/{last_trained}") # List all the saved trains
        try:
            checkpoint_list.remove("logs.csv") # Removes the logs.csv file on the list because we do not need it we only need the model folders
        except:
            pass
        try:
            least_loss_model = min(checkpoint_list, key=lambda loss_val:loss_val[-4:-1]) # Finds the model which has the least loss in the list
        except:
            print("Empty train folder path")
        least_loss_checkpoint_path = f"{self.train_path}/{last_trained}/{least_loss_model}" # The file path of the least loss tensor model
        return least_loss_checkpoint_path
