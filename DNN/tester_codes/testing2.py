import os
import xlsxwriter
import tensorflow as tf

def make_csv(train_path):

    train_folder = train_path
    last_trained_checkpoint_list = os.listdir(f"{train_path}")
    last_trained_checkpoint_list.remove('logs.csv')

    for model in last_trained_checkpoint_list:
        loaded_model = tf.keras.models.load_model(f"{train_folder}/{model}")

        relative_row_idx = 0 # this is for writing bias
        row_idx = 0
        max_row_idx = 0
        column_idx = 0

        workbook = xlsxwriter.Workbook(f"{train_folder}/{model}/saved_weights_biases.xlsx")
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


make_csv("./trained_models/model1/saved_per_train/train3")