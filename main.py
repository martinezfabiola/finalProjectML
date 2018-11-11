import csv
from os.path import dirname, join

import numpy as np
import math
from sklearn import metrics, model_selection, preprocessing

import tensorflow as tf

def dnn(args, max_it, it):
    # Load dataset
    x, y = load_vacadata()  

    # Split dataset into train / test
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=42)

    # Scale data (training set) to 0 mean and unit standard deviation.
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    feature_columns = [
        tf.feature_column.numeric_column('x', shape=np.array(x_train).shape[1:])]
    regressor = tf.estimator.DNNRegressor(
        feature_columns=feature_columns, hidden_units=args)

    # Predict.
    x_transformed = scaler.transform(x_test)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': x_transformed}, y=y_test, num_epochs=1, shuffle=False)

    error_medio = []
    i=0
    err_prev=0
    err_act=1
    #for i in range(1000):
    while(i<max_it or abs(err_prev-err_act)<0.2):
        i += 1    
        # Train.
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': x_train}, y=y_train, batch_size=1, num_epochs=None, shuffle=True)
        regressor.train(input_fn=train_input_fn, steps=it)

        predictions = regressor.predict(input_fn=test_input_fn)
        y_predicted = np.array(list(p['predictions'] for p in predictions))
        y_predicted = y_predicted.reshape(np.array(y_test).shape)

        # Score with sklearn.
        score_sklearn = metrics.mean_squared_error(y_predicted, y_test)
        print('MSE (sklearn): {0:f}'.format(score_sklearn))
        print(i,args)

        error_medio.append(score_sklearn)
        err_prev=err_act
        err_act=score_sklearn

        # Score with tensorflow.
        #scores = regressor.evaluate(input_fn=test_input_fn)
        #print('MSE (tensorflow): {0:f}'.format(scores['average_loss']))
    #print(scores)
    #for i in range(1,2000):
    #    print(regressor.evaluate(input_fn=test_input_fn, steps=1)['average_loss'])
    return error_medio, i 

def main(args):
    error_corridas=[]

    archivo = open("results_2.csv", "w")
    archivo.write("Estructura de red; MSE; Iteraciones; SQRT MSE\n")
    redes_de_prueba = []
    for i in range(2,11,2):
        redes_de_prueba.append([i])
    for i in range(4, 11, 2):
        for j in range(4,11,2):
            redes_de_prueba.append([i,j])
    for i in range(6, 11, 2):
        for j in range(6,11,2):
            for k in range(6,11,2):
                redes_de_prueba.append([i,j,k])
    for i in range(6, 11, 2):
        for j in range(6,11,2):
            for k in range(6,11,2):
              for l in range (6,11,2):
                redes_de_prueba.append([i,j,k,l])

    print(len(redes_de_prueba))

    #redes_de_prueba = [[2], [3], [10,10]]
    for est in redes_de_prueba:
        e,i=dnn(est,1, 5000)
        error_corridas.append(e)
        archivo.write(str(est)+"; "+str(e[len(e)-1])+"; "+str(i)+"; "+str(math.sqrt(e[len(e)-1]))+"\n")
    archivo.close()

    archivo_2 = open("errores_MSE.txt", "w")
    for e in error_corridas:
        archivo_2.write(str(len(e))+"\n")
        archivo_2.write(str(e)+"\n")
    #error_corridas.append(dnn([10,10]))


def load_vacadata():
    module_path = dirname(__file__)

    data_file_name = join(module_path, 'data', 'cows_pregnancy_final.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.float64)

    return data, target

if __name__ == '__main__':
    tf.app.run()