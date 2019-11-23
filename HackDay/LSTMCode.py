import tensorflow as tf
import numpy as np
import pandas as pd
import os
    
def train_test():
    
    hidden_size = 5
    sequence_length = 8
    data_dim = 2
    output_dim = 1
    learning_rate = 0.05
    
    X = tf.placeholder(tf.float32, [None, sequence_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])
    IND = tf.placeholder(tf.int32)
    
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple = True, activation=tf.tanh, reuse=tf.AUTO_REUSE)
    #cells = tf.contrib.rnn.MultiRNNCell([cell]*3, state_is_tuple = True)
    
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim,   activation_fn=None)
    
    final_info = []
    final_info.append([X[IND,0,0]])
    final_info.append([X[IND,1,0]])
    final_info.append([Y_pred[0][0]])
    Y_predicted = tf.contrib.layers.fully_connected(final_info ,output_dim,  activation_fn = None)
    
    loss = tf.reduce_mean(tf.square(Y_pred - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    path = './train_data/'
    data_list = os.listdir(path)
    for file_npy in data_list:
        input_file_name = path+file_npy
	
        input_set = np.load(input_file_name)
        input_set = np.array(input_set, dtype='float32')

        data_mean = input_set[:,2].mean()
        data_std = np.std(input_set[:2])
        input_set[:,2] = (input_set[:,2]-input_set[:,2].mean())/np.std(input_set[:2])
        
        def build_dataset(time_X_series, seq_length):
            dataX = []
            dataY = []
            for i in range(0, len(time_X_series) - seq_length):
                _x = time_X_series[i:i + seq_length, 2:4]
                _y = time_X_series[i+seq_length, 2:3]  # Next close price
                #print(_x, "->", _y)
                dataX.append(_x)
                dataY.append(_y)
            return np.array(dataX), np.array(dataY)

        trainX, trainY = build_dataset(input_set, sequence_length)

        print(file_npy)
        for i in range(2000):
            try:
                _, l = sess.run([train, loss], feed_dict={X:trainX, Y:trainY, IND:i})

                #print(Y_predicted)
                if i%40==0:
                    print(i, l)

            except:
                print("valueError")
                
        #testPredict = sess.run(Y_pred, feed_dict={X:trainX})
        #print(testPredict*data_std+data_mean)
        #print(trainY*data_std+data_mean)
    
    #val/test
    path = './test_data/'
    data_list = os.listdir(path)
    for file_npy in data_list:
        input_file_name = path+file_npy
	
        input_set = np.load(input_file_name)
        input_set = np.array(input_set, dtype='float32')

        data_mean = input_set[:,2].mean()
        data_std = np.std(input_set[:2])
        input_set[:,2] = (input_set[:,2]-input_set[:,2].mean())/np.std(input_set[:2])

        #train_size = int(len(input_set)*0.8)
        #train_set = input_set[0:train_size]
        #test_set = input_set[train_size:]
        
        def build_dataset(time_X_series, seq_length):
            dataX = []
            dataY = []
            for i in range(0, len(time_X_series) - seq_length):
                _x = time_X_series[i:i + seq_length, [2]]
                _y = time_X_series[i+seq_length,[2]]  # Next close price
                #print(_x, "->", _y)
                dataX.append(_x)
                dataY.append(_y)
            return np.array(dataX), np.array(dataY)

        testX, testY = build_dataset(input_set, sequence_length)

        testPredict = sess.run(Y_pred, feed_dict={X:testX})
        testPredict, testY = testPredict*data_std+data_mean, testY*data_std+data_mean

        df = pd.DataFrame(data=np.array(testPredict), columns=['Predict'])
        df1 = pd.DataFrame(data=np.array(testY), columns=['true'])
        df2 = pd.DataFrame(data=np.array(testPredict-testY), columns=['loss'])
        
        df.loc[:,'true'] = df1
        df.loc[:,'loss'] = df2
        
        if not os.path.exists('./out/'):
            os.mkdir('./out/')
        if not os.path.exists('./out/'+directory+'/'):
            os.mkdir('./out/'+directory+'/')
        df.to_csv('./out/'+directory+'/'+file_npy+'_output.csv',index=False)
        
train_test()
