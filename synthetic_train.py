'''
This script contains the demoo code for the synthetic experiment of our paper. 
'''

import tensorflow as tf
from tensorflow.keras.layers import Input , Dense
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error


tf.random.set_seed(111)

T = 12  # number of tasks
m = 10  # number of instances
d = 20  # dimension of features
mu = 0
b = -1
lr = 0.01  # learning rate
ITER = 150 # number of training iteration

# Set the ground truth of linear weights
w = []
# base 1
w.append(tf.ones([d], dtype=tf.float32))
w.append(tf.ones([d], dtype=tf.float32))
w.append(2*tf.ones([d], dtype=tf.float32))
w.append(2*tf.ones([d], dtype=tf.float32))
w.append(3*tf.ones([d], dtype=tf.float32))
w.append(3*tf.ones([d], dtype=tf.float32))
# base 2
w.append(b*tf.ones([d], dtype=tf.float32))
w.append(b*tf.ones([d], dtype=tf.float32))
w.append(b*2*tf.ones([d], dtype=tf.float32))
w.append(b*2*tf.ones([d], dtype=tf.float32))
w.append(b*3*tf.ones([d], dtype=tf.float32))
w.append(b*3*tf.ones([d], dtype=tf.float32))

# Create the dataset
X = tf.random.normal([m,d], 0.5, 1, tf.float32)
Y = []
for t in range(T):
    noise = tf.random.normal([m], 0, 0.01, tf.float32)
    Y.append(tf.linalg.matvec(X, w[t]) + noise)

# Generate test dataset
test_size = 40
X_test = tf.random.normal([test_size,d], 0.5, 1, tf.float32)
Y_test = []
for t in range(T):
    noise = tf.random.normal([test_size], 0, 0.01, tf.float32)
    Y_test.append(tf.linalg.matvec(X_test, w[t]) + noise)


# Build T neural networks
inp = Input(shape=(d,))

#dense = Dense(100,activation='relu')(flt)

fc_1 = Dense(1,activation='linear')(inp)

fc_2 = Dense(1,activation='linear')(inp)

fc_3 = Dense(1,activation='linear')(inp)

fc_4 = Dense(1,activation='linear')(inp)

fc_5 = Dense(1,activation='linear')(inp)

fc_6 = Dense(1,activation='linear')(inp)

fc_7 = Dense(1,activation='linear')(inp)

fc_8 = Dense(1,activation='linear')(inp)

fc_9 = Dense(1,activation='linear')(inp)

fc_10 = Dense(1,activation='linear')(inp)

fc_11 = Dense(1,activation='linear')(inp)

fc_12 = Dense(1,activation='linear')(inp)

model = Model(inp,[fc_1,fc_2,fc_3,fc_4,fc_5,fc_6,fc_7,fc_8,fc_9,fc_10,fc_11,fc_12])


##########################################################

# Initialize c matrix, which corresponds to the task relation matrix
c = tf.constant(1, shape=(T,T), dtype=tf.float32)
c_var = tf.Variable(c)

print("Training:")
loss_history = []
rmse_hist = []
mae_hist = []
optimizer1 = tf.keras.optimizers.SGD(learning_rate=lr)
optimizer2 = tf.keras.optimizers.SGD(learning_rate=0.01)

for i in range(ITER):
    print("Iteration", i+1)
    D = np.zeros((T,T))
    with tf.GradientTape() as g:
        g.watch([model.trainable_weights,c])
        c_sum = 0
        for u in range(T-1):
            for v in range(u+1,T):
                c_sum += c[u,v]
        dy_dx = 0
        for e in range(m):      
            dy_dx_e = []
            temp_x = X[e,:]
            temp_x = tf.expand_dims(temp_x, axis=0)
            for t in range(T):
                with tf.GradientTape() as gg:
                    gg.watch(temp_x)
                    y_t = model(temp_x)[t]
                dy_dx_e.append(tf.cast(gg.gradient(y_t, temp_x), tf.float32))
            for u in range(T-1):
                for v in range(u+1,T):
                    temp1 = dy_dx_e[u]
                    temp2 = dy_dx_e[v]  
                    temp = tf.norm(temp1-temp2,ord=1)
                    dy_dx += (c[u,v]/c_sum) * temp

        loss = 0
        for t in range(T):
            loss += (1/((2*m))) * tf.math.reduce_sum(tf.math.square(model(X)[t]-Y[t]))
        loss = loss/T
        if i > 10:
            mu = 1
        obj = loss + mu * dy_dx # The overall objective function is built
        print("Regress loss:",np.array(loss),"Input gradient loss:",np.array(mu*dy_dx))
  
    gradients = g.gradient(obj, [model.trainable_weights,c]) 
    # Update the weights of the model.
    optimizer1.apply_gradients(zip(gradients[0], model.trainable_weights))
    optimizer2.apply_gradients(zip([gradients[1]], [c_var]))
    c = tf.convert_to_tensor(c_var)
    c = tf.nn.relu(c)
    c_var = tf.Variable(c)

    Loss = 0
    for t in range(T):
        Loss += (1/((2*m))) * tf.math.reduce_sum(tf.math.square(model(X)[t]-Y[t]))
    Loss = Loss/T
    print("Training least square loss:", Loss.numpy())
    loss_history.append(Loss)
    
    # Test set performance
    prediction = model.predict(X_test)
    rmse = []
    mae = []
    for t in range(T):
        y_pred = np.array(prediction[t])
        label = np.array(Y_test[t])
        mse_t = mean_squared_error(label,y_pred)  
        rmse_t = np.sqrt(mse_t)
        rmse.append(rmse_t)
        mae_t = mean_absolute_error(label,y_pred)
        mae.append(mae_t)
    print("Overal Averaged Test RMSE:", np.mean(rmse))
    print("Overal Averaged Test MAE:", np.mean(mae))
    
    rmse_hist.append(np.mean(rmse))
    mae_hist.append(np.mean(mae))


print("Min RMSE iter:", np.argmin(rmse_hist))
print("Min MAE iter:", np.argmin(mae_hist))


# fig, ax = plt.subplots()
# ax.plot(loss_history)

# ax.set(xlabel='Iteration', ylabel='Loss',
#        title='Training Loss')
# ax.grid()

# plt.show()
