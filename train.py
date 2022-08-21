'''
Code for training SRDML on CIFAR-MTL
'''

import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import load_model

# change the directory to the folder where you store the base model
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

T = 10
train_size = 2000
test_size = 10000


################################
# Build model
################################

# Also, change the directory below
model_1 = load_model("fork_M1")
model_2 = load_model("fork_M2")

#############################
# Training
#############################

mu = 10**(-1)
Z = 1

c_unit = 1
c = tf.constant(c_unit, shape=(T,T), dtype=tf.float32)
c_var = tf.Variable(c)
D = np.zeros((T,T))

# Total number of epoch
EPOCH = 30
# Batch size
BS = 200
ITER = int(T*train_size/BS)
train_loss_history = []
test_loss_history = []
test_auc_history = []
test_precision_history = []
test_recall_history = []

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam()

for i in range(EPOCH):     
    print("Epoch", i+1)
    print('[',end='', flush=True)
    
    for j in range(ITER):
        if j%5 == 0:
            print('=', end='', flush=True)
        # Loss value for this batch.
        image_batch = X_train[j*BS:(j+1)*BS,:,:,:]
        image_batch = tf.convert_to_tensor(image_batch)
        with tf.GradientTape() as g:
            g.watch([model_1.trainable_weights,model_2.trainable_weights])
            # Forward pass
            feature_map = model_1(image_batch)
            y_batch = model_2(feature_map)
            # Calculate the regularization
            reg = 0
            c_sum = 0
            for u in range(T-1):
                for v in range(u+1,T):
                    c_sum += c[u,v]
            for e in range(BS):      
                dy_dx_e = []
                for t in range(T):
                    fmap = feature_map[e,:]
                    fmap = tf.expand_dims(fmap, axis=0)
                    with tf.GradientTape() as gg:
                        gg.watch(fmap)
                        y_e = model_2.layers[t+2+T](model_2.layers[t+2](model_2.layers[1](fmap)))
                    g_t = tf.cast(gg.gradient(y_e, fmap), tf.float32)
                    alpha_t = (1/Z) * tf.reduce_sum(g_t, [0,1,2])
                    dy_dx_e.append(alpha_t)
                for u in range(T-1):
                    for v in range(u+1,T):
                        temp1 = dy_dx_e[u]
                        temp2 = dy_dx_e[v]
                        temp = tf.norm(temp1 - temp2, ord=1)
                        reg += (c[u,v]/c_sum) * temp
                        D[u,v] += temp.numpy()

            # Build the overall loss function
            ce_loss = 0
            for t in range(T):
                ce_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train[t,j*BS:(j+1)*BS].astype('float32').reshape((-1,1)), logits=y_batch[t]))
            ce_loss = ce_loss/BS
            obj = ce_loss + mu * reg 

        # Get gradients of loss wrt the weights.
        gradients = g.gradient(obj, [model_1.trainable_weights,model_2.trainable_weights])
 
        # Update the weights of the model.
        optimizer.apply_gradients(zip(gradients[0], model_1.trainable_weights))
        optimizer.apply_gradients(zip(gradients[1], model_2.trainable_weights))
        
        # Update c by analytical gradient
        E = 0
        G = np.zeros((T,T))
        for u in range(T-1):
            for v in range(u+1,T):
                E += (c[u,v]/(c_sum**2)) * D[u,v]
        for u in range(T-1):
            for v in range(u+1,T):
                G[u,v] = (mu*(D[u,v]/c_sum - E)).numpy()
        D = np.zeros((T,T))
        G = tf.convert_to_tensor(G, dtype=tf.float32)
        optimizer.apply_gradients(zip([G], [c_var]))
        c = tf.convert_to_tensor(c_var)
        c = tf.nn.relu(c)
        c_var = tf.Variable(c)
        
    print(']', end='', flush=True)

    #######################################
    # Below are evaluation part
    
    # Training set performance
    y_all = model_2.predict(model_1.predict(X_train))
    ave_acc = []
    count=0
    for pred in y_all:        
        y_pred = (tf.math.sigmoid(pred)).numpy()
        label =  y_train[count,:].reshape((-1,1))
        y_pred = np.where(y_pred>0.5, 1, 0)
        wrong = np.sum(np.absolute(y_pred-label))
        acc = (T*train_size-wrong)/(T*train_size)
        ave_acc.append(acc)
        count += 1
    print("Overal Averaged Training Accuracy:", np.mean(ave_acc))
    train_loss_history.append(np.mean(ave_acc))
    
    # Test set performance
    y_all = model_2.predict(model_1.predict(X_test))
    ave_acc = []
    ave_auc = []
    ave_precision = []
    ave_recall = []
    count=0
    for pred in y_all:
        y_pred = (tf.math.sigmoid(pred)).numpy()
        label =  y_test[count,:].reshape((-1,1))
        fpr, tpr, thresholds = metrics.roc_curve(label, y_pred, pos_label=1)
        auc_t = metrics.auc(fpr, tpr)
        ave_auc.append(auc_t)
        y_pred = np.where(y_pred>0.5, 1, 0)
        wrong = np.sum(np.absolute(y_pred-label))
        acc = (test_size-wrong)/(test_size)
        print("Testing accuracy for task", count+1, "is:", np.round(100*acc,2), "%")
        ave_acc.append(acc)
        precision_t = metrics.precision_score(label, y_pred)
        recall_t = metrics.recall_score(label, y_pred)
        ave_precision.append(precision_t)
        ave_recall.append(recall_t)
        count += 1
    print("Overal Averaged Test Accuracy:", np.mean(ave_acc))
    test_loss_history.append(np.mean(ave_acc))
    test_auc_history.append(np.mean(ave_auc))
    test_precision_history.append(np.mean(ave_precision))
    test_recall_history.append(np.mean(ave_recall))

print('Max testing accuracy:', max(test_loss_history))
print('Max testing auc:', max(test_auc_history))
print('Max testing precision:', max(test_precision_history))
print('Max testing recall:', max(test_recall_history))

