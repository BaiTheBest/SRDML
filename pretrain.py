'''
Code for training the base model which will be used later by SRDML on CIFAR-MTL.
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Input ,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

from sklearn import metrics

# Change the following directory to the folder you store the generated CIFAR-MTL
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

model_1 = VGG16(include_top=False, weights='imagenet', input_shape=(64,64,3))

inp = Input(shape=(2,2,512))

flt = Flatten()(inp)

air_dense = Dense(128,activation='relu')(flt)
air_fc = Dense(1,activation='linear',name='Airplane')(air_dense)

mobile_dense = Dense(128,activation='relu')(flt)
mobile_fc = Dense(1,activation='linear',name='Automobile')(mobile_dense)

bird_dense = Dense(128,activation='relu')(flt)
bird_fc = Dense(1,activation='linear',name='Bird')(bird_dense)

cat_dense = Dense(128,activation='relu')(flt)
cat_fc = Dense(1,activation='linear',name='Cat')(cat_dense)

deer_dense = Dense(128,activation='relu')(flt)
deer_fc = Dense(1,activation='linear',name='Deer')(deer_dense)

dog_dense = Dense(128,activation='relu')(flt)
dog_fc = Dense(1,activation='linear',name='Dog')(dog_dense)

frog_dense = Dense(128,activation='relu')(flt)
frog_fc = Dense(1,activation='linear',name='Frog')(frog_dense)

horse_dense = Dense(128,activation='relu')(flt)
horse_fc = Dense(1,activation='linear',name='Horse')(horse_dense)

ship_dense = Dense(128,activation='relu')(flt)
ship_fc = Dense(1,activation='linear',name='Ship')(ship_dense)

truck_dense = Dense(128,activation='relu')(flt)
truck_fc = Dense(1,activation='linear',name='Truck')(truck_dense)

model_2 = Model(inp, [air_fc,mobile_fc,bird_fc,cat_fc,deer_fc,dog_fc,frog_fc,horse_fc,ship_fc,truck_fc])


#############################
# Training
#############################

# Total number of epoch
EPOCH = 30
# Batch size
BS = 100
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
        if j%10 == 0:
            print('=', end='', flush=True)
        # Loss value for this batch.
        image_batch = X_train[j*BS:(j+1)*BS,:,:,:]
        image_batch = tf.convert_to_tensor(image_batch)
        with tf.GradientTape() as g:
            g.watch([model_1.trainable_weights,model_2.trainable_weights])
            # Forward pass
            feature_map = model_1(image_batch)
            y_batch = model_2(feature_map)
            # Build the overall loss function
            ce_loss = 0
            for t in range(T):
                label_t = y_train[t,j*BS:(j+1)*BS].astype('float32').reshape((-1,1))
                ce_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_t, logits=y_batch[t]))
            ce_loss = ce_loss/BS
            obj = ce_loss

        # Get gradients of loss wrt the weights.
        gradients = g.gradient(obj, [model_1.trainable_weights,model_2.trainable_weights])
 
        # Update the weights of the model.
        optimizer.apply_gradients(zip(gradients[0], model_1.trainable_weights))
        optimizer.apply_gradients(zip(gradients[1], model_2.trainable_weights))
        
    print(']', flush=True)
    
    # Training set performance
    f_map = model_1.predict(X_train)
    y_all = model_2.predict(f_map)
    train_loss = 0
    for t in range(T):
        label_t = y_train[t,:].astype('float32').reshape((-1,1))
        train_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_t, logits=y_all[t]))
    train_loss = train_loss/(T*train_size)
    print("Overal Training Loss:", np.round(train_loss,3))
    
    prediction_t = model_1.predict(X_train)
    prediction = model_2.predict(prediction_t)
    ave_acc = []
    count=0
    for pred in prediction:
        y_pred = (tf.math.sigmoid(pred)).numpy()    
        label = y_train[count,:].reshape((-1,1))
        y_pred = np.where(y_pred>0.5, 1, 0)
        wrong = np.sum(np.absolute(y_pred-label))
        acc = (T*train_size-wrong)/(T*train_size)
        ave_acc.append(acc)
        count += 1
    print("Overal Averaged Training Accuracy:", np.mean(ave_acc))
    train_loss_history.append(np.mean(ave_acc))
    
    # Test set performance
    f_map = model_1.predict(X_test)
    y_all = model_2.predict(f_map)
    test_loss = 0
    for t in range(T):
        label_t = y_test[t,:].astype('float32').reshape((-1,1))
        test_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_t, logits=y_all[t]))
    test_loss = test_loss/(test_size)
    print("Overal Testing Loss:", np.round(test_loss,3))
    
    prediction_t = model_1.predict(X_test)
    prediction = model_2.predict(prediction_t)
    ave_acc = []
    ave_auc = []
    ave_precision = []
    ave_recall = []
    count=0
    for pred in prediction:
        y_pred = (tf.math.sigmoid(pred)).numpy()    
        label = y_test[count,:].reshape((-1,1))
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

fig, ax = plt.subplots()
ax.plot(train_loss_history)
ax.plot(test_loss_history)

ax.set(xlabel='Iteration', ylabel='Ave Accuracy',
       title='Learning curve')
ax.legend(['Training','Testing'])
ax.grid()

plt.show()

# Save the base model

model_1.save("fork_M1")
model_2.save("fork_M2")


