import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility
#import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation
from keras.optimizers import Adam

print('Reading Data ------------')


# Load Data
File_name = 1
pay_filename = "%d_pay_day_num.txt" % File_name
pay_data = pd.read_csv(pay_filename,names = ['ID','Pay_Times','Date'])
data_all = pay_data.iloc[:,1].values.astype('float64')

# Data normalize
data_min = np.min(data_all)
data_max = np.max(data_all)
data_all = (data_all - data_min)/ (data_max-data_min)

data = []
sequence_length = 2


for i in range(len(data_all) - sequence_length - 1):
	data.append(data_all[i: i + sequence_length + 1])

reshaped_data = np.array(data).astype('float64')

split = 0.8
np.random.shuffle(reshaped_data)
x = reshaped_data[:, :-1]
y = reshaped_data[:, -1]
split_boundary = int(reshaped_data.shape[0] * split)
train_x = x[: split_boundary]
test_x = x[split_boundary:]

train_y = y[: split_boundary]
test_y = y[split_boundary:]

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))


#print train_x
#print train_y

def build_model():
	model = Sequential()
	model.add(LSTM(input_dim=1, output_dim=7, return_sequences=False))
	model.add(Dense(output_dim=1))
	model.add(Activation('linear'))
	model.compile(loss='mse', optimizer='rmsprop')

	return model
	

model = build_model()

model.fit(train_x, train_y, batch_size=1, nb_epoch=100, validation_split=0.1)
predict = model.predict(test_x)
predict = np.reshape(predict, (predict.size, ))

#accuracy = (predict - test_y) / (predict + test_y)
predict = predict*(data_max-data_min)+data_min
test_y = test_y*(data_max-data_min)+data_min

result = np.reshape([predict,test_y], (-1,2))

np.savetxt('%d_result.csv'%File_name ,result,fmt='%d',delimiter=',')
result_data = pd.read_csv('%d_result.csv'%File_name ,names = ['Predict','Right'])

Loss = (result_data['Predict'] - result_data['Right']) /( result_data['Predict'] + result_data['Right'])

print Loss.abs().mean()

print result



#print accuracy

'''	
def train_model(train_x, train_y, test_x, test_y):
	model = build_model()

	try:
		model.fit(train_x, train_y, batch_size=512, nb_epoch=30, validation_split=0.1)
		predict = model.predict(test_x)
		predict = np.reshape(predict, (predict.size, ))
	except KeyboardInterrupt:
		print(predict)
		print(test_y)
	print(predict)
	print(test_y)
	
	except Exception as e:
		print(e)
	return predict, test_y
'''	

	


'''
# Construct X_train Y_train
seq_length = 3
Y_train = []
X_train = []
Y_train = pay_data.iloc[seq_length:,1].values

for i in range(len(Y_train)):
	X_array = pay_data.iloc[:,1].values
	X_train.append(X_array[i:(i+seq_length)])
	
#Y_pay = np.reshape(Y_pay,(55,7))
#Y_pay = Y_pay[:,:,np.newaxis]

print X_train



'''
'''
COLUMN = 1
TIME_STEPS = 3
BATCH_SIZE = 1
INPUT_SIZE = 3
OUTPUT_SIZE = 3
CELL_SIZE = 10
LR = 0.006


def get_batch(x=[]):
	global TIME_STEPS, COLUMN
	# xs shape (50batch, 20steps)
	xs = x[COLUMN,:]
	COLUMN += 1
	ys = x[COLUMN,:]	

	return [xs[:, :, np.newaxis], ys[:, :, np.newaxis]]

i = 1
pay_filename = "%d_pay_day_num.txt" % i
pay_data = pd.read_csv(pay_filename,names = ['ID','Pay_Times','Date'])
Y_pay = pay_data.iloc[0:385,3].values
Y_pay = np.reshape(Y_pay,(385,3))
Y_pay = Y_pay[:,:,np.newaxis]

#print(Y_pay)
print('Establishing ------------')	
model = Sequential()
# build a LSTM RNN
model.add(LSTM(
	batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
	output_dim=CELL_SIZE,
	return_sequences=True,      # True: output at all steps. False: output as last step.
	stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))
# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam = Adam(LR)
model.compile(optimizer=adam,
			  loss='mse',)
			
print('Training ------------')

for step in range(20):	
	X_batch, Y_batch = get_batch(Y_pay)
	cost = model.train_on_batch(X_batch, Y_batch)
	pred = model.predict(X_batch, BATCH_SIZE)
	print('train cost: ', cost)
	print('pred ac:',pred)
'''
