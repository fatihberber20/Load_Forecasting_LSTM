First, IMFs were obtained by applying variable mode decomposition method 
to our input data. The complexity of each signal was measured by calculating 
the entropy values of these IMFs separately. Signals with entropy values 
close to each other were grouped together and six new signals were created. 
After dividing our data set into two parts as 90% training and 10% testing, 
the data sets obtained with these new signals were trained with an ANN model 
with eight layers: one input layer, three LSTM layers, three simplification 
layers and one output layer. The success of the model according to the MAPE 
success criterion was found to be 0.4394. 