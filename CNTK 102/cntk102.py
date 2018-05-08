import cntk as C
import numpy as np
from matplotlib import pyplot as plt
from math import pi
from collections import defaultdict
from common import predict, particles, get_color


# GLOBALS
num_features = 2
num_classes = 5

# Helper function to generate a random data sample
def data_gen(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy. It is linearly separable.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)
    normalized = Y / num_classes
    x1 = np.random.randn(sample_size, 1) + (4 * np.sin(pi*normalized) - 2)
    x2 = np.random.randn(sample_size, 1) + (4 * np.sin(pi*normalized + pi/4) - 2) 
    X = np.hstack([x1, x2])
    # Must be float32 for cntk inputs
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)  
    return X, Y



'''
Manual Neural Network
'''
params = {}
def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    # Introduce model parameters
    weight_param = C.parameter(shape=(output_dim, input_dim), name="weights", init=C.glorot_uniform())
    bias_param = C.parameter(shape=(output_dim, 1), name="biases", init=C.glorot_uniform())
    # Reshape to facilitate matrix multiplication
    input_reshaped = C.reshape(input_var, (input_dim, 1))
    # Weighted sums
    params['w'], params['b'] = weight_param, bias_param
    part1 = C.times(weight_param, input_reshaped)
    # Add biases
    part2 = part1 + bias_param
    # Return 1-D representation
    return C.reshape(part2, (output_dim))

def dense_layer(input_var, output_dim, nonlinearity):
    return nonlinearity(linear_layer(input_var, output_dim))

def fully_connected_classifier_net(input_var, num_output_classes, hidden_layer_dim, num_hidden_layers, nonlinearity):
    h = dense_layer(input_var, hidden_layer_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        h = dense_layer(h, hidden_layer_dim, nonlinearity)
    return linear_layer(h, num_output_classes)


'''
Built-in Neural Network
'''
def create_model(features, num_hidden_layers, hidden_layer_dim):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.sigmoid):
        h = features
        for _ in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim)(h)
        last_layer = C.layers.Dense(num_classes, activation = None)
        return last_layer(h)



inputs = C.input_variable(shape=(num_features), dtype=np.float32, name="features")
# Z is the model; a composition of operation. Maps [(input_dim) -> (num_classes)]
num_hidden_layers = 2
hidden_layers_dim = 10

# Choose your model
z = create_model(inputs, num_hidden_layers, hidden_layers_dim)
z = fully_connected_classifier_net(inputs, num_classes, hidden_layers_dim, num_hidden_layers, C.sigmoid)

print(z.parameters)
label = C.input_variable(1, dtype=np.float32, name="label")
onehot = C.one_hot(label, num_classes)
loss = C.cross_entropy_with_softmax(z, onehot)
eval_error = C.classification_error(z, onehot)

# Instantiate the trainer object to drive the model training
learning_rate = 0.5
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, eval_error), [learner])

# Define a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss, eval_error = "NA", "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))

    return mb, training_loss, eval_error

# Initialize the parameters for the trainer
minibatch_size = 500
num_samples_to_train = 150000
num_minibatches_to_train = int(num_samples_to_train / minibatch_size)

# Run the trainer and perform model training
training_progress_output_freq = 50
plotdata = defaultdict(list)

for i in range(0, num_minibatches_to_train):
    features, labels = data_gen(minibatch_size, num_features, num_classes)

    # Assign the minibatch data to the input variables and train the model on the minibatch
    trainer.train_minibatch({inputs : features, label : labels})
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)


f, axarr = plt.subplots(2)
axarr[0].plot(plotdata["batchsize"], plotdata["loss"], 'b--')
axarr[0].set_xlabel('Minibatch number')
axarr[0].set_ylabel('Loss')
axarr[0].set_title('Training Loss')
axarr[1].plot(plotdata["batchsize"], plotdata["error"], 'r--')
axarr[1].set_xlabel('Minibatch number')
axarr[1].set_ylabel('Error')
axarr[1].set_title('Training Error')
f.tight_layout()
plt.show()


# Get test data
features, labels = data_gen(minibatch_size, num_features, num_classes)
print(trainer.test_minibatch({inputs: features, label: labels}))
pred = predict(features, z, inputs)
eq = pred == labels
print("Error (1-acc):", 1.0 - float(sum(eq))/len(eq))

acc_colors = [get_color(label[0], num_classes) for label in labels]
pred_colors = [get_color(y[0], num_classes) for y in pred]
plt.scatter(features[:,0], features[:,1], c=acc_colors, s=55)
plt.scatter(features[:,0], features[:,1], c=pred_colors, marker="x")
plt.show()

# Visualize boundaries
print("Displaying decision boundaries")
plt.scatter(features[:,0], features[:,1], c=acc_colors, s=55)
part, part_y = particles(500, 5, z, inputs, num_features)
part_colors = [get_color(y[0], num_classes) for y in part_y]
plt.scatter(part[:,0], part[:,1], c=part_colors, marker="x", alpha=0.5)
plt.show()
