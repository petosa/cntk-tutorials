import cntk as C
import numpy as np
from matplotlib import pyplot as plt
from math import pi

# GLOBALS
num_features = 2
num_classes = 3

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


params = {}
def linear_units(input_var, output_dim):
    input_dim = input_var.shape[0]
    # Introduce model parameters
    weight_param = C.parameter(shape=(output_dim, input_dim), name="weights")
    bias_param = C.parameter(shape=(output_dim, 1), name="biases")
    # Reshape to facilitate matrix multiplication
    input_reshaped = C.reshape(input_var, (input_dim, 1))
    # Weighted sums
    params['w'], params['b'] = weight_param, bias_param
    part1 = C.times(weight_param, input_reshaped)
    # Add biases
    part2 = part1 + bias_param
    # Return 1-D representation
    return C.reshape(part2, (num_classes))


inputs = C.input_variable(shape=(num_features), dtype=np.float32, name="features")
# Z is the model; a composition of operation. Maps [(input_dim) -> (num_classes)]
z = linear_units(inputs, num_classes)
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
minibatch_size = 250
num_samples_to_train = 50000
num_minibatches_to_train = int(num_samples_to_train / minibatch_size)


from collections import defaultdict

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

print("Optimal parameters")
print(params["w"].value)
print(params["b"].value)

print("Displaying decision boundaries")

# Get test data
features, labels = data_gen(minibatch_size, num_features, num_classes)
def get_color(label):
    return plt.cm.get_cmap("hsv", num_classes + 1)(int(label))
colors = [get_color(label[0]) for label in labels]
plt.scatter(features[:,0], features[:,1], c=colors)

# Draw decision boundaries
final_weights = params["w"].value
final_bias = params["b"].value
# Draw a boundary for each class, since boundaries are "one vs all"
for i in range(len(final_weights)):
    c1 = final_weights[i][0]
    c2 = final_weights[i][1]
    b = final_bias[i]
    # Solve for x2 in w1x1 + w2x2 + b = 0 (the boundary)
    def solver(x):
        return -c1/c2*x - b/c2
    plt.plot([-3, 3], [solver(-3), solver(3)], color=get_color(i), linewidth=3, linestyle="--")

plt.show()
