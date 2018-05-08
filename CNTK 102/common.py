import cntk as C
import numpy as np
from matplotlib import pyplot as plt


def get_color(label, num_classes):
    return plt.cm.get_cmap("hsv", num_classes + 1)(int(label))

def predict(X, model, inputs, classification=True):
    if classification:
        model = C.softmax(model)
        pred = model.eval({inputs: X})
        return np.array([[np.argmax(row)] for row in pred], dtype=np.float32)
    else:
        return model.eval({inputs: X})

    
def particles(N, bound, model, inputs, num_features, classification=True):
    part = np.random.uniform(-bound, bound, (N, num_features))
    part = part.astype(np.float32)
    part_y = predict(part, model, inputs, classification)
    return part, part_y