import matplotlib.pyplot as plt
import numpy as np
import shap
from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images[..., np.newaxis] / 255.0
test_images = test_images[..., np.newaxis] / 255.0

model_dirs = ['reference', 'baseline', 'pruned', 'distilled']

for model_dir in model_dirs:
  model = keras.models.load_model(model_dir)

  # select a set of background examples to take an expectation over
  background = train_images[np.random.choice(train_images.shape[0], 100, replace=False)]

  # explain predictions of the model on four images
  e = shap.DeepExplainer(model, background)
  # ...or pass tensors directly
  # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
  shap_values = e.shap_values(test_images[1:5])

  # plot the feature attributions
  shap.image_plot(shap_values, -test_images[1:5], show=False)
  plt.savefig(f'{model_dir}.png')