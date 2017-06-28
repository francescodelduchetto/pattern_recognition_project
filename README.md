# Pattern Recognition Project
## 

## Tensorflow installation
[Official installation instructions](https://www.tensorflow.org/install/)

## Magenta installation
We modified the original Magenta repository for our needs. The modified repository is 
[here](https://github.com/francescodelduchetto/magenta).

For installing it follow the instructions in the magenta's README.

## Training 
1. Go inside the Magenta environment with `source activate magenta`
2. If Tensorflow is installed inside a virtual environment or inside Anaconda run `source activate tensorflow`
3. Dowload the dataset you want to use for training from [google-cloud](https://console.cloud.google.com/storage/quickdraw_dataset/sketchrnn).
4. run the command `sketch_rnn_train --log_root=checkpoint_path --data_dir=dataset_path --hparams="data_set=[dataset_filename.npz]"` where `checkpoint_path` is the folder that will contain the trained model and the checkpoints during training and `dataset_path` is where the dataset is contained. 

All the parameters that can be changed are:
```python
data_set=['aaron_sheep.npz'],  # Our dataset. Can be list of multiple .npz sets.
num_steps=10000000,            # Total number of training set. Keep large.
save_every=500,                # Number of batches per checkpoint creation.
dec_rnn_size=512,              # Size of decoder.
dec_model='lstm',              # Decoder: lstm, layer_norm or hyper.
enc_rnn_size=256,              # Size of encoder.
enc_model='lstm',              # Encoder: lstm, layer_norm or hyper.
z_size=128,                    # Size of latent vector z. Recommend 32, 64 or 128.
kl_weight=0.5,                 # KL weight of loss equation. Recommend 0.5 or 1.0.
kl_weight_start=0.01,          # KL start weight when annealing.
kl_tolerance=0.2,              # Level of KL loss at which to stop optimizing for KL.
batch_size=100,                # Minibatch size. Recommend leaving at 100.
grad_clip=1.0,                 # Gradient clipping. Recommend leaving at 1.0.
num_mixture=20,                # Number of mixtures in Gaussian mixture model.
learning_rate=0.001,           # Learning rate.
decay_rate=0.9999,             # Learning rate decay per minibatch.
kl_decay_rate=0.99995,         # KL annealing decay rate per minibatch.
min_learning_rate=0.00001,     # Minimum learning rate.
use_recurrent_dropout=True,    # Recurrent Dropout without Memory Loss. Recomended.
recurrent_dropout_prob=0.90,   # Probability of recurrent dropout keep.
use_input_dropout=False,       # Input dropout. Recommend leaving False.
input_dropout_prob=0.90,       # Probability of input dropout keep.
use_output_dropout=False,      # Output droput. Recommend leaving False.
output_dropout_prob=0.90,      # Probability of output dropout keep.
random_scale_factor=0.15,      # Random scaling data augmention proportion.
augment_stroke_prob=0.10,      # Point dropping augmentation proportion.
conditional=True,              # If False, use decoder-only model.
```
and have to be specified in the `hparams` list of parameters.

To see the results of the training you can run the sketchDraw.py script which uses the trained model in order to generate the sketches.
