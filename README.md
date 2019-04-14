# Silent Error Detector

A silent error detector for 3D discrete time simulations. The detector uses a deep neural network that is trained 
on error-free and erroneous executions of an application. It is able to detect bit-flips in in-memory physical variables
of applications with different initial conditions. Unlike existing methods that require the detector to be run at each
iteration, we show that our method can identify errors even after multiple iterations.


## Install

The code is written in Python (2.7.5). The neural network is implemented with Keras (2.2.2). We use Tensorflow as backend in our
experiments.
1. Download or clone the code
2. Install dependent packages:
  numpy, keras, tensorflow-gpu (or tensorflow for CPU only), bitstring, h5py, mpi4py, scikit-image.


## Usage
<pre>
usage: detector.py [-h] (--train | --test | --detect PATH)
                            [--clean PATH_TO_CORRECT_SAMPLES] [--error PATH_TO_CORRUPTED_SAMPLES] 
                            [-n EPOCHS] [-m MODEL]
</pre>

#### Directory Structure

The training directory looks like the following, each sub-directory contains clean and error samples generated from one initial condition.
```console
foo@bar:~$ ls ./training
0/  1/  2/
foo@bar:~$ ls ./training/0/
clean/  error/
```
#### Generate clean samples
1. Run the simulation and save the checkpiont files
2. Split each checkpoint file into a number of windows (samples)
3. Put all samples into the corresponding clean directory.

#### Train on 0-delay dataset
The 0-delay corrupted dataset will be generated on the fly. No need to specifiy the directory to error samples.
The following example trains for 10 epochs and saves the model to ./my_model.h5
```console
foo@bar:~$ python detector.py --clean ./training --train -n 10 -m ./my_model.h5
```

#### Test on 0-delay dataset
Load the pre-trained model, and test it on the same 0-delay training set
```console
foo@bar:~$ python detector.py --clean ./training -m ./my_model.h5 --test
```

Load the pre-trained model, and test it on a 0-delay testing set. Note that testing directory should have the same
sturcture as the training directory.
```console
foo@bar:~$ python detector.py --clean ./testing -m ./my_model.h5 --test
```


## Examples

1. Training </br>
```console
foo@bar:~$ python detector.py --clean PATH_TO_CORRECT_SAMPLES --error PATH_TO_COCCUPTED_SAMPLES --train -n 10
```
2. Testing </br>
```console
foo@bar:~$ python detector.py --clean PATH_TO_CORRECT_SAMPLES --error PATH_TO_COCCUPTED_SAMPLES --test
```
3. Detecting </br>
```console
foo@bar:~$ python detector.py -m MY_MODEL_FILE.h5 --detect PATH
```
