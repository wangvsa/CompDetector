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

## Directory Structure

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
3. Put all samples into the corresponding clean directory

#### Generate k-delay corrupted samples
// You can skip this step if only want to train/test on 0-delay dataset. 
1. Restart the simulation from a corrupted checkpoint
2. Run it for k iterations
3. Split the output into windows and place them under ./error/k-delay/ directory, e.g.
```console
foo@bar:~$ ls ./training/0/error/
0-delay/  5-delay/
```

## Usage
<pre>
usage: detector.py [-h] (--train | --test | --detect PATH)
                   [--data PATH_TO_DATASET] 
                   [-n EPOCHS] [-m MODEL] [-k k] [-b ERROR_BIT_RANGE]
</pre>


#### Train on 0-delay dataset
Set k to 0 when training on 0-delay dataset. The 0-delay corrupted dataset will be generated on the fly
(by inserting errors into clean samples). Thus, the corrupted samples under ./training/error/ will not be loaded.
The following example trains for 10 epochs and saves the model to ./my_model.h5.
```console
foo@bar:~$ python detector.py --data ./training -k 0 --train -n 10 -m ./my_model.h5
```

Errors by default are injected by flipping one of the [0, 15] bits. However, this can be modified by -b option.
```console
foo@bar:~$ python detector.py --data ./training -k 0 --train -n 10 -b 10 -m ./my_model.h5
```
This example trains the detector with errors only injected in [0, 10] bits. Note that -b only works when k = 0.


#### Test on 0-delay dataset
Example 1. Load the pre-trained model, and test it on the same 0-delay training set.
```console
foo@bar:~$ python detector.py --data ./training -k 0 -m ./my_model.h5 --test
```

Example 2. Load the pre-trained model, and test it on a 0-delay testing set. Note that testing directory should have the same
sturcture as the training directory.
```console
foo@bar:~$ python detector.py --data ./testing -k 0 -m ./my_model.h5 --test
```

#### Train on k-delay dataset
If k is not specified, all corrupted samples under ./training/\*/error/\*/ will be loaded.
Here's an example of training (and testing) on 0-delay and 5-delay dataset.
```console
foo@bar:~$ ls ./training/0/error/
0-delay/  5-delay/
foo@bar:~$ python detector.py --data ./training -m ./my_model.h5 --train
foo@bar:~$ python detector.py --data ./training -m ./my_model.h5 --test
```


#### Detection
The --train and --test options work on splitted dataset. We can use --detect to run the detector on original checkpoint files. 
```console
foo@bar:~$ ls PATH_TO_CHECKPOINTS/
chk_0001  chk_0002  chk_0003 ...
foo@bar:~$ python detector.py -m ./my_model.h5 --detect PATH_TO_CHECKPOINTS
```

