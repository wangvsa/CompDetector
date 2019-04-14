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
usage: keras_detector_3d.py [-h] (--train | --test | --detect PATH)
                            [--clean PATH_TO_CORRECT_SAMPLES] [--error PATH_TO_CORRUPTED_SAMPLES] 
                            [-n EPOCHS] [-m MODEL]
</pre>

## Examples

1. Training </br>
```console
foo@bar:~$ python keras_detector_3d.py --clean PATH_TO_CORRECT_SAMPLES --error PATH_TO_COCCUPTED_SAMPLES --train -n 10
```
2. Testing </br>
```console
foo@bar:~$ python keras_detector_3d.py --clean PATH_TO_CORRECT_SAMPLES --error PATH_TO_COCCUPTED_SAMPLES --test
```
3. Detecting </br>
```console
foo@bar:~$ python keras_detector_3d.py -m MY_MODEL_FILE.h5 --detect PATH
```
