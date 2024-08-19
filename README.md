# Sum of Squares Circuits

This repository contains the code for reproducing the experiments of the paper [_"Sum of Squares Circuits"_].

## Project Structure

The repository is structured as follows.
The file ```requirements.txt``` contains all the required Python dependencies, which can be installed by ```pip```.
The directory ```src``` contains all the code, including utility scripts to run experiments and reproduce the plots of
the papers starting from tensorboard log files (see below). In ```src/tests``` we store sanity checks that can be run
by executing ```pytest``` at the root level.
The directory ```econfigs``` contains the configuration files for all the experiments, which consist of selections of
models, datasets and all the relevant hyperparameters.  The directories ```slurm``` and ```sge``` contain some utility scripts to execute
batches of experiments (e.g., grid  searches) on Slurm and Sun Grid Engine (SGE) clusters.

## How to Run Experiments?

### Download the Data

Each data set should be downloaded and placed in the ```./datasets``` directory,
which is the default one.

#### UCI Datasets

The continuous UCI data sets that are commonly used in the _normalizing flow_ literature, i.e.,
Power, Gas, Hepmass and MiniBoone, can be downloaded from [zenodo](https://zenodo.org/record/1161203#.Wmtf_XVl8eN).

#### Image data sets

The download of image data sets -- MNIST, FashionMNIST and CelebA -- is managed automatically thrugh ```torchvision```.

### Run the same hyperparameters grid searches

The directory ```econfigs/``` contains configuration files of the same hyperparameter grid searches we performed for all our experiments.
See below section about running grids of experiments for details.

### Run simple experiments

Simple experiments can be run by executing the Python module ```scripts.experiment.py```.
For a complete overview of the parameters to pass to it, it is suggested to read its code. 

For example, to run an experiment with a ```MPC``` having input layers computing Gaussian likeliohoods on the dataset ```Power```, you can execute
```shell
python -m scripts.experiment --dataset power --model MPC --num-units 8 \
    --optimizer Adam --learning-rate 1e-3 --batch-size 128 --verbose --device cuda
```
The ```--num-units``` argument is used to provide the number of sum, product and input units of each layer in the
tensorized circuit architecture built.

Note that the flag ```--verbose``` will enable terminal logging (e.g., to show the loss).
All the models are learned by minimizing the negative log-likelihood on the
training data with gradient descent.

In addition, to run an experiment with (sum of) complex squared PCs -- ```SOS``` --
on the data set ```Power```, you can execute
```shell
python -m scripts.experiment --dataset power --model SOS --num-units 8 --complex --num-components 4 \
    --optimizer Adam --learning-rate 1e-3 --batch-size 128 --verbose --device cuda
```
The ```--num-components``` argument is used to specify the number of squares in the SOS PC,
and the argument ```--complex``` enables complex parameters.

#### Logging Metrics and Models

To log metrics locally such as the test average log-likelihood or to observe training curves,
one can use either ```tensorboard```.
For ```tensorboard``` it is sufficient to specify
```--tboard-path /path/to/tboard-directory```
with an arbitrarily chosen path that will contain Tensorboard files.

It is possible to save the best checkpoint of the model, that will be updated only upon
an improvement of the loss on the validation data.
To enable this, you can specify
```--save-checkpoint``` and ```--checkpoint-path /path/to/checkpoints```
with a path that will contain the model's weights in the ```.pt``` PyTorch format.

### Run a Grid of Experiments

To run a batch of experiments, e.g., as to do a hyperparameters grid search,
you can use the ```scripts.grid``` module by specifying a grid configuration JSON file.
The directory ```./econfigs``` contains some examples of such configuration file.

The fields to specify are the following:

- ```common``` contains parameters to pass to ```scripts.experiment```
  that are common to each experiment of the batch. 
- ```datasets```contains the list of data sets on which each experiment instance will be executed on.
- ```grid.common``` contains a grid of hyperparameters.
  Each hyperparameter is a pair ```"name": value``` where value can be either a single value or a list of values.
  A products of lists will be performed as to retrieve all the possible configurations of hyperparameters in the grid.
- ```grid.models``` contains additional hyperparameter configurations that are specific for some
  dataset or some model. Each entry in ```grid.models``` is a dictionary from a single dataset or a list of datasets,
  to a set of maps from model names to hyperparameter configurations. The semantic is that the hyperparameters specified
  in ```grid.models``` will overwrite the ones in ```grid.common``` for some specific combination of datasets and models.

To run a batch of experiments, you can execute
```shell
python -m scripts.grid path/to/config.json
```
You can also use the flag ```--dry-run``` to just print the list of generated commands, without running them.
This is particularly useful in combination with job schedulers on clusters, e.g., Slurm.

Additionally, one can specify a number of experiments that will be distatched in parallel
(by default only one experiment will be runned) by specifying ```--num-jobs k```, where k is the maximum number
of experiments that will be "alive" at each time.

Instead of specifying parallel jobs that will be runned on the same device,
you can also specify multiple devices on which the experiments will be dispatched on.
This can be done with ```--multi-devices```.
For instance, ```--multi-devices cuda:0 cuda:2 cpu``` will dispatch three experiment at a time,
respectively on devices ```cuda:0```, ```cuda:2``` and ```cpu```.

Finally, you can specify independent repetition for each experiment of the batch,
which will append a different ```--seed``` argument for each experiment command to launch.
This can be done with, for instance, ```--num-repetitions 5```.

Disclaimer: in case of repeated runs the checkpoints that are saved are not reliable,
as they can be overwritten by repeated run.

#### Run a Grid of Experiments (on Slurm)

To run a grid of experiments on a Slurm cluster, we first need to configure some constants in the ```slurm/launch.sh```
utility scripts, such as the Slurn partition to use, the maximum number of parallel jobs, the needed resources,
and the path to a local directory of nodes in order to save model checkpoints and tensorboard logs.

Then, we need to generate the commands to dispatch and save it to a text file.
For this purpose, it is possible to use the script ```scripts.grid``` (see above) with the argument ```--dry-run```.
For instance, to generate the commands to execute for the experiments on UCI data sets, it suffices to run the command
```shell
python -m scripts.grid econfigs/image-sos-npcs.json --dry-run > exps-image-sos-npcs.txt
```
Finally, the Bash script ```slurm/launch.sh``` will automatically dispatch an array of Slurm jobs to execute.
```shell
EXPS_ID=image-sos-npcs bash slurm/launch.sh exps-image-sos-npcs.txt
```
The Slurm jobs should now appear somewhere in the queue, which can be viewed by running ```squeue```.
