# Guide on dimensions of objects in MIMOSA

## Generative models

Fundamentally, MIMOSA models try to learn *one or multiple functions* from *samples*.

The general form of the functions is:

f_k: (x1, x2, ..., xF) -> (o1, o2, ..., oF)

Where:

* `F` is the number of features, aka input-output pairs of the sample that might be 
correlated across features.
* `xi` is the inputs of feature i, which has dimension `I`.
* `oi` is the outputs of feature i, which has dimension `O`.

We learn in a multi-task setting, meaning that each task (aka *a sample in the dataset*) is an observation of the 
function we are modeling, deviated by some noise. `T` is the number of tasks in the dataset.

In most cases, there are more than one generative function to learn. MIMOSA models this by learning a mixture of multiple 
generative functions, called *mean processes* (as they are gaussian processes used as the mean of the task-gaussian 
processes). Each task is assigned to one mean-process (hard-clustering) or considered a mix of
multiple mean-processes (soft-clustering). `K` is the number of mean-processes in the mixture.

Each task is an observation of the function(s) we are trying to model, but it might be observed only at specific input 
locations, and with specific output dimensions, aka our data results from *heterogeneous sampling*. One of MIMOSA's 
strengths is that it doesn't require any constraint on the sampling of the data of each task. This means that...

* ... each task can have "missing data" (not observed at some input locations or output dimensions)
* ... input points do not need to be aligned across tasks (i.e. they can be observed at different input locations)
* ... some features can be more/less observed, or completely missing from some tasks

However, having some shared input locations across tasks helps "share information" across tasks more effectively.
Having completely distinct input locations across tasks/features will also quickly lead to computational issues, 
as we are still tied to the computational complexity of GPs (cubic in the number of input points).

## Summary of dimensions

* `I`: the dimension of each input point
* `O`: the dimension of each output point
* `F`: the number of features (input-output pairs)
* `T`: the number of tasks (samples in the dataset)
* `K`: the number of mean-processes in the mixture
* `N`: the number of input points observed by individual tasks
* `G`: the number of input point in a grid (for predictions)

## A word on the distinction between "features" and "outputs"

TLDR: features are assumed to have some *correlations* across them, while outputs are assumed to be *uncorrelated* across them.

In MIMOSA, we use the term "feature" to refer to an input-output pair that might be correlated with other input-output 
pairs across tasks. This correlation is computed directly in covariance matrices, taking the shape of block-matrices.
This is why many covariance matrices have two `F * N` or `F * G` dimensions.

Another aspect to take into account is computational complexity. If we observe two quantities, considering them as
features lead to `(2 * N, 2 * N)` covariance matrices to invert, while considering them as outputs lead to two separate 
`(N, N)` covariance matrices to invert. The complexity of the whole model can either be `bigO((F * N)^3)` or 
`bigO(O * N^3)`, which is a huge difference, especially when `N` is large.

## Diving deeper in the *dimension hell*

The summary given above is a bit of an oversimplification, because many dimension might vary (between tasks, 
features, ...), yet we store them in fixed-size arrays. This is done by padding the arrays with "missing data" values 
(NaNs) at the right places.

**Oversimplification 1: input/output dimensions varying across features**

`I` and `O` are to be understood as the maximum input and output dimensions across all features. 
Some features might have lower input/output, in which case the corresponding dimensions in the arrays are filled with NaNs.
These NaNs propagate through the computations, eventually leading to NaN predictions for the missing dimensions. This
makes sense, as we cannot predict a dimension that simply doesn't exist for a feature.

**Oversimplification 2: input points varying across tasks**

`N` is to be understood as the number of input points in the "biggest" task of the dataset. Some tasks might have lower
number of input points, in which case the corresponding elements in the arrays are filled with NaNs. 

Note that this doesn't mean that inputs/outputs are *aligned* across tasks! This would tremendously increase the size of 
each task's data, using large chunks of memory just to store NaNs. Instead, we compute **mappings** for each task, telling
us where each input point of the task is located in the full grid of input points across the dataset.

**Oversimplification 3: size of the grid**

Depending on the context, `G` either refers to the union of all input points across the dataset, or to a custom grid
of input points defined by the user for predictions. In both cases, `G` is to be understood as the maximum number of input
points across all features and tasks.

## Practical example

If you have data about the trajectory of cars across a roundabout, you could use MIMOSA to model the trajectories of the 
cars like this:

* `T` is the number of cars in the dataset
* Your input space might be time since the entry in the roundabout, therefore 1-dimensional, with `I` = 1
* `F` could be 2, with:
  * `dx` and `dy`: the change in x and y coordinates of the car at each time point, which are uncorrelated to allow for 
movement in any direction. This would lead to `O = 2`.
  * `v`: the speed of the car at each time point, which is correlated with `dx` and `dy`, but 1-dimensional.
* `N` is the maximum number of time points observed across all cars and features.
* `K` would be the number of clusters of trajectories we want to learn. For a 3-way roundabout, 9 clusters would make
sense, modeling each destination from each entry point. 

From this, you can expect to extract interesting information about the trajectories of the cars, such as:
* The typical trajectories of the cars across the roundabout (the mean processes)
* Prediction as to which exit a car will take given its entry point and its trajectory so far (the cluster assignment of the task)
* Prediction on the speed of a car from which only the change in coordinates is observed, or vice versa (the correlation between features)
* ...

## The many, many scenarios of heterogeneous sampling and parameter sharing

Prediction of GPs are ruled by hyper-parameters, which can be learnt from the data. Mixing so many scenarios enables 
modeling complex functions, but requires some configuration. Here are some configuration that changes the learning
of the model, and the dimensions of its computations.

**Hyper-parameters:**

* HPs can be shared/distinct across tasks
* HPs can be shared/distinct across features
* HPs can be shared/distinct across output dimensions
* HPs can be shared/distinct across mean-processes in the mixture*
* Some HPs can vary for each dimension of the input space (e.g: ARD length_scales)

**Sampling:**

* Inputs can be aligned or not across tasks
* Inputs can be the same or different across features
* Some features can be missing from some tasks

**Should you bother?**

For hyper-parameters: YES! You will not find the same results with different configs, and exploring various scenarios is
a good way of exploring and better understanding your data.

For sample: ...perhaps. MIMOSA will try its best to learn from what you give it. But the dimensions of some tensors might
quickly explode, i.e. if you give it completely unaligned inputs across tasks and features. If it fails to learn,
takes forever to train or simply crashes due to a lack of memory, you might want to preprocess your data to make it
a bit easier (align some inputs, break big tasks into smaller ones, remove some features, ...).