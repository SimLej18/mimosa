# MIMOSA - A Multi-Input, Multi-Output Sequence Analyser

MIMOSA is a framework for analysing functional data samples, based on Gaussian processes.

> **⚠️ Project Status**:  "MIMOSA is currently in early development, and is not yet ready for use. The API and features are subject to change."

Its core features (will) include:
* Joint learning across multiple samples (Multi-Task GPs) with heterogeneous inputs and outputs
* Clustering of samples
* Multi-dimensional inputs and outputs
* Multi-feature correlation discovery
* Probabilistic predictions with uncertainty quantification
* Adaptation to non-gaussian data and classification tasks
* Sparse learning and approximations for scalability
* Clever initialisation for hyper-parameters of the models
* Support for training and predicting on GPUs and TPUs
* Visualisation tools for data and model outputs

---
## Installation

[Package not yet available]

---
## Philosophy

We basically took a Gaussian process, and put a ton of ✨*fancy components*✨ on top of it.

* ➡️ Multi-task learning based on [Magma](https://jmlr.org/papers/v24/20-1321.html)
* ➡️ Clustering as a mixture of Magma GPs
* ➡️ Multi-dimensional inputs thanks to efficient kernels from [Kernax](https://github.com/SimLej18/kernax-ml)
* ➡️ Multi-dimensional uncorrelated outputs by simply broadcasting the kernel across output dimensions
* ➡️ Multi-feature correlation discovery by learning a [convolution process]([TODO]) with specific kernel


We use **JAX** as a back-end for every computation! 
`JIT` compilation saves us a lot of time, 
`vmap` allows us to efficiently batch over all dimensions of the problem,
`grad` spares us the hastle of implementing gradients for every component of the model, 
and XLA support allows us to train and predict on GPUs and TPUs for scalability.

A GP library is nothing without a fast and modular **Kernel library**. 
MIMOSA relies on the efficiency-focused [Kernax package](https://github.com/SimLej18/kernax-ml) for every kernel use 
throughout the algorithms.

---
## Examples

[TODO]

---
## Authors and citation

MIMOSA is primarily developed by the *Magma Task Force*, composed of:

* [Arthur Leroy](https://arthur-leroy.netlify.app/), researcher at Paris Saclay and INRAe (FR), main author of the original tool [MagmaClustR](https://arthurleroy.github.io/MagmaClustR/) and coordinator of the Task Force.
* [Simon Lejoly](https://researchportal.unamur.be/fr/persons/slejoly/), PhD student at UNamur (BE), main developper of the package and author of the [Kernax package](https://github.com/SimLej18/kernax-ml)
* [Alexia Grenouillat]([TODO]), PhD student at INSA Toulouse (FR), developper of multi-feature correlation discovery.
* [Térence Viellard]([TODO]), PhD student at INRAe (FR), working on sparse approximations and scalability.

If you use the package for your research, please consider citing the following paper:

[TODO]

If your research field could provide an interesting example of using MIMOSA, please consider publishing a toy example in
the documentation [TODO].

---
## Contributing

At this stage of development, we are not yet accepting contributions to the codebase. However, if you are interested in contributing to the project, please feel free to reach out to us via email or through our GitHub repository.

---
## License

This work is distributed under the [MIT License](https://opensource.org/licenses/MIT). See `LICENSE` for more information.

---


