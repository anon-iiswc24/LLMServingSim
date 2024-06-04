The structure of the workload input adheres to the following format. Please note that all communication sizes are measured in bytes and compute times are denoted in cycles:

* **First Line**: (DATA/HYBRID_TRANSFORMER/HYBRID_DLRM)
  * This line specifies the type of training loop parallelization. DATA refers to a purely data-parallel approach, HYBRID_TRANSFORMER denotes a hybrid-parallel approach tailored for Transformer DNN networks, while HYBRID_DLRM implies a hybrid-parallel approach fine-tuned for DLRM DNN networks.

* **Second Line**: (int)
  * This line indicates the number of layers in the DNN.

* **Subsequent Lines**: Each subsequent line describes a layer. The format of layer description  is as follows:
  * (string: **layer name**)
  * (LOCAL/REMOTE/STORAGE/INVALID: **location of input tensor**)
  * (int: **size of input tensor in bytes**)
  * (LOCAL/REMOTE/STORAGE/INVALID: **location of weight tensor**)
  * (int: **size of weight tensor in bytes**)
  * (LOCAL/REMOTE/STORAGE/INVALID: **location of output tensor**)
  * (int: **size of output tensor in bytes**)  
  * (int: **forward pass compute time**)
  * (ALLREDUCE/ALLGATHER/ALLTOALL: **forward pass communication type**)
  * (int: **forward pass communication size**)
  * (str: **additional information passed to the converter**)

*NOTE: All parameters within the brackets are defined on a single line for each layer of the DNN network.* 
