<h1 align='center'>Vision-LSTM</h1>

### This is a warehouse for Vision-LSTM-Pytorch-model, can be used to train your image-datasets for vision tasks.

<p align="center">
  <img src="https://github.com/jiaowoguanren0615/Vision-LSTM/blob/main/sample_png/ViL-configs.jpg" height="500" width="400" style="display: inline-block;" />
  <img src="https://github.com/jiaowoguanren0615/Vision-LSTM/blob/main/sample_png/Compare-otherNets.jpg" height="300" width="400" style="display: inline-block;" />
</p>

### [Vision-LSTM: xLSTM as Generic Vision Backbone](https://arxiv.org/abs/2406.04303)  

## Preparation
### Create conda virtual-environment
```bash
conda env create -f environment.yml
```

### Download the dataset: 
[flower_dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).  

## Project Structure
```
├── datasets: Load datasets
    ├── my_dataset.py: Customize reading data sets and define transforms data enhancement methods
    ├── split_data.py: Define the function to read the image dataset and divide the training-set and test-set
    ├── threeaugment.py: Additional data augmentation methods
├── models: Vision-LSTM Model
    ├── build_models.py: Construct Vision-LSTM models
    ├── vision_lstm_utils.py: Construct Vision-LSTM children modules
    ├── kan_model.py: Define KAN model replaces MLP layers
├── scheduler:
    ├──scheduler_main.py: Fundamental Scheduler module
    ├──scheduler_factory.py: Create lr_scheduler methods according to parameters what you set
    ├──other_files: Construct lr_schedulers (cosine_lr, poly_lr, multistep_lr, etc)
├── util:
    ├── engine.py: Function code for a training/validation process
    ├── LBFGS_optimizer.py: Define LBFGS_optimizer for training KAN Network  
    ├── losses.py: Knowledge distillation loss, combined with teacher model (if any)
    ├── lr_decay.py: Define "inverse_sqrt_lr_decay" function for "Adafactor" optimizer
    ├── lr_sched.py: Define "adjust_learning_rate" function
    ├── optimizer.py: Define Sophia & Adafactor optimizer(for ViT-e & PaLI-17B)
    ├── samplers.py: Define the parameter of "sampler" in DataLoader
    ├── utils.py: Record various indicator information and output and distributed environment
├── estimate_model.py: Visualized evaluation indicators ROC curve, confusion matrix, classification report, etc.
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___,  ___data_len___, ___num_workers___ and ___nb_classes___ parameters. If you want to draw the confusion matrix and ROC curve, you only need to set the ___predict___ parameter to __True__.  
Moreover, you can set the ___opt_auc___ parameter to ___True___ if you want to optimize your model for a better performance(maybe~).  

## Train this model

### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Transfer Learning:
Step 1: Write the ___pre-training weight path___ into the ___args.fintune___ in string format.  
Step 2: Modify the ___args.freeze_layers___ according to your own GPU memory. If you don't have enough memory, you can set this to True to freeze the weights of the remaining layers except the last layer of classification-head without updating the parameters. If you have enough memory, you can set this to False and not freeze the model weights.  

#### Here is an example for setting parameters:
![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/transfer_learning.jpg)


### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error.  

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Citation
```
@article{alkin2024visionlstm,
  title={Vision-LSTM: xLSTM as Generic Vision Backbone},
  author={Benedikt Alkin and Maximilian Beck and Korbinian P{\"o}ppel and Sepp Hochreiter and Johannes Brandstetter}
  journal={arXiv preprint arXiv:2406.04303},
  year={2024}
}
```
