## Leak Localization in Water Distribution Networks Using Artificial Neural Network
### Step 1：Generatioon of leak dataset, see foder 1-Dataset
1. Run simu_tool.ipynb step by step.

2. To simiplify network, run topo_delete_pipe.py&topo_delete_node.py.

    Note: The hydraulic simulation software is not available in demo.

### Step 2: Training of ANN, see foder 2-NN
1. Modify model structure in ann.py.

2. Run nn_train.ipynb step by step.

3. Run nn_draw.py to plot ANN prediction results.

#### Result: Top3 Accuracy on a network with 878 pipelines
![image](https://github.com/xdhcode/Demo_Machine_Learning/blob/main/3-Result/Pipe878_Top3_Accuracy.png)