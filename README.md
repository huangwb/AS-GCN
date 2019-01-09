# AS-GCN in Tensorflow

We provide Tensorflow implementations for the paper "Adaptive Sampling Towards Fast Graph Representation Learning". You need to install Tensorflow and other related python packages, e.g. scypy, networkx, etc. Our code is based on the orginal GCN framework, and takes inspirations from GraphSAGE and FastGCN. The core of this code is that we separate the sampling (i.e. sampler) and propagation (i.e. propagator) processes, both of which are implemented by tensorflow. 

Please note that it is possible that the results by this code would be slightly different to those reported in the paper due to the random noise and some post-modifications of the code.


## Data Preparation
We provide the datasets including cora, citeseer, pubmed. For reddit, you need to download it from http://snap.stanford.edu/graphsage/ and then transfer it to the correct format using transfer_Graph.py.


## Run the Code

1) For cora, citeseer and pubmed, try
```
python run_pubmed.py --dataset dataset_name
```
2) For reddit, try 
```
python run_reddit.py
```


## Citation
If you find this code useful for your research, please cite the paper:


```
@inproceedings{huang2018adapt,
  title={Adaptive Sampling Towards Fast Graph Representation Learning},
  author={Huang, Wenbing and Zhang, Tong and Rong, Yu and Huang, Junzhou},
  booktitle={Advances in Neural Information Processing Systems (NIPS)},
  pages={},
  year={2018}
}
```





