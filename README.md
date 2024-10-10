# **SPrint: Self-Paced Continual Learning with Adaptive Curriculum and Memory Replay**<br>
<img width="781" alt="overview5" src="https://github.com/user-attachments/assets/81cba275-1876-48d2-8daa-06d66d506e9b">

## Abstract
Continual learning addresses the challenge of learning new concepts progressively without forgetting previously acquired knowledge, a problem known as catastrophic forgetting. While existing methods have shown high performance on specific datasets or model architectures, they often lack generalizability across diverse learning scenarios. To overcome this limitation, we propose a novel continual learning method, which dynamically adjust the difficulty of both training samples and in-memory samples in a self-paced manner based on the model’s evolving learning capacity. We empirically demonstrate that SPrint robustly outperforms state-of-the-art methods despite its simplicity.

## Getting Started
### Requirements 

- python==3.7.11
- torch==1.10.0
- numpy==1.21.2
- torch_optimizer==0.3.0
- randaugment==1.0.2
- easydict==1.13
- pandas==1.1.5

### Datasets
All the datasets are saved in `dataset` directory by following formats as shown below.

```angular2html
[dataset name] 
    |_train
        |_[class1 name]
            |_00001.png
            |_00002.png 
            ...
        |_[class2 name]
            ... 
    |_test (val for ImageNet)
        |_[class1 name]
            |_00001.png
            |_00002.png
            ...
        |_[class2 name]
            ...
```

### Usage 
To run the experiments in the paper, you just run `experiment.sh`.
```angular2html
bash experiment.sh 
```
For various experiments, you should know the role of each argument. 

- `MODE`: CIL methods. Our method is called `sprint`.
  (`joint` calculates accuracy when training all the datasets at once.)
- `MEM_MANAGE`: Memory management method. [default, random, reservoir, uncertainty, prototype].
- `RND_SEED`: Random Seed Number 
- `DATASET`: Dataset name [cifar10, cifar100, imagenet100]
- `EXP`: Task setup [disjoint, blurry10]
- `MEM_SIZE`: Memory size
- `TRANS`: Augmentation. Multiple choices [cutmix, cutout, randaug, autoaug]

### Results
There are three types of logs during running experiments; logs, results, tensorboard. 
The log files are saved in `logs` directory, and the results which contains accuracy of each task 
are saved in `results` directory. 
```angular2html
root_directory
    |_ logs 
        |_ [dataset]
            |_{mode}_{mem_manage}_{stream}_msz{k}_rnd{seed_num}_{trans}.log
            |_ ...
    |_ results
        |_ [dataset]
            |_{mode}_{mem_manage}_{stream}_msz{k}_rnd{seed_num}_{trans}.npy
            |_...
```

In addition, you can also use the `tensorboard` as following command.
```angular2html
tensorboard --logdir tensorboard
```

## Acknowledgment and Modifications

This project is based on the [Rainbow Memory](https://github.com/clovaai/rainbow-memory) project by Jihwan Bang, Heesu Kim, YoungJoon Yoo, Jung-Woo Ha, and Jonghyun Choi. We have made the following modifications to the original codebase:

1. Implement our method in `methods/sprint.py`
2. Add forgetting events computation algorithm in `utils/compute_forgetting.py`

We express our gratitude to the original authors for their valuable work and insights in the field of continual learning.

## License

This project is licensed under the GNU General Public License v3.0, following the original Rainbow Memory project's license.

As per the terms of the GPL-3.0 license, we acknowledge that this work is a derivative of the Rainbow Memory project. We have retained the copyright notice and license derived from the original project.

For the original Rainbow Memory project, please refer to:
- Project: [Rainbow Memory](https://github.com/clovaai/rainbow-memory)
- Paper: [Rainbow Memory: Continual Learning with a Memory of Diverse Samples](https://openaccess.thecvf.com/content/CVPR2021/html/Bang_Rainbow_Memory_Continual_Learning_With_a_Memory_of_Diverse_Samples_CVPR_2021_paper.html)
- License: [GNU General Public License v3.0](https://github.com/clovaai/rainbow-memory/blob/master/LICENSE)

If you use this code for your research, please consider citing both our work and the Rainbow Memory paper:
```angular2
@InProceedings{Bang_2021_CVPR,
    author    = {Bang, Jihwan and Kim, Heesu and Yoo, YoungJoon and Ha, Jung-Woo and Choi, Jonghyun},
    title     = {Rainbow Memory: Continual Learning With a Memory of Diverse Samples},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {8218-8227}
}
```
