# Optimal Fast-Style Transfer
Source files of my Computer Vision Seminar Project [Optimal Real-Time Video Style Transfer](https://drive.google.com/open?id=1p2ynBRw2CiYbsb4JlxI0dEBoIttsjdRx).

Code is partially based on the ReCoNet implementation of [safwankdb](https://github.com/safwankdb/ReCoNet-PyTorch). 

A demo file `demo.py` for training and testing is provided. Additionally some pre-trained network files in `/runs/` are provided to skip training.

## Requirements:
Mandatory:
* Python (tested with 3.6)
* CUDA capable device with
  - cudatoolkit (tested with 10.0.130)
  - cudnn (tested with 7.1)
* PyTorch (tested with 1.2.0)
* numpy, imageio, OpenCV3, scikit-image, PyQt5

Optional (additional optical flow generation):
* Tensorflow (tested with 1.14.0)
* [SelFlow](https://github.com/ppliuboy/SelFlow)

## Install:
Just install all mandatory packages from the requirements list. Everything else should be handled by Python. But be sure to adjust all paths when training or testing.

## GUI
The file `fs_gui.py` provides an interface for interactive testing. You don't need to download any dataset for this to work. You can also use your webcam instead of the test video.

<div align = 'center'>
<img src = 'examples/ost_demo.gif' alt = 'GUI preview' width = '700px' height = '606px'>
</div>

## Training:
For training you will need either the [FlyingChairs2](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) (recommended), the [Hollywood2](https://www.di.ens.fr/~laptev/actions/hollywood2/) or the [Microsoft COCO](http://cocodataset.org/) dataset. You can also use your own dataset, but this requires modifying the base code. You will also need to pre-process the data using the files within `dataset-generation`. To train a new network you can use the `train_net` method within `fs_tests.py`. 

## Testing:
For testing you have to download the [Sintel](http://sintel.is.tue.mpg.de/) dataset. You will also need to pre-process the data using the files within `dataset-generation`. To test a trained network you can use the `infer_test` method within `fs_tests.py`. 

## Demo:
The demo file `demo.py` includes an example for training and testing. Be sure to adjust the standard paths of `train_net` and `infer_test` and the batch size or input resolution if GPU memory is restricted.

## Test System Info
* OS: Win10 (1909)
* CPU: AMD R5 3600
* RAM: 16GB DDR4 3200MHz CL14
* GPU: NVIDIA RTX 2080
