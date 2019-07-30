# Program notes
## Convolutional Network:
* In Pytorch, convolutional layer is defined as:
```python
nn.Conv2d(in_channels, out_channels kernel_size, stride, padding)
```
* `in_channels` is the number of incoming image feature maps.
* `out_channels` are number of kernels you will use, which in turn determine the number of feature maps. Each kernel produces a separate feature map image after scanning the input. So 32 out_channels means you will have 32 feature map images. The more kernels you have, the more features you will detect.
* Convolution layers expect 4D tensors: [batch_size x incoming channels x height x width]


* stride
* Formula to determine output size of image:
* padding

## Preprocessing Image
* Crop the health bar in MsPacman-v0 for example:
  To test image manipulations in terminal:
  ```python
  import gym
  import matplotlib.pyplot as plt # hit enter
  obs = env.reset()
  plt.show(plt.imshow(obs[0:174])) # outputs image with stats cropped 
  ```
* Next Steps: convert to PIL Image, convert to greyscale


## References:
* *Playing Atari with Deep Reinforcement Learning* - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
* *A Guide to convolution arithmetic for deep learning* - Vincent Dumoulin, Francesco Visin
