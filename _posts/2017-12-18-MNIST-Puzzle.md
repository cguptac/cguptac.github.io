---
layout: post
title: MNIST puzzle
---

A recent paper by Noroozi on [puzzle solving](https://arxiv.org/abs/1603.09246) using neural networks caught my attention.
It's a pretty neat idea, and got me thinking about how far that idea could be pushed. 
In other words, what is the largest size of puzzles a neural network could be trained to solve? And that is the most complicated boundary topology that could be learned?
Of course, to test an idea like this, I needed access to a dataset of high-resolution images which I didn't have. 
So the next best idea seemed to be the polar opposite: what is a low information dataset on which a puzzle solving network be trained?
Turns out MNIST works perfectly here!


So here's the basic idea: We're going to randomly break an image up into boxes, permute the boxes, and then stitch them together into a new image. 

```python
# we are going to try to create a jigsaw puzzle from an image. 
# the encoder is going to learn to solve the puzzle


def decompose(image, stride_i=4, stride_j=4):
    '''takes an input image and breaks it up into 4x4 blocks'''
    assert image.shape[0]%stride_i == 0
    assert image.shape[0]%stride_j == 0
    start = 0
    l = []
    for i in range(image.shape[0]//stride_i):
        for j in range(image.shape[1]//stride_j):
            l += [image[(start+stride_i*i):(start+stride_i * (i+1)), 
                         (start + stride_j*j):(start+stride_j*(j+1))]]
    shuffle(l)
    return l

def stitch(shuffled_image):
    '''takes a list of square blocks and assembles into an image'''
    assert np.sqrt(len(shuffled_image)) %1 == 0 # check that shuffled image can be made square
    N = int(np.sqrt(len(shuffled_image)))
    return np.block([shuffled_image[i*N:(i+1)*N] for i in range(N)])
```

These functions will permute the images in the lower row into images in the top row
![_config.yml]({{ base-url }}/images/blog/2017-12-18/raw_permuted.png)

The rest turns out to be fairly straightforward. A basic convolutional autoencoder, built on top of examples from the Keras website, learns to recreate permuted images into originals well enough to pass a visual test (interestingly, the loss function evaluates to a value I would consider significantly differently from 0 even when the network does pretty well visually). 

![_config.yaml]({{ base-url }}/images/blog/2017-12-18/performance.png)
The top row is the raw image, the middle row is permuted, and the bottom row is reconstructed from permuted input.

The jupyter notebook that implements this code (and a little bit extra) can be found [here](https://github.com/cguptac/blog/tree/master/convnn).
                                                                                                
### Epilogue                                                                                    
                                                                                                
Tried to recreate this with cifar-10 and failed miserably. The pixel density is really too low for the task at hand.


