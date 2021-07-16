# dwtls: Discrete Wavelet Transform LayerS

[![PyPI version](https://badge.fury.io/py/dwtls.svg)](https://badge.fury.io/py/dwtls)
| [**Paper**](https://doi.org/10.1109/TASLP.2021.3072496)
| [**Demo**](https://tomohikonakamura.github.io/Tomohiko-Nakamura/demo/MRDLA/)

`dwtls` is a downsampling/pooling layer library of discrete wavelet transform (DWT) layers with fixed and trainable wavelets presented in [1].
The use of DWT has an anti-aliasing filter and the perferct reconstruction property, at least either of which is lacked in conventional pooling layers.
The two properties enables the DWT layers to propagate entire information of an input feature.  

The library includes the following layers.
- DWT layers with fixed wavelets (e.g., Haar, CDF22, CDF26, CDF15, and DD4 wavelets)
- DWT layers with trainable wavelets (trainable DWT layers)
- Weight-normalized trainable DWT layers
The library works with Python>=3.7 and pytorch>=1.0.

## Installation
`dwtls` can be installed with:
```bash
pip install dwtls
```
or 
```bash
git clone git@github.com:TomohikoNakamura/dwtls.git
pip install -e dwtls
```

## How to use
All DWT layers are implemented as subclasses of `torch.nn.Module` of PyTorch, so we can use them in many DNNs, using PyTorch scripts.
We prepare [a short tutorial notebook](examples/tutorial.ipynb), which makes it easy to use our DWT layers in your codes.  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomohikoNakamura/dwtls/blob/master/examples/tutorial.ipynb)

## Citation
If you use `dwtls` in your work, please cite:

```bibtex
@article{TNakamura202104IEEEACMTASLP,
    author={Nakamura, Tomohiko and Kozuka, Shihori and Saruwatari, Hiroshi},
    title = {Time-Domain Audio Source Separation with Neural Networks Based on Multiresolution Analysis},
    journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    volume=29,
    pages={1687--1701},
    month=apr,
    year=2021,
    doi={10.1109/TASLP.2021.3072496},
}
```

## References
[1] Tomohiko Nakamura, Shihori Kozuka, and Hiroshi Saruwatari, “Time-Domain Audio Source Separation with Neural Networks Based on Multiresolution Analysis,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 1687–1701, Apr. 2021. [pdf](https://doi.org/10.1109/TASLP.2021.3072496), [demo](https://tomohikonakamura.github.io/Tomohiko-Nakamura/demo/MRDLA/)

## License
`dwtls` is MIT licensed (see [LICENSE](LICENSE) file).
