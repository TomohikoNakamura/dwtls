# Copyright (c) Tomohiko Nakamura
# All rights reserved.
"""Library of discrete wavelet transform layers using fixed and trainable wavelets
"""

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT(nn.Module):
    '''Discrete wavelet transform layer using fixed wavelet

    This layer uses discrete wavelet transform (DWT) for downsampling. It enables us to downsample features without losing their entire information and causing aliasing in the feature domain.

    Attributes:
        wavelet (str): Wavelet type [lazy, haar, cdf22, cdf26, cdf15, dd4]
            The "lazy" wavelet is the same as the squeezing operation.
        p_weight (torch.Tensor): Prediction weight
        p_params (dict): Convolution parameters for the prediction step
        u_weight (torch.Tensor): Update weight
        u_params (dict): Convolution parameters for the update step
        scaling_factor (torch.Tensor): Scaling factor

    Referenes:
        [1] Tomohiko Nakamura, Shihori Kozuka, and Hiroshi Saruwatari, “Time-Domain Audio Source Separation with Neural Networks Based on Multiresolution Analysis,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 1687–1701, Apr. 2021.
    '''
    def __init__(self, wavelet="haar"):
        '''

        Args:
            wavelet (str): Wavelet name (lazy, haar, cdf22, cdf26, cdf15, dd4)
        '''
        super().__init__()
        if wavelet not in ["lazy", "haar", "cdf22", "cdf26", "cdf15", "dd4"]:
            raise NotImplementedError(f'Undefined wavelet [{wavelet}]')
        self.wavelet = wavelet
        ########
        if self.wavelet in ["haar", "cdf22", "cdf26", "cdf15", "dd4"]:
            if self.wavelet in ["cdf22", "cdf26"]:
                p_weight = [0.5, 0.5, 0.0]
            elif self.wavelet in ["haar", "cdf15"]:
                p_weight = [1]
            elif self.wavelet in ["dd4"]:
                p_weight = [0, -1/16.0, 9/16.0, 9/16.0, -1.0/16][::-1]
            self.register_buffer('p_weight', torch.tensor(
                p_weight, dtype=torch.float)[None, None, :])
            self.p_params = dict(bias=None, stride=1, padding=(
                self.p_weight.shape[2]-1)//2, dilation=1, groups=1)
            #
            if self.wavelet in ["haar"]:
                u_weight = [0.5]
            elif self.wavelet in ["cdf15"]:
                u_weight = list(numpy.array([-3, 22, 128, -22, 3])/256)[::-1]
            elif self.wavelet in ["cdf22"]:
                u_weight = [0.0, 0.25, 0.25]
            elif self.wavelet in ["cdf26"]:
                u_weight = list(numpy.array([5, -39, 162, 162, -39, 5, 0])/512)[::-1]
            elif self.wavelet in ["dd4"]:
                u_weight = [-1/32, 9/32, 9/32, -1/32, 0][::-1]
            self.register_buffer('u_weight', torch.tensor(u_weight, dtype=torch.float)[None, None, :])
            self.u_params = dict(bias=None, stride=1, padding=(self.u_weight.shape[2]-1)//2, dilation=1, groups=1)
            #
            scaling_factor = numpy.sqrt(2.0)
            self.register_buffer('scaling_factor', torch.tensor(scaling_factor, dtype=torch.float))

    def split(self, x):
        '''Split step

        Args:
            x (torch.Tensor): Input feature (batch x channels x time)

        Returns:
            Tuple[torch.Tensor]: even- and odd-indexed components of `x`, each of which has (batch x channels x time/2) shape.
        '''
        even = x[:, :, ::2]
        odd = x[:, :, 1::2]
        return even, odd

    def predict(self, even):
        '''Predict odd from even

        Args:
            even (torch.Tensor): Even component (batch x ch x time)

        Return:
            torch.Tensor: Predicted odd component (batch x ch x time)
        '''
        if self.wavelet in ["haar", "cdf15"]: # Prediction filter length is one.
            return even*self.p_weight[0,0,0]
        else:
            B, C, T = even.shape
            return F.conv1d(even.reshape(B*C, 1, -1), self.p_weight, **self.p_params).reshape(B, C, -1)

    def update(self, highfreq):
        '''Smooth even from prediction error

        Args:
            highfreq (torch.Tensor): Prediction error, a.k.a. (unscaled) high-frequency component (batch x ch x time)

        Return:
            torch.Tensor: Residual for smoothed even (batch x ch x time)
        '''
        if self.wavelet in ["haar"]: # Update filter length is one.
            return highfreq*self.u_weight[0,0,0]
        else:
            B, C, T = highfreq.shape
            return F.conv1d(highfreq.reshape(B*C, 1, -1), self.u_weight, **self.u_params).reshape(B, C, -1)

    def forward(self, x, no_concat=False):
        '''Forward computation 

        Args:
            x (torch.Tensor): Input feature (batch x ch x time)
            no_concat (bool): If True, return tuple of high- and low-frequency components (default: False)
        
        Return:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: High- and low-frequency components concatenated along the channel axis (batch x ch*2 x time/2) or tuple of them (batch x ch x time/2)
        '''
        assert x.shape[-1]%2 == 0, "Time length must be even."
        if self.wavelet == "lazy":
            even, odd = x[:, :, ::2], x[:, :, 1::2]
            y = odd, even
        elif self.wavelet in ["haar", "cdf22", "cdf26", "cdf15", "dd4"]:
            even, odd = self.split(x)
            unscaled_highfreq = odd - self.predict(even)
            unscaled_lowfreq = even + self.update(unscaled_highfreq)
            y = unscaled_highfreq/self.scaling_factor, unscaled_lowfreq*self.scaling_factor
        else:
            raise NotImplementedError(f'Undefined wavelet [{self.wavelet}]')
        if no_concat:
            return y
        else:
            return torch.cat(y, dim=1)

    def inverse(self, x, no_concat=False):
        '''Inverse computation

        Args:
            x (torch.Tensor): High- and low-frequency components concatenated along the channel axis (batch x ch*2 x time/2) or tuple of them (batch x ch x time/2)
            no_concat (bool): If True, `x` is assumed to be tuple of high- and low-frequency components (default: False)
        
        Return:
            torch.Tensor: inverse DWT of x (batch x ch x time)
        '''
        if no_concat:
            assert isinstance(x, tuple), 'If no_concat=True, x must be a tuple of high- and low-frequency components.'
            highfreq, lowfreq = x
        else:
            # First and second ones are respectively high- and low-frequency components.
            C = x.shape[1] // 2
            highfreq = x[:, :C, :]
            lowfreq = x[:, C:, :]
        if self.wavelet == "lazy":
            y = torch.stack((lowfreq, highfreq), dim=-1)
            y = y.reshape(y.shape[0], y.shape[1], -1)
        elif self.wavelet in ["haar", "cdf22", "cdf26", "cdf15", "dd4"]:
            unscaled_highfreq = highfreq/self.scaling_factor
            unscaled_lowfreq = lowfreq*self.scaling_factor
            even = unscaled_lowfreq - self.update(unscaled_highfreq)
            odd = unscaled_highfreq + self.predict(even)
            y = torch.stack((even, odd), dim=-1)
            y = y.reshape(y.shape[0], y.shape[1], -1)
        else:
            raise NotImplementedError(f'Undefined wavelet [{self.wavelet}]')
        return y

class PredictUpdateStage(nn.Module):
    '''Module of predict and update steps
    '''
    def __init__(self, is_first: bool, predict_ksize=3, update_ksize=3, requires_grad={"predict": True, "update": True}, initial_values={"predict": [0,1,0], "update": [0,0.5,0]}):
        '''

        Args:
            is_first (bool): Whether first prediction and update stages are.
            predict_ksize (int): Prediction filter length
            update_ksize (int): Update filter length
            requires_grad (dict[str:bool]): Whether we wish to train prediction and update filters.
                ex.) {"predict": True, "update": True}
            initial_values (dict[str:iteratable]): Initial values of prediction and update filters.
                ex.) {"predict": [0,1,0], "update": [0,0.5,0]}
        '''
        super().__init__()
        self.is_first = is_first
        self.predict_ksize = predict_ksize
        self.update_ksize = update_ksize
        if self.predict_ksize == 1:
            if requires_grad["predict"]:
                raise ValueError
            self._predict_weight = nn.Parameter(
                torch.tensor([1 if self.is_first else 0], dtype=torch.float),
                requires_grad=requires_grad["predict"]
            )
        else:
            self._predict_weight = nn.Parameter(
                torch.from_numpy(numpy.ones((predict_ksize,), dtype='f')),
                requires_grad=requires_grad["predict"]
            )
            self._predict_weight.data.uniform_(-numpy.sqrt(3.0), numpy.sqrt(3.0))
        if self.update_ksize == 1:
            if requires_grad["update"]:
                raise ValueError
            self._update_weight = nn.Parameter(
                torch.tensor([0.5 if self.is_first else 0], dtype=torch.float),
                requires_grad=requires_grad["update"]
            )
        else:
            self._update_weight = nn.Parameter(
                torch.from_numpy(numpy.ones((update_ksize,), dtype='f')),
                requires_grad=requires_grad["update"]
            )
            self._update_weight.data.uniform_(-numpy.sqrt(3.0), numpy.sqrt(3.0))
        self.initial_values = initial_values

    @property
    def predict_weight(self):
        return self._predict_weight

    def predict_op(self, x):
        if self.predict_ksize == 1:
            return x * self._predict_weight
        else:
            return F.conv1d(x, self.predict_weight[None, None, :], None, 1, (self.predict_ksize-1)//2, 1, 1)

    @property
    def update_weight(self):
        return self._update_weight

    def update_op(self, x):
        if self.update_ksize == 1:
            return x * self._update_weight
        else:
            return F.conv1d(x, self.update_weight[None, None, :], None, 1, (self.update_ksize-1)//2, 1, 1)

    def init_params(self):
        '''Initialize prediction and update weights by `initial_values`
        '''
        initial_values = self.initial_values
        if self.initial_values is not None:
            assert self.predict_ksize==len(initial_values["predict"]), "Initial values do not match prediction filter length."
            predict_weight = torch.tensor(
                initial_values["predict"], dtype=torch.float, device=self._predict_weight.data.device
            )[:self._predict_weight.shape[0]]
            self._predict_weight.data.copy_(predict_weight)
            #
            assert self.update_ksize==len(initial_values["update"]), "Initial values do not match update filter length."
            update_weight = torch.tensor(initial_values["update"], dtype=torch.float, device=self._update_weight.data.device)[:self._update_weight.shape[0]]
            self._update_weight.data.copy_(update_weight)
            print(f"### Initialize predict & update as {self.predict_weight.data} {self.update_weight.data}")

    def forward(self, even, odd):
        '''Do forward prediction and update steps

        Args:
            even (torch.Tensor): Even-indexed component (batch*ch x 1 x time)
            odd (torch.Tensor): Odd-indexed component (batch*ch x 1 x time)
        
        Return:
            Tuple[torch.Tensor,torch.Tensor]: High- and low-frequency components (batch*ch x 1 x time)
        '''
        # predict
        highfreq = odd - self.predict_op(even)
        # update
        lowfreq = even + self.update_op(highfreq)
        # scaling
        return highfreq, lowfreq

    def inverse(self, highfreq, lowfreq):
        '''Do forward prediction and update steps

        Args:
            highfreq (torch.Tensor): High-frequency component (batch*ch x 1 x time)
            lowfreq (torch.Tensor): Low-frequency component (batch*ch x 1 x time)
        
        Return:
            Tuple[torch.Tensor,torch.Tensor]: Even- and odd-indexed components (batch*ch x 1 x time)
        '''
        # predict
        even = lowfreq - self.update_op(highfreq)
        # update
        odd = highfreq + self.predict_op(even)
        return even, odd

class WeightNormalizedPredictUpdateStage(PredictUpdateStage):
    '''Prediction and update steps for weight-normalized trainable DWT
    '''
    @property
    def predict_weight(self):
        if self.is_first:
            # The sum of coefficients equals 1.
            return self._predict_weight - (self._predict_weight.mean() - 1.0/self.predict_ksize)
        else:
            # The sum of coefficients equals 0.
            return self._predict_weight - self._predict_weight.mean()

    @property
    def update_weight(self):
        if self.is_first:
            # The sum of coefficients equals 1/2.
            return self._update_weight - (self._update_weight.mean() - 0.5/self.update_ksize)
        else:
            # The sum of coefficients equals 0.
            return self._update_weight - self._update_weight.mean()

class MultiStageLWT(nn.Module):
    '''Trainable discrete wavelet transform layer (without weight normalization of prediction and update filters)

    This layer is based on lifting scheme, so it has the perfect reconstruction property but lacks the guarantee of having an anti-aliasing filter.

    References:
        [1] Tomohiko Nakamura, Shihori Kozuka, and Hiroshi Saruwatari, “Time-Domain Audio Source Separation with Neural Networks Based on Multiresolution Analysis,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 1687–1701, Apr. 2021.
    '''
    def __init__(self, params_list):
        '''

        Args:
            params_list (list): Parameter list, whose element is the set of parameters for `PredictionUpdateStage` or `WeightNormalizedPredictionUpdateStage`.
        '''
        super().__init__()
        self.params_list = params_list
        n_stages = len(params_list)
        self.pu_stages = nn.ModuleList()
        for n in range(n_stages):
            self.pu_stages.append(
                self.get_PredictUpdateStage(is_first=n == 0, **params_list[n])
            )
        self.scaling_factor = numpy.sqrt(2.0)

    def get_PredictUpdateStage(self, is_first: bool, **kwargs):
        return PredictUpdateStage(is_first, **kwargs)

    def init_params(self):
        '''Initialize prediction and update filters
        '''
        for stage in self.pu_stages:
            stage.init_params()

    def forward(self, x, no_concat=False):
        '''Forward computation 

        Args:
            x (torch.Tensor): Input feature (batch x ch x time)
            no_concat (bool): If True, return tuple of high- and low-frequency components (default: False)
        
        Return:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: High- and low-frequency components concatenated along the channel axis (batch x ch*2 x time/2) or tuple of them (batch x ch x time/2)
        '''
        B, C, T = x.shape
        assert T % 2 == 0, f"Time length must be even but [{T}]."
        x = x.reshape(B*C, 1, T)
        # split
        even, odd = x[:, :, ::2], x[:, :, 1::2]
        h_list, l_list = [odd], [even]
        # predict, update and scaling
        for stage in self.pu_stages:
            h, l = stage(l_list[-1], h_list[-1])
            h_list.append(h)  # high pass
            l_list.append(l)  # low pass
        y = h.view(B, C, T//2)/self.scaling_factor, l.view(B, C, T//2)*self.scaling_factor
        # output
        if no_concat:
            return y
        else:
            # concat
            return torch.cat(y, dim=1)

    def inverse(self, x, no_concat=False):
        '''Inverse computation

        Args:
            x (torch.Tensor): High- and low-frequency components concatenated along the channel axis (batch x ch*2 x time/2) or tuple of them (batch x ch x time/2)
            no_concat (bool): If True, `x` is assumed to be tuple of high- and low-frequency components (default: False)
        
        Return:
            torch.Tensor: inverse DWT of x (batch x ch x time)
        '''
        if no_concat:
            assert isinstance(x, tuple), 'If no_concat=True, x must be a tuple of high- and low-frequency components.'
            highfreq, lowfreq = x
        else:
            # First and second ones are respectively high- and low-frequency components.
            highfreq, lowfreq = x[:, :x.shape[1]//2, :], x[:, x.shape[1]//2:, :]
        B, C, T = highfreq.shape
        highfreq = highfreq.reshape(B*C, 1, T)
        lowfreq = lowfreq.reshape(B*C, 1, T)
        h_list, l_list = [highfreq*self.scaling_factor], [lowfreq/self.scaling_factor]
        # unscale, update and predict
        for stage in reversed(self.pu_stages):
            h, l = stage.inverse(h_list[-1], l_list[-1])
            h_list.append(h)
            l_list.append(l)
        # merge
        even = l_list[-1]
        odd = h_list[-1]
        # batch*ch x 1 x time x 2 (even, odd)
        out = torch.stack((even, odd), dim=-1)
        out = out.view(B, C, T*2)
        return out

class WeightNormalizedMultiStageLWT(MultiStageLWT):
    '''Weight-normalized trainable discrete wavelet transform layer

    This layer has both the perfect reconstruction property and an anti-aliasing filter owing to the lifting scheme and the proposed weight normalization of the prediction and update filters.

    References:
        [1] Tomohiko Nakamura, Shihori Kozuka, and Hiroshi Saruwatari, “Time-Domain Audio Source Separation with Neural Networks Based on Multiresolution Analysis,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 1687–1701, Apr. 2021.
    '''
    def get_PredictUpdateStage(self, is_first: bool, **kwargs):
        return WeightNormalizedPredictUpdateStage(is_first, **kwargs)
