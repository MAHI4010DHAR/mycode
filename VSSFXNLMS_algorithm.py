import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import progressbar

#------------------------------------------------------------------------------
# Class: VSSFXNLMS algorithm
#------------------------------------------------------------------------------
class VSSFXNLMS():

    def __init__(self, Len, mu_init=0.1, alpha=0.1, beta=0.1):
        self.Wc = torch.zeros(1, Len, requires_grad=True, dtype=torch.float)
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
        self.mu = mu_init  # Initial step size
        self.alpha = alpha  # Adaptation parameter for step size
        self.beta = beta  # Adaptation parameter for step size

    def feedforward(self, Xf):
        self.Xd = torch.roll(self.Xd, 1, 1)
        self.Xd[0, 0] = Xf
        yt = self.Wc @ self.Xd.t()
        power = self.Xd @ self.Xd.t()  # FxNLMS different from FxLMS
        return yt, power

    def LossFunction(self, y, d, power):
        e = d - y  # disturbance-control signal
        return e**2 / (2 * power), e

    def adapt_step_size(self, e, Xd):
        self.mu = self.alpha * e**2 / (Xd @ Xd.t() + self.beta)

    def _get_coeff_(self):
        return self.Wc.detach().numpy()

#------------------------------------------------------------------------------
# Function : train_vssfxnlms_algorithm()
#------------------------------------------------------------------------------
def train_vssfxnlms_algorithm(Model, Ref, Disturbance):

    bar = progressbar.ProgressBar(maxval=2*Disturbance.shape[0], \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    optimizer = optim.SGD([Model.Wc], lr=Model.mu)

    bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[0]
    for itera in range(len_data):
        # Feedforward
        xin = Ref[itera]
        dis = Disturbance[itera]
        y, power = Model.feedforward(xin)
        loss, e = Model.LossFunction(y, dis, power)

        # Adapt step size
        Model.adapt_step_size(e, Model.Xd)

        # Update optimizer with new step size
        for param_group in optimizer.param_groups:
            param_group['lr'] = Model.mu

        # Progress shown
        bar.update(2*itera + 1)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.item())

        # Progress shown
        bar.update(2*itera + 2)
    bar.finish()
    return Erro_signal

#------------------------------------------------------------
# Function : Generating the testing broadband noise
#------------------------------------------------------------
def Generating_broadband_noise_waveform_tensor(Wc_F, Seconds, fs):
    filter_len = 1024
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window='hamming', fs=fs)
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y = signal.lfilter(bandpass_filter, 1, xin)
    yout = y[filter_len:]
    # Standardize
    yout = yout / np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)
