import torch
import numpy as np
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class RBF(torch.nn.Module):
    def __init__(self, parametrized=False, act_dim=None, hidden_sizes=None, activation=torch.nn.ReLU, adaptive_sig=1, num_particles=None, sigma=None, device=None):
        super(RBF, self).__init__()
        self.parametrized = parametrized
        self.num_particles = num_particles
        self.sigma = sigma
        self.device = device
        self.adaptive_sig = adaptive_sig
        if parametrized:
            self.log_std_layer = mlp([num_particles*num_particles] + list(hidden_sizes) + [act_dim] , activation)
            self.log_std_min = 2
            self.log_std_max = -20
            

    def forward(self, input_1, input_2,  h_min=1e-3):
        _, out_dim1 = input_1.size()[-2:]
        _, out_dim2 = input_2.size()[-2:]
        num_particles = input_2.size()[-2]
        assert out_dim1 == out_dim2

        # print(input_1[0])
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1) # [100,10,10]
        dist_sq = dist_sq.unsqueeze(-1)
        
        # print('Sigma : ', self.sigma)
        if self.sigma is not None:
            sigma = torch.tensor(self.sigma).reshape(-1, 1, 1, 1).to(self.device)
        elif self.parametrized == False:
            if self.adaptive_sig == 1:
                # print('###################################### kernel 11111')
                # Get median.
                median_sq = torch.median(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1)[0]
                median_sq = median_sq.reshape(-1,1,1,1)
                h = median_sq / (2 * np.log(num_particles + 1.))
                sigma = torch.sqrt(h)
            elif self.adaptive_sig == 2:
                # print('######################################  kernel 22222')
                median_sq = torch.quantile(dist_sq.detach().reshape(-1, num_particles*num_particles), 0.25, interpolation='lower', dim=1)
                median_sq = median_sq.reshape(-1,1,1,1)
                h = median_sq / (2 * np.log(num_particles + 1.))
                sigma = torch.sqrt(h)
            elif self.adaptive_sig == 3:
                # print('######################################  kernel 33333')
                median_sq = torch.mean(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1)
                median_sq = median_sq.reshape(-1,1,1,1)
                h = median_sq / (2 * np.log(num_particles + 1.))
                sigma = torch.sqrt(h)
            elif self.adaptive_sig == 4:
                # print('######################################  kernel 44444')
                median_sq = torch.mean(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1) / 2
                median_sq = median_sq.reshape(-1,1,1,1)
                h = median_sq / (2 * np.log(num_particles + 1.))
                sigma = torch.sqrt(h)
            elif self.adaptive_sig == 5: 
                dist_sum = dist_sq.detach().reshape(-1, num_particles*num_particles).sum(-1)
                dist_sum = dist_sum.reshape(-1,1,1,1)
                sigma = dist_sum / (4 * (2 * np.log(num_particles) + 1))
                # sigma = torch.sqrt(sigma)
                # print("sum:{}, sigma:{}".format(dist_sum.min(),sigma.min()))
            else:
                a_mean = input_1.mean(1)
                # print(a_mean.size(), input_1.size())
                sigma = (input_1 - a_mean.unsqueeze(1)).pow(2).sum(-1).sum(-1) / (num_particles-1)
                print("sigma:{}".format(sigma))
                sigma = sigma.reshape(-1,1,1,1)
                sigma = torch.sqrt(sigma)
        else:
            log_std = self.log_std_layer(dist_sq)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            sigma = torch.exp(log_std)
        # self.sigma_debug = torch.mean(sigma).detach().cpu().item()
        # print('***** sigma ', sigma[0])

        gamma = (1.0 / (1e-8 + 2 * sigma**2))
        kappa = (-gamma * dist_sq).exp() 
        # print(diff.size(),gamma.size(),kappa.size()) # [16,16,16,6],[16,1,1,1],[16,16,16,1]
        kappa_grad = -2. * (diff * gamma) * kappa
        
        return kappa.squeeze(-1), diff, dist_sq.squeeze(-1), gamma.squeeze(-1), kappa_grad
