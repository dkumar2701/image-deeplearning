import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class diffusion_process(nn.Module):
    
    def __init__(self, time_steps, beta_1, beta_T):
        super().__init__()
        """
        beta_1 and beta_T are the variances for q(x_1|x_0) and q(x_T|x_{T-1}) respectively
        beta_T should be larger, and we'll linearly interpolate between these to set the variances of the diffusion process.
        """
        # linear schedule
        # self.variance_schedule = torch.linspace(beta_1, beta_T, time_steps)   #could be learned (Kingma et al), but we'll start with fixed.
        
        # cosine schedule
        s = 0.008
        t = torch.linspace(0, time_steps, steps=time_steps+1)
        f = torch.cos(((t / time_steps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_hat = f / f[0]
        betas = 1 - (alphas_hat[1:] / alphas_hat[:-1])
        self.variance_schedule = torch.clip(betas, 0, beta_T)

        self.T = time_steps
        
        
    def forward_var(self, t):
        """
        The variance schedule records the variances of q(x_{t}| x_{t-1}) - this returns the variance of q(x_t| x_0)
        where both x_0 and t are batches rather than an isolated datapoint.
        """
        ones = torch.ones(self.variance_schedule.size()) 
        alphas = ones - self.variance_schedule
        alpha_bars = torch.cumprod(alphas, dim=0)

        return 1- alpha_bars[t]
        
        
        
    def forward(self, x, t):
        """
        starting with an input batch of images x and a timestep t between 0 and 1, we return a sample x_{t} from q(x_t | x) and the corresponding log probability.
        We'll always work in closed form rather than performing a sequence of samples.
        """

        var = self.forward_var(t).view(-1,1,1,1)
        sqrt_var = torch.sqrt(var)
        alpha_bar = torch.sqrt(1-var)
        
        epsilon = torch.randn_like(x)
        x_t = alpha_bar*x + sqrt_var*epsilon

        return x_t, epsilon



class denoise_net(nn.Module):
  """
  This class basically organizes some utilities we'll use in the backward process. It doesn't include anything of import about the UNET.
  """
  def __init__(self, dif_proc, net):
      super().__init__()
      self.dif_proc = dif_proc
      self.net = net
        
  def forward(self, z_t, t):
      #should take in a pair (z_t, t) and spit out epsilon(z_t, t)
      #epsilon should be a U-net neural network, as in Ho et al.
      return self.net(z_t, t)
        
  def denoise(self, z_t, t):
      #should take in a pair (z_t, t) and spit out the neural networks reconstruction of x from z_t
      var_t = self.dif_proc.forward_var(t) #Equation 3
      alpha_t = (1-var_t)**.5
                
      return (z_t - var_t*self.net(z_t,t))/alpha_t #Equation 10






class diff_model:
    
    def __init__(self, dif_proc, denoiser, device = "cuda"):
        super().__init__()
        
        self.device = device
        self.dif_proc = dif_proc
        self.denoiser = denoiser 
        self.T = dif_proc.T
        
    
    def generate(self, batch_size=64):
        """
        This function generates a batch of images following algorithm 2 from Ho et al.
        """
        with torch.no_grad():
            im = torch.randn((batch_size, 3, 32, 32)).to(self.device)
    
            for i in reversed(range(self.T)):
                if i > 1:
                    z = torch.randn_like(im).to(self.device)
                else:
                    z = torch.zeros(im.size()).to(self.device)
                t = (torch.ones(batch_size)*i).long()
                alpha = (1 - self.dif_proc.variance_schedule[t]).view(-1,1,1,1).to(self.device)
                alpha_bar = (1 - self.dif_proc.forward_var(t)).view(-1,1,1,1).to(self.device)
                sigma = torch.sqrt(1-alpha) #this is a choice - see section 3.2 of Ho et al

                im = 1/torch.sqrt(alpha)*(im - ((1-alpha)/torch.sqrt(1-alpha_bar))*self.denoiser.net(im, t)) + sigma*z

            im = (torch.clamp(im, -1, 1)+1)/2
            im = (im*255).type(torch.uint8)
    
            return im




    
    def conditional_generate(self, label, batch_size=64):
        """
        This function generates a batch of images following algorithm 2 from Ho et al.
        """
        with torch.no_grad():
            im = torch.randn((batch_size, 3, 32, 32)).to(self.device)
    
            for i in reversed(range(self.T)):
                if i > 1:
                    z = torch.randn_like(im).to(self.device)
                else:
                    z = torch.zeros(im.size()).to(self.device)
                y = label
                t = (torch.ones(batch_size)*i).long()
                alpha = (1 - self.dif_proc.variance_schedule[t]).view(-1,1,1,1).to(self.device)
                alpha_bar = (1 - self.dif_proc.forward_var(t)).view(-1,1,1,1).to(self.device)
                sigma = torch.sqrt(1-alpha) #this is a choice - see section 3.2 of Ho et al

                pred_noise = self.denoiser.net(im, t, y)

                im = 1/torch.sqrt(alpha)*(im - ((1-alpha)/torch.sqrt(1-alpha_bar))*pred_noise) + sigma*z

            im = (torch.clamp(im, -1, 1)+1)/2
            im = (im*255).type(torch.uint8)
    
            return im
         
        
        
        
    def gaussian_kl(self, mean_1, logvar_1, mean_2, logvar_2, reduction = None):
        """
        Given two Gaussian distributions p_1 and p_2, returns D_{KL}(p_1 || p_2)
        """
       
        D = 0.5 * (-1.0 + logvar_2 - logvar_1 + torch.exp(logvar_1 - logvar_2)
            + ((mean_1-mean_2)**2) * torch.exp(-logvar_2))
        
        if reduction:
            return torch.sum(D)
        else:
            return D
        
        
        
        
        
    def posterior_parameters(self, x_0, x_t, t):
        """
        returns the mean and log-variance of q(x_{t-1} | x_t, x_0) according to equation (7) in Ho et al
        """
        beta_t = self.dif_proc.variance_schedule[t]
        alpha_t = 1-beta_t
        
        alpha_bar_t = 1-self.dif_proc.forward_var(t)
        alpha_bar_s = 1-self.dif_proc.forward_var(t-1)
        
        coef_1 = (alpha_bar_s**.5)*beta_t/(1-alpha_bar_t)
        
        coef_2 = (alpha_t**.5)*(1-alpha_bar_s)/(1-alpha_bar_t)
        
        mean = coef_1*x_0 + coef_2*x_t
        var = (1-alpha_bar_s)/(1-alpha_bar_t)*beta_t
        
        return mean, torch.log(var)
        
        
              
        
    def model_distribution_parameters(self, x_t, t):
        """
        returns the mean and log-variance of p(x_{t-1}|x_{t}) according to equation (11) for the mean, and the choice of sigma^2_{t} = beta_{t} as in
        the first paragraph of 3.2
        """
        
        beta_t = self.dif_proc.variance_schedule[t]
        alpha_bar_t = 1- self.dif_proc.forward_var(t)
        
        epsilon_theta = self.denoiser(x_t, t)
        
        mean = (1/(1-beta_t)**.5)*(x_t - beta_t/((1-alpha_bar_t)**.5)*epsilon_theta)
        
        return mean, torch.log(beta_t)
        
              
        
    def L_t(self, x_0, x_t, t):
        """
        Given a clean image x_0 and a noisy version x_t, return D_{KL}(q(x_{t-1}| x_t, x_0) || p(x_{t-1}|x_t)), or ||eps - eps_theta||^2
        """

        mu_q, logvar_q = self.posterior_parameters(x_0, x_t, t)       
        mu_p, logvar_p = self.model_distribution_parameters(x_t, t)
        
        return self.gaussian_kl(mu_q, logvar_q, mu_p, logvar_p, reduction = True)
    
    