import torch


def weighted_mse_loss(input_tensor, target_tensor, weight=1):
  observation_dim = input_tensor.size()[-1]
  streched_tensor = ((input_tensor - target_tensor) ** 2).view(
      -1, observation_dim)
  entry_num = float(streched_tensor.size()[0])
  non_zero_entry_num = torch.sum(streched_tensor[:, 0] != 0).float()
  weighted_tensor = torch.mm(
      ((input_tensor - target_tensor)**2).view(-1, observation_dim),
      (torch.diag(weight.float().view(-1))))
  return torch.mean(
      weighted_tensor) * weight.nelement() * entry_num / non_zero_entry_num


def sigma2_prior_loss(num_non_zero, sigma_alpha, sigma_beta, sigma2):
  return ((2 * sigma_alpha + num_non_zero + 2) /
          (2 * num_non_zero) * torch.log(sigma2)).sum() + (
              sigma_beta / (sigma2 * num_non_zero)).sum()


def regularization_loss(params, weight):
  l2_reg = 0
  for param in params:
    l2_reg += torch.norm(param)
  return weight * l2_reg
