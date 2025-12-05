import torch
from torch import nn
from abc import ABCMeta, abstractmethod
from .scheduler_utils import LinearScheduler, ConstantScheduler, scheduler_dict


def real(tensor):
    if not torch.is_complex(tensor):
        return tensor
    else:
        return tensor.real


def imag(tensor):
    if not torch.is_complex(tensor):
        return 0.
    else:
        return tensor.imag


def pairwise_cos_sim(x):
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)  # --> B,(6/3)
    # multiply row i with row j using transpose
    # element wise product
    pairwise_sim = torch.matmul(x_norm, torch.transpose(x_norm, 0, 1))
    return pairwise_sim


class Loss(nn.Module, metaclass=ABCMeta):
    def __init__(self, key):
        super(Loss, self).__init__()
        self.key = key

    @abstractmethod
    def forward(self, model_output):
        ...

class L2Loss(Loss):
    def __init__(self, key):
        super(L2Loss, self).__init__(key)

    def forward(self, model_output):
        pred_key = model_output[self.key]
        gt_key = model_output[self.key + '_gt']

        return    ( (real(pred_key) - real(gt_key)) ** 2
                +   (imag(pred_key) - imag(gt_key)) ** 2 ).mean()

class MultiChoiceL2Loss(Loss):
    def __init__(self, key):
        super(MultiChoiceL2Loss, self).__init__(key)

    def forward(self, model_output):
        pred_key = model_output[self.key] # (K B) 1 H W
        gt_key = model_output[self.key + '_gt'] # B 1 H W
        pred_key = pred_key.reshape(-1, *gt_key.shape)
        loss = ( (real(pred_key) - real(gt_key)) ** 2
                +   (imag(pred_key) - imag(gt_key)) ** 2 ).mean(axis=(-1, -2, -3)) # K B
        min_loss, min_idxs = torch.min(loss, dim=0)
        model_output['activated_paths'] = min_idxs.long()
        return min_loss.mean()

class L2LossPreFlip(Loss):
    def __init__(self, key, mask=False, hf_increase=False, hf_coeff=1e0, mask_rad=0.75, beta=3e-5):
        super(L2LossPreFlip, self).__init__(key)
        self.use_mask = mask
        self.hf_increase = hf_increase
        self.hf_coeff = hf_coeff
        self.mask_rad = mask_rad
        self.beta = beta

    def forward(self, model_output):
        pred_key = model_output[self.key]  # 2 * B, 1, img_sz, img_sz
        pred_prob = model_output['prob']
        
        gt_key = model_output[self.key + '_gt'].permute(1, 0, 2, 3)  # 1, B, img_sz, img_sz
        K = pred_key.shape[0] // gt_key.shape[1] // 2
        if self.use_mask:
            img_sz = pred_key.shape[-1]
            mask = (torch.linspace(-1, 1, img_sz)[None] ** 2 + torch.linspace(-1, 1, img_sz)[:, None] ** 2) <\
                   self.mask_rad
            mask = mask.to(pred_key.device)
            pred_key_masked = mask * pred_key
            gt_key_masked = mask * gt_key
        else:
            pred_key_masked = pred_key
            gt_key_masked = gt_key

        if self.hf_increase:
            img_sz = pred_key.shape[-1]
            abs_freq = torch.linspace(-1, 1, img_sz)[None] ** 2 + torch.linspace(-1, 1, img_sz)[:, None] ** 2
            abs_freq = abs_freq.to(pred_key.device)
            abs_freq = 1. + self.hf_coeff * abs_freq
            pred_key_weighted = abs_freq * pred_key_masked
            gt_key_weighted = abs_freq * gt_key_masked
        else:
            pred_key_weighted = pred_key_masked
            gt_key_weighted = gt_key_masked

        pred_double = pred_key_weighted.reshape(K, 2, -1, pred_key.shape[2], pred_key.shape[3]) # K, 2, B, img_sz, img_sz
        # Unflip
        pred_double[:, 1] = torch.flip(pred_double[:, 1], [-1, -2])
        # Place DC at expected position in fourier
        # if self.key == 'fproj':
        #     pred_double[1] = torch.roll(pred_double[1], shifts=(1, 1), dims=(1, 2))

        distance_double = torch.mean((real(pred_double) - real(gt_key_weighted)) ** 2 +
                                   (imag(pred_double) - imag(gt_key_weighted)) ** 2,
                                     (-1, -2)) # K 2 B 
        distance_double = distance_double.reshape(K*2, -1)
        
        min_argmin = torch.min(distance_double, 0)
        min_distances = min_argmin[0] # B

        # Keep track of activated paths
        activated_paths = min_argmin[1] # B
        
        model_output["activated_paths"] = activated_paths.long()
        model_output["mean_diff_paths"] = torch.abs(distance_double[0] - distance_double[1]).mean()
        model_output['num_modes'] = K
        return min_distances.mean()

class PriorLoss(Loss):
    def __init__(self, beta=3e-5):
        super(PriorLoss, self).__init__('prob')
        self.beta = beta
    
    def forward(self, model_output):
        pred_prob = model_output['prob']
        if 'activated_paths' in model_output:
            pred_prob = pred_prob[model_output['activated_paths']]
        return (self.beta.update() * pred_prob).mean()
        
class L1Loss(Loss):
    def __init__(self, key):
        super(L1Loss, self).__init__(key)

    def forward(self, model_output):
        pred_key = model_output[self.key]
        gt_key = model_output[self.key + '_gt']

        return (   torch.abs(real(pred_key) - real(gt_key))
                 + torch.abs(imag(pred_key) - imag(gt_key)) ).mean()


class ContrastiveLoss(Loss):
    def __init__(self, key):
        super(ContrastiveLoss, self).__init__(key)

    def forward(self, model_output):
        code = model_output[self.key]

        return ((pairwise_cos_sim(code) - torch.eye(code.shape[0]).to(code.device)) ** 2).mean()


def loss_factory(config):
    """
    Creates a dictionary of losses from a parser.

    Parameters
    ----------
    config: namespace

    Returns
    -------
    loss_dict: Dictionary
    """
    loss_dict = {}

    # Data Loss
    if config.data_loss_domain == 'primal':
        if config.data_loss_norm == 'L2':
            # loss_dict['data_term'] = L2Loss('proj')
            loss_dict['data_term'] = MultiChoiceL2Loss('proj')
        elif config.data_loss_norm == 'L1':
            loss_dict['data_term'] = L1Loss('proj')
        elif config.data_loss_norm == 'symloss':
            loss_dict['data_term'] = L2LossPreFlip(
                'proj',
                mask=config.use_masked_loss,
                hf_increase=False,
                mask_rad=config.mask_rad_loss
            )
        else:
            print("Unrecognized data loss norm.")
    elif config.data_loss_domain == 'fourier':
        if config.data_loss_norm == 'L2':
            # loss_dict['data_term'] = L2Loss('fproj')
            loss_dict['data_term'] = MultiChoiceL2Loss('fproj')
        elif config.data_loss_norm == 'L1':
            loss_dict['data_term'] = L1Loss('fproj')
        elif config.data_loss_norm == 'symloss':
            loss_dict['data_term'] = L2LossPreFlip(
                'fproj',
                mask=config.use_masked_loss,
                hf_increase=config.hf_increase_loss
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    beta_scheduler = scheduler_dict[config.beta_scheduler](v0=0, v1=config.beta, num_steps=config.beta_scheduler_steps) 
    loss_dict['prior'] = PriorLoss(beta=beta_scheduler)

    # Contrastive Loss
    if not config.so3_parameterization == 'gt' and config.use_contrastive_loss:
        loss_dict['contrastive_term'] = ContrastiveLoss('latent_code_prerot')

    return loss_dict
