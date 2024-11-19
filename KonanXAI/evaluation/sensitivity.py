import warnings
from typing import Optional
import torch

from KonanXAI.utils.evaluation import heatmap_postprocessing, postprocessed_ig
from .base import Metric
from torch.nn.modules import Module
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
__all__ = ["Sensitivity"]
class Sensitivity(Metric):
    """
    Computes the complexity of attributions.
    
    Given `attributions`, calculates a fractional contribution distribution `prob_mass`,
    ``prob_mass[i] = hist[i] / sum(hist)``. where ``hist[i] = histogram(attributions[i])``.

    The complexity is defined by the entropy,
    ``evaluation = -sum(hist * ln(hist))``
    
    
    Args:
        model (Model): The model used for evaluation
        explainer (Optional[Explainer]): The explainer used for evaluation.
        n_iter (Optional[int]): The number of iterations for perturbation.
        epsilon (Optional[float]): The magnitude of random uniform noise.
    """
    def __init__(
        self,
        model: Module,
        n_iter: Optional[int] = 8,
        epsilon: Optional[float] = 0.2,
    ):
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def evaluate(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        explainer: object,
        attributions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.
            attributions (Optional[torch.Tensor]): The attributions of the inputs.

        Returns:
            torch.Tensor: The result of the metric evaluation.
        """
        if attributions is None:
            attributions = explainer.calculate(inputs) 
        if isinstance(targets, int):
            targets = (targets,)
        evaluations = []
        for inp, target, attr in zip(inputs, targets, attributions):
            # Add random uniform noise which ranges [-epsilon, epsilon]
            perturbed = torch.stack([inp]*self.n_iter)
            noise = (
                torch.rand_like(perturbed).to(self.device) * self.epsilon * 2 \
                - self.epsilon
            )
            perturbed += noise
            # Get perturbed attribution results
            perturbed_attr = []
            for perturb in perturbed:
                if explainer.type in ["kernelshap", "lime"]:
                    perturb = transforms.ToPILImage()(perturb.detach().cpu())#Image.fromarray(np.uint8((perturb*255).detach().cpu().numpy().transpose(1,2,0)))
                    heatmap = explainer.calculate(inputs = perturb,targets = target)
                else:
                    heatmap = explainer.calculate(inputs = perturb.unsqueeze(0).to(self.device),targets = target)
                heatmap = heatmap_postprocessing(explainer.type, attr.shape, heatmap)
                perturbed_attr.extend(heatmap)
            perturbed_attr = torch.stack(perturbed_attr).to(self.device)
            # Get maximum of the difference between the perturbed attribution and the original attribution
            attr_norm = torch.linalg.norm(attr).to(self.device)
            attr = attr.to(self.device)
            attr_diff = attr - perturbed_attr
            sens = max([torch.linalg.norm(diff)/attr_norm for diff in attr_diff])
            evaluations.append(sens)
        return torch.stack(evaluations).to(self.device)