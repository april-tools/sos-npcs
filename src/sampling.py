from typing import Optional, Union

import torch

from models import PC


@torch.no_grad()
def inverse_transform_sample(
    model: PC,
    *,
    vdomain: int,
    num_samples: int = 1,
    device: Optional[Union[int, str, torch.device]] = None
) -> torch.Tensor:
    model.eval()
    model = model.to(device)
    num_variables = model.num_variables
    num_channels = model.num_channels
    if num_channels != 1:
        raise NotImplementedError()
    samples = torch.zeros(
        (num_samples, num_channels, num_variables), dtype=torch.int64, device=device
    )
    working_samples = torch.zeros(
        (num_samples, vdomain, num_channels, num_variables),
        dtype=torch.int64,
        device=device,
    )
    prev_log_scores = model.log_partition()  # (num_samples = 1 at the first step, 1)

    for i in range(num_variables):
        # compute \log \sum_{a_{i+1}, ..., a_d} c(a_1, ..., a_d)
        working_samples[:, :, 0, i] = torch.arange(vdomain, device=device).unsqueeze(
            dim=0
        )
        if i < num_variables - 1:
            log_scores = model.log_integrated_score(
                working_samples.view(-1, num_channels, num_variables),
                variables=tuple(j for j in range(num_variables) if j > i),
            )  # (num_samples * vdomain, 1)
        else:
            log_scores = model.log_score(
                working_samples.view(-1, num_channels, num_variables),
            )
        log_scores = log_scores.squeeze(dim=-1).view(
            num_samples, vdomain
        )  # (num_samples, vdomain)

        # sample X_i ~ p(X_i | X_1 = a_1, ..., X_{i-1} = a_{i-1})
        conditional_probs = torch.nan_to_num(torch.exp(log_scores - prev_log_scores))
        u = torch.rand(conditional_probs.shape[0], 1, device=device)
        sample_i = vdomain - (u <= torch.cumsum(conditional_probs, dim=1)).long().sum(
            dim=1
        )  # (num_samples,)
        samples[:, 0, i] = sample_i
        sample_idx_range = torch.arange(num_samples, device=device)
        prev_log_scores = log_scores[sample_idx_range, sample_i].unsqueeze(dim=1)
        working_samples[sample_idx_range, :, 0, i] = sample_i.unsqueeze(dim=1)

    return samples
