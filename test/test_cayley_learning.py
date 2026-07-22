import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.quantized_linear import NVFP4Linear


if __name__ == "__main__":

    batch_size = 10
    seq_len = 20
    hidden_dim_1 = 30
    hidden_dim_2 = 40
    x = torch.randn(batch_size, seq_len, hidden_dim_1)

    linear_ori = nn.Linear(hidden_dim_1, hidden_dim_2, bias=True)
    with torch.no_grad():
        y_ori = linear_ori(x)

    quantization_error_info = {}
    loss_fn = F.mse_loss

    criteria = ['activation', 'parameter', 'output', 'combined']
    # criteria = ['output']
    results = {}

    for criterion in criteria:
        
        linear_cayley = NVFP4Linear(
            hidden_dim_1, hidden_dim_2, bias=True, 
            rotation="cayley", permutation="random", 
            block_size=16, quantize=True, 
            layer_prefix="base"
        )
        linear_cayley.load_state_dict(linear_ori.state_dict(), strict=False)
        optimizer = torch.optim.Adam([linear_cayley.rotation.K], lr=1e-3)
        
        loss_act_history = []
        loss_parm_history = []
        loss_act_in_record_history = []
        loss_parm_in_record_history = []
        loss_out_history = []
        grad_norm_history = []
        
        for i in range(200):

            optimizer.zero_grad()
            quantization_error_info.clear()

            y_cayley = linear_cayley(x, quantization_error_info=quantization_error_info)
            
            loss_act, loss_parm = linear_cayley.get_differentiable_quantization_error(loss_fn)
            loss_out = loss_fn(y_cayley, y_ori)
            # loss = loss_out
            
            if criterion == 'activation':
                loss = loss_act
            elif criterion == 'parameter':
                loss = loss_parm
            elif criterion == 'output':
                loss = loss_out
            elif criterion == 'combined':
                loss = loss_act + loss_parm + loss_out

            loss.backward()
            
            grad_norm = linear_cayley.rotation.K.grad.norm().item() if linear_cayley.rotation.K.grad is not None else 0
            grad_norm_history.append(grad_norm)
            
            optimizer.step()
            
            loss_act_in_record_history.append(quantization_error_info['base.input.nvfp4_act_error_mse'])
            loss_parm_in_record_history.append(quantization_error_info['base.weight.nvfp4_error_mse'])
            loss_act_history.append(loss_act.item())
            loss_parm_history.append(loss_parm.item())
            loss_out_history.append(loss_out.item())
            
            # if i % 20 == 0:
            # print(f"Iter {i}: loss_act={loss_act.item():.3e}, loss_parm={loss_parm.item():.3e}, loss_out={loss_out.item():.3e}, grad_norm={grad_norm:.3e}")
            # print(f"Iter {i}: loss_act={loss_act:.3e}, loss_parm={loss_parm:.3e}, loss_out={loss_out.item():.3e}, grad_norm={grad_norm:.3e}")
        
        results[criterion] = {
            'loss_act': loss_act_history,
            'loss_parm': loss_parm_history,
            'loss_out': loss_out_history,
            'grad_norm': grad_norm_history,
            'loss_act_in_record': loss_act_in_record_history,
            'loss_parm_in_record': loss_parm_in_record_history
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    colors = ['blue', 'orange', 'green', 'red']
    markers = ['-', '--', '-.', ':']

    for ax, (criterion, data), color, marker in zip(axes.flatten(), results.items(), colors, markers):
        ax.plot(data['loss_act'], label='Activation Error', color='blue', linestyle=marker)
        ax.plot(data['loss_parm'], label='Parameter Error', color='orange', linestyle=marker)
        ax.plot(data['loss_out'], label='Output Error', color='green', linestyle=marker)
        # ax.plot(data['loss_act_in_record'], label='Activation Error in Record', color='red', linestyle=marker)
        # ax.plot(data['loss_parm_in_record'], label='Parameter Error in Record', color='purple', linestyle=marker)
        ax.set_title(f'Training with {criterion} loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

    # plt.savefig('nvfp4_quantization_loss_comparison.png', dpi=150)
    # print("\nLoss comparison plot saved to nvfp4_quantization_loss_comparison.png")

    # fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

    # for ax, (criterion, data), color, marker in zip(axes2.flatten(), results.items(), colors, markers):
    #     ax.plot(data['grad_norm'], label='Grad Norm', color=color, linestyle=marker)
    #     ax.set_title(f'Gradient Norm - {criterion}')
    #     ax.set_xlabel('Iteration')
    #     ax.set_ylabel('Gradient Norm')
    #     ax.legend()
    #     ax.grid(True)
    #     ax.set_yscale('log')

    # plt.tight_layout()
    # plt.show()
    # plt.savefig('nvfp4_quantization_grad_norm.png', dpi=150)
    # print("Gradient norm plot saved to nvfp4_quantization_grad_norm.png")

    # fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # for criterion, data, color, marker in zip(criteria, results.values(), colors, markers):
    #     ax1.plot(data['loss_act'], label=criterion, color=color, linestyle=marker)
    # ax1.set_title('Activation Quantization Error Comparison')
    # ax1.set_xlabel('Iteration')
    # ax1.set_ylabel('Loss')
    # ax1.legend()
    # ax1.grid(True)
    # ax1.set_yscale('log')

    # for criterion, data, color, marker in zip(criteria, results.values(), colors, markers):
    #     ax2.plot(data['loss_parm'], label=criterion, color=color, linestyle=marker)
    # ax2.set_title('Parameter Quantization Error Comparison')
    # ax2.set_xlabel('Iteration')
    # ax2.set_ylabel('Loss')
    # ax2.legend()
    # ax2.grid(True)
    # ax2.set_yscale('log')

    # for criterion, data, color, marker in zip(criteria, results.values(), colors, markers):
    #     ax3.plot(data['loss_out'], label=criterion, color=color, linestyle=marker)
    # ax3.set_title('Output Error Comparison')
    # ax3.set_xlabel('Iteration')
    # ax3.set_ylabel('Loss')
    # ax3.legend()
    # ax3.grid(True)
    # ax3.set_yscale('log')

    # plt.tight_layout()
    # plt.savefig('nvfp4_quantization_loss_separate.png', dpi=150)
    # print("Separate loss comparison plot saved to nvfp4_quantization_loss_separate.png")

    # plt.show()