
# coding: utf-8
import os

import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np

from .layers import MVCompressor, ResidualCompressor, Mask
from .flow import Network

device = torch.device("cuda:0")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.FlowNet = Network()
        for p in self.FlowNet.parameters():
            p.requires_grad = True
            
        self.mv_compressor = MVCompressor()
        self.residual_compressor = ResidualCompressor()
        self.masknet = Mask()
    
    def forward(self, x_before, x_current, x_after, train):
        
        
        N, C, H, W = x_current.size()
        num_pixels = N * H * W
        
        flow_ba = self.estimate_flow(x_after, x_before) / 2.
        flow_ab = self.estimate_flow(x_before, x_after) / 2.
        
        flow_bc = self.estimate_flow(x_current, x_before)
        flow_ac = self.estimate_flow(x_current, x_after)
        
        diff_flow = torch.cat([(flow_bc - flow_ba), (flow_ac - flow_ab)], dim=1)
        flow_result = self.mv_compressor(diff_flow)
        
        flow_bc_hat, flow_ac_hat = torch.chunk(flow_result["x_hat"] + torch.cat([flow_ba, flow_ab], dim=1), 2, dim=1)
        
        fw, bw = self.backwarp(x_before, flow_bc_hat), self.backwarp(x_after, flow_ac_hat)
        
        mask = self.masknet(torch.cat([fw, bw], dim=1)).repeat([1, 3, 1, 1])
        
        x_current_hat = mask*fw + (1.0 - mask)*bw
        x_current_hat_avg = (fw + bw)/2.
        
        residual = x_current - x_current_hat
        residual_result = self.residual_compressor(residual)
        residual_hat = residual_result["x_hat"]
        
        residual_avg = x_current - x_current_hat_avg
        residual_result_avg = self.residual_compressor(residual_avg)
        residual_hat_avg = residual_result_avg["x_hat"]
        
        x_current_final = residual_hat + x_current_hat
        x_current_final_avg = residual_hat_avg + x_current_hat_avg
        
        size_flow = sum(
            (torch.log(likelihoods).sum() / (-math.log(2)))
            for likelihoods in flow_result["likelihoods"].values()
        )
        
        size_residual = sum(
            (torch.log(likelihoods).sum() / (-math.log(2)))
            for likelihoods in residual_result["likelihoods"].values()
        )
        
        size_residual_avg = sum(
            (torch.log(likelihoods).sum() / (-math.log(2)))
            for likelihoods in residual_result_avg["likelihoods"].values()
        )
        
                
        

        return x_current_final, x_current_final_avg, size_flow.item() + size_residual.item(), size_flow.item() + size_residual_avg.item(), flow_bc, flow_ac, flow_bc_hat, flow_ac_hat, mask, fw, bw, x_current_hat, x_current_hat_avg
        
    
    def estimate_flow(self, tenFirst, tenSecond):
        
        h = tenFirst.size()[2]
        w = tenFirst.size()[3]
        h_p = int(math.floor(math.ceil(h / 32.0) * 32.0))
        w_p = int(math.floor(math.ceil(w / 32.0) * 32.0))
        

        tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenFirst, size=(h_p, w_p), mode='bilinear', align_corners=False)
        tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenSecond, size=(h_p, w_p), mode='bilinear', align_corners=False)

        tenFlow = torch.nn.functional.interpolate(input=self.FlowNet(tenPreprocessedFirst, tenPreprocessedSecond), size=(h, w), mode='bilinear', align_corners=False)
        tenFlow[:, 0, :, :] *= float(w) / float(w_p)
        tenFlow[:, 1, :, :] *= float(h) / float(h_p)

        return tenFlow
    
    
    def backwarp(self, tenInput, tenFlow):
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid = torch.cat([ tenHor, tenVer ], 1).to(device).float()
        # end

        tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

        return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
