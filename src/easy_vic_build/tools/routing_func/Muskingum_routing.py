# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import torch
import torch.nn as nn


class DifferentiableMuskingumGrid(nn.Module):
    def __init__(self, grid_shape, dt=1.0):
        """
        Differentiable Muskingum routing model for grid-based hydrological simulation.
        
        Args:
            grid_shape (tuple): Dimensions of the grid (height, width)
            dt (float): Time step duration (hours)
        """
        super().__init__()
        
        self.grid_shp = grid_shape
        self.dt = dt
        
        # Trainable parameters for each grid cell
        self.K = nn.Parameter(torch.full(grid_shape, 10.0))  # Storage constant (hours), initialized to 10
        self.x = nn.Parameter(torch.full(grid_shape, 0.2))  # Weighting factor (0-0.5), initialized to 0.2
        
        # Flow direction matrix (fixed, can be derived from DEM)
        self.register_buffer("flow_dir", self._init_flow_dir())
        
        # Precompute direction offsets for vectorized routing
        # di: row offsets for all 9 possible directions (including center)
        di = torch.tensor([[-1,-1,-1],  # NW, N, NE
                          [ 0, 0, 0],   # W, CENTER, E
                          [ 1, 1, 1]])  # SW, S, SE
        
        # dj: column offsets for all 9 possible directions
        dj = torch.tensor([[-1, 0, 1],  # NW, N, NE
                          [-1, 0, 1],   # W, CENTER, E
                          [-1, 0, 1]])  # SW, S, SE
        
        # Convert flow directions to coordinate offsets
        self.register_buffer('di_grid', di[self.flow_dir])
        self.register_buffer('dj_grid', dj[self.flow_dir])
        
    
    def _init_flow_dir(self):
        """
        Initialize static flow direction matrix using D8 encoding.
        Note: Simplified to rightward flow - replace with actual DEM-derived directions.
        
        D8 Encoding:
            32  64  128
            16   0    1
             8   4    2
        """
        flow_dir = torch.zeros(self.grid_shape, dtype=torch.long)
        flow_dir[:, :-1] = 6  # 6 represents rightward flow in this example
        return flow_dir
    
    def forward(self, Qin_grid, n_steps):
        """
        Perform grid-based Muskingum routing.
        
        Args:
            Qin_grid (torch.Tensor): Initial inflow grid (B, C, H, W)
                B: batch size, C: channels (typically 1)
            n_steps (int): Number of time steps to simulate
            
        Returns:
            torch.Tensor: Outflow sequence (B, n_steps, H, W)
        """
        B, C, H, W = Qin_grid.shape
        Qout = torch.zeros(B, n_steps, H, W, device=Qin_grid.device)
        Qout[:, 0] = Qin_grid.squeeze(1)  # Initialize first time step
        
        # Precompute Muskingum coefficients (differentiable operations)
        denom = self.K * (1 - self.x) + 0.5 * self.dt
        C0 = (-self.K * self.x + 0.5 * self.dt) / denom  # (H, W)
        C1 = (self.K * self.x + 0.5 * self.dt) / denom   # (H, W)
        C2 = (self.K * (1 - self.x) - 0.5 * self.dt) / denom  # (H, W)
        
        # Create grid indices for vectorized operations
        i = torch.arange(H, device=Qin_grid.device)[:, None]  # Row indices (H,1)
        j = torch.arange(W, device=Qin_grid.device)[None, :]  # Column indices (1,W)
        
        # Time-stepping loop
        for t in range(1, n_steps):
            Q_prev = Qout[:, t-1]
            
            # Calculate downstream indices with boundary checking
            i_dn = torch.clamp(i + self.di_grid, 0, H-1)
            j_dn = torch.clamp(j + self.dj_grid, 0, W-1)
            
            # Vectorized Muskingum routing equation
            Qout[:, t] = (C0 * Q_prev + 
                         C1 * Q_prev + 
                         C2 * Qout[:, t-1, i_dn, j_dn])
        
        return Qout