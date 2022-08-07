# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 13:57:26 2022

@author: Bart de Koning
"""

# TODO: (possibly ever)
# - Format docstrings for automatic documentation generation (with Sphinx)
# - Create support for periodic knot vectors

import torch

# Standard hyper-parameter values
degree_standard           = 3
n_control_points_standard = 10
device_standard           = torch.device("cuda:0") if torch.cuda.is_available() \
                             else torch.device("cpu")
param_min_standard        = 0
param_max_standard        = 1
float_type                = torch.float32 # !!!: Not used


def description_creator(prop_dict, object_name = ""):
    """Creates a description of an object from a dictionary of properties."""
    
    s = f"{object_name}(\n\t"
    
    for prop, val in prop_dict.items():
        new = f"{prop} = {val},\n"
        new = new.replace("\n","\n\t")
        
        s += new
        
    s = s[:-3] + "\n)"
    
    return s


class Knot_vector():
    
    def __init__(self,knots,
                 multiplicities = None,
                 is_open        = False,
                 is_equispaced  = False,
                 degree         = degree_standard):
        """Knot vector class."""
        
        self.device        = knots.device
        self.is_open       = is_open
        self.is_equispaced = is_equispaced
        self.degree        = degree
        
        # Check whether vector consists of nondecreasing values
        assert (knots.sort().values == knots).all()
        
        # Individual knots without multiplicities
        self.knots = knots
        
        if multiplicities is None:
            self.multiplicities = torch.ones_like(knots, dtype = torch.uint32)
        else:
            self.multiplicities = multiplicities
            
        if is_equispaced:
            self.knot_span_len = knots[1] - knots[0]
            
        self.update_knot_vector_full()
            
            
    def update_knot_vector_full(self):
        """Knot_distances structure:
            --> dim 0
            |   dim 1 (size degree)
            V
            u1-u0,        u2-u1,          ..., u(n_control_points+1) - u(n_control_points)
            u2-u0,        u3-u2,          ..., u(n_control_points+2) - u(n_control_points)
            :
            u(degree)-u0, u(degree+1)-u1, ..., u(n_control_points+degree) - u(n_control_points)
            """
        
        # Create full knot vector with multiplicities included
        parts = []
        
        for knot, multiplicity in zip(self.knots,self.multiplicities):
            parts.append(torch.full((multiplicity,), knot, device = self.device))
            
        self.vector_full = torch.cat(parts)
        
        # Indices of entries in the full knot vector 
        # in the self.knots vector without repetition
        self.knot_vector_indices = torch.tensor(
                                     sum([m*[i] for i,m in enumerate(self.multiplicities)],[]),
                                     device = self.device)
        
        
        # Distances between knots used in B-spline basis function computation
        self.knot_distances = torch.tile(-self.vector_full[:-self.degree],
                                         (self.degree,1)).T
        
        
        self.n_basis_functions = len(self.vector_full) - (self.degree+1)
        
        for i in range(self.degree):
            self.knot_distances[:,i] += self.vector_full[i+1:i+2+self.n_basis_functions]
        
    
    def __getitem__(self,x):
        """Make this class directly access the knot vector by subscription."""
        return self.knots[x]

        
    def __str__(self):
        return description_creator(dict(knots          = self.knots,
                                        multiplicities = self.multiplicities,
                                        open           = self.is_open,
                                        equispaced     = self.is_equispaced),
                                   object_name = "knot_vector")
    
    
    
    def get_knot_span_indices(self,u):
        
        if self.is_equispaced:
            knot_span_indices = ((u-self.knots[0])/self.knot_span_len).ceil().long()
                
        else:
            knot_span_indices = torch.searchsorted(self.knots,u)
            
        cumulative = torch.cat([torch.tensor([self.multiplicities[0]], device = self.device), 
                                torch.cumsum(self.multiplicities,0)])
        
        return cumulative[knot_span_indices]


    @staticmethod
    def make_open(degree           = degree_standard,
                  n_control_points = n_control_points_standard,
                  device           = device_standard,
                  param_min        = param_min_standard,
                  param_max        = param_max_standard,
                  mode             = "equispaced"):
        """Create an open and equispaced knot_vector:
            [param_min, ...., param_min, <linear from param_min to param_max>, param_max,...,param_max].
            param_min and param_max are repeated degree+1 times."""
            
        if mode == "equispaced":
            knots = torch.linspace(param_min,param_max, n_control_points - degree + 1,
                                   device = device)
            
        elif mode == "random":
            
            # Create tensor of random nondecreasing values between param_min and param_max
            values = torch.rand(n_control_points - degree + 1,
                                device = device)
            
            csum   = torch.cumsum(values,0)
            knots  = (csum-csum.min())*(param_max-param_min)/(csum.max()-csum.min()) + param_min
            
        else:
            raise ValueError(f"\"{mode}\" is not a valid open knot vector creation mode.")
        
        multiplicities = torch.ones_like(knots, dtype = torch.int32,
                                         device = device)
        
        multiplicities[[0,-1]] = degree+1
        
        return Knot_vector(knots,
                           multiplicities = multiplicities, 
                           is_open        = True, 
                           is_equispaced  = (mode == "equispaced"),
                           degree         = degree)
    
    

class Basis_functions():
    
    def __init__(self,
                 knot_vector):
        """B-spline basis functions class."""
        
        self.degree      = knot_vector.degree
        self.device      = knot_vector.device
        self.knot_vector = knot_vector
            
        self.n_basis_functions = knot_vector.n_basis_functions
            
            
    def __str__(self):
        return description_creator(dict(degree      = self.degree,
                                        knot_vector = self.knot_vector),
                                   object_name = "basis_functions")
    
    
    
    def _uncompress_basis_function_values(self,values,knot_span_indices):
        """Calling this class for certain inputs u only yields values for the basis functions
        whose support include u. This method extends that to an output with zeros for all
        other basis functions (mainly useful for plotting the basis functions)."""
        
        values_uncompressed = torch.zeros((len(values),self.n_basis_functions),
                                           device = self.device)
        
        u_indics = torch.arange(len(values), device = self.device)

        for i in range(self.degree+1):
            values_uncompressed[u_indics,knot_span_indices+i-self.degree-1] = values[:,i]
            
        return values_uncompressed
    
    
    def __call__(self,u,
                    derivative_order         : int  = 0,
                    return_knot_span_indices : bool = False,
                    knot_span_indices               = None,
                    uncompress               : bool = True):
        """Evaluate the basis functions."""
        
        if knot_span_indices is None:
            knot_span_indices = self.knot_vector.get_knot_span_indices(u)
            
        len_u = len(u)
        
        # Calculate the relevant distances between the inputs and the knots
        # TODO: Goes wrong if knot vector is not open?
        u_knot_diffs = torch.tile(u,
                                  (2*self.degree,1)).T
        
        for i in range(-self.degree,self.degree):
            u_knot_diffs[:,i+self.degree] -= self.knot_vector.vector_full[knot_span_indices+i]
        
        # Create tensor with all relevant knot span indices
        knot_span_indices_shifted = torch.tile(knot_span_indices,
                                               (self.degree+1,1)).T
        
        knot_span_indices_shifted += torch.arange(-self.degree,1, 
                                                  device = self.device)[None,:]
        
        
        # Recursively compute the basis function values for increasing
        # degrees
        # TODO: Make it so that (optionally) all lower order derivative values are also returned?
        values_prev_degree = torch.ones((len_u,1), device = self.device)
        
        for current_degree in range(1,self.degree+1):
        
            values_current_degree = torch.zeros((len_u,current_degree+1),
                                                device = self.device)
            
            if self.degree - current_degree < derivative_order:
                part1 = (values_prev_degree *
                    current_degree/self.knot_vector.knot_distances[knot_span_indices_shifted[:,self.degree-current_degree:self.degree],current_degree-1]
                )
                part2 = (values_prev_degree *
                    current_degree/self.knot_vector.knot_distances[knot_span_indices_shifted[:,self.degree-current_degree:self.degree],current_degree-1]
                )
                
            else:
                part1 = (values_prev_degree *
                    u_knot_diffs[:,self.degree-current_degree:self.degree]/self.knot_vector.knot_distances[knot_span_indices_shifted[:,self.degree-current_degree:self.degree],current_degree-1]
                )
                part2 = (values_prev_degree *
                    u_knot_diffs[:,self.degree:self.degree+current_degree]/self.knot_vector.knot_distances[knot_span_indices_shifted[:,self.degree-current_degree:self.degree],current_degree-1]
                )
            
            
            values_current_degree[:,1:]   = part1
            values_current_degree[:,:-1] -= part2
    
            values_prev_degree = values_current_degree.clone()
            
            
        if uncompress:
            out = self._uncompress_basis_function_values(values_prev_degree, knot_span_indices)
        else:
            out = values_prev_degree
                        
        out_all = [out]
            
        if return_knot_span_indices:
            out_all.append(knot_span_indices)
            
        return out_all
        
    
    
    def eval_grid(self,
                  n : int  = 100,
                  **kwargs):
        
        return self(torch.linspace(self.knot_vector.knots[0],
                                   self.knot_vector.knots[-1],n,
                                   device = self.device),
                    **kwargs)[0]
    

    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    kv = Knot_vector.make_open(degree=2, mode = "random")
    bf = Basis_functions(kv)
    print(bf)
    
    fig,ax = plt.subplots(dpi = 100)
    
    ax.set_title(f"Basis functions of degree {bf.degree}")
    
    N = 1000
    u = torch.linspace(0,1,N)
    
    basis_functions_eval = bf.eval_grid(n=N).cpu()
    
    for i in range(bf.n_basis_functions):
        ax.plot(u, basis_functions_eval[:,i])
        
    ax.vlines(kv.knots.cpu(),0,1, color = "k", ls = ":")
        
    fig.tight_layout()
    
    
    fig_d,ax_d = plt.subplots(dpi = 100)
    
    ax_d.set_title(f"Basis functions of degree {bf.degree} derivatives")
    
    basis_functions_eval_d = bf.eval_grid(n=N, derivative_order = 1).cpu()
    
    for i in range(bf.n_basis_functions):
        ax_d.plot(u.cpu(), basis_functions_eval_d[:,i])
        
    fig_d.tight_layout()
        
    # Second derivative test
    fig_dd, ax_dd = plt.subplots(dpi = 100)
    ax_dd.set_title(f"Basis functions of degree {bf.degree} second derivatives")
    
    basis_functions_eval_dd = bf.eval_grid(n=N, derivative_order = 2).cpu()
    
    for i in range(bf.n_basis_functions):
        ax_dd.plot(u.cpu(), basis_functions_eval_dd[:,i])
    
    fig_dd.tight_layout()
    
    