# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 19:33:27 2022

@author: Bart de Koning
"""

from sys     import path
from pathlib import Path

path_here  = str(Path(__file__).parent.resolve())
path.append(path_here)

import torch
import basis_functions

from string import ascii_lowercase, ascii_uppercase


def Marsden(knot_vector):
    """Marsden's identity defines control points in one dimension such that x=u."""
    
    # TODO: Does not support non-open knot vectors?
    
    n_control_points = len(knot_vector.knots) + knot_vector.degree -1
    control_points   = torch.zeros(n_control_points)
    
    for i in range(n_control_points):
        control_points[i] = knot_vector.vector_full[i+1:i+1+knot_vector.degree].sum()
    
    return control_points/knot_vector.degree



class NURBS_object():
    
    def __init__(self,
                 control_net_shape,
                 include_weights = False,
                 n_inputs        = 1,
                 n_outputs       = 2,
                 device          = basis_functions.device_standard):
        """Base class for NURBS objects for any number of inputs and outputs."""
        
        self._object_name = "NURBS_object"
        
        self.device = device
        
        # If True, this is a NURBS object, otherwise it is a B-spline object
        self.include_weights = include_weights
        
        # The number of inputs of the NURBS parametrization
        # This also determines the dimensionality of the parameterized object
        self.n_inputs = n_inputs
        
        # The number of outputs of the NURBS parameterization
        # This also determines the dimensionality of the space the parameterized
        # object is embedded in (and the number of coordinates per control point)
        self.n_outputs = n_outputs
        
        assert len(control_net_shape) == n_inputs, \
            f"The control net shape length {control_net_shape} must be of length n_inputs = {n_inputs}."
            
        self.control_net_shape = control_net_shape
        
        self.basis_function_sets      = []
        self.control_point_coord_sets = []
        
        # For calculating the knot span products
        # 'a' represents the number of input variables
        # The uppercase letters represent the number of nonzero basis functions (degree + 1)
        # The lowercase letters represent the different derivative orders
        self.einsum_string  = ",".join(f"a{S}{s}" for S,s in zip(ascii_uppercase[1:1+n_inputs],
                                                                 ascii_lowercase[1:1+n_inputs]))
        self.einsum_string += "->a" + ascii_uppercase[1:1+n_inputs] + ascii_lowercase[1:1+n_inputs]
        
        # Basis function values for certain inputs can be stored in this dictionary
        # so that if you want to evaluate this NURBS object for the same inputs multiple
        # times and only the control points change, you don't have to calculate the same
        # basis function values every time.
        self.basis_function_memory = dict()
        
        
    def get_property_dict(self):
        pd =  dict(n_inputs          = self.n_inputs,
                   n_outputs         = self.n_outputs,
                   include_weights   = self.include_weights,
                   control_net_shape = self.control_net_shape)
    
        for i, bfs in enumerate(self.basis_function_sets):
            pd[f"basis_functions (dim {i})"] = bfs
            
        return pd
            
    def __str__(self):
        return basis_functions.description_creator(self.get_property_dict(),
                                                   object_name = self._object_name)
        
        
    def set_parameters(self,
                       weights : torch.Tensor   = None,
                       basis_function_sets      = None,
                       control_point_coord_sets = None):
        """
        Weights             : torch.Tensor of shape self.control_net_shape
        basis_function_sets : iterable of basis_functions.Basis_functions agreeing with self.control_net_shape
        control_point_coords: iterable of length self.n_outputs with torch.Tensor of shape self.control_net_size4
        """
        
        if not weights is None:
            
            if not self.include_weights:
                print("""Warning: weights set but this object currently does not include weights
                      (to change this set the attribute include_weights to True)""")
            
            assert torch.tensor(weights.shape == self.control_net_shape).all(), \
                "There must be precisely one weight per control point."
                
            self.weights = weights.to(self.device)
            
        if not basis_function_sets is None:
            
            assert len(basis_function_sets) == self.n_inputs, \
                """There must be precisely one basis function set per input dimension."""
            
            for i, basis_function_set in enumerate(basis_function_sets):
                
                assert basis_function_set.n_basis_functions == self.control_net_shape[i], \
                    f"""The number of basis functions and the control net shape at dimension {i} do not agree
                    (got {basis_function_set.n_basis_functions} and {self.control_net_shape[i]})."""
                    
                assert basis_function_set.device == self.device, \
                    f"""The basis function set at dimension {i} is not on the same device
                    as this NURBS_object."""
                    
            self.basis_function_sets = basis_function_sets
            
        if not control_point_coord_sets is None:
            
            assert len(control_point_coord_sets) == self.n_outputs, \
                f"""There must be a set of control point coordinates for each of the {self.n_outputs}
                output dimenions."""
                
            for i,control_point_coord_set in enumerate(control_point_coord_sets):
                
                assert (control_point_coord_set.shape == self.control_net_shape), \
                    f"""The control net coordinates at dimension {i} must be of shape 
                    control_net_shape = {self.control_net_shape}"""
                
                control_point_coord_sets[i] = control_point_coord_set.to(self.device)
                
            self.control_point_coord_sets = control_point_coord_sets
        
        
    def __call__(self,*inputs_all,
                 derivative_orders = None,
                 to_memory         = None,
                 from_memory       = None,
                 outputs_include   = "all"):
        """
        Evaluate the NURBS_object as a function.
        Note that the inputs must be in the domain of the basis functions.
        """
        
        # TODO: Add weights incorporation (with and without derivatives)
        
        if derivative_orders is None:
            derivative_orders = self.n_inputs*[0]
            
            
        if from_memory is None:
        
            # Check whether the correct amount of input variables is given
            assert len(inputs_all) == self.n_inputs
            
            basis_function_values = []
            
            for inputs,basis_function_set,derivative_order in zip(inputs_all, 
                                                                   self.basis_function_sets,
                                                                   derivative_orders):
                basis_function_values.append(basis_function_set(inputs,
                                                                return_knot_span_indices = True,
                                                                derivative_order         = derivative_order,
                                                                uncompress               = False))
                
            if not to_memory is None:
                
                # !!!: Note that here the basis function values and not the basis function products are stored.
                # This is a choice to do some more computational work per call in favour of saving memory.
                self.basis_function_memory[to_memory] = basis_function_values
                
        else:
            basis_function_values = self.basis_function_memory[from_memory]
    
    
        # Shape: (len(inputs[.]),self.n_inputs)    
        knot_span_product_indices = torch.stack([bfv[1] for bfv in basis_function_values],
                                                dim = -1)
             
        # Shape: (1, self.n_inputs, degree_1+1, degree_2+1, ..., degree_{self.n_inputs}+1)
        control_point_indices_per_input = torch.stack(torch.meshgrid([torch.arange(-(bfs.degree+1),0, device = self.device)
                                                                      for bfs in self.basis_function_sets],
                                                                      indexing = "ij")).unsqueeze(dim=0)
        
        knot_span_product_indices_unsqueezed = knot_span_product_indices.clone()
        
        for i in range(self.n_inputs):
            knot_span_product_indices_unsqueezed = knot_span_product_indices_unsqueezed.unsqueeze(dim=-1)
        
        # Shape: (len(inputs[.]), self.n_inputs, degree_1+1, degree_2+1, ..., degree_{self.n_inputs}+1)
        control_point_indices_per_input = control_point_indices_per_input + knot_span_product_indices_unsqueezed
        
        # Shape: (len(inputs[.]), degree_1+1, degree_2+1, ..., degree_{self.n_inputs}+1,
        #         derivative_orders[0]+1, ..., derivative_orders[self.n_inputs-1]+1)
        enumerator = torch.einsum(self.einsum_string,
                                  *[bfv[0] for bfv in basis_function_values])
        
        if outputs_include == "all":       
            control_points_stacked = torch.stack(self.control_point_coord_sets,
                                                 dim = -1)
            
        else:
            control_points_stacked = torch.stack([self.control_point_coord_sets[i] for i in outputs_include],
                                                 dim = -1)
        
        # Shape: (len(inputs[.]), degree_1+1, degree_2+1, ..., degree_{self.n_inputs}+1, self.n_outputs)
        # Note: if outputs_include is specified the leength of this iterable determines the size of the
        # last dimension
        control_points_per_input = control_points_stacked.__getitem__(control_point_indices_per_input.split(1,dim=1)).squeeze(dim=1)
        
        # Add dimensions to control_points_per_input for the sum below, corresponding to the
        # various derivative order combinations
        for i in range(self.n_inputs):
            control_points_per_input = control_points_per_input.unsqueeze(dim=-2)
        
        # Shape: (len(inputs[.]), derivative_orders[0]+1, ..., derivative_orders[self.n_inputs-1]+1, 
        #         self.n_outputs)
        # Note: see note above for size of last dimension
        out = torch.sum(enumerator.unsqueeze(dim=-1) * control_points_per_input,
                        axis = tuple(range(1,self.n_inputs+1)))
        
        return out

    
    def eval_grid(self,n=100):
        
        if not hasattr(n, "__iter__"):
            n = self.n_inputs*[n]
            
        basis_function_values = []
            
        for n_,basis_function_set in zip(n,self.basis_function_sets):
            basis_function_values.eval_grid(n = n_, return_knot_span_indices = True)
            
        # TODO: construct output
            
            
            
            


class Curve(NURBS_object):
    
    def __init__(self,
                 n_control_points = 10,
                 n_outputs        = 2):
        """NURBS_object for curves with an arbitrary number of outputs."""
        
        super().__init__((n_control_points,),
                         n_inputs  = 1,
                         n_outputs = n_outputs)
        
    def __call__(self,*args,**kwargs):
        return super().__call__(*args,**kwargs).squeeze()
    
    
    def get_length(self, n = 100,
                   values = None,
                   **kwargs):
        """Compute the length of this curve with a piece-wise linear approximation."""
        
        if values is None:
            
            # TODO: In the future use NURBS_object.eval_grid
            u = torch.linspace(self.basis_function_sets[0].knot_vector.knots[0],
                               self.basis_function_sets[0].knot_vector.knots[-1],
                               n, device = self.device)
            values = self(u)
            
        return (values[1:] - values[:-1]).norm(dim=1).sum()
        
        
        
    
    
class Curve_2D(Curve):

    def __init__(self,**kwargs):
        """Curve in 2 dimensions."""
        super().__init__(**kwargs,
                         n_outputs = 2)
        
        self._object_name = "curve_2D"
        
    def normals(self,u):
        """Compute normal vectors to the curve."""
        
        deriv_values = self(u, derivative_orders = [1])[:,1]
        normals      = torch.zeros_like(deriv_values)
        
        normals[:,0] = -deriv_values[:,1]
        normals[:,1] =  deriv_values[:,0]
        
        return normals/normals.norm(dim=1,keepdim=True)
        
        
class Curve_3D(Curve):
    
    def __init__(self, **kwargs):
        """Curve in 3 dimensions."""
        super().__init__(**kwargs,
                         n_outputs = 3)
        
        self._object_name = "curve_3D"
        
        
        
class Surface_3D(NURBS_object):
    
    def __init__(self,**kwargs):
        
        super().__init__(n_inputs  = 2,
                         n_outputs = 3,
                         **kwargs)
        
        self._object_name = "surface_3D"
        
    def __call__(self,*args,**kwargs):
        return super().__call__(*args,**kwargs).squeeze()
        
    # TODO: Implement getting triangle mesh
    # TODO: Implement getting surface area from triangle mesh
    # TODO: Implement getting normals using cross product of derivatives
            
        

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    # 2D curve example
    C_2D = Curve_2D()
    
    kv   = basis_functions.Knot_vector.make_open()
    bf   = basis_functions.Basis_functions(kv)
    
    C_2D.set_parameters(
                      basis_function_sets      = [bf],
                      control_point_coord_sets = [torch.linspace(0,1,10),
                                                  torch.rand(C_2D.control_net_shape)])
    
    print(C_2D)
    
    u = torch.linspace(0,1, 100, device = C_2D.device)
    
    curve_2D = C_2D(u).cpu()
    
    fig_2D,ax_2D = plt.subplots(dpi = 100)
    
    ax_2D.plot(curve_2D[:,0],
                curve_2D[:,1],
                label = f"length = {C_2D.get_length().item():.3f}")
    
    ax_2D.plot(C_2D.control_point_coord_sets[0].cpu(),
                C_2D.control_point_coord_sets[1].cpu(), 
                marker = ".")
    
    normals = C_2D.normals(u[::10]).cpu()
    
    ax_2D.quiver(curve_2D[::10,0],
                  curve_2D[::10,1],
                  normals[:,0],
                  normals[:,1],
                  label = "Normals")
    
    ax_2D.legend()
    ax_2D.set_aspect("equal")
    
    # 3D curve example
    n_control_points = 20
    
    kv   = basis_functions.Knot_vector.make_open(n_control_points = n_control_points)
    bf   = basis_functions.Basis_functions(kv)
    
    control_points_z = torch.linspace(0,1,n_control_points)
    control_points_x = control_points_z*torch.cos(20*control_points_z)
    control_points_y = control_points_z*torch.sin(20*control_points_z)
    
    C_3D = Curve_3D(n_control_points = n_control_points)
    C_3D.set_parameters(
                      basis_function_sets      = [bf],
                      control_point_coord_sets = [control_points_x,
                                                  control_points_y,
                                                  control_points_z])
    
    print(C_3D)
    
    fig_3D = plt.figure(dpi = 100)
    ax_3D  = fig_3D.add_subplot(projection = "3d")
    
    curve_3D = C_3D(u).cpu().numpy()
    
    ax_3D.plot(curve_3D[:,0],
                curve_3D[:,1],
                curve_3D[:,2],
                label = f"length = {C_3D.get_length().item():.3f}")
    
    ax_3D.plot(C_3D.control_point_coord_sets[0].cpu().numpy(),
                C_3D.control_point_coord_sets[1].cpu().numpy(),
                C_3D.control_point_coord_sets[2].cpu().numpy(),
                marker = ".")
    
    ax_3D.legend()
    
    # 3D surface example
    n_control_points_1 = 15
    n_control_points_2 = 25
    
    degree_1 = 4
    degree_2 = 6
    
    kv_1 = basis_functions.Knot_vector.make_open(n_control_points = n_control_points_1,
                                                 degree           = degree_1)
    bf_1 = basis_functions.Basis_functions(kv_1)
    
    kv_2 = basis_functions.Knot_vector.make_open(n_control_points = n_control_points_2,
                                                 degree           = degree_2)
    bf_2 = basis_functions.Basis_functions(kv_2)
    
    S_3D = Surface_3D(control_net_shape = (n_control_points_1,
                                           n_control_points_2))
    
    control_points_x = torch.linspace(-1,1, n_control_points_1)
    control_points_y = torch.linspace(-1,1, n_control_points_2)
    
    control_points_x, control_points_y = torch.meshgrid(control_points_x,
                                                        control_points_y,
                                                        indexing = 'ij')
    
    control_points_z = control_points_x**2 -control_points_y**2
    
    S_3D.set_parameters(basis_function_sets      = [bf_1,bf_2],
                        control_point_coord_sets = [control_points_x,
                                                    control_points_y,
                                                    control_points_z])
    
    print(S_3D)
    
    u = torch.linspace(0,1, 100, device = S_3D.device)
    v = u.clone()
    
    U,V = torch.meshgrid(u,v, indexing = 'ij')
    
    surface_3D = S_3D(U.reshape(-1),
                      V.reshape(-1)).cpu().numpy()
    
    fig_3D = plt.figure(dpi = 100)
    ax_3D  = fig_3D.add_subplot(projection = "3d")
    
    ax_3D.plot_surface(surface_3D[:,0].reshape(100,100),
                       surface_3D[:,1].reshape(100,100),
                       surface_3D[:,2].reshape(100,100))
    
    ax_3D.plot_wireframe(control_points_x.numpy(),
                         control_points_y.numpy(),
                         control_points_z.numpy(),
                         color = "C1")
    
    
    
        
    