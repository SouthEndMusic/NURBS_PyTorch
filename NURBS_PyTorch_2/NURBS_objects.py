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
        
        self._object_name = "NURBS_object" # Overwritten in child classes
        
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
        einsum_string_call  = ",".join(f"a{S}{s}" for S,s in zip(ascii_uppercase[1:1+n_inputs],
                                                                 ascii_lowercase[1:1+n_inputs]))
        einsum_string_call += "->a"
        einsum_string_call += ascii_uppercase[1:1+n_inputs] + ascii_lowercase[1:1+n_inputs]
        
        self.einsum_string_call = einsum_string_call
        
        einsum_string_grid = ",".join(f"{a}{S}{s}" for a,S,s in zip(ascii_lowercase[13:13+n_inputs],
                                                                    ascii_uppercase[:n_inputs],
                                                                    ascii_lowercase[:n_inputs]))
        
        einsum_string_grid += "->"
        einsum_string_grid += ascii_lowercase[13:13+n_inputs] + \
                              ascii_uppercase[:n_inputs] + \
                              ascii_lowercase[:n_inputs]
                              
        self.einsum_string_grid = einsum_string_grid
        
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
        
        # Create list of derivative orders 0 if list is not given
        if derivative_orders is None:
            derivative_orders = self.n_inputs*[0]
            
            
        if from_memory is None:
        
            # Check whether the correct amount of input variables is given
            assert len(inputs_all) == self.n_inputs
            
            basis_function_values = []
            
            # Compute basis function values
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
        
        
        input_size = basis_function_values[0][0].shape[0]
        
        # Create Boolean array control_points_with support that indicates per input
        # which control points are associated with basis functions that have this input
        # in their support. Shape:
        # (n_inputs, *self.control_net_shape)
        for i,(bfv,bfs) in enumerate(zip(basis_function_values,
                                         self.basis_function_sets)):
            
            control_indices_dim_i = torch.zeros(input_size,self.control_net_shape[i],
                                                dtype = torch.bool, device = self.device)
            
            indics1 = torch.arange(input_size, device = self.device).tile(bfs.degree+1,1).T
            indics2 = bfv[1][:,None] + torch.arange(-(bfs.degree+1),0, device = self.device)[None,:]
            
            control_indices_dim_i[indics1,indics2] = True
            
            if i == 0:
                control_points_with_support = control_indices_dim_i
            else:
                for j in range(i):
                    control_indices_dim_i = control_indices_dim_i.unsqueeze(dim=1)
                    
                control_points_with_support = control_points_with_support.unsqueeze(dim=-1) & \
                                              control_indices_dim_i
             

        enumerator = torch.einsum(self.einsum_string_call,
                                  *[bfv[0] for bfv in basis_function_values])
        
        if outputs_include == "all":       
            control_points_stacked = torch.stack(self.control_point_coord_sets,
                                                 dim = -1)
            
        else:
            control_points_stacked = torch.stack([self.control_point_coord_sets[i] for i in outputs_include],
                                                 dim = -1)

        
        degrees_plus_1           = [bfs.degree+1 for bfs in self.basis_function_sets]
        control_points_per_input = control_points_stacked.tile(input_size,
                                                               *(1+self.n_inputs)*[1])[control_points_with_support].reshape(input_size,
                                                                                                                            *degrees_plus_1,
                                                                                                                            control_points_stacked.shape[-1])
        
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
    
    
    def eval_grid(self,n=100,
                  derivative_orders = None,
                  to_memory         = None,
                  from_memory       = None,
                  outputs_include   = 'all',
                  construct_output  = True):
        
        if not hasattr(n, "__getitem__"):
            n = self.n_inputs*[n]
            
        basis_function_values = []
            
        if derivative_orders is None:
            derivative_orders = self.n_inputs*[0]
            
        if from_memory is None:
            
            basis_function_values = []
            
            for n_, basis_function_set,derivative_order in zip(n,
                                                           self.basis_function_sets,
                                                           derivative_orders):
                
                basis_function_values.append(basis_function_set.eval_grid(n=n_,
                                                                          return_knot_span_indices = True,
                                                                          derivative_order = derivative_order,
                                                                          uncompress = False))
                
            if not to_memory is None:
                
                # !!!: Note that here the basis function values and not the basis function products are stored.
                # This is a choice to do some more computational work per call in favour of saving memory.
                self.basis_function_memory[to_memory] = basis_function_values
                
        else:
            basis_function_values_all = self.basis_function_memory[from_memory]
            basis_function_values     = []
            
            for i in range(self.n_inputs):        
                basis_function_values.append([basis_function_values_all[i][0][:,:,:derivative_orders[i]+1],
                                              basis_function_values_all[i][1]])
            
            
        if not construct_output:
            return
            
        # Shape of basis_function_values:
        # [(n[0],degree_1+1,derivative_orders[0]+1),
        #  ...,
        #  (n[self.n_inputs-1], degree_{self.n_inputs}+1,derivative_orders[self.n_inputs-1]+1)]

        
        for i,(bfv,bfs) in enumerate(zip(basis_function_values,
                                         self.basis_function_sets)):
            
            control_indices_dim_i = torch.zeros(n[i],self.control_net_shape[i],
                                                dtype = torch.bool, device = self.device)
            
            indics1 = torch.arange(n[i], device = self.device).tile(bfs.degree+1,1).T
            indics2 = bfv[1][:,None] + torch.arange(-(bfs.degree+1),0, device = self.device)[None,:]
            
            control_indices_dim_i[indics1,indics2] = True
            
            if i == 0:
                control_points_with_support = control_indices_dim_i
            else:
                for j in range(i):
                    control_indices_dim_i = control_indices_dim_i.unsqueeze(dim=0)
                    control_indices_dim_i = control_indices_dim_i.unsqueeze(dim=j+2)
                        
                control_points_with_support = control_points_with_support.unsqueeze(dim=i)
                control_points_with_support = control_points_with_support.unsqueeze(dim=-1)
                    
                control_points_with_support = control_points_with_support & \
                                              control_indices_dim_i
        
        # Shape of knot_span_product_indices after unsqueeze:
        # (n[0], n[1], ..., n[self.n_inputs-1], self.n_inputs,*self.n_inputs*[1])
        # for i in range(self.n_inputs):
        #     knot_span_product_indices = knot_span_product_indices.unsqueeze(dim = -1)
            
        # Shape: (*n, self.n_inputs, degree_1+1,...,degree_{self.n_inputs}+1,
        # *derivative_orders)
        enumerator = torch.einsum(self.einsum_string_grid,
                                  *[bfv[0] for bfv in basis_function_values])
        
        if outputs_include == "all":       
            control_points_stacked = torch.stack(self.control_point_coord_sets,
                                                 dim = -1)
            
        else:
            control_points_stacked = torch.stack([self.control_point_coord_sets[i] for i in outputs_include],
                                                 dim = -1)
        
        # Shape: (*n, degree_1+1, degree_2+1, ..., degree_{self.n_inputs}+1, self.n_outputs)
        # Note: if outputs_include is specified the length of this iterable determines the size of the
        # last dimension
        # control_points_per_input = control_points_stacked.__getitem__(control_point_indices_per_input.split(1,dim=self.n_inputs)).squeeze(dim=self.n_inputs)
        degrees_plus_1           = [bfs.degree+1 for bfs in self.basis_function_sets]
        control_points_per_input = control_points_stacked.expand(*n,
                                                                 *control_points_stacked.shape)[control_points_with_support].reshape(*n,
                                                                                                                            *degrees_plus_1,
                                                                                                                            control_points_stacked.shape[-1])
        
        
        # Add dimensions to control_points_per_input for the sum below, corresponding to the
        # various derivative order combinations
        for i in range(self.n_inputs):
            control_points_per_input = control_points_per_input.unsqueeze(dim=-2)
        
        # Shape: (*n, derivative_orders[0]+1, ..., derivative_orders[self.n_inputs-1]+1, 
        #         self.n_outputs)
        # Note: see note above for size of last dimension
        out = torch.sum(enumerator.unsqueeze(dim=-1) * control_points_per_input,
                        axis = tuple(range(self.n_inputs,2*self.n_inputs)))
        
        return out


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
            values = self.eval_grid(n=n, **kwargs).squeeze()
            
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
    
    def normals(self,u,v,
                return_locs = False):
        
        values = self(u,v, derivative_orders = [1,1])
        
        deriv_u = values[:,1,0]
        deriv_v = values[:,0,1]
        
        normals  = torch.cross(deriv_u,deriv_v,dim=1)
        normals /= normals.norm(dim=1,keepdim=True)
        
        if return_locs:
            return values[:,0,0], normals
        else:
            return normals
        
        
    def triangle_mesh(self,n,
                      **kwargs):
        
        if not hasattr(n, "__getitem__"):
            n = self.n_inputs*[n]
        
        # Number of faces and vertices
        n_faces    = 2*(n[0]-1)*(n[1]-1)
        n_vertices = n[0]*n[1]
        
        indices_0 = torch.arange(n[0], device = self.device).expand(n[1],n[0]).T
        indices_1 = torch.arange(n[1], device = self.device).expand(*n)
        indices   = n[0]*indices_1 + indices_0
        
        del indices_0, indices_1
        
        # Create array for the faces defined by vertex point index triples
        faces = torch.zeros((n_faces,3), dtype = torch.int, device = self.device)
        
        # Array that defines one corner of every triangle on the surface
        face_corners = indices[:-1,:-1].reshape(-1)
        
        # First half of the triangles
        faces[:n_faces//2,0] = face_corners
        faces[:n_faces//2,1] = face_corners + 1
        faces[:n_faces//2,2] = face_corners + n[0]

        del face_corners

        # Second half of the triangles
        faces[n_faces//2:,0] = faces[:n_faces//2,1]
        faces[n_faces//2:,1] = faces[:n_faces//2,2] + 1
        faces[n_faces//2:,2] = faces[:n_faces//2,2]
        
        vertices = self.eval_grid(n,**kwargs).squeeze().reshape(n_vertices,3)
        
        return vertices,faces
    
    def area(self,n,
             return_mesh = False,
             **kwargs):
        """Approximate the surface area from a triangle mesh."""
        
        vertices, faces = self.triangle_mesh(n,**kwargs)
        faces           = faces.long()
        v1              = vertices[faces[:,1]] - vertices[faces[:,0]]
        v2              = vertices[faces[:,2]] - vertices[faces[:,0]]
        
        area = torch.cross(v1,v2).norm(dim=1).sum()/2
        
        if return_mesh:
            return vertices, faces, area
        else:
            return area

    
    
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
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
    
    surface_3D = S_3D.eval_grid().squeeze().cpu()
    
    fig          = plt.figure(dpi = 100)
    ax_eval_grid = fig.add_subplot(121,projection = "3d")
    
    ax_eval_grid.plot_surface(surface_3D[:,:,0],
                              surface_3D[:,:,1],
                              surface_3D[:,:,2])
    
    ax_eval_grid.plot_wireframe(control_points_x.numpy(),
                                control_points_y.numpy(),
                                control_points_z.numpy(),
                                color = "C1")
    
    N = 100
    u    = torch.linspace(0,1,N, device = S_3D.device)
    v    = torch.linspace(0,1,N, device = S_3D.device)
    u,v  = torch.meshgrid(u,v, indexing = 'ij')
    u    = u.reshape(-1)
    v    = v.reshape(-1)
    locs = S_3D(u,v).reshape(N,N,3).cpu()
    
    ax_call = fig.add_subplot(122, projection = "3d")
    
    ax_call.plot_surface(locs[:,:,0],
                         locs[:,:,1],
                         locs[:,:,2])
    
    S_3D.triangle_mesh((50,60))
    

