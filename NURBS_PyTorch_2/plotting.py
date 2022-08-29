# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 07:57:27 2022

@author: Bart de Koning
"""

import torch
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
from dash import dcc, html
import dash
from dash.dependencies import Input, Output

# TODO: ...
# - In the title, call the objects NURBS of weights are included, and show the weights
#   in the control point colors
# - Fix index error when selecting
# - Look of control net: Have only the for movement selected control points be bright, unselected a bit brighter than now
# - Obtain better size/shape of app/plot based on contents

def plotter(obj,*args,**kwargs):
    
    assert hasattr(obj, "_object_name"), \
        "Object to plot must have \"_object_name\" attribute."
    
    if obj._object_name == "curve_2D":
        return Curve_2D_interactor(obj,*args,**kwargs)
    
    elif obj._object_name == "surface_3D":
        return Curve_3D_interactor(obj,*args,**kwargs)
        
    
    else:
        raise TypeError(f"There is no plotter implementation for a NURBS object of type {obj._object_name}.")
        
        

class Curve_2D_interactor():
    
    def __init__(self,
                 curve,
                 curve_color         = 'blue',
                 control_point_color = 'rgba(255,0,0,0.8)',
                 title               = None,
                 n_points_eval       = 100,
                 modify_figure       = None,
                 graph_shape         = (500,500)):
        
        self.device        = curve.device
        self.curve         = curve
        self.app           = JupyterDash(__name__)
        self.n_points_eval = n_points_eval
        self.figure        = None
        self.modify_figure = modify_figure
        
        degree           = curve.basis_function_sets[0].degree
        n_control_points = curve.control_net_shape[0]
        
        self.selection = torch.zeros(n_control_points,
                                     device = curve.device,
                                     dtype = torch.bool)
        
        layout_elems = [
            html.H1(f"Interactive B-spline curve of degree {degree}" if title is None else title),
            dcc.Graph(id='graph')]

        self.app.layout = html.Div(layout_elems)
        
        self.get_callback()
        
        self.app.run_server(mode='inline')
        
        # Plot properties
        self.graph_shape         = graph_shape
        self.curve_color         = curve_color
        self.control_point_color = control_point_color 
        
        
    def update_selection(self,selection):
        
        box_data       = selection
        self.box_min_x = min(box_data['x0'],box_data['x1'])
        self.box_max_x = max(box_data['x0'],box_data['x1'])
        self.box_min_y = min(box_data['y0'],box_data['y1'])
        self.box_max_y = max(box_data['y0'],box_data['y1'])
        
        cp_x = self.curve.control_point_coord_sets[0]
        cp_y = self.curve.control_point_coord_sets[1]
        
        self.selection = (cp_x >= self.box_min_x) & (cp_x < self.box_max_x) & \
                         (cp_y >= self.box_min_y) & (cp_y < self.box_max_y)
        
        self.control_points_selection_x = self.curve.control_point_coord_sets[0][self.selection].clone()
        self.control_points_selection_y = self.curve.control_point_coord_sets[1][self.selection].clone()
        
        
    def create_figure(self):
        
        #TODO: Use eval_grid
        knots = self.curve.basis_function_sets[0].knot_vector.knots
        u     = torch.linspace(knots[0],
                               knots[-1],
                              self.n_points_eval, device = self.device)
        
        with torch.no_grad():
            Eval = self.curve(u, to_memory = "for_plotting").cpu()
        
        figure      = go.FigureWidget(layout = dict(width  = self.graph_shape[0],
                                                    height = self.graph_shape[1]))

        figure.update_layout(dragmode = "select")

        figure.add_scatter(            
            mode       = "lines",
            x          = Eval[:,0],
            y          = Eval[:,1],
            name       = "B-spline curve",
            line_color = self.curve_color)

        figure.add_scatter(
            mode       = "lines+markers",
            x          = self.curve.control_point_coord_sets[0].detach().cpu(),
            y          = self.curve.control_point_coord_sets[1].detach().cpu(),
            name       = "Control net",
            marker_size = 10,
            line_color = self.control_point_color)

        figure.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1)
        
        if not self.modify_figure is None:
            self.modify_figure(figure)
        
        return figure
        
        
        
    def update_figure(self,
                      figure = None):
        
        if figure is None:
            figure = self.figure
        
        if figure is None:
            print('fail')
            return
        
        with torch.no_grad():
            Eval = self.curve(from_memory = "for_plotting").cpu()
        
        figure['data'][0]['x'] = Eval[:,0]
        figure['data'][0]['y'] = Eval[:,1]
        
        figure['data'][1]['x'] = self.curve.control_point_coord_sets[0].detach().cpu()
        figure['data'][1]['y'] = self.curve.control_point_coord_sets[1].detach().cpu()
        
        
    
    def get_callback(self):
        
        @self.app.callback(
            [Output('graph', 'figure')],
            [Input('graph', 'relayoutData'),
             Input('graph', 'figure')]
        )
        def callback(data,figure):

            if not figure:
                
                figure      = self.create_figure()
                self.figure = figure

            else:
                
                if 'selections' in data: 
                    if len(data['selections']) == 0:
                        return dash.no_update
                        
                    selection = data['selections'][0]

                    if 'type' in selection:
                        self.update_selection(selection)
                        
                    else:
                        return dash.no_update
                
                elif 'selections[0].x0' in data:
                    Dx = max(data['selections[0].x0'],data['selections[0].x1']) - self.box_max_x
                    Dy = max(data['selections[0].y0'],data['selections[0].y1']) - self.box_max_y
                    
                    self.curve.control_point_coord_sets[0][self.selection] = self.control_points_selection_x + Dx
                    self.curve.control_point_coord_sets[1][self.selection] = self.control_points_selection_y + Dy
                    
                    self.update_figure(figure = figure)
                    
                else:
                    return dash.no_update

            return [figure]
        
        
        
class Curve_3D_interactor():
    
    def __init__(self,
                 surface,
                 surface_color       = 'blue',
                 surface_opacity     =  0.5,
                 control_point_color = 'rgba(255,0,0,0.8)',
                 control_point_size  = 3,
                 title               = None,
                 n_points_eval       = 100,
                 modify_figure       = None,
                 graph_shape         = (500,500),
                 slider_r            = 1,
                 cbar_range          = None,
                 control_net         = True):
        
        self.device        = surface.device
        self.surface       = surface
        self.app           = JupyterDash(__name__)
        self.n_points_eval = n_points_eval
        self.modify_figure = modify_figure
        
        degrees           = tuple([bfs.degree for bfs in surface.basis_function_sets])
        
        self.selection = torch.zeros(surface.control_net_shape,
                                     device = self.device,
                                     dtype = torch.bool)
        
        layout_elems = [
            html.H1(f"Interactive B-spline surface of degrees {degrees}" if title is None else title),
            dcc.Graph(id='graph3d'),
            dcc.Slider(-slider_r, slider_r, marks=None, value=0, id = 'slider',
                       step = 2*slider_r/25, updatemode = 'drag'),
            dcc.Graph(id='graph2d')]
        
        self.app.layout = html.Div(layout_elems)
        
        self.get_callback()
        self.app.run_server(mode='inline')
        
        # Plot properties
        self.control_net         = control_net
        self.cbar_range          = cbar_range
        self.prev_value_slider   = 0
        self.graph_shape         = graph_shape
        self.surface_opacity     = surface_opacity
        self.surface_color       = surface_color
        self.control_point_color = control_point_color
        self.control_point_size  = control_point_size
        
        
        
    def update_selection(self,selection):
        
        box_data       = selection
        self.box_min_x = min(box_data['x0'],box_data['x1'])
        self.box_max_x = max(box_data['x0'],box_data['x1'])
        self.box_min_y = min(box_data['y0'],box_data['y1'])
        self.box_max_y = max(box_data['y0'],box_data['y1'])
        
        cp_x = self.surface.control_point_coord_sets[0]
        cp_y = self.surface.control_point_coord_sets[1]
        
        selection_new = (cp_x >= self.box_min_x) & (cp_x < self.box_max_x) & \
                         (cp_y >= self.box_min_y) & (cp_y < self.box_max_y)
                         
        selection_changed = ~(selection_new == self.selection).all()
        self.selection    = selection_new
        
        if selection_changed:
            self.control_points_selection_z = self.surface.control_point_coord_sets[2][self.selection].clone()
    
        return selection_changed
        
    def create_figures(self):
        
        with torch.no_grad():
            Eval = self.surface.eval_grid(n=self.n_points_eval,
                                          to_memory = 'for_grid').squeeze().cpu()
            
            X = Eval[:,:,0]
            Y = Eval[:,:,1]
            Z = Eval[:,:,2]
            
        figure2d = go.FigureWidget(layout = dict(width  = self.graph_shape[0],
                                                 height = self.graph_shape[1]))
            
        figure3d = go.FigureWidget(layout = dict(width  = self.graph_shape[0],
                                                 height = self.graph_shape[1]))
        
        
        
        figure2d.update_layout(dragmode = "select")
        
        # Adding surface
        figure3d.add_surface(x = X,
                             y = Y,
                             z = Z,
                             name = "B-spline surface",
                             opacity = self.surface_opacity,
                             colorscale = 'jet',
                             cmin = self.cbar_range[0],
                             cmax = self.cbar_range[1],
                             )
        
        figure3d.update_traces(colorbar_len = 0.8)
        
        line_marker = dict(color='#0066FF', width=2)
        
        # Adding control net        
        for i in range(self.surface.control_net_shape[0]):
            
            x = self.surface.control_point_coord_sets[0][i].cpu()
            y = self.surface.control_point_coord_sets[1][i].cpu()
            z = self.surface.control_point_coord_sets[2][i].cpu()
            
            if self.control_net:
                figure3d.add_scatter3d(mode = "markers+lines",
                                     x = x,
                                     y = y,
                                     z = z,
                                     name   = "Control net",
                                     marker = dict(size = self.control_point_size),
                                     line   = line_marker,
                                     showlegend = (i == 0))
            
            figure2d.add_scatter(mode = "markers+lines",
                                 x = x,
                                 y = y,
                                 name = "Control net projection",
                                 line = line_marker,
                                 showlegend = (i == 0))
            
        for j in range(self.surface.control_net_shape[1]):
            
            x = self.surface.control_point_coord_sets[0][:,j].cpu()
            y = self.surface.control_point_coord_sets[1][:,j].cpu()
            z = self.surface.control_point_coord_sets[2][:,j].cpu()
            
            if self.control_net:
                figure3d.add_scatter3d(mode = "markers+lines",
                                     x = x,
                                     y = y,
                                     z = z,
                                     name   = "Control net",
                                     marker = dict(size = self.control_point_size),
                                     line   = line_marker,
                                     showlegend = False)
            
            figure2d.add_scatter(mode = "markers+lines",
                                 x = x,
                                 y = y,
                                 name = "Control net projection",
                                 line = line_marker,
                                 showlegend = False)
            
        figure2d.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1)
        
        return figure2d, figure3d
    
    
    def update_figure3d(self,figure3d):
        
        with torch.no_grad():
            Eval = self.surface.eval_grid(n = self.n_points_eval,
                                          from_memory = 'for_grid').squeeze().cpu()                        

        # Surface update
        figure3d['data'][0]['z'] = Eval[:,:,2]
        
        control_points_z = self.surface.control_point_coord_sets[2].cpu()
        
        # Control net update
        if self.control_net:
        
            for i in range(self.surface.control_net_shape[0]):
                
                z = control_points_z[i]
                figure3d['data'][i+1]['z'] = z
                
            for j in range(self.surface.control_net_shape[1]):
                
                z = control_points_z[:,j]
                figure3d['data'][j+self.surface.control_net_shape[0]+1]['z'] = z
        
    def get_callback(self):
        
        @self.app.callback(
            [Output('graph2d', 'figure'), 
             Output('graph3d', 'figure'),
             Output('slider','value')],
            [Input('graph2d', 'relayoutData'),
             Input('slider', 'value'),
             Input('graph2d', 'figure'),
             Input('graph3d', 'figure')]
        )
        def callback(data,value_slider,figure2d,figure3d):
            
            if not figure3d:
                
                figure2d, figure3d = self.create_figures()
                
            else:
                
                if 'selections' in data:
                    if len(data['selections']) == 0:
                        return dash.no_update
                     
                    else:
                        selection = data['selections'][0]
    
                        if 'type' in selection:
                            if self.update_selection(selection):
                                value_slider = 0
                        
                if value_slider != self.prev_value_slider:
                    
                    self.surface.control_point_coord_sets[2][self.selection] = \
                        self.control_points_selection_z + value_slider
                   
                    self.update_figure3d(figure3d)
                    self.prev_value_slider = value_slider
                
            return [figure2d,figure3d,value_slider]
        