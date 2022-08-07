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

# TODO:
# - Fix index error when selecting
# - Look of control net: Have only the for movement selected control points be bright, unselected a bit brighter than now
# - Obtain better size/shape of app/plot based on contents

def plotter(obj,*args,**kwargs):
    
    if obj._object_name == "curve_2D":
        return Curve_2D_interactor(obj,*args,**kwargs)
    
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
                 interval            = False,
                 interval_length     = 1000):
        
        self.device        = curve.device
        self.curve         = curve
        self.app           = JupyterDash(__name__)
        self.n_points_eval = n_points_eval
        self.figure        = None
        self.modify_figure = modify_figure
        
        degree           = curve.basis_function_sets[0].degree
        n_control_points = curve.control_net_shape[0]
        
        self.selection = torch.zeros(n_control_points,
                                     device = curve.device)
        
        layout_elems = [
            html.H1(f"Interactive B-spline curve of degree {degree}" if title is None else title),
            dcc.Graph(id='graph')]
        
        if interval:
            layout_elems.append(dcc.Interval(id='interval', interval = interval_length,
                         n_intervals = 0))

        self.app.layout = html.Div(layout_elems)
        
        self.get_callback()
        
        self.app.run_server(mode='inline')
        
        # Plot properties
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
        
        #TODO: Use eval_grid (or even more clever: compute the basis function products only once since they
        # do not change)
        knots = self.curve.basis_function_sets[0].knot_vector.knots
        u     = torch.linspace(knots[0],
                               knots[-1],
                              self.n_points_eval, device = self.curve.device)
        
        with torch.no_grad():
            Eval = self.curve(u, to_memory = "for_plotting").cpu()
        
        figure      = go.FigureWidget()

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