import os
import pandas as pd
import plotly.express as px
import numpy as np

def plot(frame, x_axis, y_axis, color, size, lable, marker, year,
         city, program, orient, wwr, glazing, ext_thick, 
         ext_mat, in_thick, in_mat, shgc, shading, inf_rate,
         gheight=600, gwidth=600, filename='plot'):

    hd = ['City', 'Program', 'Orientation', 'WWR', 'Glazing', 'exterior_t (in)',
          'exterior_mat', 'interior_t (in)', 'interior_mat', 'SHGC', 'Shading',
          'Inf rate', 'payback_linear'
            ]

    ld = {'City':{'list':city, 'name': 'City', 'float':False},
            'Program':{'list':program, 'name': 'Program', 'float':False},
            'Orientation':{'list':orient, 'name': 'Orientation', 'float':False},
            'Glazing':{'list':glazing, 'name': 'Glazing', 'float':False},
            'ext_mat':{'list':ext_mat, 'name': 'exterior_mat', 'float':False},
            'in_mat':{'list':in_mat, 'name': 'interior_mat', 'float':False},
            'SHGC':{'list':shgc, 'name': 'SHGC', 'float':True},
            'Shading':{'list':shading, 'name': 'Shading', 'float':True},
            'Inf rate':{'list':inf_rate, 'name': 'Inf rate', 'float':True},
            'WWR':{'list':wwr, 'name': 'WWR', 'float':True}, 
            'in_thick':{'list':in_thick, 'name': 'interior_t (in)', 'float':True},
            'ext_thick':{'list':ext_thick, 'name': 'exterior_t (in)', 'float':True},
            }
    
    df = frame
    for key in ld:
        masks = []
        for value in ld[key]['list']:
            if ld[key]['float']:
                value = float(value)
            mask = df[ld[key]['name']] == value
            masks.append(mask)
        df = df[np.logical_or.reduce(masks)]

    if color == 'None':
        color = None

    if lable == 'None':
        lable = None

    if size == 'None':
        size = None

    if y_axis == 'Total GWP non-linear 2% (kg CO2e / m2) N year':
        y_axis = 'Total GWP non-linear 2% (kg CO2e / m2) {} year'.format(year)
    elif y_axis == 'Total GWP non-linear 3% (kg CO2e / m2) N year':
        y_axis = 'Total GWP non-linear 3% (kg CO2e / m2) {} year'.format(year) 
    elif y_axis == 'Total GWP non-linear 5% (kg CO2e / m2) N year':
        y_axis = 'Total GWP non-linear 5% (kg CO2e / m2) {} year'.format(year)        
    elif y_axis == 'Total GWP Linear (kg CO2e / m2) N year':
        y_axis = 'Total GWP Linear (kg CO2e / m2) {} year'.format(year)     

    if x_axis == 'Total GWP non-linear 2% (kg CO2e / m2) N year':
        x_axis = 'Total GWP non-linear 2% (kg CO2e / m2) {} year'.format(year)
    elif x_axis == 'Total GWP non-linear 3% (kg CO2e / m2) N year':
        x_axis = 'Total GWP non-linear 3% (kg CO2e / m2) {} year'.format(year) 
    elif x_axis == 'Total GWP non-linear 5% (kg CO2e / m2) N year':
        x_axis = 'Total GWP non-linear 5% (kg CO2e / m2) {} year'.format(year) 
    elif x_axis == 'Total GWP Linear (kg CO2e / m2) N year':
        x_axis = 'Total GWP Linear (kg CO2e / m2) {} year'.format(year)  
    
    if y_axis == 'Op GWP non-linear 2% (kg CO2e / m2) N year':
        y_axis = 'Op GWP non-linear 2% (kg CO2e / m2) {} year'.format(year)
    elif y_axis == 'Op GWP non-linear 3% (kg CO2e / m2) N year':
        y_axis = 'Op GWP non-linear 3% (kg CO2e / m2) {} year'.format(year) 
    elif y_axis == 'Op GWP non-linear 5% (kg CO2e / m2) N year':
        y_axis = 'Op GWP non-linear 5% (kg CO2e / m2) {} year'.format(year)      
    elif y_axis == 'Op GWP Linear (kg CO2e / m2) N year':
        y_axis = 'Op GWP Linear (kg CO2e / m2) {} year'.format(year)  

    if x_axis == 'Op GWP non-linear 2% (kg CO2e / m2) N year':
        x_axis = 'Op GWP non-linear 2% (kg CO2e / m2) {} year'.format(year)
    elif x_axis == 'Op GWP non-linear 3% (kg CO2e / m2) N year':
        x_axis = 'Op GWP non-linear 3% (kg CO2e / m2) {} year'.format(year) 
    elif x_axis == 'Op GWP non-linear 5% (kg CO2e / m2) N year':
        x_axis = 'Op GWP non-linear 5% (kg CO2e / m2) {} year'.format(year)    
    elif x_axis == 'Op GWP Linear (kg CO2e / m2) N year':
        x_axis = 'Op GWP Linear (kg CO2e / m2) {} year'.format(year)  

    if color =='WWR':
        df['WWR'] = df['WWR'].astype(str)
        color_seq=['rgb(68,1,84)', 'rgb(59,81,138)', 'rgb(35,144,140)',
                   'rgb(96,200,96)', 'rgb(253,231,37)']
    elif color =='Inf rate':
        df['Inf rate'] = df['Inf rate'].astype(str)
        color_seq = px.colors.qualitative.Vivid

    elif color == 'Program':
        color_seq = px.colors.qualitative.Set1[3:]

    else:
        # color_seq=['red', 'green', 'blue', 'goldenrod', 'magenta']
        color_seq = px.colors.qualitative.T10
    
    orders={'Program': ['residential', 'office'],
            'Glazing': ['Double', 'Triple'],
            'Orientation': ['n', 'e', 's', 'w'],
            'WWR': ['0.0', '0.2', '0.4', '0.6', '0.8'],
            'Inf rate': ['0.00059', '0.0003', '0.00015'],
            }

    fig = px.scatter(df,
                     x=x_axis,
                     y=y_axis,
                     color=color,
                     size=size,
                     symbol=marker,
                     size_max=12,
                     text=lable,
                     hover_data=hd,
                     labels=None,
                     color_continuous_scale='Viridis',
                     category_orders=orders,
                     color_discrete_sequence=color_seq,
                    )
    

    string = None # 'Envelope carbon emissions'
    fig.update_layout(title={'text':string},
                        hovermode='closest',
                        autosize=False,
                        height=gheight,
                        width=gwidth,
                        paper_bgcolor = 'rgba(0,0,0,0)',
                        plot_bgcolor = 'rgba(0,0,0,0)',
                        showlegend=False,
                                #  legend=dict(x=0,
                                #  y=1.,
                                #  traceorder='normal',
                                #  font=dict(size=12,)
                                # )
                    )
    
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', title='',
                     gridcolor='lightgray', gridwidth=.1, ticks='outside', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', title='',
                     gridcolor='lightgray', gridwidth=.1, ticks='outside', mirror=True)
    fig.update_traces(textposition='top right')
    fig.update_traces(marker_sizemin=5, marker_line_color='black')
    fig.update(layout_coloraxis_showscale=False)
    # fig.show()

    # here = os.path.dirname(__file__)
    # path = os.path.join(here, '{}.pdf'.format(filename))
    fig.write_image(filename, engine='orca')
    # plotly.io.write_image(fig, path, format = 'pdf')
    return fig


if __name__ == '__main__':
    for i in range(50): print('')

    here = os.path.dirname(__file__)

    filepath = os.path.join(here,'assemblies_data_metric.csv')
    frame = pd.read_csv(filepath)
    cities = ['Seattle', 'New York', 'Los Angeles', 'Atlanta', 'Milwaukee', 'San Antonio']
    # cities = ['Seattle']
    # for o in ['n', 's', 'w', 'e']:
    for city in cities:
        y_axis = 'Total GWP Linear (kg CO2e / m2) N year'
        y_axis = 'Total GWP non-linear 5% (kg CO2e / m2) N year'
        # y_axis = 'Operational (kg CO2e / m2 * year)'
        x_axis = 'Embodied (kg CO2e / m2)'
        color = 'Orientation'
        size = 'Wall R (m2 K / W)'
        lable = 'None'
        marker = 'Shading'
        year = 50
        city = [city]
        program = ['residential']
        orient = ['s', 'w', 'e']
        wwr = ['.4']  #['.2', '.4', '.6', '.8']  # ['.0', '.2', '.4', '.6', '.8']  # 
        glazing = ['Double']
        ext_thick = ['0', '.5', '1', '2', '3', '4', '5', '6']
        ext_mat = ['EPS', 'Polyiso']
        in_thick = ['0', '3.5', '5.5', '7.25', '9.25', '4', '6', '8']
        in_mat = ['Fiberglass', 'Cellulose']
        shgc = ['0.25', '0.6']
        shading = ['0.0', '2.5']  # ['0.0']  # 
        inf_rate = ['0.00059'] # ['0.00059', '0.0003', '0.00015']
    
        filename = 'shading_{}.pdf'.format(city[0])
        fp = os.path.join(os.path.dirname(__file__), 'shading', filename)
        plot(frame, x_axis, y_axis, color, size, lable, marker,  year,
            city, program, orient, wwr, glazing, ext_thick, 
            ext_mat, in_thick, in_mat, shgc, shading, inf_rate, filename=fp)
