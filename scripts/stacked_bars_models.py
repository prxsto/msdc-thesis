import os
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import numpy as np

def plot(frame, cities, program, orient, wwr, glazing, ext_thick, 
         ext_mat, in_thick, in_mat, shgc, shading, inf_rate,
         gheight=650, gwidth=900, filename='plot'):

    ld = {'City':{'list':cities, 'name': 'City', 'float':False},
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
    
    data_dict = df.to_dict()

    d = {'Seattle':        {'n':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'e':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            's':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'w':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0}},
        'Atlanta':         {'n':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'e':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            's':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'w':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0}},
        'San Antonio'     :{'n':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'e':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            's':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'w':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0}},
        'Los Angeles':     {'n':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'e':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            's':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'w':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0}},
        'New York':        {'n':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'e':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            's':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'w':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0}},
        'Milwaukee':       {'n':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'e':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            's':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0},
                            'w':{'heat':0, 'cool':0, 'light':0, 'hw':0, 'eq':0}},
    }            
    
    colors = px.colors.qualitative.G10

    ck = 'Cooling EUI (kWh / m2 * year)'
    hk = 'Heating EUI (kWh / m2 * year)'
    lk = 'Lighting EUI (kWh / m2 * year)'
    tk = 'Operational (kg CO2e / m2 * year)'

    keys = data_dict['Orientation'].keys()

    for k in keys:
        city = data_dict['City'][k]
        ori = data_dict['Orientation'][k]
        cool = data_dict[ck][k]
        heat = data_dict[hk][k]
        light = data_dict[lk][k]
        tot = data_dict[tk][k]

        city_gwp = data_dict['city_gwp'][k]
        
        if program == ['residential']:
            hw = (3985 / data_dict['area'][k]) * city_gwp
            eq = (3397 / data_dict['area'][k]) * city_gwp
        else:
            hw = 0
            eq = tot - ((cool + heat + light) * city_gwp)

        

        d[city][ori]['cool'] = cool * city_gwp
        d[city][ori]['heat'] = heat * city_gwp
        d[city][ori]['light'] = light * city_gwp
        d[city][ori]['hw'] = hw
        d[city][ori]['eq'] = eq

    orientations = ['n', 'e', 's', 'w']
    heat = {'n':[], 'e':[], 's':[], 'w':[]}
    cool = {'n':[], 'e':[], 's':[], 'w':[]}
    light = {'n':[], 'e':[], 's':[], 'w':[]}
    hw = {'n':[], 'e':[], 's':[], 'w':[]}
    eq = {'n':[], 'e':[], 's':[], 'w':[]}
    for o in orientations:
        for c in cities:
            heat[o].append(d[c][o]['heat'])
            cool[o].append(d[c][o]['cool'])
            light[o].append(d[c][o]['light'])
            hw[o].append(d[c][o]['hw'])
            eq[o].append(d[c][o]['eq'])
    
    # hw = 3985 / data_dict['area'][k]
    # eq = 3397 / data_dict['area'][k]

    bars = []
    for i, o in enumerate(orientations):

        y1 = heat[o]
        y2 = cool[o]
        y3 = light[o]
        y4 = hw[o]
        y5 = eq[o]

        base2 = y1
        base3 = np.add(y1, y2)
        base4 = np.add(base3, y3)
        base5 = np.add(base4, y4)
        base6 = np.add(base5, y5)

        b1 = go.Bar(name= 'heating',
                    x=cities,
                    y=y1,
                    text=np.round(y1, 0),
                    offsetgroup=i,
                    marker_color=colors[1],
                )

        b2 = go.Bar(name='cooling',
                    x=cities,
                    y=y2,
                    text=np.round(y2, 0),
                    offsetgroup=i,
                    base=base2,
                    marker_color=colors[0],
                )

        b3 = go.Bar(name='lighting',
                    x=cities,
                    y=y3,
                    text=np.round(y3,0),
                    offsetgroup=i,
                    base=base3,
                    marker_color=colors[2],
                )
        
        b4 = go.Bar(name='hot water',
                    x=cities,
                    y=y4,
                    text=np.round(y4,0),
                    offsetgroup=i,
                    base=base4,
                    marker_color=colors[5],
        )

        b5 = go.Bar(name='eq',
                    x=cities,
                    y=y5,
                    text=np.round(y5,0),
                    offsetgroup=i,
                    base=base5,
                    marker_color=colors[7],
        )

        b6 = go.Bar(name='total',
                    x=cities,
                    y=[5,5,5,5,5,5],
                    text=np.round(base6,0),
                    offsetgroup=i,
                    base=base6,
                    marker_color='rgba(0,0,0,0)',
                )

        if program == ['residential']:
            bars.extend([b1, b2, b3, b4, b5, b6])
        else:
            bars.extend([b1, b2, b3, b5, b6])

    fig = go.Figure(data=bars, layout=None)

    string = None # 'Envelope carbon emissions'
    fig.update_layout(title={'text':string},
                        hovermode='closest',
                        autosize=False,
                        height=gheight,
                        width=gwidth,
                        paper_bgcolor = 'rgba(0,0,0,0)',
                        plot_bgcolor = 'rgba(0,0,0,0)',
                        showlegend=False,
                    )
    
    

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', title='',
                     gridcolor='lightgray', gridwidth=.1, ticks='outside', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', title='',
                     gridcolor='lightgray', gridwidth=.1, ticks='outside', mirror=True)

    fig.update_yaxes(range=[-10,110])

    fig.update_traces(textfont_color="darkslategray", textposition='inside',
                      textangle=0, insidetextanchor="middle", marker_line_color='rgb(255,255,255)')
    fig.update_layout(uniformtext_minsize=9, uniformtext_mode='hide')

    fig.update(layout_coloraxis_showscale=False)
    fig.write_image(filename, engine='orca')
    # fig.show()  


if __name__ == '__main__':
    import studio2021
    for i in range(50): print('')

    here = os.path.dirname(__file__)

    filepath = os.path.join(here,'assemblies_data_metric.csv')
    frame = pd.read_csv(filepath)


    cities = ['Seattle', 'New York', 'Los Angeles', 'Atlanta', 'Milwaukee', 'San Antonio']
    cities = ['Los Angeles', 'Seattle', 'New York', 'Atlanta', 'San Antonio', 'Milwaukee']


    # program = ['residential']
    # orient = ['n', 's', 'w', 'e']
    # wwr = ['.4']  # ['.0', '.2', '.4', '.6', '.8']
    # glazing = ['Double']
    # ext_thick = ['0', '.5', '1', '2', '3', '4', '5', '6']
    # ext_mat = ['EPS', 'Polyiso']
    # in_thick = ['3.5']  #['0', '3.5', '5.5', '7.25', '9.25', '4', '6', '8']
    # in_mat = ['Fiberglass', 'Cellulose']
    # shgc = ['0.25', '0.6']
    # shading = ['0.0']  # ['0.0', '2.5']
    # inf_rate  = ['0.00059'] # ['0.00059', '0.0003', '0.00015']

    program = ['office']  # ['residential']
    orient = ['n', 's', 'w', 'e']
    wwr = ['.4']  #['.2', '.4', '.6', '.8']  # ['.0', '.2', '.4', '.6', '.8']  # 
    glazing = ['Double']
    ext_thick = ['0']
    ext_mat = ['EPS', 'Polyiso']
    in_thick = ['6']  # ['6'] ['3,6']
    in_mat = ['Fiberglass', 'Cellulose']
    shgc = ['0.25']
    shading = ['0.0']  # ['0.0']  # 
    inf_rate = ['0.00059'] # ['0.00059', '0.0003', '0.00015']


    filename = 'bars_{}.pdf'.format(program[0])
    fp = os.path.join(os.path.dirname(__file__), 'bars', filename)

    plot(frame, cities, program, orient, wwr, glazing, ext_thick, 
        ext_mat, in_thick, in_mat, shgc, shading, inf_rate, filename=fp)
