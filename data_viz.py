from matplotlib.pyplot import inferno
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
# import pickle

# load pickled dataframe to visualize
# pickle_in = open('energy_data.pkl', 'rb')
energy_data = pd.read_csv('all_sims.csv')
# labels_drop = ['filename', 'num_adiabatic', 'rear_setback', 'side_setback', 
#                 'structure_setback', 'area_buildable', 'area_buildable', 'surf_tot',
#                 'surf_glaz', 'surf_opaq', 'volume', 'surf_vol_ratio', 'cooling', 'heating', 
#                 'lighting', 'equipment', 'water', 'eui_kwh', 'carbon', 'kg_CO2e']
# energy_data.drop(labels=labels_drop, axis=1, inplace=True)

# energy_data.to_csv('energy_data.csv', sep=',')
# print(energy_data.head())
# print(energy_data.shape)
energy_data = energy_data.dropna(how='any',axis=0)
energy_data.drop(energy_data.index[energy_data['eui_kbtu'] < 20], inplace=True)
print(energy_data.head())
print(energy_data.shape)
# print(energy_data['eui_kbtu'].min())
# print(energy_data['eui_kbtu'].max())
# print(energy_data['size'].min())
# print(energy_data['size'].max())
# print(energy_data['footprint'].min())
# print(energy_data['footprint'].max())
# print(energy_data['eui_kbtu'].min())
# print(energy_data['eui_kbtu'].max())


# 1867 missing ['surf_glaz'], ['surf_opaq']


# fig = px.parallel_coordinates(energy_data, 
#                             color="eui_kbtu", 
#                             labels={
#                                     "site": "Site",
#                                     "size": "Size (sqft)", 
#                                     "footprint": "Footprint (sqft)",
#                                     "height": "Height (ft)", 
#                                     "num_stories": "Number of stories", 
#                                     "num_units": "Number of units",
#                                     "inf_rate": "Infiltration rate",
#                                     "orientation": "Orientation to existing",
#                                     "wwr": "WWR",
#                                     "frame": "Frame",
#                                     "polyiso_t": "Polyiso Insulation (in)",
#                                     "cellulose_t": "Cellulose Insulation (in)",
#                                     "setback": "Setbacks",
#                                     "assembly_r": "Wall Assembly R",
#                             },
#                             color_continuous_scale=px.colors.diverging.Tealrose,
#                             color_continuous_midpoint=2)
# fig.show()


para_coord = go.Figure(data=go.Parcoords(
    line=dict(color=energy_data['eui_kbtu'],
              colorscale='Viridis',
              showscale=False,
              cauto=True
              ),
    dimensions=list([
        dict(range=[-.25, 3.25],
             tickvals=[-.25, 0, 1, 2, 3, 3.25],
             ticktext=[' ', 0, 1, 2, 3, ' '],
             label="Site", values=energy_data['site']),
        dict(range=[-.25, 1.25],
             tickvals=[-.25, 0, 1, 1.25],
             ticktext=[' ', 'Existing', 'Proposed', ' '],
             label='Setback', values=energy_data['setback']),
        dict(range=[.75, 2.25],
             tickvals=[.75, 1, 2, 2.25],
             ticktext=[' ', 1, 2, ' '],
             label='# Stories', values=energy_data['num_stories']),
        dict(range=[.75, 2.25],
             tickvals=[.75, 1, 2, 2.25],
             ticktext=[' ', 1, 2, ' '],
             label='# Units', values=energy_data['num_units']),
        dict(range=[0, 1000],
             tickvals=[0, 200, 400, 600, 800, 1000],
             label='Footprint', values=energy_data['footprint']),
        dict(range=[0, 1000],
             tickvals=[0, 200, 400, 600, 800, 1000],
             label='Size', values=energy_data['size']),
        dict(range=[.75, 4.25],
             tickvals=[.75, 1, 2, 3, 4, 4.25],
             ticktext=[' ', 'N', 'S', 'E', 'W', ' '],
             label='Orientation', values=energy_data['orientation']),
        dict(range=[0, 50],
             tickvals=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
             ticktext=[' ', 5, 10, 15, 20, 25, 30, 35, 40, 45, ' '],
             label='R-Assembly', values=energy_data['assembly_r']),
        dict(range=[0, 1],
             tickvals=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
             ticktext=[' ', .1, .2, .3, .4, .5, .6, .7, .8, .9, ' '],
             label='WWR', values=energy_data['wwr']),
        dict(range=[0, .001],
             tickvals=[0, .0001, .0002, .0003, .0004, .0005, .0006, .0007,
                       .0008, .0009, .001],
             ticktext=[' ', .0001, .0002, .0003, .0004, .0005, .0006, .0007,
                       .0008, .0009, ' '],
             #  tickvals=[0, .00015, .00059, .001],
             label='Inf. Rate', values=energy_data['inf_rate'])
     #    dict(range=[50, 160],
     #         tickvals=[45, 50, 60, 70, 80, 90, 100, 
     #                   110, 120, 130, 140, 150, 160],
     #         ticktext=[' ', 50, 60, 70, 80, 90, 100,
     #                   110, 120, 130, 140, 150, ' '],
     #         label="EUI (kBTU/sqft)", values=energy_data['eui_kbtu'])
        ])
    )
)
para_coord.update_layout(
    autosize=False,
    width=1920/2,
    height=850/2,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4),
    # paper_bgcolor='Black',
    # plot_bgcolor='Black',
    title=dict(
        text='Simulation Inputs')
)

# pickle_out = open('para_coord.pkl','wb')
# pickle.dump(para_coord, pickle_out)
# pickle_out.close()
    
para_coord.show()

# para_cats = px.parallel_categories(energy_data, 
#             dimensions=['site', 'orientation', 'size', 'footprint', 'num_stories',
#                         'num_units', 'inf_rate', 'wwr', 'setback'],
#             color="eui_kbtu", color_continuous_scale=px.colors.sequential.Inferno)
#             # labels={'sex':'Payer sex', 
#             #         'smoker':'Smokers at the table', 
#             #         'day':'Day of week'})
# para_cats.show()
