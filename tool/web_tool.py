import csv
from operator import index
from numpy import full, poly
import streamlit as st
import pandas as pd
import pickle
import plotly as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import xgboost as xgb
import datetime
import make_mesh
from math import sqrt

# import sys
# sys.path.append('presrton/stuff/here')

# from filename import preston_plotter

# load pickled xgboost model to predict EUI
pickle_in = open('xgboost_reg.pkl', 'rb')
regressor = pickle.load(pickle_in)

# load pickled parallel coordinate chart
pickle_in = open('para_coord.pkl', 'rb')
para_coord = pickle.load(pickle_in)

# dictionaries to translate between user input and prediction input values
infd = {
    'Standard': 0.00059,
    'Passive house': 0.00015
}
oriend = {
    'North': 1,
    'South': 2,
    'East': 3,
    'West': 4
}
setbackd = {
    'Existing': 0,
    'Proposed': 1
}
typologyd = {
    '1 Unit, 1 Story': {'num_units': 1, 'num_stories': 1},
    '1 Unit, 2 Stories': {'num_units': 1, 'num_stories': 2},
    '2 Units, 1 Story': {'num_units': 2, 'num_stories': 1}
}
sited = {
    'Corner with alley':0,
    'Corner without alley':1,
    'Infill with alley':2,
    'Infill without alley':3
}
plotd = {
    'Lot type': 'site',
    'Infiltration rate': 'inf_rate',
    'Orientation': 'orientation',
    'Setbacks': 'setback',
    'Floor area': 'size',
    'WWR': 'wwr',
    'R-assembly': 'assembly_r',
    'EUI': 'eui_kbtu',
    'CO2': 'annual_carbon',
    'Cost': 'annual_cost'
}

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
 
def calc_r(polyiso_t, cellulose_t):
    """Calculates wall assembly R value.

    Args:
        polyiso_t (float): thickness of polyiso insulation (inches)
        cellulose_t (float): thickness of cellulose insulation (inches)

    Returns:
        assembly_r (float): wall assembly R value
    """
    air_gap = 1.0
    cladding = 0.61 #aluminum
    ply = 0.63
    ext_air_film = .17
    int_air_film = .68
    
    assembly_r = ext_air_film + cladding + air_gap + polyiso_t*7 + ply + cellulose_t*3.5 + int_air_film
    
    return assembly_r

def create_input_df(site, size, footprint, height, num_stories, num_units, inf_rate, orientation, wwr, 
                    setback, assembly_r, surf_vol_ratio):
    """Takes user input from Streamlit sliders and creates 1D dictionary from variables, then converts to DataFrame

    Args:
        site (int): site to simulate, currently 0-3 and modeled in Grasshopper TODO: allow for selection of site type by user
        size (int): total square footage of DADU
        footprint (int): first floor square footage of DADU; footprint == size for 1 story DADU
        height (int): height of DADU; 10 for 1 story, 20 for 2
        num_stories (int): number of stories (1,2)
        num_units (int): number of units (1,2); cannot have 2 units in 2 story DADU
        inf_rate (float): infiltration rate (standard or passive house)
        orientation (int): orientation of existing house to DADU (0,1,2,3; N,S,E,W)
        wwr (float): window-to-wall ratio (0.0<=1.0)
        setback (int): existing or revised, lax lot setbacks
        assembly_r (float): total r value of wall assembly
        surf_vol_ratio (float): ratio of total surface area to volume

    Returns:
        pred_input (DataFrame): dataframe of shape (1,15)
    """
    inputs = {
        'site': [site], 'size': [size], 'footprint': [footprint], 'height': [height], 'num_stories': [num_stories], 'num_units': [num_units],
        'inf_rate': [inf_rate], 'orientation': [orientation], 'wwr': [wwr], 'setback': [setback], 'assembly_r': [assembly_r], 
        'surf_vol_ratio': [surf_vol_ratio]
    }
    pred_input = pd.DataFrame(inputs)
    print(pred_input.shape)
    return pred_input

def predict_eui(pred_input):
    """Predicts energy use intensity of DADU from user input.

    Args:
        pred_input (DataFrame): DataFrame of shape (1,15) containing user input for prediction

    Returns:
        prediction (float): energy use intensity (EUI; kBTU/ft^2)
    """
    pred_inputDM = xgb.DMatrix(pred_input)
    prediction = regressor.predict(pred_inputDM)
    return prediction

def user_favorites(results, count, favorites): #TODO
    """Takes user's list of favorites appends results of each to new dataframe. Allows user to download csv 
    with all of their top picks.

    Args:
        full_df (DataFrame): DataFrame containing results of favorited simulation
        favorites (List): List containing all of the runs which are flagged as favorites
    """
    fav_df = pd.DataFrame()
    for row in favorites:
        df = df.append(row, ignore_index=True)
    return fav_df

def percent_change(old, new):
    pc = round((new - old) / abs(old) * 100, 2)
    return pc
    
def web_tool():
    st.title('DADU Energy Simulator')
    col1, col2 = st.columns([1, 2])

    with col1:
        with st.expander('Documentation'):
            st.markdown('asdf \n asdogihasodhgaspodighaosdhaiosdgaips \n' +
                        'asidubgaiosudbgaosdngoiaSDDG \n aosidhgaoishdgoaisdgpia \n' +
                        'asduigahsdoiughasoidghasipdgasdg') # TODO
        st.header('Results')
    if 'results' not in st.session_state:
        st.session_state.results = pd.DataFrame()
        
    count = len(st.session_state.results.index)

    # if 'favorites' not in st.session_state:
    #     st.session_state.favorites = pd.DataFrame() #TODO
    
    # constants
    kgCO2e = .135669
    kwh_cost = .1189
    
    if count >= 1:
        rounded_eui = st.session_state.results.iat[count - 1, st.session_state.results.columns.get_loc('eui_kbtu')]
        rounded_eui_kwh = st.session_state.results.iat[count - 1, st.session_state.results.columns.get_loc('eui_kwh')]
        rounded_co2 = st.session_state.results.iat[count - 1, st.session_state.results.columns.get_loc('annual_carbon')]
        rounded_cost = st.session_state.results.iat[count - 1, st.session_state.results.columns.get_loc('annual_cost')]
    
    # create sidebar form
    st.sidebar.header('Prediction Input')
    with st.sidebar.form(key='user_input'):
        # sidebar dropdowns
        site = st.selectbox('Lot type', options=['Corner with alley', 'Corner without alley', 
                                                     'Infill with alley', 'Infill without alley'], 
                                index=3, help='Select the type of lot that your existing dwelling belongs to')
        typology = st.selectbox('DADU typology', ['1 Unit, 1 Story', '1 Unit, 2 Stories', 
                                        '2 Units, 1 Story'], index=0, 
                                        help='Select the number of stories and units')
        inf_rate = infd[st.selectbox('Infiltration rate (cubic m/s?)', # double check units TODO 
                                             ['Standard', 'Passive house'], 
                                             help='Select either standard infiltration rate or passive house (extremely tight enclosure)',
                                             index=0, key='inf')]
        orientation = oriend[st.selectbox('Orientation',
                                                  ['North', 'South', 'East', 'West'], 
                                                  help='Select the direction of existing dwelling to DADU',
                                                  key='orientation')]
        setback = setbackd[st.selectbox('Land use setbacks',
                                                ['Existing', 'Proposed'], 
                                                help='Select either existing (2022), or proposed (more lenient) setbacks',
                                                key='setback')] 
        
        # sidebar sliders
        size = st.slider('Total floor area (sqft)', 100, 1000,
                                value=400, 
                                help='Select the total square footage of floor (maximum floor area per Seattle code is 1000sqft)',
                                step=10, key='sqft')
        wwr = st.slider('Window-to-wall ratio (WWR)', 0.0, 0.9, 
                                help='Window to wall ratio is the ratio between glazing (window) surface area and opaque surface area',
                                value=.4, key='wwr')
        polyiso_t = st.slider('Polyiso insulation depth (inches)', 0.0, 1.0, 
                                      help='Select amount of polyiso insulation in wall assembly',
                                      step=.25, value=.75, key='polyiso')
        cellulose_t = st.slider('Cellulose insulation depth (inches)', 0.0, 10.0, 
                                        help='Select amount of cellulose insulation in wall assembly',
                                        step=.5, value=8.0, key='cellulose')
            
        site = sited[site]

        num_stories = typologyd[typology]['num_stories']
        num_units = typologyd[typology]['num_units']
        if num_stories == 1:
            height = 10
            footprint = size
        else:
            height = 20
            footprint = size / 2.

        assembly_r = round(float(calc_r(polyiso_t, cellulose_t)), 2)
        
        length = sqrt(footprint)
        volume = (length ** 2) * height 
        surf_area = ((length ** 2) * 2) + (4 * (length * height))
        surf_vol_ratio = surf_area / volume
        
        # show r-assembly value
        # st.write('R-assembly:', str(assembly_r), '(units)') #TODO
        
        # submit user prediction
        activate = st.form_submit_button(label='Predict', 
                            help='Click \"Predict\" once you have selected your desired options')
        # if st.button('Favorite', help=
        #     'Add to list of favorite combinations to easily return to result'):
        # csv_favs = convert_df(user_favorites(results, count))
        # pass #TODO
    
    with st.sidebar:    
        now = datetime.datetime.now()
        file_name_all = 'results_' + (now.strftime('%Y-%m-%d_%H_%M')) + '.csv'
        csv_all = convert_df(st.session_state.results)
        st.download_button('Download All Results',
                        data=csv_all, file_name=file_name_all,
                        help='Download a .CSV spreadsheet with all simulation data from current session')

        # file_name_favs = 'favorites_' + \
        #     (now.strftime('%Y-%m-%d_%H_%M')) + '.csv'  # TODO
        # csv_favs = convert_df(st.session_state.favorites)
        # st.download_button('Download Favorited Results',
        #                 data=csv_favs, file_name=file_name_favs)
        
        # clear results
        clear_res = st.button('Clear results',
                              help='Delete all previous prediction data from current session')
        advanced_toggle = st.checkbox('Advanced view',
                                    help='Enables advanced user view')
            
    if activate:
        count = len(st.session_state.results.index) + 1
            
        pred_input = create_input_df(site, size, footprint, height, num_stories, num_units, inf_rate, orientation, wwr,
                                setback, assembly_r, surf_vol_ratio)

        eui = predict_eui(pred_input)
        co2 = eui * 3.2 * size * 0.09290304 * kgCO2e 
        cost = eui * 3.2 * size * 0.09290304 * kwh_cost / 12 
        eui_kwh = eui * 3.2
        
        rounded_eui = round(float(eui), 2)
        rounded_eui_kwh = round(float(eui_kwh), 2)
        rounded_co2 = round(float(co2), 2)
        rounded_cost = round(float(cost), 2)
        
        outcomes_dict = {
            'site': site,
            'size': size,
            'footprint': footprint,
            'height': height,
            'num_stories': num_stories,
            'num_units': num_units,
            'inf_rate': inf_rate,
            'orientation': orientation,
            'wwr': wwr,
            'setback': setback,
            'assembly_r': assembly_r,
            'surf_vol_ratio': surf_vol_ratio,
            'eui_kwh': rounded_eui_kwh,
            'eui_kbtu': rounded_eui,
            'annual_carbon': rounded_co2,
            'annual_cost': rounded_cost
        }
        outcomes = pd.DataFrame(outcomes_dict, index=[0])
        st.session_state.results = st.session_state.results.append(outcomes, ignore_index=True)
        
    with col1:
        if count == 0:
            st.metric('Predicted EUI', ' ')
            st.write('\n' + '\n')
            st.metric('Predicted Operational Carbon', ' ')
            st.write('\n' + '\n')
            st.metric('Predicted monthly energy cost', ' ') 
            st.write('\n' + '\n')
            
        if count == 1:
            eui_kwh = rounded_eui * 3.2
            rounded_eui_kwh = round(float(eui_kwh), 2)
            st.metric('Predicted EUI', str(rounded_eui) + ' kBTU/sqft')
            st.metric('Predicted Operational Carbon', str(rounded_co2) + ' kgCO2')
            st.metric('Predicted monthly energy cost', '$' + str(rounded_cost))  
            
        if count > 1:
            eui_kwh = rounded_eui * 3.2
            rounded_eui_kwh = round(float(eui_kwh), 2)
            
            d_eui_kbtu = percent_change(
                st.session_state.results.iat[count - 2, st.session_state.results.columns.get_loc('eui_kbtu')], rounded_eui)
            d_carbon = percent_change(
                st.session_state.results.iat[count - 2, st.session_state.results.columns.get_loc('annual_carbon')], rounded_co2)
            d_cost = percent_change(
                st.session_state.results.iat[count - 2, st.session_state.results.columns.get_loc('annual_cost')], rounded_cost)

            st.metric('Predicted EUI', str(rounded_eui) + ' kBTU/sqft', delta=str(d_eui_kbtu) + ' %', delta_color='inverse')
            st.metric('Predicted annual operational carbon', str(rounded_co2) + ' kgCO2', delta=str(d_carbon) + ' %', delta_color='inverse')
            st.metric('Predicted monthly energy cost', '$' + str(rounded_cost), delta=str(d_cost) + ' %', delta_color='inverse')  
        
        
    # model viewer    
    with col2:    
        mesh = make_mesh.make_mesh(size, wwr, num_stories, num_units)
        st.plotly_chart(mesh, use_container_width=True)
        
    # under advanced, allow use to see their prediction plotted against all simulated data for validation
    # plot with selectable axes to compare results // pareto frontier
    # allow hover to show design parameters if unable to set sliders and things to input values when result is selected
    
    with st.container():
        # st.subheader('Plot options:')
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1: 
            x_axis_data = st.selectbox('X-Axis', options=['Lot type', 'Infiltration rate', 'Orientation',
                                    'Setbacks', 'Floor area', 'WWR', 'R-assembly'], index=6, help='Select data feature to display on X axis')   
        with s_col2:
            y_axis_data = st.selectbox('Y-Axis', options=['EUI', 'CO2', 'Cost'], index=0, help='Select data feature to display on Y axis')
        
        with s_col3:
            colorby = st.selectbox('Color by', options=['Lot type', 'Infiltration rate', 'Orientation',
                                    'Setbacks', 'Floor area', 'WWR', 'R-assembly'], help='Select data feature to color markers by')                    
            
        if count > 0:
            
            # st.write('haha')
            x = st.session_state.results[plotd[x_axis_data]]
            y = st.session_state.results[plotd[y_axis_data]]
            color = st.session_state.results[plotd[colorby]]
            
            scatter = go.Scattergl(x=x, 
                                 y=y,
                                 marker_color=color,
                                 text=color,
                                 mode='markers',
                                #  hovertemplate='wwr: %{wwr}, floor area: %{size}',
                                 marker= {
                                     'size': 12,
                                     'colorscale': 'Viridis',
                                     'showscale': True
                                 }
                                 )
            fig = go.Figure(data=scatter)
            
            fig.update_xaxes(title_text=x_axis_data)
            fig.update_yaxes(title_text=y_axis_data)
            
            fig.update_layout(hovermode='closest',
                                clickmode='event',
                                margin={'pad':10,
                                        'l':50,
                                        'r':50,
                                        'b':50,
                                        't':50},
                                font=dict(
                                    size=18,
                                    color="black")
                                )
            st.plotly_chart(fig, use_container_width=True)
            
            
            # line_results = st.session_state.results['eui_kbtu']
            # line_results_carbon = st.session_state.results['annual_carbon']
            
            # double = make_subplots(specs=[[{"secondary_y": True}]])
            # double.add_trace(go.Scatter(x=list(range(line_results.shape[0])),
            #                             y=line_results,
            #                             hovertext='kBTU/ft2',
            #                             hoverinfo='y+text',
            #                             marker={'size': 14},
            #                             mode='lines+markers',
            #                             # marker_symbol='line-ns',
            #                             # marker_line_width=2,
            #                             # name='EUI (kBTU/ft2)'
            #                             ),
            #                 secondary_y=False
            #                 )
            
            # double.add_trace(go.Scatter(x=list(range(line_results_carbon.shape[0])),
            #                         y=line_results_carbon,
            #                         hovertext='kgCO2',
            #                         hoverinfo='y+text',
            #                         marker={'size': 14,
            #                                 'color': 'red'},
            #                         mode='lines+markers',
            #                         # marker_symbol='line-ns',
            #                         # marker_line_width=2,
            #                         # name='EUI (kBTU/ft2)'
            #                         ),
            #                 secondary_y=True
            #                 )
            
            # double.update_xaxes(title_text='Result History')
            # double.update_yaxes(title_text='EUI (kBTU/sqft)', secondary_y=False)
            # double.update_yaxes(title_text='kgCO2 (annual)', secondary_y=True)
            
            # double.update_layout(
            #     hovermode='x unified',
            #     clickmode='event',
            #     margin={'pad':10,
            #             'l':50,
            #             'r':50,
            #             'b':100,
            #             't':100},
            #     font=dict(
            #         size=18,
            #         color="black"
            #     )
            # )
            
            # st.plotly_chart(double, use_container_width=True)
            
    if clear_res:
        st.session_state.results = st.session_state.results[0:0]

    if advanced_toggle:
        st.write(st.session_state.results)
            
# /Users/preston/Documents/GitHub/msdc-thesis/tool/results
st.set_page_config(layout='wide')

if __name__=='__main__':
    for i in range(50): print('')
    web_tool()

# to run: streamlit run /Users/preston/Documents/GitHub/msdc-thesis/tool/web_tool.py 

# TODO
# ask tomas about discrepency of EUI between 1 and 2 story with everything else constant
# advanced view: show dataframe, hover info (SA and glaz SA on model), 3d pareto??
# implement favorites feature and downloading of favorites