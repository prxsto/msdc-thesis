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
import energy_model as em


# from studio2021 import streamlit_dictionaries

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
    'Existing': {
        'Infill with alley': {'rear': 0, 'side': 5, 'structure': 5},
        'Infill without alley': {'rear': 5, 'side': 5, 'structure': 5},
        'Corner with alley': {'rear': 0, 'side': 5, 'structure': 5},
        'Corner without alley': {'rear': 5, 'side': 5, 'structure': 5}
    },
    'Proposed': {
        'Infill with alley': {'rear': 0, 'side': 5, 'structure': 5},
        'Infill without alley': {'rear': 0, 'side': 5, 'structure': 5},
        'Corner with alley': {'rear': 0, 'side': 5, 'structure': 5},
        'Corner without alley': {'rear': 0, 'side': 5, 'structure': 5}
    }
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
                    frame, polyiso_t, cellulose_t, rear_setback, side_setback, structure_setback, assembly_r):
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
        frame (int): wood frame type (0,1,2,3; 2x4, 2x6, 2x8, 2x10)
        polyiso_t (float): thickness of polyiso insulation (0<=1; inches)
        cellulose_t (float): thickness of cellulose insulation (0<=10; inches)
        rear_setback (int): rear setback; minimum distance between DADU and rear lot line
        side_setback (int): side setback; minimum distance between DADU and side lot lines
        structure_setback (int): structure setback; minimum distance between DADU and closest part of existing house
        assembly_r (float): total r value of wall assembly

    Returns:
        pred_input (DataFrame): dataframe of shape (1,15)
    """
    inputs = {
        'site': [site], 'size': [size], 'footprint': [footprint], 'height': [height], 'num_stories': [num_stories], 'num_units': [num_units],
        'inf_rate': [inf_rate], 'orientation': [orientation], 'wwr': [wwr], 'frame': [frame], 'polyiso_t': [polyiso_t], 'cellulose_t': [cellulose_t], 
        'rear_setback': [rear_setback], 'side_setback': [side_setback], 'structure_setback': [structure_setback], 'assembly_r': [assembly_r]
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
    prediction = regressor.predict(pred_input)
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
    # inital values 
    kgCO2e = 0.135669 
    kwh_cost = .1189
    init_eui = 125.23
    init_eui_kwh = 400.74
    init_co2 = 2020.41
    init_cost = 1770.69
    
    # count = 0
    
    entry_dict = {
        'site': 3,
        'size': 400,
        'footprint': 400,
        'height': 10,
        'num_stories': 1,
        'num_units': 1,
        'inf_rate': .00059,
        'orientation': 1,
        'wwr': .4,
        'frame': 2,
        'polyiso_t': .75,
        'cellulose_t': 8,
        'rear_setback': 0,
        'side_setback': 5,
        'structure_setback': 5,
        'assembly_r': 36.34,
        'eui_kwh': init_eui_kwh,
        'eui_kbtu': init_eui,
        'annual_carbon': init_co2,
        'annual_cost': init_cost
    }
    
    if 'results' not in st.session_state:
        st.session_state.results = pd.DataFrame()
        # st.session_state.results = st.session_state.results.append(
        #     entry_dict, ignore_index=True)
    if 'favorites' not in st.session_state:
        st.session_state.favorites = pd.DataFrame()
        
    if ' count' not in st.session_state:
        st.session_state.count = 0
        
    def increment_counter():
        st.session_state.count += 1
    
    st.title('DADU Energy Simulator')
    col1, col2 = st.columns(2)
    col1.header('Results')
    col2.header('Options')
    # col3.header('3D Viewer')
        
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
                                                key='setback')][site]        
        
        # sidebar sliders
        size = st.slider('Total floor area (sqft)', 100, 1000,
                                value=400, 
                                help='Select the total square footage of floor (maximum floor area per Seattle code is 1000sqft)',
                                step=10, key='sqft')
        wwr = st.slider('Window-to-wall ratio (WWR)', 0.0, 1.0, 
                                help='Window to wall ratio is the ratio between glazing (window) surface area and opaque surface area',
                                value=.4, key='wwr')
        polyiso_t = st.slider('Polyiso insulation depth (inches)', 0.0, 1.0, 
                                      help='Select amount of polyiso insulation in wall assembly',
                                      step=.25, value=.75, key='polyiso')
        cellulose_t = st.slider('Cellulose insulation depth (inches)', 0.0, 10.0, 
                                        help='Select amount of cellulose insulation in wall assembly',
                                        step=.5, value=8.0, key='cellulose')

        if cellulose_t < 5.5:
            frame = 0
        elif cellulose_t >= 5.5 and cellulose_t < 7.25:
            frame = 1
        elif cellulose_t >= 7.25 and cellulose_t < 9.25:
            frame = 2
        else:
            frame = 3
            
        site  = sited[site]
        
        rear_setback = setback['rear'] 
        side_setback = setback['side']
        structure_setback = setback['structure']

        num_stories = typologyd[typology]['num_stories']
        num_units = typologyd[typology]['num_units']
        if num_stories == 1:
            height = 10
            footprint = size
        else:
            height = 20
            footprint = size / 2.

        assembly_r = round(float(calc_r(polyiso_t, cellulose_t)), 2)
        
        # submit user prediction
        activate = st.form_submit_button(label='Predict', 
                            help='Click \"Predict\" once you have selected your desired options', on_click=increment_counter)
        
    if activate:
        increment_counter()
        pred_input = create_input_df(site, size, footprint, height, num_stories, num_units, inf_rate, orientation, wwr,
                                frame, polyiso_t, cellulose_t, rear_setback, side_setback, structure_setback, assembly_r)

        eui = predict_eui(pred_input)
        co2 = eui * 3.2 * size * 0.09290304 * kgCO2e #TODO check these calculations
        cost = eui * 3.2 * size * 0.09290304 * kwh_cost / 12 #TODO check these calculations also
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
            'frame': frame,
            'polyiso_t': polyiso_t,
            'cellulose_t': cellulose_t,
            'rear_setback': rear_setback,
            'side_setback': side_setback,
            'structure_setback': structure_setback,
            'assembly_r': assembly_r,
            'eui_kwh': rounded_eui_kwh,
            'eui_kbtu': rounded_eui,
            'annual_carbon': rounded_co2,
            'annual_cost': rounded_cost
        }
        outcomes = pd.DataFrame(outcomes_dict, index=[0])
        st.session_state.results = st.session_state.results.append(outcomes, ignore_index=True)
        # st.write(st.session_state.results)
        
    with col2:
        if st.button('Favorite', help=
            'Add to list of favorite combinations to easily return to result'):
            # csv_favs = convert_df(user_favorites(results, count))
            pass #TODO
        
        now = datetime.datetime.now()
        file_name_all = 'results_' + (now.strftime('%Y-%m-%d_%H_%M')) + '.csv'
        csv_all = convert_df(st.session_state.results)
        st.download_button('Download All Results',
                           data=csv_all, file_name=file_name_all)

        file_name_favs = 'favorites_' + \
            (now.strftime('%Y-%m-%d_%H_%M')) + '.csv'  # TODO
        csv_favs = convert_df(st.session_state.favorites)
        st.download_button('Download Favorited Results',
                           data=csv_favs, file_name=file_name_favs)
        
        # clear results
        clear_res = st.button('Clear results')
        show_dataframe = st.checkbox('Show dataframe', value=False)
        advanced_toggle1 = st.checkbox('Toggle advanced view',
                                      help='Enables advanced user view') #TODO

    if clear_res:
        # st.session_state.results = pd.DataFrame()
        # last_row = st.session_state.results.iloc[len(st.session_state.results.index) - 1:, :]
        st.session_state.results = st.session_state.results.loc[0]
        # st.session_state.results = st.session_state.results.append
        #     entry_dict, ignore_index=True)

    if show_dataframe:
        st.write(st.session_state.results)
        
    with col1:
        if not activate:
        # if count == 0:
            # st.metric('Predicted EUI', str(init_eui) + ' kBTU/sqft ' + str(init_eui_kwh) + ' kWh/m2')
            # st.metric('Predicted EUI', str(init_eui) + ' kBTU/sqft')
            # st.metric('Predicted Operational Carbon', str(init_co2) + ' kgCO2')
            # st.metric('Predicted monthly energy cost', '$' + str(init_cost / 12)) 
            st.metric('Predicted EUI', ' ')
            st.write('\n')
            st.write('\n')
            st.metric('Predicted Operational Carbon', ' ')
            st.write('\n')
            st.write('\n')
            st.metric('Predicted monthly energy cost', ' ') 
            st.write('\n')
            st.write('\n')
            st.write(st.session_state.count)    
        # elif count > 0:
        if activate:
            if st.session_state.count <= 1:
                eui_kwh = rounded_eui * 3.2
                rounded_eui_kwh = round(float(eui_kwh), 2)
                # st.metric('Predicted EUI', str(rounded_eui) + ' kBTU/sqft ' + str(rounded_eui_kwh) + ' kWh/m2')
                st.metric('Predicted EUI', str(rounded_eui) + ' kBTU/sqft')
                st.metric('Predicted Operational Carbon', str(rounded_co2) + ' kgCO2')
                st.metric('Predicted monthly energy cost', '$' + str(rounded_cost))  
                st.write(st.session_state.count)
            elif st.session_state.count > 1:
                eui_kwh = rounded_eui * 3.2
                rounded_eui_kwh = round(float(eui_kwh), 2)
                
                d_eui_kbtu = percent_change(
                    st.session_state.results.iat[st.session_state.count, st.session_state.results.columns.get_loc('eui_kbtu')], rounded_eui)
                # d_eui_kwh = percent_change(
                #     st.session_state.results.iat[st.session_state.count, st.session_state.results.columns.get_loc('eui_kwh')], rounded_eui_kwh)
                d_carbon = percent_change(
                    st.session_state.results.iat[st.session_state.count, st.session_state.results.columns.get_loc('annual_carbon')], rounded_co2)
                d_cost = percent_change(
                    st.session_state.results.iat[st.session_state.count, st.session_state.results.columns.get_loc('annual_cost')], rounded_cost)
                
                # st.metric('Predicted EUI', str(rounded_eui) + ' kBTU/sqft ' + str(rounded_eui_kwh) + ' kWh/m2')
                st.metric('Predicted EUI', str(rounded_eui) + ' kBTU/sqft', delta=d_eui_kbtu, delta_color='inverse')
                st.metric('Predicted Operational Carbon', str(rounded_co2) + ' kgCO2', delta=d_carbon, delta_color='inverse')
                st.metric('Predicted monthly energy cost', '$' + str(rounded_cost), delta=d_cost, delta_color='inverse')  
                st.write(st.session_state.count)
    # with col3:

    advanced_toggle=True
    if advanced_toggle:
        with st.container():
            if activate:

                line_results = st.session_state.results['eui_kbtu']
                line_results_carbon = st.session_state.results['annual_carbon']
                
                double = make_subplots(specs=[[{"secondary_y": True}]])
                double.add_trace(go.Scatter(x=list(range(line_results.shape[0])),
                                            y=line_results,
                                            hovertext='kBTU/ft2',
                                            hoverinfo='y+text',
                                            marker={'size': 14},
                                            mode='lines+markers',
                                            # marker_symbol='line-ns',
                                            # marker_line_width=2,
                                            # name='EUI (kBTU/ft2)'
                                            ),
                                secondary_y=False
                                )
                
                double.add_trace(go.Scatter(x=list(range(line_results_carbon.shape[0])),
                                        y=line_results_carbon,
                                        hovertext='kgCO2',
                                        hoverinfo='y+text',
                                        marker={'size': 14,
                                                'color': 'red'},
                                        mode='lines+markers',
                                        # marker_symbol='line-ns',
                                        # marker_line_width=2,
                                        # name='EUI (kBTU/ft2)'
                                        ),
                                secondary_y=True
                                )
                
                double.update_xaxes(title_text='Result History')
                double.update_yaxes(title_text='EUI (kBTU/sqft)', secondary_y=False)
                double.update_yaxes(title_text='kgCO2 (annual)', secondary_y=True)
                
                double.update_layout(
                    # title={'text':'Result History',
                    #        'x': .5,
                    #        'xanchor':'center'
                    # },
                    hovermode='x unified',
                    clickmode='event',
                    margin={'pad':10,
                            'l':50,
                            'r':50,
                            'b':100,
                            't':100},
                    font=dict(
                        size=18,
                        color="black"
                    )
                )
                # scatter = double.data[0]
                
                # # define callback function to return to prediction at point
                # def callback_predict(trace, points, selector):
                #     ind = scatter.marker.x
                #     prev_predict = st.session_state.results.loc(ind)
                #     predict_eui(prev_predict)
                #     st.write([trace, points, selector])
                
                # # callback function to return to selected value's prediction
                # scatter.on_click(callback_predict)

                # display = go.Figure(scatter)
                # display.update_layout(
                #     # title='Result History', 
                #     xaxis_title='Result',
                #     yaxis_title='EUI (kBTU/sqft)',
                #     hovermode='closest',
                #     clickmode='event',
                #     font=dict(
                #         size=18,
                #         color="black"
                #         )
                #     )  
                # carbon_chart.update_layout(
                #     xaxis_title='Result',
                #     yaxis_title='Annual Carbon (kgCO2)',
                # )
                
                st.plotly_chart(double, use_container_width=True)

st.set_page_config(layout='wide')
    
#TODO add st.metric, show delta from last simulation ran- need to add button then, dont show delta if no button before prediction

if __name__=='__main__':
    for i in range(50): print('')
    web_tool()

# to run: streamlit run /Users/preston/Documents/GitHub/msdc-thesis/tool/web_tool.py
