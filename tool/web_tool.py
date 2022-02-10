import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import xgboost
import datetime

# from studio2021 import streamlit_dictionaries

# load pickled xgboost model to predict EUI
pickle_in = open('xgboost_reg.pkl', 'rb')
regressor = pickle.load(pickle_in)

# dictionaries to translate between user input and prediction input values
infd = {
    'standard': 0.00059,
    'passive house': 0.00015
}
oriend = {
    'North': 1,
    'South': 2,
    'East': 3,
    'West': 4
}
setbackd = {
    'existing': {
        0: {'rear': 0, 'side': 5, 'structure': 5},
        1: {'rear': 5, 'side': 5, 'structure': 5},
        2: {'rear': 0, 'side': 5, 'structure': 5},
        3: {'rear': 5, 'side': 5, 'structure': 5}
    },
    'proposed': {
        0: {'rear': 0, 'side': 5, 'structure': 5},
        1: {'rear': 0, 'side': 5, 'structure': 5},
        2: {'rear': 0, 'side': 5, 'structure': 5},
        3: {'rear': 0, 'side': 5, 'structure': 5}
    }
}
typologyd = {
    '1 unit, 1 story': {'num_units': 1, 'num_stories': 1},
    '1 unit, 2 stories': {'num_units': 1, 'num_stories': 2},
    '2 units, 1 story': {'num_units': 2, 'num_stories': 1}
}

EUI = 0

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

def web_tool():
    st.title('DADU Energy Simulator')
    col1, col2 = st.columns(2)
    col1.header('Results')
    col2.header('Energy Indicators')
    
    if (st.sidebar.button('Predict', key='pred_button')):
        activate = True
    else:
        activate = False
    
    # sidebar sliders
    site = st.sidebar.slider('site', 0, 3, key='site')
    size = st.sidebar.slider('square footage', 100, 1000,
                            value=400, step=10, key='sqft')
    typology = st.sidebar.select_slider('DADU typology',
                                        ['1 unit, 1 story', '1 unit, 2 stories', '2 units, 1 story'], key='typology')
    num_stories = typologyd[typology]['num_stories']
    num_units = typologyd[typology]['num_units']

    inf_rate = infd[st.sidebar.select_slider(
        'infiltration rate', ['standard', 'passive house'], key='inf')]
    orientation = oriend[st.sidebar.select_slider('direction existing to DADU',
                                                ['North', 'South', 'East', 'West'], key='orientation')]
    wwr = st.sidebar.slider('wwr', 0.0, 1.0, key='wwr')
    # frame = framed[st.sidebar.select_slider(
    #     'frame size', ['2x4', '2x6', '2x8', '2x10'], key='frame')]
    polyiso_t = st.sidebar.slider('polyiso_t', 0, 1, step=1, key='polyiso')
    cellulose_t = st.sidebar.slider('cellulose_t', 0, 10, key='cellulose')
    if cellulose_t < 5.5:
        frame = 0
    elif cellulose_t >= 5.5 and cellulose_t < 7.25:
        frame = 1
    elif cellulose_t >= 7.25 and cellulose_t < 9.25:
        frame = 2
    else:
        frame = 3
    
    setback = setbackd[st.sidebar.select_slider('land use setbacks',
                                                ['existing', 'proposed'], key='setback')][site]
    rear_setback = setback['rear']
    side_setback = setback['side']
    structure_setback = setback['structure']

    if num_stories == 1:
        height = 10
        footprint = size
    else:
        height = 20
        footprint = size/2.

    assembly_r = calc_r(polyiso_t, cellulose_t)
    # sur_area =
    # volume = footprint * height
    # surf_vol_ratio = surf_area / volume
    
    # from arcgis import GIS TODO
    # footprint_max = area_buildable from GIS data based on city land use code
    # st.slider('square footage', 100, footprint_max, value=footprint_max/2, step=10, key='sqft')

    # sur_area =
    # volume = footprint * height
    # surf_vol_ratio = surf_area / volume
    # create df from user input
    if activate:
        pred_input = create_input_df(site, size, footprint, height, num_stories, num_units, inf_rate, orientation, wwr,
                                 frame, polyiso_t, cellulose_t, rear_setback, side_setback, structure_setback, assembly_r)

        EUI = predict_eui(pred_input)
        kgCO2e = 0.135669
        CO2 = EUI * kgCO2e * 3.2 * size * 0.09290304
        full_df = pred_input
        full_df['EUI'] = EUI
        full_df['CO2'] = CO2

        rounded_eui = round(float(EUI), 2)
        rounded_co2 = round(float(CO2), 2)
        
        with col1:
            st.metric('Predicted EUI', rounded_eui)
        with col2:
            st.metric('Predicted kgCO2e (operational)', rounded_co2)
        
        csv = convert_df(full_df)
        now = datetime.datetime.now()
        file_name = 'results_' + (now.strftime('%Y-%m-%d_%H_%M')) + '.csv'
        st.download_button('Download Results CSV', data=csv, file_name=file_name)


    #     if EUI == 0:
    #         EUI_last = 0
    #         CO2_last = 0
    #     if EUI != None:
    #         EUI_delta = (EUI - EUI_last) / EUI_last *100
    #         CO2_delta = (CO2 - CO2_last) / CO2_last *100
    #     else:
    #         EUI_delta = None
    #         CO2_delta = None
    #     col1.table(full_df)
    #     with col2:
    #         if EUI_delta == None:
    #             st.metric('Predicted EUI', EUI)
    #             st.metric('Predicted kgCO2e (operational)', CO2)
    #         else:
    #             st.metric('Predicted EUI', EUI, EUI_delta)
    #             st.metric('Predicted kgCO2e (operational)', CO2, CO2_delta)
    
    # if 'EUI' in locals():
    #     print('in locals')
    #     EUI_last = EUI
    #     CO2_last = CO2
    
    
    # download results
    # if 'full_df' in locals():
    #     st.download_button('Download Results File', full_df)
    # else:
    #     st.markdown('please generate prediction')
    
    
    
    #add option to use button to predict instead of real-time
    #add st.metric, show delta from last simulation ran- need to add button then, dont show delta if no button before prediction
    #add download option for output dataframe

if __name__=='__main__':
    for i in range(50): print('')
    web_tool()

# to run: streamlit run /Users/preston/Documents/GitHub/msdc-thesis/tool/web_tool.py
