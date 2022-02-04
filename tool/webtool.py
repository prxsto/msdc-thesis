import streamlit as st
import pandas as pd
import numpy as np
# from studio2021 import streamlit_dictionaries

def calc_r(polyiso_t, cellulose_t):
    air_gap = 1.0
    cladding = 0.61 #aluminum
    ply = 0.63
    
    assembly_r = cladding + air_gap + polyiso_t*7 + ply + cellulose_t*3.5 #shouldn't it include air films???
    
    return assembly_r

infd = {
    'standard':0.00059,
    'passive house':0.00015
}

oriend = {
    'North':1,
    'South':2,
    'East':3,
    'West':4
}

framed = {
    '2x4':0,
    '2x6':1,
    '2x8':2,
    '2x10':3
}

setbackd = {
    'existing':{
        0: {'rear':0, 'side':5, 'structure':5},
        1: {'rear':5, 'side':5, 'structure':5},
        2: {'rear':0, 'side':5, 'structure':5},
        3: {'rear':5, 'side':5, 'structure':5}
    },
    'proposed':{
        0: {'rear':0, 'side':5, 'structure':5},
        1: {'rear':0, 'side':5, 'structure':5},
        2: {'rear':0, 'side':5, 'structure':5},
        3: {'rear':0, 'side':5, 'structure':5}
    }
}

typologyd = {
    '1 unit, 1 story': {'num_units':1, 'num_stories':1},
    '1 unit, 2 stories': {'num_units':1, 'num_stories':2},
    '2 units, 1 story': {'num_units':2, 'num_stories':1}
}

st.title('DADU Energy Simulator')

# sidebar sliders
site = st.sidebar.slider('site', 0, 3, key='site')
size = st.sidebar.slider('square footage', 100, 1000, value=400, step=10, key='sqft')
typology = st.sidebar.select_slider('DADU typology', 
                                    ['1 unit, 1 story', '1 unit, 2 stories', '2 units, 1 story'], key='typology')
num_stories = typologyd[typology]['num_stories']
num_units = typologyd[typology]['num_units']

inf_rate = infd[st.sidebar.select_slider('infiltration rate', ['standard','passive house'], key='inf')]
orientation = oriend[st.sidebar.select_slider('direction existing to DADU', 
                                              ['North', 'South', 'East', 'West'], key='orientation')]
wwr = st.sidebar.slider('wwr', 0.0, 1.0, key='wwr')
frame = framed[st.sidebar.select_slider('frame size', ['2x4','2x6','2x8','2x10'], key='frame')]
polyiso_t = st.sidebar.slider('polyiso_t', 0, 1, step=1, key='polyiso')
cellulose_t = st.sidebar.slider('cellulose_t', 0, 10, key='cellulose')
setback = setbackd[st.sidebar.select_slider('land use setbacks', 
                                            ['existing','proposed'], key='setback')][site]
rear_setback = setback['rear']
side_setback = setback['side']
structure_setback = setback['structure']

if num_stories == 1:
    height = 10
    footprint = size
else:
    height = 20
    footprint = size/2.

    """
    from arcgis import GIS
    footprint_max = area_buildable from GIS data based on city land use code
    st.slider('square footage', 100, footprint_max, value=footprint_max/2, step=10, key='sqft')
    """

assembly_r = calc_r(polyiso_t, cellulose_t)

# sur_area = 
# volume = footprint * height
# surf_vol_ratio = surf_area / volume


# download results
# st.download_button()

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     def lowercase(x): return str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data


# data_load_state = st.text('Loading data...')
# data = load_data(10000)
# data_load_state.text("Done! (using st.cache)")

# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(
#     data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)