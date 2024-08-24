import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

cam_id_turns_map = {
    'Stn_HD_1' : ['BC','BE','DE','DA','FA','FC'],
    'Sty_Wll_Ldge_FIX_3' : ['BA','AB'],
    'SBI_Bnk_JN_FIX_1' : ['AB','AC','CA','CB','BA','BC'],
    'SBI_Bnk_JN_FIX_3' : ['AB','BA'],
    '18th_Crs_BsStp_JN_FIX_2' : ['AB','BA'],
    '18th_Crs_Bus_Stop_FIX_2' : ['AB','AD','EB','ED','CD'],
    'Ayyappa_Temple_FIX_1' : ['AB','BA'],
    'Devasandra_Sgnl_JN_FIX_1' : ['BA','AB'],
    'Devasandra_Sgnl_JN_FIX_3' : ['AB','AD','DA','DB','CD','CA'],
    'Mattikere_JN_FIX_1' : ['AB','AD','DA','DB','CD','CA'],
    'Mattikere_JN_FIX_2' : ['DA','DC','CA','BC','BD'],
    'Mattikere_JN_FIX_3' : ['AB','BA'],
    'Mattikere_JN_HD_1' : ['AC','BA','BD','CA','CD'],
    'HP_Ptrl_Bnk_BEL_Rd_FIX_2' : ['AB','BA'],
    'Kuvempu_Circle_FIX_1' : ['BA'],
    'Kuvempu_Circle_FIX_2' : ['BA'],
    'MS_Ramaiah_JN_FIX_1' : ['AC','AB'],
    'MS_Ramaiah_JN_FIX_2' : ['HA','HC','HE','FG','FA','FC','DA','DG','DE','BC','BE','BG'],
    'Ramaiah_BsStp_JN_FIX_1' : ['BA','AB'],
    'Ramaiah_BsStp_JN_FIX_2' : ['AB','BA'],
    '18th_Crs_BsStp_JN_FIX_1' : [],
    '18th_Crs_Bus_Stop_FIX_1' : [],
    'MS_Ramaiah_JN_FIX_3' : [],
    'Udayagiri_JN_FIX_2' : [],
    'Dari_Anjaneya_Temple': ['CD','BC','BD','AC','AD'],
    'Nanjudi_House': ['CA','CB','BC','AC'],
    'Buddha_Vihara_Temple': ['AB','BA'],
    'Sundaranagar_Entrance': ['AB','BA'],
    'ISRO_Junction': ['AC','AB','BD','BC'],
    '80ft_Road': ['BA','AB'],
}

vehicle_class_rmap = {
    0:'Bicycle',
    1:'Bus',
    2:'Car',
    3:'LCV',
    4:'Three Wheeler',
    5:'Truck',
    6:'Two Wheeler',
} 
    
def detect_turns(cam_id, output_json = "output.json"):

    turns_list = cam_id_turns_map[cam_id]

    tvdf1 = pd.read_csv('vid_1.csv')
    tvdf2 = pd.read_csv('vid_2.csv')
    tvdf2['fn'] += tvdf1['fn'].max()
    vdf = pd.concat([tvdf1, tvdf2], ignore_index=True)
    vdf.reset_index(names='v_id', inplace=True)

    vdf['theta'] = np.arctan2(vdf['delta_y'], vdf['delta_x'])
    vdf['cos_theta'] = np.cos(vdf['theta'])
    vdf['sin_theta'] = np.sin(vdf['theta'])
    vdf['angle'] = np.rad2deg(vdf['theta'])

    agc = AgglomerativeClustering(n_clusters=len(turns_list))
    vdf['cluster'] = agc.fit_predict(vdf[['cos_theta', 'sin_theta']])
    cluster_centers = vdf.groupby('cluster')['angle'].median().reset_index()
    cluster_centers.sort_values(by='angle', key=lambda x: (x - 90) % 360, inplace=True)
    cluster_centers.reset_index(drop=True, inplace=True)
    cluster_centers['Turns'] = turns_list
    cluster_map = cluster_centers[['cluster','Turns']].set_index('cluster').to_dict()['Turns']

    vdf['Turning Patterns'] = vdf['cluster'].map(cluster_map)
    gdf = vdf.groupby(['Turning Patterns', 'vehicle'])[['v_id']].count().reset_index()
    pgdf = gdf.pivot_table(values='v_id', index='Turning Patterns', columns='vehicle')

    vtypes = list(vehicle_class_rmap.values())
    for vtype in vtypes:
        if vtype not in pgdf.columns:
            pgdf[vtype] = np.nan 

    pgdf = pgdf.T
    for turn in turns_list:
        if turn not in pgdf.columns:
            pgdf[turn] = np.nan  
    pgdf = pgdf.T 

    fdf = pgdf.loc[turns_list][vtypes].fillna(0).astype(int)

    counts = fdf.T.to_dict()

    fn_min = 0
    fn_max = vdf['fn'].max()
    fn_max = round(fn_max / 100)*100
    parts = 15
    step = fn_max//parts

    mcounts = {}
    for idx, fn in enumerate(np.arange(fn_min, fn_max, step)):
        mvdf = vdf.loc[(vdf['fn'] > fn) & (vdf['fn'] <= fn + step)].copy()
        mgdf = mvdf.groupby(['Turning Patterns', 'vehicle'])[['v_id']].count().reset_index()
        mpgdf = mgdf.pivot_table(values='v_id', index='Turning Patterns', columns='vehicle')

        for vtype in vtypes:
            if vtype not in mpgdf.columns:
                mpgdf[vtype] = np.nan 

        mpgdf = mpgdf.T
        for turn in turns_list:
            if turn not in mpgdf.columns:
                mpgdf[turn] = np.nan
        mpgdf = mpgdf.fillna(0).T 

        mfdf = mpgdf.loc[turns_list][vtypes].fillna(0).astype(int)
        mcounts[idx] = mfdf.values.reshape(-1)

    mdf = pd.DataFrame.from_dict(mcounts, orient='columns')

    pcounts = {}
    # Parameters for ARIMA (p, d, q)
    order = (1, 1, 1)  # You can adjust the p, d, q parameters according to your data

    for idx, row in mdf.iterrows():
        y = row.values
        
        # Fit the ARIMA model
        model = ARIMA(y, order=order)
        model_fit = model.fit()
        
        # Forecast the next 'parts + 1' points
        forecast = model_fit.forecast(steps=parts + 1)
        
        # Store the predictions in pcounts
        pcounts[idx] = forecast

    pdf = pd.DataFrame.from_dict(pcounts, orient='index')
    pdf['pred'] = pdf.sum(axis=1).astype(int)
    pred = pdf['pred'].values
    pred[pred<0] = 0

    pred_df = pd.DataFrame(pred.reshape(len(turns_list),len(vtypes)), index=turns_list, columns=vtypes)
    pred_counts = pred_df.T.to_dict()

    # Predicting counts for future where N knowns = N unknowns is currently not known in mathematics
    output = {cam_id:{"Cumulative Counts":counts, "Predicted Counts":pred_counts}}

    # Writing to a JSON file
    with open(output_json, 'w') as file:
        json.dump(output, file, indent=4)
