import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

print('turns list update')

cam_id_turns_map = {
    'Stn_HD_1' : ['BC','BE','DE','DA','FA','FC'],
    'Sty_Wll_Ldge_FIX_3' : ['AB','BA'],
    'SBI_Bnk_JN_FIX_1' : ['AB','AC','BA','BC','CA','CB'],
    'SBI_Bnk_JN_FIX_3' : ['AB','BA'],
    '18th_Crs_BsStp_JN_FIX_2' : ['AB','BA'],
    '18th_Crs_Bus_Stop_FIX_2' : ['AB','AD','CD','EB','ED'],
    'Ayyappa_Temple_FIX_1' : ['AB','BA'],
    'Devasandra_Sgnl_JN_FIX_1' : ['AB','BA'],
    'Devasandra_Sgnl_JN_FIX_3' : ['AB','AD','CA','CD','DA','DB'],
    'Mattikere_JN_FIX_1' : ['AB','AD','CA','CD','DA','DB'],
    'Mattikere_JN_FIX_2' : ['BC','BD','CA','DA','DC'],
    'Mattikere_JN_FIX_3' : ['AB','BA'],
    'Mattikere_JN_HD_1' : ['AC','BA','BD','CA','CD'],
    'HP_Ptrl_Bnk_BEL_Rd_FIX_2' : ['AB','BA'],
    'Kuvempu_Circle_FIX_1' : ['BA'],
    'Kuvempu_Circle_FIX_2' : ['BA'],
    'MS_Ramaiah_JN_FIX_1' : ['AB','AC'],
    'MS_Ramaiah_JN_FIX_2' : ['BC','BE','BG','DA','DE','DG','FA','FC','FG','HA','HC','HE'],
    'Ramaiah_BsStp_JN_FIX_1' : ['AB','BA'],
    'Ramaiah_BsStp_JN_FIX_2' : ['AB','BA'],
    '18th_Crs_BsStp_JN_FIX_1' : [],
    '18th_Crs_Bus_Stop_FIX_1' : [],
    'MS_Ramaiah_JN_FIX_3' : [],
    'Udayagiri_JN_FIX_2' : [],
    'Dari_Anjaneya_Temple': ['BC','BD','AD','AC','CD'],
    'Nanjudi_House': ['AC','BC','CA','CB'],
    'Buddha_Vihara_Temple': ['AB','BA'],
    'Sundaranagar_Entrance': ['AB','BA'],
    'ISRO_Junction': ['AB','AC','BC','BD'],
    '80ft_Road': ['AB','BA'],
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
    # flipped = True if cam_id == '18th_Crs_BsStp_JN_FIX_2' else False

    tvdf1 = pd.read_csv('vid_1.csv')
    tvdf2 = pd.read_csv('vid_2.csv')
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
    # cluster_centers['Turns'] = turns_list[::-1] if flipped else turns_list
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
    # Predicting counts for future where N knowns = N unknowns is currently not known in mathematics
    output = {cam_id:{"Cumulative Counts":counts, "Predicted Counts":counts}}

    # Writing to a JSON file
    with open(output_json, 'w') as file:
        json.dump(output, file, indent=4)
