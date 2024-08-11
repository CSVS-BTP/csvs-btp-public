import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

cam_id_turns_map = {
    'Stn_HD_1' : {'turns':['BC','BE','DE','DA','FA','FC'], 'flipped':False},
    'Sty_Wll_Ldge_FIX_3' : {'turns':['AB','BA'], 'flipped':False},
    'SBI_Bnk_JN_FIX_1' : {'turns':['AB','AC','BA','BC','CA','CB'], 'flipped':False},
    'SBI_Bnk_JN_FIX_3' : {'turns':['AB','BA'], 'flipped':False},
    '18th_Crs_BsStp_JN_FIX_2' : {'turns':['AB','BA'], 'flipped':True},
    '18th_Crs_Bus_Stop_FIX_2' : {'turns':['AB','AD','CD','EB','ED'], 'flipped':False},
    'Ayyappa_Temple_FIX_1' : {'turns':['AB','BA'], 'flipped':False},
    'Devasandra_Sgnl_JN_FIX_1' : {'turns':['AB','BA'], 'flipped':False},
    'Devasandra_Sgnl_JN_FIX_3' :  {'turns':['AB','AD','CA','CD','DA','DB'], 'flipped':False},
    'Mattikere_JN_FIX_1' : {'turns':['AB','AD','CA','CD','DA','DB'], 'flipped':False},
    'Mattikere_JN_FIX_2' : {'turns':['BC','BD','CA','DA','DC'], 'flipped':False},
    'Mattikere_JN_FIX_3' :  {'turns':['AB','BA'], 'flipped':False},
    'Mattikere_JN_HD_1' : {'turns':['AC','BA','BD','CA','CD'], 'flipped':False},
    'HP_Ptrl_Bnk_BEL_Rd_FIX_2' : {'turns':['AB','BA'], 'flipped':False},
    'Kuvempu_Circle_FIX_1' : {'turns':['BA'], 'flipped':False},
    'Kuvempu_Circle_FIX_2' : {'turns':['BA'], 'flipped':False},
    'MS_Ramaiah_JN_FIX_1' : {'turns':['AB','AC'], 'flipped':False},
    'MS_Ramaiah_JN_FIX_2' : {'turns':['BC','BE','BG','DA','DE','DG','FA','FC','FG','HA','HC','HE'], 'flipped':False},
    'Ramaiah_BsStp_JN_FIX_1' : {'turns':['AB','BA'], 'flipped':False},
    'Ramaiah_BsStp_JN_FIX_2' : {'turns':['AB','BA'], 'flipped':False},
    '18th_Crs_BsStp_JN_FIX_1' : [],
    '18th_Crs_Bus_Stop_FIX_1' : [],
    'MS_Ramaiah_JN_FIX_3' : [],
    'Udayagiri_JN_FIX_2' : [],
}

vehicle_class_rmap = {
    0:'Car',
    1:'Bus',
    2:'Truck',
    3:'Three-Wheeler',
    4:'Two-Wheeler',
    5:'LCV',
    6:'Bicycle'
}

# Function to calculate perpendicular distance from a point to a line segment
def segment_distance(x1, y1, x2, y2, x0, y0):
    # Vector AB
    AB = np.array([x2 - x1, y2 - y1])
    # Vector AP
    AP = np.array([x0 - x1, y0 - y1])
    # Vector BP
    BP = np.array([x0 - x2, y0 - y2])
    
    # Dot products
    AB_AB = np.dot(AB, AB)
    AB_AP = np.dot(AB, AP)
    AB_BP = np.dot(AB, BP)
    
    if AB_AB == 0:
        return np.linalg.norm(AP)  # A and B are the same point
    t = AB_AP / AB_AB
    if t < 0.0:
        return np.linalg.norm(AP)  # Closest to A
    elif t > 1.0:
        return np.linalg.norm(BP)  # Closest to B
    else:
        nearest = np.array([x1, y1]) + t * AB
        return np.linalg.norm(nearest - np.array([x0, y0]))

# Function to calculate perpendicular distance from a point to a line segment
def segment_distance(x1, y1, x2, y2, x0, y0):
    AB = np.array([x2 - x1, y2 - y1])
    AP = np.array([x0 - x1, y0 - y1])
    BP = np.array([x0 - x2, y0 - y2])
    
    AB_AB = np.dot(AB, AB)
    AB_AP = np.dot(AB, AP)
    AB_BP = np.dot(AB, BP)
    
    if AB_AB == 0:
        return np.linalg.norm(AP)
    t = AB_AP / AB_AB
    if t < 0.0:
        return np.linalg.norm(AP)
    elif t > 1.0:
        return np.linalg.norm(BP)
    else:
        nearest = np.array([x1, y1]) + t * AB
        return np.linalg.norm(nearest - np.array([x0, y0]))
    
def detect_turns(cam_id, output_json = "output.json"):

    turns_list = cam_id_turns_map[cam_id]['turns']
    flipped = cam_id_turns_map[cam_id]['flipped']

    tvdf1 = pd.read_csv('vid_1.csv')
    tvdf2 = pd.read_csv('vid_2.csv')
    vdf = pd.concat([tvdf1, tvdf2], ignore_index=True)
    vdf.reset_index(names='v_id', inplace=True)

    vdf['r'] = np.sqrt(vdf['delta_x']**2 + vdf['delta_y']**2)
    vdf['theta'] = np.rad2deg(np.arctan2(vdf['delta_y'], vdf['delta_x']))
    r_max = vdf['r'].max()
    vdf['r'] /= r_max

    lines = []
    agc = AgglomerativeClustering(n_clusters=len(turns_list))
    vdf['cluster'] = agc.fit_predict(vdf[['r','theta']])
    cluster_centers = vdf.groupby('cluster')['theta'].median().reset_index()
    for idx, row in cluster_centers.iterrows():
        angle = row['theta']
        theta = np.deg2rad(angle)
        x1  = int((r_max*np.cos(theta)).round())
        y1 = int((r_max*np.sin(theta)).round())
        line = (x1, y1)
        lines.append(line)

    lines = lines[::-1] if flipped else lines

    # Calculate and plot the shortest perpendiculars
    turning_patterns = []
    for index, row in vdf.iterrows():
        x0, y0 = row['delta_x'], row['delta_y']
        min_distance = float('inf')
        closest_line = None

        for i, (x, y) in enumerate(lines):
            distance = segment_distance(0, 0, x, y, x0, y0)
            if distance < min_distance:
                min_distance = distance
                closest_line = turns_list[i]

        turning_patterns.append(closest_line)

    vdf['Turning Patterns'] = turning_patterns
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

    fdf = pgdf.loc[turns_list][vtypes].fillna(0).astype(int).reset_index()

    counts = fdf.T.to_dict()
    # Predicting counts for future where N knowns = N unknowns is currently not known in mathematics
    output = {"Cam_ID":{"Cumulative Counts":counts, "Predicted Counts":counts}}

    # Writing to a JSON file
    with open(output_json, 'w') as file:
        json.dump(output, file, indent=4)
