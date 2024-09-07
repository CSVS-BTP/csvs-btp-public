import os
import ast
import itertools
import json
import cv2
import torch
import pandas as pd
import numpy as np

# Check if GPU is available
print("Checking GPU availability...")
if torch.cuda.is_available():
    print("GPU is available")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")

vehicle_list = ['Bicycle', 'Bus', 'Car', 'LCV', 'Three-Wheeler', 'Truck', 'Two-Wheeler']

def pair_buckets_with_reuse(bucket1, bucket2):
    max_length = max(len(bucket1), len(bucket2))
    # Repeat the smaller bucket elements to match the size of the larger bucket
    if len(bucket1) < max_length:
        bucket1 = list(itertools.islice(itertools.cycle(bucket1), max_length))
    elif len(bucket2) < max_length:
        bucket2 = list(itertools.islice(itertools.cycle(bucket2), max_length))

    # Generate all permutations of the first bucket
    permutations = itertools.permutations(bucket1)

    # Pair each permutation with the other bucket
    pairings = []
    for perm in permutations:
        pairings.append(list(zip(perm, bucket2)))

    return pairings

def reid_vehicles(cam_ids, videos, output_dir = '/app/data/CSVS'):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    images_dir = os.path.join(output_dir, 'Images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
    for vehicle in vehicle_list:
        if not os.path.isdir(os.path.join(images_dir, vehicle)):
            os.mkdir(os.path.join(images_dir, vehicle))

    matrices_dir = os.path.join(output_dir, 'Matrices')
    if not os.path.isdir(matrices_dir):
        os.mkdir(matrices_dir)

    vdfs = []
    for cam_id in cam_ids:
        vdf = pd.read_csv(f'{cam_id}.csv')
        vdf.reset_index(names=['v_id'], inplace=True)
        vdf['cam_id'] = cam_id
        vdfs.append(vdf)

    avdf = pd.concat(vdfs, ignore_index=True)

    avdf['features'] = avdf['features'].apply(lambda x: torch.tensor(ast.literal_eval(x)).to(device))
    avdf['features'] = avdf['features'].apply(lambda x: x/x.norm())

    scols = ['v_id', 'v_idt', 'cls_id', 'features', 'Turning Patterns', 'cam_id']
    dfa_list = []
    for i in range(len(cam_ids)):
        for j in range(i+1, len(cam_ids)):
            df1 = avdf.loc[avdf['cam_id'] == cam_ids[i]]
            df2 = avdf.loc[avdf['cam_id'] == cam_ids[j]]
            df1_tp = sorted(df1['Turning Patterns'].unique())
            df2_tp = sorted(df2['Turning Patterns'].unique())
            pairs = pair_buckets_with_reuse(df1_tp, df2_tp)
            dfs_dict = {}
            for tp_i in range(len(df1_tp)):
                for tp_j in range(len(df2_tp)):
                    df1_tp_i = df1.loc[df1['Turning Patterns'] == df1_tp[tp_i]].copy()
                    df2_tp_j = df2.loc[df2['Turning Patterns'] == df2_tp[tp_j]].copy()
                    df1_tp_i['v_idt'] = list(range(df1_tp_i.shape[0]))
                    df2_tp_j['v_idt'] = list(range(df2_tp_j.shape[0]))
                    dfm = pd.merge(df1_tp_i[scols], df2_tp_j[scols], how='inner', on=['cls_id'], suffixes=['_1', '_2'])
                    dfm['similarity'] = dfm.apply(lambda x: torch.nn.functional.cosine_similarity(x['features_1'], x['features_2'], dim=0).item(), axis=1).round(2)
                    dfm.drop(columns=['features_1', 'features_2'], inplace=True)
                    dfm = dfm.loc[dfm['similarity'] > 0.45].copy()
                    dfm.sort_values(by=['similarity', 'v_id_1', 'v_id_2'], ascending=[False, True, True], inplace=True)
                    v1s = []
                    v2s = []
                    indices = []
                    for idx,row in dfm.iterrows():
                        v1 = row['v_id_1']
                        v2 = row['v_id_2']
                        v1t = row['v_idt_1']
                        v2t = row['v_idt_2']
                        if v1 not in v1s and v2 not in v2s and abs(v1t-v2t) < 50:
                            v1s.append(v1)
                            v2s.append(v2)
                            indices.append(idx)
                    dfm = dfm.loc[indices].copy()
                    dfm.sort_values(by=['v_id_1', 'v_id_2'], ascending=[True,True], inplace=True)
                    dfm.reset_index(drop=True, inplace=True)
                    dfs_dict[(df1_tp[tp_i], df2_tp[tp_j])] = dfm
            max_match = 0
            max_pair = None
            for pair in pairs:
                matched = 0
                for p in pair:
                    matched += dfs_dict[p].shape[0]
                if matched > max_match:
                    max_match = matched
                    max_pair = pair
            dfs_list = []
            for p in max_pair:
                dft = dfs_dict[p]
                dfs_list.append(dft)
            dfs = pd.concat(dfs_list, ignore_index=True)
            dfs.sort_values(by=['v_id_1', 'v_id_2'], ascending=[True,True], inplace=True)
            dfs.reset_index(drop=True, inplace=True)
            dfa_list.append(dfs)
    dfa = pd.concat(dfa_list, ignore_index=True)
    dfa.reset_index(names=['v_id'], inplace=True)

    avcols = ['v_id', 'cam_id', 'fn', 'vehicle', 'x', 'y', 'w', 'h']
    fdf_list = []
    for i in range(len(cam_ids)):
        for j in range(i+1, len(cam_ids)):
            dfa1 = dfa.loc[dfa['cam_id_1'] == cam_ids[i]][['v_id', 'v_id_1', 'cam_id_1']]
            dfa2 = dfa.loc[dfa['cam_id_2'] == cam_ids[j]][['v_id', 'v_id_2', 'cam_id_2']]
            fdf1 = dfa1.merge(avdf[avcols], how='left', left_on=['v_id_1', 'cam_id_1'], right_on=['v_id', 'cam_id'])
            fdf1.drop(columns=['v_id_1', 'cam_id_1'], inplace=True)
            fdf1.rename(columns={'v_id_x': 'v_id', 'v_id_y': 'v_id_orig'}, inplace=True)
            fdf1.sort_values(by='fn', ascending=True, inplace=True)
            fdf1.reset_index(drop=True, inplace=True)
            fdf2 = dfa2.merge(avdf[avcols], how='left', left_on=['v_id_2', 'cam_id_2'], right_on=['v_id', 'cam_id'])
            fdf2.drop(columns=['v_id_2', 'cam_id_2'], inplace=True)
            fdf2.rename(columns={'v_id_x': 'v_id', 'v_id_y': 'v_id_orig'}, inplace=True)
            fdf2.sort_values(by='fn', ascending=True, inplace=True)
            fdf2.reset_index(drop=True, inplace=True)
            fdf_list.append(fdf1)
            fdf_list.append(fdf2)
    fdf = pd.concat(fdf_list, ignore_index=True)

    for idx, cam_id in enumerate(cam_ids):
        video_cap = cv2.VideoCapture(videos[idx])
        fdfc = fdf.loc[fdf['cam_id'] == cam_id]
        for fn in sorted(fdfc['fn'].unique()):
            fdfcf = fdfc.loc[fdfc['fn'] == fn]
            for ind, row in fdfcf.iterrows():
                v_id = row['v_id']
                vehicle = row['vehicle']
                x1 = row['x'] - row['w']//2
                y1 = row['y'] - row['h']//2
                x2 = row['x'] + row['w']//2
                y2 = row['y'] + row['h']//2

                video_cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, frame = video_cap.read()
                if not ret:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
                text = f'{vehicle}_{cam_id}_{fn}_{v_id}'
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                box_coords = ((x1, y1 - text_height - 10), (x1 + text_width, y1))
                cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                output_file = os.path.join(images_dir, vehicle, f'{text}.jpg')
                cv2.imwrite(output_file, frame)

    rdf_dict = {}
    matrix_dict = {}
    for vehicle in vehicle_list:
        fdfv = fdf.loc[fdf['vehicle'] == vehicle]
        if fdfv.shape[0] == 0:
            matrix_dict[vehicle] = np.zeros((len(cam_ids), len(cam_ids))).astype(int).tolist()
        else:
            fdfvp = fdfv.pivot_table(index='v_id', columns='cam_id', values='fn')
            rdf_dict[vehicle] = {}
            for i in range(len(cam_ids)):
                for j in range(len(cam_ids)):
                    if i == j:
                        continue
                    cam_id_1 = cam_ids[i]
                    cam_id_2 = cam_ids[j]
                    if cam_id_1 not in fdfvp.columns or cam_id_2 not in fdfvp.columns:
                        continue
                    trdf = fdfvp[[cam_id_1, cam_id_2]].dropna()
                    first = trdf.loc[trdf[cam_id_1]==trdf.min(axis=1)].shape[0]
                    rdf_dict[vehicle][f'{cam_id_1},{cam_id_2}'] = first
            matrix = np.zeros((len(cam_ids), len(cam_ids)))
            reid_value = 0
            for idx1, cam_id_1 in enumerate(cam_ids):
                for idx2, cam_id_2 in enumerate(cam_ids):
                    key = f'{cam_id_1},{cam_id_2}'
                    if key in rdf_dict[vehicle]:
                        reid_value = rdf_dict[vehicle][key]
                        matrix[idx1, idx2] = reid_value
            matrix_dict[vehicle] = matrix.astype(int).tolist()
        output_file = os.path.join(matrices_dir, f'{vehicle}.json')
        with open(output_file, 'w') as f:
            json.dump(matrix_dict[vehicle], f)