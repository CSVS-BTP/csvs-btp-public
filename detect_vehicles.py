import torch
from ultralytics import YOLOWorld
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

# Load a pretrained YOLOv8x-worldv2 model
model = YOLOWorld("yolov8x-worldv2.pt")

print('Model updated')
vehicle_list = [
    'vehicle/Bicycle',
    'vehicle/Bus',
    'vehicle/Car',
    'vehicle/LCV',
    'vehicle/LCV/Ambulance',
    'vehicle/LCV/Police-Van',
    'vehicle/LCV/Patrol-Jeep',
    'vehicle/LCV/Goods-Transporter',
    'vehicle/LCV/Mini-Bus',
    'vehicle/LCV/Mini-Van',
    'vehicle/LCV/Tempo-Traveller',
    'vehicle/Three Wheeler',
    'vehicle/Truck',
    'vehicle/Truck/Goods-Transporter',
    'vehicle/Truck/Lorry',
    'vehicle/Truck/Garbage-Truck',
    'vehicle/Truck/Water-Tanker',
    'vehicle/Two Wheeler',
    'vehicle/Two Wheeler/Scooter',
    'vehicle/Two Wheeler/Motorbike',
    'vehicle/Two Wheeler/EV',
]
model.set_classes(vehicle_list)

# Define vehicle class map and IDs
vehicle_class_rmap = {
    0:'Bicycle',
    1:'Bus',
    2:'Car',
    3:'LCV',
    4:'Three Wheeler',
    5:'Truck',
    6:'Two Wheeler',
} 

vehicle_class_map = {v:k for k,v in vehicle_class_rmap.items()}
cls_id_map = {idx:vehicle_class_map[vehicle.split('/')[1]] for idx, vehicle in enumerate(vehicle_list)}

def detect_vehicles(video_file, csv_file='vehicles.csv'):
    
    results = model.track(
        source=video_file,
        device=device,
        imgsz = (1080,1920),
        conf = 0.1,
        iou = 0.4,
        max_det=100,
        agnostic_nms = True,
        vid_stride = 1,
        stream = True,
        tracker = "custom_botsort.yaml",
        persist = True,
        verbose = False
    )

    ob_id = 0
    instance_dict = {}

    for fn, result in enumerate(results):
        boxes = result.boxes

        for ob in range(boxes.shape[0]):
            cls_id = int(boxes.cls[ob].item())
            v_id = int(boxes.id[ob].item()) if boxes.id is not None else None

            x, y, w, h = map(int, boxes.xywh[ob])

            features_dict = {
                "fn": fn,
                "v_id": v_id,
                "cls_id": cls_id_map[cls_id],
                "cn_x": x,
                "cn_y": y,
            }
            instance_dict[ob_id] = features_dict
            ob_id += 1

    idf = pd.DataFrame(instance_dict).T
    idf['fn'] = idf['fn'].astype(int)
    idf['cls_id'] = idf['cls_id'].astype(int)
    idf['v_id'] = idf['v_id'].astype(float)

    tdf_drop = idf.groupby('v_id')['fn'].count().reset_index()
    vdrop = tdf_drop.loc[tdf_drop['fn'] < 2]['v_id'].values.tolist()
    idf.drop(idf.loc[idf['v_id'].isin(vdrop + [np.nan])].index, inplace=True)
    idf.reset_index(drop=True, inplace=True)

    vidfx = idf.groupby('v_id')['cn_x'].aggregate(['first','last']).reset_index()
    vidfy = idf.groupby('v_id')['cn_y'].aggregate(['first','last']).reset_index()
    vdf = pd.merge(vidfx, vidfy, how='inner', on='v_id', suffixes=['_x', '_y'])
    vdf['delta_x'] = vdf['first_x'] - vdf['last_x']
    vdf['delta_y'] = vdf['first_y'] - vdf['last_y']
    vids = vdf.loc[(vdf['delta_x'].abs()>45) & (vdf['delta_x'].abs()>80)]['v_id']

    vdf = vdf.loc[vdf['v_id'].isin(vids)]
    idf = idf.loc[idf['v_id'].isin(vids)]

    tvtype_df = idf.groupby(['v_id'])['cls_id'].value_counts().reset_index()
    idxs = tvtype_df.groupby(['v_id'])['count'].idxmax()

    for v_id in tvtype_df['v_id'].unique():
        ttvtype_df = tvtype_df.loc[tvtype_df['v_id']==v_id]
        cls_auto = ttvtype_df['cls_id'] == 4
        if cls_auto.any():
            if ttvtype_df[cls_auto]['count'].iloc[0] > 0:
                idx = ttvtype_df[ttvtype_df['cls_id'] == 4].index
                idxs[v_id] = idx[0]
                
    vdf['cls_id'] = tvtype_df.loc[idxs]['cls_id'].values
    vdf['vehicle'] = vdf['cls_id'].map(vehicle_class_rmap)

    cols = ['first_x', 'first_y', 'last_x', 'last_y', 'delta_x', 'delta_y', 'vehicle']
    tvdf = vdf[cols]
    tvdf.to_csv(csv_file, index=False)