import torch
from ultralytics import YOLOWorld
import pandas as pd

# Check if GPU is available
print("Checking GPU availability...")
if torch.cuda.is_available():
    print("GPU is available")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8s-worldv2.pt")

vehicle_list = [
    'vehicle/Car',
    'vehicle/Bus',
    'vehicle/Truck',
    'vehicle/Three Wheeler',
    'vehicle/Two Wheeler',
    'vehicle/Small Four Wheeler',
    'vehicle/Bicycle'
]
model.set_classes(vehicle_list)

# Define vehicle class map and IDs
vehicle_class_rmap = {
    0:'Car',
    1:'Bus',
    2:'Truck',
    3:'Three-Wheeler',
    4:'Two-Wheeler',
    5:'LCV',
    6:'Bicycle'
} 

rectangles_dict = {
    "a": {"tl": {"x": 0, "y": 450}, "br": {"x": 1020, "y": 1080}},
    "b": {"tl": {"x": 0, "y": 150}, "br": {"x": 1020, "y": 450}},
    "c": {"tl": {"x": 400, "y": 0}, "br": {"x": 1020, "y": 450}},
    "d": {"tl": {"x": 1020, "y": 0}, "br": {"x": 1720, "y": 450}},
    "e": {"tl": {"x": 1020, "y": 150}, "br": {"x": 1920, "y": 450}},
    "f": {"tl": {"x": 1020, "y": 450}, "br": {"x": 1920, "y": 1080}},
}

def detect_vehicles(video_file, csv_file='counts.csv'):
    results = model.track(
        source=video_file,
        device=device,
        imgsz = (640,1120),
        conf = 0.25,
        iou = 0.5,
        agnostic_nms = True,
        vid_stride=1,
        stream=True,
        tracker="custom_bytetrack.yaml",
        persist=True,
        verbose=False
    )

    ob_id = 0
    instance_dict = {}

    for fn, result in enumerate(results):
        boxes = result.boxes

        for ob in range(boxes.shape[0]):
            cls_id = int(boxes.cls[ob].item())
            v_id = int(boxes.id[ob].item()) if boxes.id is not None else None
            # features = boxes.data[ob].cpu()

            x1, y1, x2, y2 = map(int, boxes.xyxy[ob])
            top_left = {"x": x1, "y": y1}
            bottom_right = {"x": x2, "y": y2}
            x, y, w, h = map(int, boxes.xywh[ob])
            centre = {"x": x, "y": y}

            features_dict = {
                "fn": fn,
                "v_id": v_id,
                "cls_id": cls_id,
                # "features": features,
                "tl": top_left,
                "br": bottom_right,
                "cn":centre
            }
            instance_dict[ob_id] = features_dict
            ob_id += 1

    idf = pd.DataFrame(instance_dict).T
    idf['fn'] = idf['fn'].astype(int)
    idf['cls_id'] = idf['cls_id'].astype(int)
    idf['v_id'] = idf['v_id'].astype(float)

    vblx = idf['tl'].apply(lambda x:x['x'])
    vbly = idf['br'].apply(lambda x:x['y'])
    vtlx = idf['tl'].apply(lambda x:x['x'])
    vtly = idf['tl'].apply(lambda x:x['y'])
    vtrx = idf['br'].apply(lambda x:x['x'])
    vtry = idf['tl'].apply(lambda x:x['y'])
    vbrx = idf['br'].apply(lambda x:x['x'])
    vbry = idf['br'].apply(lambda x:x['y'])
    vbcx = idf['cn'].apply(lambda x:x['x'])
    vbcy = idf['cn'].apply(lambda x:x['y'])

    bbox_dict = {}
    for bbid, bbox in rectangles_dict.items():
        rtlx = bbox["tl"]["x"]
        rtly = bbox["tl"]["y"]
        rbrx = bbox["br"]["x"]
        rbry = bbox["br"]["y"]

        # Create a mask for each corner point being inside the bounding box
        bl_inside = (rtlx <= vblx) & (vblx <= rbrx) & (rtly <= vbly) & (vbly <= rbry)
        tl_inside = (rtlx <= vtlx) & (vtlx <= rbrx) & (rtly <= vtly) & (vtly <= rbry)
        tr_inside = (rtlx <= vtrx) & (vtrx <= rbrx) & (rtly <= vtry) & (vtry <= rbry)
        br_inside = (rtlx <= vbrx) & (vbrx <= rbrx) & (rtly <= vbry) & (vbry <= rbry)
        cn_inside = (rtlx <= vbcx) & (vbcx <= rbrx) & (rtly <= vbcy) & (vbcy <= rbry)
        any_inside = tl_inside | br_inside | bl_inside | tr_inside | cn_inside
   
        bbox_dict[bbid] = any_inside

    bbox_df = pd.DataFrame(bbox_dict)
    bbox_df['v_id'] = idf['v_id']
    bbox_df = bbox_df.dropna(subset='v_id').reset_index(names='ob_id')
    bbox_df['v_id'] = bbox_df['v_id'].astype(int)

    r_df = bbox_df.drop(columns='ob_id').groupby('v_id').max().reset_index(drop=True)
    r_df.loc[(r_df['b']) & ((r_df['c']) & (r_df['e'])), 'c'] = False
    r_df.loc[(r_df['d']) & ((r_df['e']) & (r_df['a'])), 'e'] = False
    r_df.loc[(r_df['f']) & ((r_df['a']) & (r_df['c'])), 'a'] = False

    r_df.loc[(r_df.sum(axis=1)==2) & (r_df['a']) & (r_df['b']), 'c'] = True
    r_df.loc[(r_df.sum(axis=1)==2) & (r_df['e']) & (r_df['f']), 'd'] = True

    r_df.loc[(r_df['b']) & (r_df['e']), ['a','d','c','f']] = False
    r_df.loc[(r_df['d']) & (r_df['a']), ['b','c','e','f']] = False
    r_df.loc[(r_df['f']) & (r_df['c']), ['a','b','d','e']] = False
    r_df.loc[(r_df['b']) & (r_df['c']), ['a','d','e','f']] = False
    r_df.loc[(r_df['d']) & (r_df['e']), ['a','b','c','f']] = False
    r_df.loc[(r_df['f']) & (r_df['a']), ['b','c','d','e']] = False

    r_df.loc[(r_df.sum(axis=1)==1) & (r_df['b']), 'c'] = True
    r_df.loc[(r_df.sum(axis=1)==1) & (r_df['c']), 'f'] = True
    r_df.loc[(r_df.sum(axis=1)==1) & (r_df['f']), 'a'] = True
    r_df.loc[(r_df.sum(axis=1)==1) & (r_df['a']), 'd'] = True
    r_df.loc[(r_df.sum(axis=1)==1) & (r_df['d']), 'e'] = True
    r_df.loc[(r_df.sum(axis=1)==1) & (r_df['e']), 'b'] = True

    vtype_df = pd.merge(bbox_df['v_id'], idf['cls_id'], how='inner', right_index=True, left_index=True)
    tvtype_df = vtype_df.groupby(['v_id'])['cls_id'].value_counts().reset_index()
    idxs = tvtype_df.groupby(['v_id'])['count'].idxmax()

    for v_id in tvtype_df['v_id'].unique():
        ttvtype_df = tvtype_df.loc[tvtype_df['v_id']==v_id]
        cls_auto = ttvtype_df['cls_id'] == 3
        cls_lcv = ttvtype_df['cls_id'] == 5
        if cls_auto.any():
            if ttvtype_df[cls_auto]['count'].iloc[0] > 0:
                idx = ttvtype_df[ttvtype_df['cls_id'] == 3].index
                idxs[v_id] = idx[0]
        if cls_lcv.any():
            if ttvtype_df[cls_lcv]['count'].iloc[0] > 2:
                idx = ttvtype_df[ttvtype_df['cls_id'] == 5].index
                idxs[v_id] = idx[0]
                
    r_df['cls_id'] = tvtype_df.loc[idxs]['cls_id'].values
    r_df['vehicle'] = r_df['cls_id'].map(vehicle_class_rmap)

    column_pairs = [('b', 'c'),
                    ('b', 'e'),
                    ('d', 'e'),
                    ('d', 'a'),
                    ('f', 'a'),
                    ('f', 'c')]

    t_df = pd.DataFrame(vehicle_class_rmap.values(), columns=['vehicle'])
    for vehicle in r_df['vehicle'].unique():
        tv_df = r_df.loc[r_df['vehicle']==vehicle]
        # Iterate through each pair and count how many times both columns are True
        for pair in column_pairs:
            col1, col2 = pair
            t_df.loc[t_df['vehicle']==vehicle, ''.join(pair).upper()] = ((tv_df[col1]) & (tv_df[col2])).sum()

    f_df = t_df.copy().rename(columns={"vehicle": "Turning Patterns"})
    f_df = f_df.set_index("Turning Patterns").T
    f_df = f_df.fillna(0).astype(int).reset_index(names='Turning Patterns')
    f_df.to_csv(csv_file, index=False)
