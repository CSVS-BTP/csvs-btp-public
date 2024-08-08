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

# Load a pretrained YOLOv8l-worldv2 model
model = YOLOWorld("yolov8l-worldv2.pt")

vehicle_list = [
    "vehicle/Car",
    "vehicle/Bus",
    "vehicle/Truck",
    "vehicle/Three Wheeler",
    "vehicle/Two Wheeler",
    "vehicle/Small Four Wheeler",
    "vehicle/Bicycle",
]
model.set_classes(vehicle_list)

# Define vehicle class map and IDs
vehicle_class_rmap = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Three-Wheeler",
    4: "Two-Wheeler",
    5: "LCV",
    6: "Bicycle",
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
        imgsz = (640,1280),
        conf = 0.25,
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

    idf['cn_x'] = idf['cn'].apply(lambda p:p['x'])
    idf['cn_y'] = idf['cn'].apply(lambda p:p['y'])

    vidfx = idf.groupby('v_id')['cn_x'].aggregate(['first','last']).reset_index()
    vidfy = idf.groupby('v_id')['cn_y'].aggregate(['first','last']).reset_index()
    vdf = pd.merge(vidfx, vidfy, how='inner', on='v_id', suffixes=['_x', '_y'])
    vdf['delta_x'] = vdf['first_x'] - vdf['last_x']
    vdf['delta_y'] = vdf['first_y'] - vdf['last_y']

    tvtype_df = idf.groupby(['v_id'])['cls_id'].value_counts().reset_index()
    idxs = tvtype_df.groupby(['v_id'])['count'].idxmax()

    for v_id in tvtype_df['v_id'].unique():
        ttvtype_df = tvtype_df.loc[tvtype_df['v_id']==v_id]
        cls_auto = ttvtype_df['cls_id'] == 3
        cls_lcv = ttvtype_df['cls_id'] == 5
        if cls_auto.any():
            if ttvtype_df[cls_auto]['count'].iloc[0] > 0:
                idx = ttvtype_df[ttvtype_df['cls_id'] == 3].index
                idxs[v_id] = idx[0]
        elif cls_lcv.any():
            if ttvtype_df[cls_lcv]['count'].iloc[0] > 2:
                idx = ttvtype_df[ttvtype_df['cls_id'] == 5].index
                idxs[v_id] = idx[0]
                
    vdf['cls_id'] = tvtype_df.loc[idxs]['cls_id'].values
    vdf['vehicle'] = vdf['cls_id'].map(vehicle_class_rmap)
    
    # Coordinates for lines
    lines = [
        (-2000, 380),
        (-2000, -100),
        (-1000, -300),
        (1200, -400),
        (1600, 0),
        (1200, 280)
    ]
    line_names = ['BC', 'BE', 'DE', 'DA', 'FA', 'FC']

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
                closest_line = line_names[i]

        turning_patterns.append(closest_line)

    vdf['Turning Patterns'] = turning_patterns
   
    gdf = vdf.groupby(['Turning Patterns', 'vehicle'])[['v_id']].count().reset_index()
    pgdf = gdf.pivot_table(values='v_id', index='Turning Patterns', columns='vehicle')

    vtypes = list(vehicle_class_rmap.values())    
    for vtype in vtypes:
        if vtype not in pgdf.columns:
            pgdf[vtype] = np.nan 

    pgdf = pgdf.T
    turns_order = ['BC','BE','DE','DA','FA','FC']
    for turn in turns_order:
        if turn not in pgdf.columns:
            pgdf[turn] = np.nan  
    pgdf = pgdf.T 

    fdf = pgdf.loc[turns_order][vtypes]
    fdf = fdf.fillna(0).astype(int).reset_index()
    fdf.to_csv(csv_file, index=False)
