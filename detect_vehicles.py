import torch
from ultralytics import YOLO
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import inception_v3
from sklearn.cluster import AgglomerativeClustering

# Check if GPU is available
print("Checking GPU availability...")
if torch.cuda.is_available():
    print("GPU is available")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")

# Load a pretrained YOLOv8s model
model = YOLO("best.pt")

# Load a pre-trained Inception_v3 model
feature_extractor = inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
# Remove the final classification layer
feature_extractor.classifier = torch.nn.Identity()
feature_extractor.eval()
feature_extractor.to(device)

# Define image transformation
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Define vehicle class map and IDs
vehicle_class_rmap = {
    0:'Bicycle',
    1:'Bus',
    2:'Car',
    3:'LCV',
    4:'Three-Wheeler',
    5:'Truck',
    6:'Two-Wheeler',
} 
# Define turns map for each cam_id
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

def detect_vehicles(cam_id, video_file):

    vcam_id = video_file.split('/')[-1].split('.')[0]
    print(vcam_id)
    if vcam_id not in cam_id_turns_map:
        cam_id_turns_map[vcam_id] = ['AB', 'BA']
    cam_id_turns_map[cam_id] = cam_id_turns_map[vcam_id]

    csv_file=f'{cam_id}.csv'
    
    results = model.track(
        source=video_file,
        device=device,
        imgsz = 1152,
        conf = 0.1,
        iou = 0.5,
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

        frame = result.orig_img
        boxes = result.boxes

        for ob in range(boxes.shape[0]):
            cls_id = int(boxes.cls[ob].item())
            v_id = int(boxes.id[ob].item()) if boxes.id is not None else None

            x, y, w, h = map(int, boxes.xywh[ob])
            x1, y1, x2, y2 = map(int, boxes.xyxy[ob])

            ob_crop = frame[y1:y2, x1:x2]
            if ob_crop.size == 0:
                continue

            # Apply image transformations and extract features
            roi = preprocess(ob_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                features = feature_extractor(roi).squeeze().cpu()
                
            features_dict = {
                "fn": fn,
                "v_id": v_id,
                "cls_id": cls_id,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "features": features,
            }
            instance_dict[ob_id] = features_dict
            ob_id += 1

    idf = pd.DataFrame(instance_dict).T
    idf['fn'] = idf['fn'].astype(int)
    idf['cls_id'] = idf['cls_id'].astype(int)
    idf['v_id'] = idf['v_id'].astype(float)

    tdf_drop = idf.groupby('v_id')['fn'].count().reset_index()
    vdrop = tdf_drop.loc[tdf_drop['fn'] < 2]['v_id'].values.tolist() # minimum of 2 instances filter
    idf.drop(idf.loc[idf['v_id'].isin(vdrop + [np.nan])].index, inplace=True)
    idf.reset_index(drop=True, inplace=True)

    vidfx = idf.groupby('v_id')['x'].aggregate(['first','last']).reset_index()
    vidfy = idf.groupby('v_id')['y'].aggregate(['first','last']).reset_index()
    vidft = idf.groupby('v_id')['fn'].median().astype(int).reset_index()
    vdf = pd.merge(vidfx, vidfy, how='inner', on='v_id', suffixes=['_x', '_y'])
    vdf = pd.merge(vdf, vidft, how='inner', on='v_id')
    vdf['delta_x'] = vdf['first_x'] - vdf['last_x']
    vdf['delta_y'] = vdf['first_y'] - vdf['last_y']
    vids = vdf.loc[(vdf['delta_x'].abs()>5) & (vdf['delta_x'].abs()>5)]['v_id'] # Ignore vehicles which haven't moved at least 5 pixels

    vdf = vdf.loc[vdf['v_id'].isin(vids)]
    idf = idf.loc[idf['v_id'].isin(vids)]

    tvtype_df = idf.groupby(['v_id'])['cls_id'].value_counts().reset_index()
    idxs = tvtype_df.groupby(['v_id'])['count'].idxmax()
                
    vdf['cls_id'] = tvtype_df.loc[idxs]['cls_id'].values
    vdf['vehicle'] = vdf['cls_id'].map(vehicle_class_rmap)

    dcols = ['first_x', 'last_x', 'first_y', 'last_y']
    tvdf = vdf.drop(columns = dcols).merge(idf.drop(columns=['cls_id']), on=['v_id', 'fn'])
    tvdf.drop(columns=['v_id'], inplace=True)

    turns_list = cam_id_turns_map[cam_id]

    tvdf.reset_index(names='v_id', inplace=True)

    tvdf['theta'] = np.arctan2(tvdf['delta_y'].astype(int), tvdf['delta_x'].astype(int))
    tvdf['cos_theta'] = np.cos(tvdf['theta'])
    tvdf['sin_theta'] = np.sin(tvdf['theta'])
    tvdf['angle'] = np.rad2deg(tvdf['theta'])

    agc = AgglomerativeClustering(n_clusters=len(turns_list))
    tvdf['cluster'] = agc.fit_predict(tvdf[['cos_theta', 'sin_theta']])
    cluster_centers = tvdf.groupby('cluster')['angle'].median().reset_index()
    cluster_centers.sort_values(by='angle', key=lambda x: (x - 90) % 360, inplace=True)
    cluster_centers.reset_index(drop=True, inplace=True)
    cluster_centers['Turns'] = turns_list
    cluster_map = cluster_centers[['cluster','Turns']].set_index('cluster').to_dict()['Turns']

    tvdf['Turning Patterns'] = tvdf['cluster'].map(cluster_map)

    cols = ['fn',
    'cls_id',
    'vehicle',
    'features',
    'x',
    'y',
    'w',
    'h',
    'Turning Patterns']

    tvdf['features'] = tvdf['features'].apply(lambda x: x.tolist())
    tvdf[cols].to_csv(csv_file, index=False)