import torch
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
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

# Define vehicle class map and IDs
## We currently don't have LCV and Three-Wheeler
vehicle_class_map = {1: "Bicycle", 2: "Car", 3: "Two-Wheeler", 5: "Bus", 7: "Truck"}
vehicle_class_ids = list(vehicle_class_map.keys())
vehicle_class_rmap = {
    "Car": 2,
    "Bus": 5,
    "Truck": 7,
    "Three-Wheeler": 4,
    "Two-Wheeler": 5,
    "LCV": 6,
    "Bicycle": 1,
}

model = YOLO("yolov10n.pt")
conf = 0.25

# Load a pre-trained ResNet18 model
feature_extractor = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
# Remove the final classification layer
feature_extractor.classifier = torch.nn.Identity()
feature_extractor.eval()
feature_extractor.to(device)

# Define image transformation
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# High overlap
rectangles_dict = {
    "a": {"tl": {"x": 0, "y": 500}, "br": {"x": 1020, "y": 1080}},
    "b": {"tl": {"x": 0, "y": 150}, "br": {"x": 1020, "y": 500}},
    "c": {"tl": {"x": 350, "y": 0}, "br": {"x": 1020, "y": 500}},
    "d": {"tl": {"x": 1020, "y": 0}, "br": {"x": 1600, "y": 500}},
    "e": {"tl": {"x": 1020, "y": 150}, "br": {"x": 1920, "y": 500}},
    "f": {"tl": {"x": 1020, "y": 500}, "br": {"x": 1920, "y": 1080}},
}

def apply_column_overrides(df, column_pairs, dist_map, dist_mapr):
    df_copy = df.copy()
    for _, row in df_copy.iterrows():
        for idx in range(len(column_pairs)):
            src_col, dst_col = column_pairs[idx]
            # Check if source column has True value
            if row[src_col]:
                # Check if destination columns have True values
                if row[dst_col[0]] and row[dst_col[1]]:
                    # Determine which destination column to override based on distance
                    max_dist = max(dist_map[src_col+dst_col[0]], dist_map[src_col+dst_col[1]])
                    dst_col_to_set = dist_mapr[max_dist][-1]
                    # Apply the override rules
                    row[dst_col_to_set] = False
        
    return df_copy

# Create ordinal difference mapping
def create_distance_map(columns):
    dist_map = {}
    col_list = list(columns)
    for i, col1 in enumerate(col_list):
        for j, col2 in enumerate(col_list):
            if i != j:
                # Compute absolute difference between positions
                dist = abs(i - j)
                dist_map[col1 + col2] = dist
    return dist_map

def detect_vehicles(video_file, csv_file='counts.csv'):
    results = model(
        source=video_file,
        conf=conf,
        device=device,
        classes=vehicle_class_ids,
        vid_stride=1,
        stream=True,
        verbose=False,
    )

    ob_id = 0
    instance_dict = {}

    for fn, result in enumerate(results):
        frame = result.orig_img
        boxes = result.boxes
        names = result.names

        # tic = time.time()
        for ob in range(boxes.shape[0]):
            cls_id = int(boxes.cls[ob].item())
            if cls_id not in vehicle_class_ids:
                continue

            x1, y1, x2, y2 = map(int, boxes.xyxy[ob])
            top_left = {"x": x1, "y": y1}
            bottom_right = {"x": x2, "y": y2}

            ob_crop = frame[y1:y2, x1:x2]
            if ob_crop.size == 0:
                continue

            # Apply image transformations and extract features
            roi = preprocess(ob_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                features = feature_extractor(roi).squeeze().cpu()
                # features = feature_extractor(roi).squeeze().cpu().numpy()

            features_dict = {
                "fn": fn,
                "name": vehicle_class_map[cls_id],
                "cls_id": cls_id,
                "features": features,
                "tl": top_left,
                "br": bottom_right,
            }
            instance_dict[ob_id] = features_dict
            ob_id += 1

    idf = pd.DataFrame(instance_dict).T

    # Convert features to a single tensor for faster computation
    features_tensor = torch.tensor(np.stack(idf["features"].values)).to(device)

    # Initialize variables
    thresh = 50.0
    n_features = features_tensor.shape[1]
    vdf = torch.empty((0, n_features), dtype=torch.float32).to(device)
    count_dict = {}

    # Process each set of features
    for ob_id, features in enumerate(features_tensor):
        if vdf.shape[0] == 0:
            # Initialize with the first set of features if empty
            vdf = torch.cat([vdf, features.unsqueeze(0)]).to(device)
            count_dict[0] = [idf.index[ob_id]]
        else:
            # Compute Euclidean distances using broadcasting
            distances = torch.sqrt(((vdf - features) ** 2).sum(dim=1)).to(device)
            matches = distances < thresh
            if matches.any():
                vid = matches.nonzero(as_tuple=True)[0][0].item()
            else:
                vid = len(vdf)
                vdf = torch.cat([vdf, features.unsqueeze(0)]).to(device)
            if vid not in count_dict:
                count_dict[vid] = []
            count_dict[vid].append(idf.index[ob_id])

    cdf = pd.DataFrame.from_dict(count_dict, orient="index")

    vtlx = idf["tl"].apply(lambda x: x["x"])
    vtly = idf["tl"].apply(lambda x: x["y"])
    vbrx = idf["br"].apply(lambda x: x["x"])
    vbry = idf["br"].apply(lambda x: x["y"])

    bbox_dict = {}
    for bbid, bbox in rectangles_dict.items():
        rtlx = bbox["tl"]["x"]
        rtly = bbox["tl"]["y"]
        rbrx = bbox["br"]["x"]
        rbry = bbox["br"]["y"]

        # Create a mask for each corner point being inside the bounding box
        tl_inside = (rtlx <= vtlx) & (vtlx <= rbrx) & (rtly <= vtly) & (vtly <= rbry)
        br_inside = (rtlx <= vbrx) & (vbrx <= rbrx) & (rtly <= vbry) & (vbry <= rbry)

        # Aggregate the results, assuming you want to know if any corner is inside
        any_inside = tl_inside | br_inside
        bbox_dict[bbid] = any_inside

    bbox_df = pd.DataFrame(bbox_dict)

    # Flatten cdf from wide to long format, preserving the original row indices
    cdf_melted = cdf.reset_index().melt(
        id_vars="index", var_name="Variable", value_name="Value"
    )
    cdf_melted = cdf_melted.drop(
        columns="Variable"
    )  # We don't need the variable column anymore

    # Filter out only the rows in cdf_melted where 'Value' matches any index in idf
    filtered_cdf = cdf_melted[cdf_melted["Value"].isin(idf.index)]

    # Create vids list initialized to False
    vids = [False] * len(idf)

    # Update vids based on filtered_cdf
    # This will set True at the position in vids if the corresponding idf index was found in any row of cdf
    for value, group in filtered_cdf.groupby("Value"):
        if value in idf.index:
            vids[int(value)] = (
                True  # Set True where the idf index matches the Value in cdf
            )

    # If you need the index of cdf where each idf index was found, we need to adjust this:
    vids = [None] * len(idf)
    for value, group in filtered_cdf.groupby("Value"):
        if value in idf.index:
            vids[int(value)] = group["index"].tolist()[
                0
            ]  # Store the list of cdf indices where each idf index was found

    bbox_df["v_id"] = vids

    tr_df = bbox_df.groupby("v_id").max().reset_index(drop=True)

    columns = tr_df.columns

    # Create the distance map
    dist_map = create_distance_map(columns)

    # Create reverse mapping for distances
    dist_mapr = {dist: pair for pair, dist in dist_map.items()}


    # Define column pairs for overriding (Update column pairs according to rules)
    src_dst_pairs = [("b", ("c", "e")), ("d", ("e", "a")), ("f", ("a", "c"))]

    r_df = apply_column_overrides(tr_df, src_dst_pairs, dist_map, dist_mapr)
    vtype_df = pd.merge(
        bbox_df["v_id"], idf["name"], how="inner", right_index=True, left_index=True
    )
    r_df["vehicle"] = vtype_df.groupby("v_id")["name"].max()

    column_pairs = [
        ("b", "c"),
        ("b", "e"),
        ("d", "e"),
        ("d", "a"),
        ("f", "a"),
        ("f", "c"),
    ]

    # Initialize a dictionary to hold the counts of each combination
    combination_counts = {pair: 0 for pair in column_pairs}

    t_df = pd.DataFrame(vehicle_class_rmap.keys(), columns=["vehicle"])
    for vehicle in r_df["vehicle"].unique():
        tv_df = r_df.loc[r_df["vehicle"] == vehicle]
        # Iterate through each pair and count how many times both columns are True
        for pair in column_pairs:
            col1, col2 = pair
            combination_counts[pair] = (
                (tv_df[col1] == True) & (tv_df[col2] == True)
            ).sum()
            t_df.loc[t_df["vehicle"] == vehicle, "".join(pair).upper()] = (
                combination_counts[pair]
            )

    f_df = t_df.copy().rename(columns={"vehicle": "Turning Patterns"})
    f_df = f_df.set_index("Turning Patterns").T
    f_df = f_df.fillna(0).astype(int).reset_index(names='Turning Patterns')
    f_df.to_csv(csv_file, index=False)
