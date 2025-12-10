class Config:
    
    model_name = "google/owlvit-base-patch16"
    model_family = "owlvit"
    device = "cuda"
    box_threshold = 0.3
    
    _root_path = f"/Users/user/Desktop/DataProcessor"
    
    frames_dir = f"{_root_path}/Segmentor/data/video"
    meta_path = f"{frames_dir}/metadata.json"
    rs_path = f"{_root_path}/VisualProcessor/result_store"