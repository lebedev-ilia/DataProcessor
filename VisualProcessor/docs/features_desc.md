object_detection:
    Модели:
        YOLO("yolo11x.pt")

    Для каждого кадра:
        "class": "person",
        "conf": 0.9593384861946106,
        "box": [43.025848388671875, 180.14501953125, 1077.9505615234375, 1914.7108154296875]

    Алгоритм:
        for start in range(0, len(frame_indices), batch_size):
            end = start + batch_size
            batch_indices = frame_indices[start:end]

            frames = [frame_manager.get(i) for i in batch_indices]

            preds = model.predict(
                frames, 
                stream=False,
                verbose=False
            )

            for frame_index, pred in zip(batch_indices, preds):
                results[frame_index] = []
                boxes = pred.boxes.cpu()

                for box in boxes:
                    conf = float(box.conf)
                    if conf < threshold:
                        continue

                    cls_id = int(box.cls)
                    cls_name = names.get(cls_id, str(cls_id))
                    xyxy = box.xyxy.tolist()[0]

                    results[frame_index].append({
                        "class": cls_name,
                        "conf": conf,
                        "box": xyxy
                    })

scene_classification:
    Модели:
        categories     - categories_places365.txt
        clip_model     - CLIPModel.from_pretrained("clip_vit_base_patch32")
        clip_processor - CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model          - default: resnet18, resnet50 | timm: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, convnext_tiny, convnext_small, convnext_base, vit_base_patch16_224, vit_large_patch16_224, regnetx_002, regnetx_004, regnetx_006, resnet50, resnet101
    Выход:
        {
            "scene_1": {                                     - формируеться как - f"{scene_label}_{last_frame_num}"
                "indices": ["...indices..."],
                "mean_score": 0.913928210735321,
                "mean_indoor": 1.0,
                "mean_outdoor": 0.0,
                "mean_nature": 0.0,
                "mean_urban": 1.0,
                "mean_morning": 0.23584198906735004,
                "mean_day": 0.267079332173882,
                "mean_evening": 0.2817385487868892,
                "mean_night": 0.21534012997187874,
                "mean_aesthetic_score": 0.5977593447480883,
                "mean_luxury_score": 0.682619475892612,
                "mean_cozy": 0.1042156663856336,
                "mean_scary": 0.08533628497804914,
                "mean_epic": 0.06016373275113957,
                "mean_neutral": 0.7502843354429517,
                "mean_openness": 0.6850459717782398,
                "mean_clutter": 0.07841400738536154,
                "mean_depth_cues": 0.259309675461476
            },
            "scene_2": {...},
        }

    indices
        Индексы фрэймов на сцену 
    mean_score
        увереность модели по кадрам усредненая по сцене
    mean_indoor
    mean_outdoor
        Значения усредненные по сцене
        Для одного фрэйма:
            indoor_keywords = ["room", "bedroom", "kitchen", "bathroom", "living", "dining", "office","hall", "corridor", "staircase", "attic", "basement", "garage", "shop","store", "mall", "restaurant", "cafe", "bar", "pub", "hospital", "school","classroom", "library", "museum", "theater", "cinema", "gym", "stadium","airport", "station", "subway", "train", "bus", "indoor"]
            outdoor_keywords = ["outdoor", "street", "road", "highway", "bridge", "park", "garden","forest", "beach", "mountain", "desert", "field", "farm", "lake", "river","ocean", "sea", "sky", "cloud", "sunset", "sunrise", "outdoor"]
            scene_label - предсказаный моделью label, например: "aquarium", "arcade"
            label_lower = scene_label.lower()
            indoor_score = sum(1 for keyword in indoor_keywords if keyword in label_lower)
            outdoor_score = sum(1 for keyword in outdoor_keywords if keyword in label_lower)
            total = indoor_score + outdoor_score
            if total == 0:
                if any(word in label_lower for word in ["room", "hall", "indoor"]):
                    return {"indoor": 0.7, "outdoor": 0.3}
                else:
                    return {"indoor": 0.5, "outdoor": 0.5}
            indoor_prob = indoor_score / total
            outdoor_prob = outdoor_score / total
    mean_nature
    mean_urban
        Значения усредненные по сцене
        Для одного фрэйма:
            nature_keywords = ["forest", "jungle", "wood", "tree", "beach", "coast", "shore", "mountain","hill", "valley", "desert", "field", "meadow", "grass", "flower", "garden","park", "lake", "river", "stream", "waterfall", "ocean", "sea", "island","cave", "canyon", "cliff", "rock", "snow", "ice", "sky", "cloud", "sunset","sunrise", "nature", "wild", "natural"]
            urban_keywords = ["city", "urban", "street", "road", "avenue", "boulevard", "alley", "plaza","square", "building", "skyscraper", "tower", "bridge", "highway", "subway","station", "airport", "mall", "shop", "store", "restaurant", "cafe", "bar","hotel", "office", "factory", "warehouse", "parking", "lot", "urban"]
            label_lower = scene_label.lower()
            nature_score = sum(1 for keyword in nature_keywords if keyword in label_lower)
            urban_score = sum(1 for keyword in urban_keywords if keyword in label_lower)
            total = nature_score + urban_score
            if total == 0:
                return {"nature": 0.5, "urban": 0.5}
            nature_prob = nature_score / total
            urban_prob = urban_score / total
    mean_morning
    mean_day
    mean_evening
    mean_night
        Значения усредненные по сцене
        Для одного фрэйма:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].flatten()
        warm_pixels = np.sum((hue < 30) | (hue > 150))
        warm_ratio = warm_pixels / len(hue) if len(hue) > 0 else 0
        morning_score = mean_brightness * 0.6 * (1 + warm_ratio * 0.5)
        day_score = mean_brightness * (1 - warm_ratio * 0.3)
        evening_score = (1 - mean_brightness) * 0.7 * (1 + warm_ratio * 0.8)
        night_score = (1 - mean_brightness) * (1 - warm_ratio * 0.5)
        total = morning_score + day_score + evening_score + night_score
        if total == 0:
            return {"morning": 0.25, "day": 0.25, "evening": 0.25, "night": 0.25}
        return {
            "morning": morning_score / total,
            "day": day_score / total,
            "evening": evening_score / total,
            "night": night_score / total
        }
    mean_aesthetic_score
        Значение усреднено по сцене
        Для одного кадра:
            если включена clip модель:
                texts = ["aesthetic beautiful scene","professional photography","ugly unappealing scene","amateur photography"]
                inputs = _clip_processor(text=texts,images=image,return_tensors="pt",padding=True).to(device)
                with torch.no_grad():
                    outputs = _clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                aesthetic_score = (probs[0][0] + probs[0][1]).item()
                return float(aesthetic_score)
            Если не включена clip (эвристика):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_score = min(1.0, laplacian_var / 500.0)
                contrast = np.std(gray) / 255.0
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_flat = rgb.reshape(-1, 3)
                std_r = np.std(rgb_flat[:, 0])
                std_g = np.std(rgb_flat[:, 1])
                std_b = np.std(rgb_flat[:, 2])
                colorfulness = (std_r + std_g + std_b) / 3.0 / 255.0
                mean_brightness = np.mean(gray) / 255.0
                brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2.0
                aesthetic = (sharpness_score * 0.3 + contrast * 0.3 + colorfulness * 0.2 + brightness_score * 0.2)
                return float(np.clip(aesthetic, 0.0, 1.0))
    mean_luxury_score
        Значение усреднено по сцене
        Для одного кадра:
            если включена clip модель:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                texts = ["luxury expensive high-end scene","premium elegant sophisticated","cheap low-quality scene","budget affordable scene"]
                inputs = _clip_processor(text=texts,images=image,return_tensors="pt",padding=True).to(device)
                with torch.no_grad():
                    outputs = _clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                luxury_score = (probs[0][0] + probs[0][1]).item()
                return float(luxury_score)
            Если не включена clip (эвристика):
                label_lower = scene_label.lower()
                luxury_keywords = ["luxury", "premium", "elegant", "sophisticated", "high-end", "expensive"]
                label_score = 0.3 if any(kw in label_lower for kw in luxury_keywords) else 0.0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                quality_score = min(1.0, sharpness / 500.0) * 0.4
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                std_colors = np.std(rgb.reshape(-1, 3), axis=0)
                color_score = np.mean(std_colors) / 255.0 * 0.3
                return float(np.clip(label_score + quality_score + color_score, 0.0, 1.0))
    mean_cozy
    mean_scary
    mean_epic
    mean_neutral
        Значения усреднены по сцене
        Для одного кадра:
            если включена clip модель:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                texts = ["cozy warm comfortable scene","scary frightening dark scene","epic grand majestic scene","neutral ordinary scene"]
                inputs = _clip_processor(text=texts,images=image,return_tensors="pt",padding=True).to(device)
                with torch.no_grad():
                    outputs = _clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                return {"cozy": float(probs[0][0].item()),"scary": float(probs[0][1].item()),"epic": float(probs[0][2].item()),"neutral": float(probs[0][3].item())}
            Если не включена clip (эвристика):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray) / 255.0
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hue = hsv[:, :, 0].flatten()
                warm_ratio = np.sum((hue < 30) | (hue > 150)) / len(hue) if len(hue) > 0 else 0
                cozy_score = mean_brightness * 0.6 * (1 + warm_ratio * 0.5)
                contrast = np.std(gray) / 255.0
                scary_score = (1 - mean_brightness) * 0.7 * (1 + contrast * 0.5)
                dynamic_range = (np.max(gray) - np.min(gray)) / 255.0
                epic_score = mean_brightness * 0.8 * (1 + dynamic_range * 0.3)
                total = cozy_score + scary_score + epic_score
                if total == 0:
                    return {"cozy": 0.33, "scary": 0.33, "epic": 0.34, "neutral": 0.0}
                return {"cozy": float(cozy_score / total),"scary": float(scary_score / total),"epic": float(epic_score / total),"neutral": float(1.0 - (cozy_score + scary_score + epic_score) / total)}
    mean_openness
    mean_clutter
    mean_depth_cues
        Значения усреднены по сцене
        Для одного кадра:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            top_portion = gray[:height//3, :]
            top_brightness = np.mean(top_portion) / 255.0
            openness = top_brightness * 0.6 + (1 - np.std(gray) / 255.0) * 0.4
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            clutter = min(1.0, edge_density * 2.0)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            depth_cues = min(1.0, np.mean(gradient_magnitude) / 100.0)
            return {"openness": float(np.clip(openness, 0.0, 1.0)),"clutter": float(np.clip(clutter, 0.0, 1.0)),"depth_cues": float(np.clip(depth_cues, 0.0, 1.0))}
        
face_detection:
    Модели:
        insightface.app.FaceAnalysis
    Выход:
        {
            "frames_with_face": [
                0,
                5,
                ...
            ]
        }
    Для одного кадра:
        faces = face_app.get(frame_bgr)
        if not faces:
            return False
        best = max(safe_det_score(f) for f in faces)
        return best >= thr

detalize_face_modules:
    Модели:
    Выход:
        {
            "0": [
                    {
                        "frame_index": 0,
                        "face_index": 0,
                        "bbox": [399.2803649902344, 415.4458312988281, 835.0166015625, 874.3395385742188],
                        "geometry": {
                            "face_bbox_area": 199956.61699487362,
                            "face_relative_size": 0.09642969569578376,
                            "face_box_ratio": 0.9495363080033962,
                            "face_bbox_position": {"cx": 617.1484832763672, "cy": 644.8926849365234},
                            "face_dist_from_center": 0.16896567724351552,
                            "face_rotation_in_frame": -8.832769393920898,
                            "aspect_ratio_stability": 0.0,
                            "jaw_width": 330.4817810058594,
                            "cheekbone_width": 432.2557373046875,
                            "cheek_width": 424.3172912597656,
                            "forehead_height": 166.32730102539062,
                            "face_shape_vector": [630.7454223632812, ...and 68 values... 432.874267578125]
                        },
                        "pose": {
                            "yaw": 6.300705890788989,
                            "pitch": -17.35517438503278,
                            "roll": -8.068668365478516,
                            "avg_pose_angle": -6.374378953240768,
                            "head_pose_variability": 0.0,
                            "pose_stability_score": 0.8599843135380225,
                            "head_turn_frequency": 0.0,
                            "attention_to_camera_ratio": 0.9384570096135878,
                            "looking_direction_vector": [188.4417724609375, -26.714111328125, 46.70124053955078]
                        },
                        "quality": {
                            "face_blur_score": 0.06484930049356688,
                            "sharpness_score": 0.00678416907787323,
                            "texture_quality": 0.17633497723734562,
                            "focus_metric": 0.17633497723734562,
                            "noise_level": 0.038173782825469973,
                            "motion_blur_score": 0.9610904197038599,
                            "artifact_score": 0.8236650227626544,
                            "resolution_of_face": 199956.609375,
                            "face_visibility_ratio": 0.09642969202112268,
                            "occlusion_score": 0.9035703079788773,
                            "quality_proxy_score": 0.15942458808571086
                        },
                        "lighting": {
                            "average_lighting_on_face": 0.5535850375306373,
                            "light_uniformity_score": 0.9283180386412377,
                            "face_contrast": 0.7607843137254902,
                            "white_balance_shift": {"r_minus_g": -0.325473830279182, "r_minus_b": -0.1098240571863511, "g_minus_b": 0.21564977309283087},
                            "skin_color_vector": [91.00738525390625, 149.82424926757812],
                            "highlight_intensity": 0.9607843137254902,
                            "shadow_depth": 0.0,
                            "glare_score": 0.8352941176470589,
                            "shading_score": 0.9254901960784314,
                            "skin_tone_index": 1,
                            "lighting_proxy_score": 0.7520862893497242,
                            "zone_lighting": {"forehead_brightness": 0.4262352438534007, "cheek_brightness": 0.6280032288794424, "chin_brightness": 0.4654477287741268}
                        },
                        "skin": {
                            "makeup_presence_prob": 0.0,
                            "lipstick_intensity": 0.0,
                            "eye_shadow_prob": 0.0,
                            "skin_smoothness": 0.6753284717684971,
                            "skin_defect_score": 48.07609064390233,
                            "beard_prob": 0.602679947023671,
                            "mustache_prob": 0.4218759629165697,
                            "face_hair_density": 0.5122779549701204,
                            "eyebrow_shape_vector": [57.296722412109375, 137.78384399414062, 402.7089538574219, 82.4332275390625],
                            "eyelid_openness": 23.402135848999023
                        },
                        "accessories": {
                            "glasses_prob": 0.15,
                            "sunglasses_prob": 0.7283823529395642,
                            "mask_prob": 0.09963344644481419,
                            "hat_prob": 0.3827916733725834,
                            "helmet_prob": 0.0,
                            "earrings_presence": false,
                            "earrings_prob": 0.02601626014145019,
                            "jewelry_probability": 0.24869254758225162,
                            "hair_accessories": 0.0
                        },
                        "eyes": {
                            "eye_opening_ratio": {"left": 23.23587417602539, "right": 23.568397521972656, "average": 23.402135848999023},
                            "blink_rate": 0.0,
                            "blink_intensity": 0.0,
                            "gaze_vector": [0.10475023421914995, -0.2982941465317203, 0.9487085909677143],
                            "gaze_at_camera_prob": 0.7899764703070337,
                            "attention_score": 12.09605615965303,
                            "eye_redness_prob": 1.0,
                            "iris_position": {"left": 0.28678393061378304, "right": 0.2929455951149514}
                        },
                        "motion": {
                            "face_speed": 0.0,
                            "face_acceleration": 0.0,
                            "micro_expression_rate": 0.0,
                            "jaw_movement_intensity": 0.0,
                            "eyebrows_motion_score": 0.0,
                            "mouth_motion_score": 0.0,
                            "head_motion_energy": 0.0,
                            "talking_motion_score": 0.0
                        },
                        "structure": {
                            "face_mesh_vector": [0.02699216641485691, ...298 values... -98.80367279052734],
                            "identity_shape_vector": [0.02699216641485691, ...148 values... -109.82438659667969],
                            "expression_vector": [-0.36291444301605225, ...148 values... 11.020713806152344],
                            "jaw_pose_vector": [6.300705890788989, -17.35517438503278, -8.068668365478516],
                            "eye_pose_vector": [-17.35517438503278, -8.068668365478516],
                            "mouth_shape_params": [0.9495363235473633, 1.0],
                            "face_symmetry_score": 0.7370345642991275,
                            "face_uniqueness_score": 0.21873119473457336,
                            "identity_params_count": 150,
                            "expression_params_count": 150,
                            "face_dimensions": {"width": 0.9495363235473633, "height": 1.0, "depth": 677.1336059570312, "aspect_ratio": 0.9495363235473633}
                        },
                        "professional": {
                            "face_quality_score": 0.012766196089568305,
                            "perceived_attractiveness_score": 0.5,
                            "emotion_intensity": 0.0,
                            "lip_reading_features": {"mouth_openness": 0.0, "mouth_motion_intensity": 0.0, "jaw_movement": 0.0, "speech_activity_prob": 0.0},
                            "fatigue_score": 0.31152648970569663,
                            "fatigue_breakdown": {
                                "eye_fatigue": 0.30282177305977004,
                                "pose_fatigue": 0.01846289711592366,
                                "motion_fatigue": 0.8,
                                "temporal_fatigue": 0.0,
                                "eye_closedness": 0.0,
                                "head_pitch_down": 0.0,
                                "low_motion_speed": 1.0,
                                "blink_abnormality": 1.0
                            },
                            "engagement_level": 1.0,
                            "alertness_score": 1.0,
                            "expressiveness_score": 0.0
                        },
                        "lip_reading": {
                            "mouth_width": 149.02044677734375,
                            "mouth_height": 3.4132251739501953,
                            "mouth_area": 399.4851891593844,
                            "mouth_aspect_ratio": 0.022904408406787324,
                            "lip_separation": 3.4132251739501953,
                            "lip_asymmetry": 5.119696455245047e-08,
                            "lip_contour_compactness": 222.96712394074865,
                            "lip_prominence_ratio": 0.8110154271125793,
                            "phoneme_features": {"round_shape": 0.022904408406787324, "wide_shape": 1.0, "narrow_shape": 0.9770955915932127, "open_shape": 0.022904408406787324},
                            "mouth_motion_intensity": 0.0,
                            "width_velocity": 0.0,
                            "height_velocity": 0.0,
                            "area_velocity": 0.0,
                            "cycle_strength": 0.0,
                            "speech_activity_prob": 0.0,
                            "width_variance": 0.0,
                            "height_variance": 0.0,
                            "area_variance": 0.0,
                            "separation_variance": 0.0
                        },
                        "face_3d": {
                            "face_mesh_vector": [0.11840604245662689, ...298 values... 1.7961621284484863],
                            "identity_shape_vector": [-0.5973265171051025, ...198 values... 0.0],
                            "expression_vector": [5.086263143994074e-08, ...48 values... 0.0],
                            "jaw_pose_vector": [6.300705890788989, -17.35517438503278, -8.068668365478516],
                            "eye_pose_vector": [-17.35517438503278, -8.068668365478516],
                            "mouth_shape_params": [154.00621032714844, 8.721385955810547, -6.754088401794434, 17.658456019199924],
                            "face_symmetry_score": 0.5040507710822972,
                            "face_uniqueness_score": 1.0,
                            "mesh_num_vertices": 100,
                            "identity_params_count": 100,
                            "expression_params_count": 50
                        }
                    }
                ],
            "5": [...]
        }

    Фичи:

        frame_index
            индекс кадра относительно всех кадров видео
        face_index
            индекс лица (в одном кадре может быть более одного лица)
        bbox
            бокс лица
        geometry
            face_bbox_area
                width = float(bbox[2] - bbox[0])
                height = float(bbox[3] - bbox[1])
                area = width * height

            face_relative_size
                face_relative_size = area / (frame_width * frame_height + 1e-6),

            face_box_ratio
                ratio = width / max(height, 1e-6)

            face_bbox_position
                cx = float(bbox[0] + width / 2)
                cy = float(bbox[1] + height / 2)
                face_bbox_position = {"cx": cx, "cy": cy},

            face_dist_from_center
                frame_cx = frame_width / 2
                frame_cy = frame_height / 2
                face_dist_from_center = float(np.linalg.norm([cx - frame_cx, cy - frame_cy]) / max(frame_width, frame_height))

            face_rotation_in_frame
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                LANDMARKS = {"left_eye_outer": 33, "right_eye_outer": 263}
                for face_idx, landmark_list in enumerate(results.multi_face_landmarks):
                    if not landmark_list or not landmark_list.landmark:
                        return np.zeros((0, 3), dtype=np.float32)
                    diag = math.sqrt(width**2 + height**2)
                    coords = np.array([[lm.x * width, lm.y * height, lm.z * (diag if scale_z else 1.0)] for lm in landmark_list.landmark], dtype=np.float32)
                    left = coords[LANDMARKS["left_eye_outer"], :2]
                    right = coords[LANDMARKS["right_eye_outer"], :2]
                    face_rotation_in_frame = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))

            aspect_ratio_stability
                _aspect_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_size))
                _aspect_history[face_idx].append(ratio)
                history = _aspect_history[face_idx]
                stability = (float(1.0 - (np.std(history) / (np.mean(history) + 1e-5))) if len(history) > 3 else 0.0)

            jaw_width
                left_jaw = 172
                right_jaw = 397
                def _safe_distance(coords, i, j):
                    """Безопасное вычисление расстояния между landmarks."""
                    if i >= len(coords) or j >= len(coords):
                        return 0.0
                    return float(np.linalg.norm(coords[i][:2] - coords[j][:2]))
                jaw_width = _safe_distance(coords, left_jaw, right_jaw)

            cheekbone_width
                left_zygom = 127
                right_zygom = 356
                cheekbone_width = _safe_distance(coords, left_zygom, right_zygom)

            cheek_width
                left_cheek = 234
                right_cheek = 454
                cheek_width = _safe_distance(coords, left_cheek, right_cheek)

            forehead_height
                left_brow = 70
                right_brow = 301
                forehead_point = (coords[left_brow] + coords[right_brow]) / 2
                forehead_height = float(np.linalg.norm(forehead_point[:2] - nose_tip[:2]))

            face_shape_vector
                FACE_OVAL_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                face_shape_vector = coords[FACE_OVAL_LANDMARKS, :2].flatten().tolist()
        pose
            yaw
                LANDMARKS = {"nose_tip": 1, "chin": 152, "left_eye_inner": 133, "left_eye_outer": 33, "right_eye_inner": 362, "right_eye_outer": 263,}
                def _average_coords(coords: np.ndarray, indices: List[int]) -> np.ndarray:
                    if len(indices) == 0 or coords.size == 0:
                        return np.zeros((coords.shape[1],), dtype=np.float32)
                    return np.mean(coords[indices, :], axis=0)
                left_eye = _average_coords(coords, [LANDMARKS["left_eye_inner"], LANDMARKS["left_eye_outer"]])
                right_eye = _average_coords(coords, [LANDMARKS["right_eye_inner"], LANDMARKS["right_eye_outer"]])
                nose = coords[LANDMARKS["nose_tip"], :3]
                chin = coords[LANDMARKS["chin"], :3]
                forehead = coords[10, :3] if 10 < len(coords) else nose.copy()
                left_ear = coords[234, :3] if 234 < len(coords) else left_eye.copy()
                right_ear = coords[454, :3] if 454 < len(coords) else right_eye.copy()
                skull_width = np.linalg.norm(right_ear[:2] - left_ear[:2]) + 1e-6
                mid_eye = (left_eye + right_eye) / 2
                nose_horizontal_offset = nose[0] - mid_eye[0]
                yaw_2d = np.degrees(np.arctan2(nose_horizontal_offset, skull_width * 0.55))
                mid_face_z = (left_eye[2] + right_eye[2] + chin[2]) / 3
                depth_offset = nose[2] - mid_face_z
                yaw_3d = np.degrees(np.arctan2(nose_horizontal_offset, abs(depth_offset) + skull_width * 0.35))
                yaw = yaw_2d * 0.7 + yaw_3d * 0.3

            pitch
                eye_y = mid_eye[1]
                chin_y = chin[1]
                face_vertical_span = abs(chin_y - eye_y) + 1e-6
                expected_nose_y = eye_y + face_vertical_span * 0.35
                nose_vertical_offset = nose[1] - expected_nose_y
                pitch_2d = np.degrees(np.arctan2(nose_vertical_offset, face_vertical_span * 0.8))
                forehead_chin_vertical = forehead[1] - chin[1]
                pitch_3d = np.degrees(np.arctan2(forehead_chin_vertical, abs(depth_offset) + face_vertical_span * 0.3))
                pitch = pitch_2d * 0.65 + pitch_3d * 0.35

            roll
                roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

            avg_pose_angle
                avg_pose_angle = float(np.mean([yaw, pitch, roll]))

            head_pose_variability
                _pose_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
                if face_idx not in _pose_history:
                    _pose_history[face_idx] = deque(maxlen=30)
                current_pose = {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}
                _pose_history[face_idx].append(current_pose)
                hist = list(_pose_history[face_idx])
                if len(hist) > 1:
                    variability = float(np.mean([np.std([p["yaw"] for p in hist]), np.std([p["pitch"] for p in hist]), np.std([p["roll"] for p in hist])]))
                else:
                    variability = 0.0

            pose_stability_score
                pose_stability_score = float(np.clip(1 - abs(yaw) / 45.0, 0.0, 1.0))

            head_turn_frequency
                if len(hist) >= 3:
                    yaw_diffs = [abs(hist[i]["yaw"] - hist[i - 1]["yaw"]) for i in range(1, len(hist))]
                    turns = [1.0 if diff > 7.0 else 0.0 for diff in yaw_diffs]
                    turn_frequency = float(np.mean(turns))
                else:
                    turn_frequency = 0.0

            attention_to_camera_ratio
                attention_to_camera_ratio = float(np.exp(-((yaw / 25.0) ** 2)))

            looking_direction_vector
                looking_direction_vector = [float(right_eye[0] - left_eye[0]), float(right_eye[1] - left_eye[1]), float(right_eye[2] - left_eye[2])]

        quality
            face_blur_score
                x_min, y_min, x_max, y_max = map(int, bbox)
                if x_max < x_min:
                    x_min, x_max = x_max, x_min
                if y_max < y_min:
                    y_min, y_max = y_max, y_min
                h, w = frame.shape[:2]
                x_min = max(0, min(w - 1, x_min))
                x_max = max(0, min(w,     x_max))
                y_min = max(0, min(h - 1, y_min))
                y_max = max(0, min(h,     y_max))
                if (x_max - x_min) < 2 or (y_max - y_min) < 2:
                    return np.zeros((1, 1, 3), dtype=frame.dtype)
                roi = frame[y_min:y_max, x_min:x_max].copy()
                
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_score = float(np.clip(blur_raw / 300.0, 0.0, 1.0))

            sharpness_score
                sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
                sharpness = float(np.mean(np.abs(sobel)))
                sharpness_score = float(np.clip(sharpness / 200.0, 0.0, 1.0))

            texture_quality
                texture_quality = float(np.std(gray) / 255.0)

            focus_metric
                mean, std = cv2.meanStdDev(gray)
                focus_metric = float(np.clip(std[0][0] / 255.0, 0.0, 1.0))

            noise_level
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                noise = float(np.std(gray.astype(np.float32) - blurred.astype(np.float32)))
                noise_level = float(np.clip(noise / 40.0, 0.0, 1.0))

            motion_blur_score
                motion_blur_score = float(np.clip(1.0 - blur_raw / 500.0, 0.0, 1.0))

            artifact_score
                artifact_score = float(np.clip(1.0 - texture_quality, 0.0, 1.0))

            resolution_of_face
                frame_area = frame_shape[0] * frame_shape[1]
                face_area = max((bbox[2] - bbox[0]), 1) * max((bbox[3] - bbox[1]), 1)
                resolution_of_face = float(face_area)

            face_visibility_ratio
                ratio = face_area / max(frame_area, 1)
                face_visibility_ratio = float(np.clip(ratio, 0.0, 1.0))

            occlusion_score
                occlusion_score = float(np.clip(1.0 - face_visibility_ratio, 0.0, 1.0))

            quality_proxy_score
                quality_proxy_score = float(np.clip(0.4 * blur_score + 0.3 * sharpness_score + 0.2 * focus_metric + 0.1 * (1.0 - noise_level), 0.0, 1.0))
        lighting
            average_lighting_on_face
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                value = hsv[:, :, 2].astype(np.float32)
                avg_light = float(np.mean(value) / 255.0)

            light_uniformity_score
                h, w = roi.shape[:2]
                mid_x = w // 2
                left_mean = float(np.mean(value[:, :mid_x])) if mid_x > 1 else 0
                right_mean = float(np.mean(value[:, mid_x:])) if w - mid_x > 1 else 0
                uniformity = float(1.0 - abs(left_mean - right_mean) / 255.0)
                uniformity = float(np.clip(uniformity, 0.0, 1.0))

            face_contrast
                p5 = np.percentile(value, 5)
                p95 = np.percentile(value, 95)
                contrast = float(np.clip((p95 - p5) / 255.0, 0.0, 1.0))

            white_balance_shift
                mean_rgb = np.mean(roi.reshape(-1, 3), axis=0).astype(np.float32)
                b, g, r = mean_rgb
                white_balance_shift = {"r_minus_g": float((r - g) / 255.0), "r_minus_b": float((r - b) / 255.0), "g_minus_b": float((g - b) / 255.0)}

            skin_color_vector
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                L_channel = lab[:, :, 0].astype(np.float32)
                A_channel = lab[:, :, 1].astype(np.float32)
                B_channel = lab[:, :, 2].astype(np.float32)
                skin_vector = [float(np.mean(A_channel)), float(np.mean(B_channel))]

            highlight_intensity
            shadow_depth
            glare_score
            shading_score
                highlight = float(np.max(value) / 255.0)
                shadow = float(np.min(value) / 255.0)
                glare_score = float(np.clip(p95 / 255.0, 0.0, 1.0))
                shading_score = float(np.clip(1.0 - p5 / 255.0, 0.0, 1.0))

            skin_tone_index
                mean_L = float(np.mean(L_channel))
                if mean_L > 80:
                    skin_tone_index = 1
                elif mean_L > 70:
                    skin_tone_index = 2
                elif mean_L > 60:
                    skin_tone_index = 3
                elif mean_L > 50:
                    skin_tone_index = 4
                elif mean_L > 40:
                    skin_tone_index = 5
                else:
                    skin_tone_index = 6

            lighting_proxy_score
                lighting_proxy = float(np.clip(0.4 * avg_light + 0.3 * uniformity + 0.2 * contrast + 0.1 * (1.0 - shadow), 0.0, 1.0))

            zone_lighting
                LANDMARKS = {"nose_tip": 1, "chin": 152, "forehead": 10}
                zone_lighting = {}
                if coords_roi is not None and coords_roi.shape[0] > 0:
                    try:
                        yy = np.clip(coords_roi[:, 1].astype(int), 0, h - 1)
                        forehead_y = yy[LANDMARKS["forehead"]]
                        nose_y = yy[LANDMARKS["nose_tip"]]
                        chin_y = yy[LANDMARKS["chin"]]

                        top = max(0, int(forehead_y * 0.6))
                        mid = int(nose_y)
                        bot = int(chin_y)

                        if top < forehead_y:
                            zone = value[:forehead_y]
                            if zone.size > 0:
                                zone_lighting["forehead_brightness"] = float(np.mean(zone) / 255.0)

                        if forehead_y < mid:
                            zone = value[forehead_y:mid]
                            if zone.size > 0:
                                zone_lighting["cheek_brightness"] = float(np.mean(zone) / 255.0)

                        if mid < bot:
                            zone = value[mid:bot]
                            if zone.size > 0:
                                zone_lighting["chin_brightness"] = float(np.mean(zone) / 255.0)
                    except Exception:
                        pass
                if zone_lighting:
                    result["zone_lighting"] = zone_lighting
        skin
            makeup_presence_prob
            lipstick_intensity
            eye_shadow_prob
                LANDMARKS = {"upper_lip": 13, "lower_lip": 14,}
                bbox_x_min, bbox_y_min = bbox[0], bbox[1]
                coords_roi = coords.copy()
                coords_roi[:, 0] -= bbox_x_min
                coords_roi[:, 1] -= bbox_y_min
                lip_idx = [LANDMARKS["upper_lip"], LANDMARKS["lower_lip"]]
                lip_coords = coords_roi[lip_idx, :2]
                lip_x1 = int(max(0, np.min(lip_coords[:, 0])))
                lip_y1 = int(max(0, np.min(lip_coords[:, 1])))
                lip_x2 = int(min(w - 1, np.max(lip_coords[:, 0])))
                lip_y2 = int(min(h - 1, np.max(lip_coords[:, 1])))
                lip_roi = roi[lip_y1:lip_y2, lip_x1:lip_x2] if lip_x2 > lip_x1 and lip_y2 > lip_y1 else None
                if lip_roi is not None and lip_roi.size > 0:
                    lip_hsv = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2HSV)
                    lip_sat = float(np.mean(lip_hsv[:, :, 1] / 255.0))
                else:
                    lip_sat = 0.0
                lipstick_intensity = float(np.clip((lip_sat - avg_sat) * 1.5, 0.0, 1.0))
                makeup_presence_prob = lipstick_intensity
                eye_shadow_prob = float(lipstick_intensity * 0.4)

            skin_smoothness
            skin_defect_score
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()

                skin_smoothness = float(1 / (1 + laplacian_var * 0.1))
                skin_defect_score = float(min(max(laplacian_var * 10, 0), 100))

            beard_prob
            mustache_prob
            face_hair_density
            eyebrow_shape_vector
            eyelid_openness
                LANDMARKS = {
                    "chin": 152,
                    "left_brow": 70,
                    "right_brow": 300
                }
                chin_idx = LANDMARKS["chin"]
                chin_y = int(coords_roi[chin_idx, 1])
                lower_roi = roi[chin_y: h] if 0 < chin_y < h else roi

                if lower_roi.size > 0:
                    lower_lab = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2LAB)
                    L_lower = lower_lab[:, :, 0]
                    darkness = 1.0 - float(np.mean(L_lower) / 255.0)
                    texture = float(cv2.Laplacian(cv2.cvtColor(lower_roi, cv2.COLOR_BGR2GRAY),
                                                cv2.CV_64F).var() / 50.0)
                    beard_prob = float(np.clip(0.6 * darkness + 0.4 * texture, 0.0, 1.0))
                else:
                    beard_prob = 0.0

                mustache_prob = float(beard_prob * 0.7)
                face_hair_density = float((beard_prob + mustache_prob) / 2)
                eyebrow_pts = coords_roi[[LANDMARKS["left_brow"], LANDMARKS["right_brow"]], :2]
                eyebrow_shape_vector = eyebrow_pts.flatten().tolist()
                eyelid_openness = eye_opening(coords_roi)

        accessories
            glasses_prob
            sunglasses_prob
                h, w = roi.shape[:2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                glasses_prob = 0.0
                sunglasses_prob = 0.0
                try:
                    ex1, ey1, ex2, ey2 = eye_box(coords_roi)
                    ex1, ey1 = int(max(0, ex1)), int(max(0, ey1))
                    ex2, ey2 = int(min(w - 1, ex2)), int(min(h - 1, ey2))
                    eye_roi = gray[ey1:ey2, ex1:ex2] if ex2 > ex1 and ey2 > ey1 else np.array([], dtype=np.uint8)
                    if eye_roi.size:
                        edges = cv2.Canny(eye_roi, 50, 150)
                        edge_density = float(np.sum(edges > 0) / (eye_roi.size + 1e-6))
                        glasses_prob = float(np.clip(edge_density * 3.0, 0.0, 1.0))
                        mean_eye = float(np.mean(eye_roi))
                        sunglasses_prob = float(np.clip((1.0 - (mean_eye / 255.0)) * 1.2, 0.0, 1.0))
                        if sunglasses_prob > 0.4 and edge_density > 0.02:
                            sunglasses_prob = float(np.clip(sunglasses_prob + edge_density * 0.5, 0.0, 1.0))
                            glasses_prob = max(glasses_prob, 0.15)
                except Exception:
                    pass

            mask_prob
                mask_prob = 0.0
                try:
                    lx1, ly1, lx2, ly2 = lower_face_box(coords_roi)
                    lx1, ly1 = int(max(0, lx1)), int(max(0, ly1))
                    lx2, ly2 = int(min(w - 1, lx2)), int(min(h - 1, ly2))
                    lower_roi = roi[ly1:ly2, lx1:lx2] if lx2 > lx1 and ly2 > ly1 else np.array([], dtype=np.uint8)
                    if lower_roi.size:
                        lower_hsv = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2HSV)
                        v = lower_hsv[:, :, 2].astype(np.float32) / 255.0
                        v_std = float(np.std(v))
                        v_mean = float(np.mean(v))
                        color_homogeneity = 1.0 - v_std
                        coverage = float(lower_roi.size / (roi.size + 1e-6))
                        mask_prob = float(np.clip(color_homogeneity * 1.2 * coverage, 0.0, 1.0))
                except Exception:
                    pass

            hat_prob
            helmet_prob
                top_strip = roi[0:max(1, h // 6), :, :]
                top_gray = cv2.cvtColor(top_strip, cv2.COLOR_BGR2GRAY) if top_strip.size else np.array([], dtype=np.uint8)
                hat_prob = 0.0
                helmet_prob = 0.0
                if top_gray.size:
                    top_mean = float(np.mean(top_gray))
                    top_std = float(np.std(top_gray))
                    hat_prob = float(np.clip((1.0 - top_std / (np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) + 1e-6)) * (1.0 - top_mean / 255.0), 0.0, 1.0))
                    helmet_prob = float(np.clip(hat_prob * 0.4 if top_mean < 80 and top_std < 30 else 0.0, 0.0, 1.0))

            earrings_presence
                earrings_prob = 0.0
                try:
                    left_cheek = coords_roi[LANDMARKS["left_cheek"], :2].astype(int)
                    right_cheek = coords_roi[LANDMARKS["right_cheek"], :2].astype(int)
                    ear_y = int((left_cheek[1] + right_cheek[1]) / 2)
                    ear_region_size = min(30, max(10, w // 8))
                    def detect_ear_spot(cx):
                        x1 = max(0, cx - ear_region_size)
                        x2 = min(w, cx + ear_region_size)
                        y1 = max(0, ear_y - ear_region_size // 2)
                        y2 = min(h, ear_y + ear_region_size // 2)
                        region = roi[y1:y2, x1:x2]
                        if region.size == 0:
                            return 0.0
                        gray_r = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                        bright_frac = float(np.sum(gray_r > np.mean(gray_r) * 1.25) / (gray_r.size + 1e-6))
                        return bright_frac
                    left_spot = detect_ear_spot(int(left_cheek[0]))
                    right_spot = detect_ear_spot(int(right_cheek[0]))
                    earrings_prob = float(np.clip((left_spot + right_spot) / 2.0, 0.0, 1.0))
                except Exception:
                    pass
                earrings_presence = bool(earrings_prob > 0.03)


            earrings_prob

            jewelry_probability
                jewelry_probability = 0.0
                try:
                    chin_y = int(coords_roi[LANDMARKS["chin"], 1])
                    neck_y = min(h - 1, chin_y + int(h * 0.06))
                    if 0 <= neck_y < h:
                        neck_h = min(30, h - neck_y)
                        neck_roi = roi[neck_y:neck_y + neck_h, :]
                        if neck_roi.size:
                            neck_gray = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)
                            jewelry_probability = float(np.clip(np.std(neck_gray) / 80.0, 0.0, 1.0))
                except Exception:
                    pass

            hair_accessories
                hair_accessories = 0.0
                try:
                    forehead_y = int(coords_roi[LANDMARKS["forehead"], 1])
                    hair_top_y = max(0, forehead_y - 20)
                    hair_roi = roi[hair_top_y:forehead_y, :]
                    if hair_roi.size:
                        gray_hr = cv2.cvtColor(hair_roi, cv2.COLOR_BGR2GRAY)
                        hr_std = float(np.std(gray_hr))
                        whole_std = float(np.std(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
                        if whole_std > 1e-6:
                            hair_accessories = float(np.clip(max(0.0, (hr_std - whole_std * 1.2) / 60.0), 0.0, 1.0))
                except Exception:
                    pass

        eyes
            eye_opening_ratio
                LANDMARKS = {"left_eye_upper": 159, "left_eye_lower": 145, "right_eye_upper": 386, "right_eye_lower": 374}
                left_open = safe_distance(coords, LANDMARKS["left_eye_upper"], LANDMARKS["left_eye_lower"])
                right_open = safe_distance(coords, LANDMARKS["right_eye_upper"], LANDMARKS["right_eye_lower"])
                avg_open = (left_open + right_open) / 2.0
                "eye_opening_ratio": {"left": float(left_open), "right": float(right_open), "average": float(avg_open)}

            blink_rate
                def _estimate_blink_rate(history: deque) -> float:
                    """Оценка частоты моргания на основе истории открытия глаз."""
                    if len(history) < 2:
                        return 0.0
                    diffs = np.diff(list(history))
                    threshold = -0.5 * np.std(list(history)) if np.std(list(history)) > 0 else -0.01
                    blinks = np.sum(diffs < threshold)
                    return float(blinks / max(len(history), 1))

            blink_intensity
                blink_rate = _estimate_blink_rate(self._blink_history[face_idx])
                blink_intensity = float(np.std(list(self._blink_history[face_idx]))) if len(self._blink_history[face_idx]) > 1 else 0.0

            gaze_vector
                yaw = float(np.radians(pose.get("yaw", 0.0)))
                pitch = float(np.radians(pose.get("pitch", 0.0)))
                gaze_x = np.sin(yaw) * np.cos(pitch)
                gaze_y = np.sin(pitch)
                gaze_z = np.cos(yaw) * np.cos(pitch)
                gaze_vector = [float(gaze_x), float(gaze_y), float(gaze_z)]

            gaze_at_camera_prob
                gaze_at_camera_prob = float(np.clip(1 - abs(pose.get("yaw", 0)) / 30.0, 0.0, 1.0))

            attention_score
                attention_score = float((gaze_at_camera_prob + avg_open) / 2.0)

            eye_redness_prob
            iris_position
                eye_redness_prob = float(np.clip((left_open + right_open) / 40.0, 0.0, 1.0))
                left_width = safe_distance(coords, LANDMARKS["left_eye_inner"], LANDMARKS["left_eye_outer"])
                right_width = safe_distance(coords, LANDMARKS["right_eye_inner"], LANDMARKS["right_eye_outer"])
                left_iris = left_open / max(left_width, 1e-6)
                right_iris = right_open / max(right_width, 1e-6)
                iris_position = {"left": float(np.clip(left_iris, 0.0, 1.0)), "right": float(np.clip(right_iris, 0.0, 1.0)),}

        motion
            face_speed
                coords = data["coords"]
                geometry = data["geometry"] # из модуля geometry
                center = geometry["face_bbox_position"]["cx"], geometry["face_bbox_position"]["cy"]
                mouth_gap = safe_distance(coords, LANDMARKS["upper_lip"], LANDMARKS["lower_lip"])
                jaw_distance = safe_distance(coords, LANDMARKS["chin"], LANDMARKS["nose_tip"])
                eyebrow_height = coords[LANDMARKS["forehead"], 1] - coords[LANDMARKS["left_brow"], 1]
                eye_opening_val = eye_opening(coords)
                if face_idx not in _temporal_state_history:
                    _temporal_state_history[face_idx] = deque(maxlen=int(fps // 2))
                history = _temporal_state_history[face_idx]
                history.append({"center": center, "mouth_gap": mouth_gap, "jaw_distance": jaw_distance, "eyebrow_height": eyebrow_height, "eye_opening": eye_opening_val})
                centers = np.array([s["center"] for s in history])
                displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
                speed_threshold = 0.5
                filtered_displacements = displacements[displacements > speed_threshold]
                face_speed = float(np.mean(filtered_displacements)) if filtered_displacements.size else 0.0

            face_acceleration
                face_acceleration = float(np.mean(np.diff(filtered_displacements))) if filtered_displacements.size > 1 else 0.0

            micro_expression_rate
                mouth_gaps = np.array([s["mouth_gap"] for s in history])
                mouth_deltas = np.abs(np.diff(mouth_gaps))
                motion_threshold = 0.3
                filtered_mouth = mouth_deltas[mouth_deltas > motion_threshold]
                mouth_motion = float(np.mean(filtered_mouth)) if filtered_mouth.size else 0.0
                micro_expression_rate = mouth_motion + eyebrow_motion

            jaw_movement_intensity
                jaw_distances = np.array([s["jaw_distance"] for s in history])
                jaw_deltas = np.abs(np.diff(jaw_distances))
                filtered_jaw = jaw_deltas[jaw_deltas > motion_threshold]
                jaw_movement_intensity = float(np.mean(filtered_jaw)) if filtered_jaw.size else 0.0

            eyebrows_motion_score
                eyebrow_heights = np.array([s["eyebrow_height"] for s in history])
                eyebrow_deltas = np.abs(np.diff(eyebrow_heights))
                filtered_eyebrow = eyebrow_deltas[eyebrow_deltas > motion_threshold]
                eyebrow_motion = float(np.mean(filtered_eyebrow)) if filtered_eyebrow.size else 0.0

            mouth_motion_score
                mouth_motion = float(np.mean(filtered_mouth)) if filtered_mouth.size else 0.0

            head_motion_energy
                eye_openings = np.array([s["eye_opening"] for s in history])
                eye_deltas = np.abs(np.diff(eye_openings))
                filtered_eye = eye_deltas[eye_deltas > motion_threshold]
                head_motion = float(np.mean(filtered_eye)) if filtered_eye.size else 0.0

            talking_motion_score
                talking_motion_score = mouth_motion

        structure
            face_mesh_vector
                normalized = coords[:, :3].copy()
                normalized[:, 0] -= np.mean(normalized[:, 0])
                normalized[:, 1] -= np.mean(normalized[:, 1])
                normalized[:, :2] /= np.max(np.ptp(normalized[:, :2], axis=0) + 1e-6)
                face_mesh_vector = normalized[:100, :].flatten().tolist()

            identity_shape_vector
                identity_shape_vector = normalized[:50, :3].flatten().tolist()

            expression_vector
                expression_vector = (normalized[50:100, :3] - normalized[:50, :3]).flatten().tolist()

            jaw_pose_vector
                jaw_pose_vector = [pose["yaw"], pose["pitch"], pose["roll"]] # из модуля pose

            eye_pose_vector
                eye_pose_vector = [pose["pitch"], pose["roll"]] # из модуля pose

            mouth_shape_params
                mouth_shape_params = [
                    float(np.max(normalized[:, 0]) - np.min(normalized[:, 0])),
                    float(np.max(normalized[:, 1]) - np.min(normalized[:, 1])),
                ]

            face_symmetry_score
            face_uniqueness_score
                symmetry = 1.0 - float(np.mean(np.abs(normalized[:, 0] + normalized[::-1, 0])) / (np.max(np.abs([:, 0])) + 1e-6))
                uniqueness = float(np.std(normalized[:, :2]))

            identity_params_count
            expression_params_count
                identity_params_count = len(identity_shape_vector)
                expression_params_count = len(expression_vector)

            face_dimensions
                face_width = float(np.max(normalized[:, 0]) - np.min(normalized[:, 0]))
                face_height = float(np.max(normalized[:, 1]) - np.min(normalized[:, 1]))
                face_depth = float(np.max(normalized[:, 2]) - np.min(normalized[:, 2])) if normalized.shape[1] > 2 else 0.0
                face_aspect_ratio = face_width / max(face_height, 1e-6)
                face_dimensions = {"width": face_width, "height": face_height, "depth": face_depth, "aspect_ratio": face_aspect_ratio}


        professional
            face_quality_score
                quality = data["quality"] # из модуля quality
                blur = quality.get("face_blur_score", 0.0)
                sharpness = quality.get("sharpness_score", 0.0)
                face_quality = float(np.clip((sharpness * 2.0) / (1.0 + np.log1p(blur)), 0.0, 1.0))

            perceived_attractiveness_score
                perceived_attractiveness = float(np.clip(quality.get("face_contrast", 0.5), 0.0, 1.0))

            emotion_intensity
                motion = data["motion"]
                emotion_intensity = float(motion.get("mouth_motion_score", 0.0) + motion.get("eyebrows_motion_score", 0.0))
                emotion_intensity = np.clip(emotion_intensity, 0.0, 1.0)

            lip_reading_features
                lip_reading = data.get("lip_reading", {})
                if lip_reading:
                    lip_reading_features = {
                        "mouth_openness": lip_reading.get("mouth_area", 0.0),
                        "mouth_motion_intensity": lip_reading.get("mouth_motion_intensity", 0.0),
                        "jaw_movement": motion.get("jaw_movement_intensity", 0.0),
                        "speech_activity_prob": lip_reading.get("speech_activity_prob", 0.0),
                        "phoneme_features": lip_reading.get("phoneme_features", {}),
                        "cycle_strength": lip_reading.get("cycle_strength", 0.0),
                    }
                else:
                    mouth_motion = motion.get("talking_motion_score", 0.0)
                    jaw_motion = motion.get("jaw_movement_intensity", 0.0)
                    lip_reading_features = {
                        "mouth_openness": mouth_motion,
                        "mouth_motion_intensity": mouth_motion,
                        "jaw_movement": jaw_motion,
                        "speech_activity_prob": float(np.clip(mouth_motion * 2.0, 0.0, 1.0)),
                    }

            fatigue_score
                eyes = data["eyes"]
                eye_opening_data = eyes.get("eye_opening_ratio", {})
                if isinstance(eye_opening_data, dict):
                    avg_eye_opening = eye_opening_data.get("average", 0.5)
                    left_eye_opening = eye_opening_data.get("left", 0.5)
                    right_eye_opening = eye_opening_data.get("right", 0.5)
                else:
                    avg_eye_opening = float(eye_opening_data) if isinstance(eye_opening_data, (int, float)) else 0.5
                    left_eye_opening = right_eye_opening = avg_eye_opening
                blink_rate = eyes.get("blink_rate", 0.0)
                eye_closedness = 1.0 - np.clip(avg_eye_opening / 20.0, 0.0, 1.0)
                eye_asymmetry = abs(left_eye_opening - right_eye_opening) / max(max(left_eye_opening, right_eye_opening), 1e-6)
                blink_rate_normalized = np.clip(blink_rate * 60.0, 0.0, 30.0) / 30.0  # Нормализуем до 0-1
                blink_deviation = abs(blink_rate_normalized - 0.5) * 2.0  # Отклонение от нормы
                eye_fatigue_score = float(eye_closedness * 0.5 + eye_asymmetry * 0.2 + blink_deviation * 0.3)
                
                head_pitch = pose.get("pitch", 0.0)
                head_pitch_normalized = np.clip(head_pitch / 30.0, 0.0, 1.0)  # Наклон вниз = усталость
                pose_variability = pose.get("head_pose_variability", 0.0)
                pose_stability = 1.0 - np.clip(pose_variability / 15.0, 0.0, 1.0)
                attention_to_camera = pose.get("attention_to_camera_ratio", 0.5)
                attention_loss = 1.0 - attention_to_camera
                pose_fatigue_score = float(head_pitch_normalized * 0.4 + (1.0 - pose_stability) * 0.3 + attention_loss * 0.3)

                face_speed = motion.get("face_speed", 0.0)
                face_acceleration = motion.get("face_acceleration", 0.0)
                speed_normalized = np.clip(face_speed / 10.0, 0.0, 1.0)
                low_speed_indicator = 1.0 - speed_normalized
                motion_activity = motion.get("micro_expression_rate", 0.0)
                motion_activity_normalized = np.clip(motion_activity, 0.0, 1.0)
                low_activity_indicator = 1.0 - motion_activity_normalized
                acceleration_variance = abs(face_acceleration)
                irregular_motion = np.clip(acceleration_variance / 5.0, 0.0, 1.0)
                motion_fatigue_score = float(low_speed_indicator * 0.4 + low_activity_indicator * 0.4 + irregular_motion * 0.2)

                current_fatigue_indicators = {"eye_closedness": eye_closedness, "head_pitch": head_pitch_normalized, "motion_speed": speed_normalized, "attention": attention_to_camera}
                if face_idx not in self._fatigue_history:
                    self._fatigue_history[face_idx] = deque(maxlen=int(self.fps * 5))
                history = self._fatigue_history[face_idx]
                history.append(current_fatigue_indicators)
                temporal_fatigue_score = 0.0
                if len(history) >= int(self.fps * 2):  # Минимум 2 секунды истории
                    eye_closedness_history = [f["eye_closedness"] for f in history]
                    motion_speed_history = [f["motion_speed"] for f in history]
                    attention_history = [f["attention"] for f in history]
                    if len(eye_closedness_history) > 1:
                        eye_trend = (eye_closedness_history[-1] - eye_closedness_history[0]) / len(eye_closedness_history)
                        motion_trend = (1.0 - motion_speed_history[-1]) - (1.0 - motion_speed_history[0])
                        attention_trend = attention_history[0] - attention_history[-1]
                        temporal_fatigue_score = float(np.clip((eye_trend * 0.5 + motion_trend * 0.3 + attention_trend * 0.2) * 2.0, 0.0, 1.0))
                fatigue_score = float(np.clip(eye_fatigue_score * 0.35 + pose_fatigue_score * 0.30 + motion_fatigue_score * 0.25 + temporal_fatigue_score * 0.10, 0.0, 1.0))

            fatigue_breakdown
                fatigue_breakdown = {
                    "eye_fatigue": float(eye_fatigue_score),
                    "pose_fatigue": float(pose_fatigue_score),
                    "motion_fatigue": float(motion_fatigue_score),
                    "temporal_fatigue": float(temporal_fatigue_score),
                    "eye_closedness": float(eye_closedness),
                    "head_pitch_down": float(head_pitch_normalized),
                    "low_motion_speed": float(low_speed_indicator),
                    "blink_abnormality": float(blink_deviation),
                }

            engagement_level
                gaze_score = eyes.get("gaze_at_camera_prob", 0.5)
                attention = eyes.get("attention_score", 0.5)
                micro_expressions = motion.get("micro_expression_rate", 0.0)
                mouth_motion = lip_reading_features.get("mouth_motion_intensity", motion.get("talking_motion_score", 0.0))
                engagement_level = float(np.clip(
                    gaze_score * 0.35 +
                    attention * 0.25 +
                    min(micro_expressions, 1.0) * 0.2 +
                    min(mouth_motion * 2.0, 1.0) * 0.1 +
                    avg_eye_opening * 0.1,
                    0.0, 1.0
                ))

            alertness_score
            expressiveness_score
                alertness_score = float(np.clip(avg_eye_opening * 0.7 + face_speed / 15.0 * 0.3, 0.0, 1.0))
                expressiveness_score = float(np.clip(emotion_intensity + micro_expressions, 0.0, 1.0))

        lip_reading
            mouth_width
            mouth_height
            mouth_area
            mouth_aspect_ratio
                LANDMARKS = {"mouth_left": 61, "mouth_right": 291}
                mouth_left_idx = LANDMARKS.get("mouth_left", 61)
                mouth_right_idx = LANDMARKS.get("mouth_right", 291)
                mouth_width = safe_distance(coords, mouth_left_idx, mouth_right_idx)
                mouth_width = safe_distance(coords, mouth_left_idx, mouth_right_idx)
                mouth_height = safe_distance(coords, upper_lip_idx, lower_lip_idx)
                mouth_area = np.pi * (mouth_width / 2.0) * (mouth_height / 2.0)
                mouth_aspect_ratio = mouth_height / max(mouth_width, 1e-6)

            lip_separation
            lip_asymmetry
                if (mouth_left_idx < len(coords) and mouth_right_idx < len(coords) and upper_lip_idx < len(coords) and lower_lip_idx < len(coords)):
                    mouth_center = ((coords[mouth_left_idx][:2] + coords[mouth_right_idx][:2]) / 2.0)
                    upper_lip_center = coords[upper_lip_idx][:2]
                    lower_lip_center = coords[lower_lip_idx][:2]
                    lip_separation = np.linalg.norm(upper_lip_center - lower_lip_center)
                    left_half_width = np.linalg.norm(mouth_center - coords[mouth_left_idx][:2])
                    right_half_width = np.linalg.norm(coords[mouth_right_idx][:2] - mouth_center)
                    lip_asymmetry = abs(left_half_width - right_half_width) / max(mouth_width, 1e-6)
                else:
                    lip_separation = 0.0
                    lip_asymmetry = 0.0
                    mouth_center = np.array([0.0, 0.0])

            lip_contour_compactness
            lip_prominence_ratio
                LANDMARKS = {
                    "upper_lip": 13,
                    "lower_lip": 14,
                    "mouth_left": 61,
                    "mouth_right": 291
                }
                lip_contour_points = []
                for key in ["mouth_left", "upper_lip", "mouth_right", "lower_lip"]:
                    idx = LANDMARKS.get(key)
                    if idx is not None and idx < len(coords):
                        lip_contour_points.append(coords[idx][:2])
                if len(lip_contour_points) >= 4:
                    lip_contour_points = np.array(lip_contour_points)
                    perimeter = (
                        np.linalg.norm(lip_contour_points[1] - lip_contour_points[0]) +
                        np.linalg.norm(lip_contour_points[2] - lip_contour_points[1]) +
                        np.linalg.norm(lip_contour_points[3] - lip_contour_points[2]) +
                        np.linalg.norm(lip_contour_points[0] - lip_contour_points[3])
                    )
                    contour_compactness = (perimeter ** 2) / max(mouth_area, 1e-6)
                    upper_lip_height = np.linalg.norm(lip_contour_points[1] - mouth_center)
                    lower_lip_height = np.linalg.norm(lip_contour_points[3] - mouth_center)
                    lip_prominence_ratio = upper_lip_height / max(lower_lip_height, 1e-6)
                else:
                    contour_compactness = 0.0
                    lip_prominence_ratio = 1.0

            phoneme_features
                phoneme_features = {
                    "round_shape": float(np.clip(mouth_aspect_ratio, 0.0, 1.0)),  # Округлая форма (О, У)
                    "wide_shape": float(np.clip(mouth_width / max(mouth_height, 1e-6), 0.0, 2.0) / 2.0),  # Широкая форма (И, Э)
                    "narrow_shape": float(np.clip(1.0 - mouth_aspect_ratio, 0.0, 1.0)),  # Узкая форма
                    "open_shape": float(np.clip(lip_separation / max(mouth_width, 1e-6), 0.0, 1.0)),  # Открытый рот (А, Э)
                }
            
            mouth_motion_intensity
            width_velocity
            height_velocity
            area_velocity
            cycle_strength
            speech_activity_prob
            width_variance
            height_variance
            area_variance
            separation_variance
                current_frame_features = {
                    "mouth_width": float(mouth_width),
                    "mouth_height": float(mouth_height),
                    "mouth_area": float(mouth_area),
                    "lip_separation": float(lip_separation),
                    "mouth_aspect_ratio": float(mouth_aspect_ratio),
                }
                if face_idx not in self._lip_history:
                    self._lip_history[face_idx] = deque(maxlen=int(self.fps * 2))
                history = self._lip_history[face_idx]
                history.append(current_frame_features)
                if len(history) >= 2:
                    widths = [f["mouth_width"] for f in history]
                    heights = [f["mouth_height"] for f in history]
                    areas = [f["mouth_area"] for f in history]
                    separations = [f["lip_separation"] for f in history]
                    width_var = float(np.var(widths))
                    height_var = float(np.var(heights))
                    area_var = float(np.var(areas))
                    separation_var = float(np.var(separations))
                    width_velocity = float(np.mean(np.abs(np.diff(widths)))) if len(widths) > 1 else 0.0
                    height_velocity = float(np.mean(np.abs(np.diff(heights)))) if len(heights) > 1 else 0.0
                    area_velocity = float(np.mean(np.abs(np.diff(areas)))) if len(areas) > 1 else 0.0
                    mouth_motion_intensity = float((width_velocity + height_velocity + area_velocity) / 3.0)
                    if len(areas) >= 4:
                        autocorr = np.correlate(areas, areas, mode='full')
                        mid = len(autocorr) // 2
                        if mid > 0:
                            autocorr_peaks = autocorr[mid+1:mid+min(len(areas)//2, 30)]
                            if len(autocorr_peaks) > 0:
                                cycle_strength = float(np.max(autocorr_peaks) / max(autocorr[mid], 1e-6))
                            else:
                                cycle_strength = 0.0
                        else:
                            cycle_strength = 0.0
                    else:
                        cycle_strength = 0.0
                    motion_from_motion_module = motion.get("mouth_motion_score", 0.0)
                    talking_from_motion = motion.get("talking_motion_score", 0.0)
                    speech_activity_prob = float(np.clip(
                        (mouth_motion_intensity * 0.4 + 
                        cycle_strength * 0.3 + 
                        talking_from_motion * 0.3) * 2.0,
                        0.0, 1.0
                    ))
                else:
                    width_var = height_var = area_var = separation_var = 0.0
                    width_velocity = height_velocity = area_velocity = 0.0
                    mouth_motion_intensity = 0.0
                    cycle_strength = 0.0
                    speech_activity_prob = motion.get("speech_activity_prob", 0.0)
        face_3d
            face_mesh_vector
                coords = data["coords"]  # (N, 3) - 3D landmarks
                mean_coords = np.mean(coords[:, :3], axis=0)
                centered_coords = coords[:, :3] - mean_coords
                scale = np.std(centered_coords) + 1e-6
                normalized_coords = centered_coords / scale
                max_landmarks_for_mesh = min(n_landmarks, 200)
                face_mesh_vector_3d = normalized_coords[:max_landmarks_for_mesh, :3].flatten()
                if len(face_mesh_vector_3d) > 300:
                    step = len(face_mesh_vector_3d) // 300
                    face_mesh_vector_3d = face_mesh_vector_3d[::step][:300]
                face_mesh_vector = face_mesh_vector_3d.tolist()

            identity_shape_vector
                n_landmarks = len(coords)
                identity_landmarks = min(100, n_landmarks)
                identity_coords = normalized_coords[:identity_landmarks, :3]
                identity_shape_vector = [
                    float(np.mean(identity_coords[:, 0])),
                    float(np.std(identity_coords[:, 0])),
                    float(np.mean(identity_coords[:, 1])),
                    float(np.std(identity_coords[:, 1])),
                    float(np.mean(identity_coords[:, 2])),
                    float(np.std(identity_coords[:, 2])),
                ]
                if identity_coords.shape[0] > 3:
                    cov_matrix = np.cov(identity_coords.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                    n_pcs = min(self.n_identity_params - 6, len(eigenvalues))
                    if n_pcs > 0:
                        sorted_idx = np.argsort(eigenvalues)[::-1]
                        top_eigenvectors = eigenvectors[:, sorted_idx[:n_pcs]]
                        projected = identity_coords @ top_eigenvectors  # исправлено
                        identity_shape_vector.extend([float(x) for x in projected.flatten()[:n_pcs * 3]])
                while len(identity_shape_vector) < self.n_identity_params:
                    identity_shape_vector.append(0.0)
                identity_shape_vector = identity_shape_vector[:self.n_identity_params]

            expression_vector
                expression_landmarks = min(150, n_landmarks)
                expression_coords = normalized_coords[:expression_landmarks, :3]
                neutral_coords = np.mean(expression_coords, axis=0)
                expression_deviations = expression_coords - neutral_coords
                expression_vector = [
                    float(np.mean(expression_deviations[:, 0])),
                    float(np.std(expression_deviations[:, 0])),
                    float(np.mean(expression_deviations[:, 1])),
                    float(np.std(expression_deviations[:, 1])),
                    float(np.mean(expression_deviations[:, 2])),
                    float(np.std(expression_deviations[:, 2])),
                ]
                if expression_deviations.shape[0] > 3:
                    expr_cov = np.cov(expression_deviations.T)
                    expr_eigenvalues, expr_eigenvectors = np.linalg.eigh(expr_cov)
                    n_expr_pcs = min(self.n_expression_params - 6, len(expr_eigenvalues))
                    if n_expr_pcs > 0:
                        sorted_idx = np.argsort(expr_eigenvalues)[::-1]
                        top_eigenvectors = expr_eigenvectors[:, sorted_idx[:n_expr_pcs]]
                        projected = expression_deviations @ top_eigenvectors  # исправлено
                        expression_vector.extend([float(x) for x in projected.flatten()[:n_expr_pcs * 3]])
                while len(expression_vector) < self.n_expression_params:
                    expression_vector.append(0.0)
                expression_vector = expression_vector[:self.n_expression_params]

            jaw_pose_vector
            eye_pose_vector
                yaw = pose.get("yaw", 0.0)
                pitch = pose.get("pitch", 0.0)
                roll = pose.get("roll", 0.0)
                jaw_pose_vector = [float(yaw), float(pitch), float(roll)]
                eye_pose_vector = [float(pitch), float(roll)]

            mouth_shape_params
                mouth_landmarks_indices = [13, 14, 61, 291]
                mouth_coords = [coords[idx, :3] for idx in mouth_landmarks_indices if idx < len(coords)]
                if len(mouth_coords) >= 2:
                    mouth_coords = np.array(mouth_coords)
                    mouth_width = float(np.linalg.norm(mouth_coords[2] - mouth_coords[3]) if len(mouth_coords) >= 4 else 0.0)
                    mouth_height = float(np.linalg.norm(mouth_coords[0] - mouth_coords[1]))
                    mouth_depth = float(np.mean(mouth_coords[:, 2]))
                    mouth_shape_params = [
                        mouth_width,
                        mouth_height,
                        mouth_depth,
                        float(mouth_width / max(mouth_height, 1e-6)),
                    ]
                else:
                    mouth_shape_params = [0.0, 0.0, 0.0, 0.0]
                
            face_symmetry_score
            face_uniqueness_score
                face_center_x = np.mean(normalized_coords[:, 0])
                left_half = normalized_coords[normalized_coords[:, 0] < face_center_x]
                right_half = normalized_coords[normalized_coords[:, 0] >= face_center_x]
                if len(left_half) > 0 and len(right_half) > 0:
                    right_half_mirrored = right_half.copy()
                    right_half_mirrored[:, 0] = face_center_x - (right_half_mirrored[:, 0] - face_center_x)
                    min_len = min(len(left_half), len(right_half_mirrored))
                    if min_len > 0:
                        left_sampled = left_half[:min_len]
                        right_sampled = right_half_mirrored[:min_len]
                        symmetry_error = np.mean(np.abs(left_sampled - right_sampled))
                        symmetry_score = float(1.0 / (1.0 + symmetry_error))
                    else:
                        symmetry_score = 0.5
                else:
                    symmetry_score = 0.5
                uniqueness_score = float(np.std(normalized_coords.flatten()))

            mesh_num_vertices
            identity_params_count
            expression_params_count
                mesh_num_vertices = len(face_mesh_vector) // 3
                identity_params_count = len(identity_shape_vector)
                expression_params_count = len(expression_vector)

emotion_face
    Модели    

behavioral
    Выход
        {
            "frame_results": {
                "0": {
                "hand_gestures": [],
                "num_hands": 0,
                "body_language": {
                    "posture": "standing",
                    "open_posture": false,
                    "closed_posture": false,
                    "power_pose": false,
                    "rigidity": false,
                    "relaxed": true,
                    "forward_lean": false,
                    "backward_lean": false,
                    "balance_offset": 0.09563497453927994
                },
                "speech_behavior": {
                    "mouth_width": 146.7745327949524,
                    "mouth_height": 40.26090903289151,
                    "mouth_area": 5909.27611320273,
                    "speech_activity": 0.0,
                    "mouth_open_ratio": 0.2743044604961338
                },
                "engagement": {
                    "engagement_score": 0.514121276140213,
                    "engagement_variation": 0.0,
                    "engagement_peaks": 0,
                    "engagement_consistency": 1.0,
                    "factors": {
                    "eye_contact": 0.756485104560852,
                    "head_movement": 0.5,
                    "gesture_activity": 0.3,
                    "open_posture": 0.5
                    }
                },
                "confidence": {
                    "confidence_score": 0.5825288377702236,
                    "dominance_score": 0.5825288377702236,
                    "confidence_variability": 0.0,
                    "confidence_peak_moments": 0,
                    "factors": {
                    "open_posture": 0.3,
                    "head_straight": 0.8515381664037704,
                    "confident_gestures": 0.5,
                    "shoulder_level": 0.678577184677124
                    }
                },
                "stress": {
                    "stress_level": 0.0,
                    "anxiety_score": 0.0,
                    "stress_indicators": {
                    "closed_posture": {
                        "present": false,
                        "intensity": 0.0
                    },
                    "rigidity": {
                        "present": false,
                        "intensity": 0.0
                    }
                    },
                    "stress_count": 0
                },
                "timestamp": 0.0
                },
                "5": {...},
            },
            "aggregated": {
                "avg_engagement": 0.5616714629215923,
                "avg_confidence": 0.5996919194115251,
                "avg_stress": 0.47414710188366255,
                "max_engagement": 0.816409562031428,
                "max_confidence": 0.814064122736454,
                "max_stress": 0.6666666666666666,
                "gesture_statistics": {
                "hands_on_hips": 2,
                "open_palm": 1
                },
                "posture_statistics": {
                "standing": 150,
                "sitting": 14
                }
            }
        }
    Фичи
        frame_results
            hand_gestures
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                 def _get_finger_states(self, hand_landmarks):
                    finger_tips = [4, 8, 12, 16, 20]
                    finger_pips = [2, 6, 10, 14, 18]
                    states = {}
                    for i, (tip_idx, pip_idx) in enumerate(zip(finger_tips, finger_pips)):
                        tip = hand_landmarks.landmark[tip_idx]
                        pip = hand_landmarks.landmark[pip_idx]
                        if i == 0:  # thumb
                            states['thumb'] = tip.x < pip.x if tip.x < 0.5 else tip.x > pip.x
                        else:
                            states[['index', 'middle', 'ring', 'pinky'][i-1]] = tip.y < pip.y
                    return states
                
                def _is_pointing(self, hand_landmarks, pose_landmarks=None):
                    states = self._get_finger_states(hand_landmarks)
                    return states.get('index', False) and not any([
                        states.get('middle', False),
                        states.get('ring', False),
                        states.get('pinky', False)
                    ])
                
                def _is_open_palm(self, hand_landmarks, pose_landmarks=None):
                    states = self._get_finger_states(hand_landmarks)
                    return all(states.values())
                
                def _is_hands_on_hips(self, hand_landmarks, pose_landmarks):
                    if pose_landmarks is None:
                        return False
                    wrist = hand_landmarks.landmark[0]
                    hip_idx = 23 if wrist.x < 0.5 else 24
                    
                    if hip_idx < len(pose_landmarks.landmark):
                        hip = pose_landmarks.landmark[hip_idx]
                        distance = abs(wrist.y - hip.y)
                        return distance < 0.1
                    
                    return False
                
                def _is_self_touch(self, hand_landmarks, pose_landmarks):
                    if pose_landmarks is None:
                        return False
                    wrist = hand_landmarks.landmark[0]
                    face_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # примерные индексы лица
                    return wrist.y < 0.3
                
                def _is_fist(self, hand_landmarks, pose_landmarks=None):
                    states = self._get_finger_states(hand_landmarks)
                    return not any(states.values())
                
                def _is_thumbs_up(self, hand_landmarks, pose_landmarks=None):
                    states = self._get_finger_states(hand_landmarks)
                    thumb_tip = hand_landmarks.landmark[4]
                    thumb_mcp = hand_landmarks.landmark[2]
                    return states.get('thumb', False) and thumb_tip.y < thumb_mcp.y
                
                def _is_thumbs_down(self, hand_landmarks, pose_landmarks=None):
                    states = self._get_finger_states(hand_landmarks)
                    thumb_tip = hand_landmarks.landmark[4]
                    thumb_mcp = hand_landmarks.landmark[2]
                    return states.get('thumb', False) and thumb_tip.y > thumb_mcp.y
                
                def _is_victory(self, hand_landmarks, pose_landmarks=None):
                    states = self._get_finger_states(hand_landmarks)
                    return states.get('index', False) and states.get('middle', False) and not any([
                        states.get('ring', False),
                        states.get('pinky', False)
                    ])
                
                def _is_ok(self, hand_landmarks, pose_landmarks=None):
                    states = self._get_finger_states(hand_landmarks)
                    index_tip = hand_landmarks.landmark[8]
                    thumb_tip = hand_landmarks.landmark[4]
                    distance = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
                    return states.get('thumb', False) and distance < 0.05 and not states.get('index', False)
                
                def _is_rock(self, hand_landmarks, pose_landmarks=None):
                    states = self._get_finger_states(hand_landmarks)
                    return states.get('index', False) and states.get('pinky', False) and not any([
                        states.get('middle', False),
                        states.get('ring', False)
                    ])
                
                def _is_call_me(self, hand_landmarks, pose_landmarks=None):
                    states = self._get_finger_states(hand_landmarks)
                    pinky_tip = hand_landmarks.landmark[20]
                    thumb_tip = hand_landmarks.landmark[4]
                    distance = np.sqrt((pinky_tip.x - thumb_tip.x)**2 + (pinky_tip.y - thumb_tip.y)**2)
                    return states.get('pinky', False) and distance < 0.05
                
                def _is_love(self, hand_landmarks, pose_landmarks=None):
                    states = self._get_finger_states(hand_landmarks)
                    return states.get('index', False) and states.get('middle', False) and not any([
                        states.get('ring', False),
                        states.get('pinky', False)
                    ])

                self.gesture_types = {
                    'pointing': self._is_pointing,
                    'open_palm': self._is_open_palm,
                    'hands_on_hips': self._is_hands_on_hips,
                    'self_touch': self._is_self_touch,
                    'fist': self._is_fist,
                    'thumbs_up': self._is_thumbs_up,
                    'thumbs_down': self._is_thumbs_down,
                    'victory': self._is_victory,
                    'ok': self._is_ok,
                    'rock': self._is_rock,
                    'call_me': self._is_call_me,
                    'love': self._is_love
                }
                def classify_gesture(self, hand_landmarks, pose_landmarks=None):
                    for gesture_name, check_func in self.gesture_types.items():
                        try:
                            if check_func(hand_landmarks, pose_landmarks):
                                return gesture_name
                        except:
                            continue
                    return 'unknown'
                hands_results = self.hands.process(rgb_frame)
                results = {}
                hand_gestures = []
                hand_landmarks_list = []
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        hand_landmarks_list.append(hand_landmarks)
                        gesture = self.gesture_classifier.classify_gesture(
                            hand_landmarks,
                            pose_results.pose_landmarks
                        )
                        hand_gestures.append(gesture)

            num_hands
                len(hand_landmarks_list)

            body_language
                pose_results = self.pose.process(rgb_frame)
                if pose_results.pose_landmarks:
                    body_language = self.body_analyzer.analyze_posture(
                        pose_results.pose_landmarks,
                        frame.shape
                    )
                    results['body_language'] = body_language


