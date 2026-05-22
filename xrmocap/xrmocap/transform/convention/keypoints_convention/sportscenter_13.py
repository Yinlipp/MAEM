# 13 keypoints convention for SportCenter dataset
# This convention defines 13 body keypoints commonly used in sports pose estimation

SPORTSCENTER_13_KEYPOINTS = [
    'head_top',           # 0: top of head
    'right_shoulder',     # 2: right shoulder
    'right_elbow',        # 3: right elbow
    'right_wrist',        # 4: right wrist
    'left_shoulder',      # 5: left shoulder
    'left_elbow',         # 6: left elbow
    'left_wrist',         # 7: left wrist
    'right_hip',          # 8: right hip
    'right_knee',         # 9: right knee
    'right_ankle',        # 10: right ankle
    'left_hip',           # 11: left hip
    'left_knee',          # 12: left knee
    'left_ankle',         # 13: left ankle (note: index goes 0-12 for 13 points)
]

# Alternative naming that might match better with existing conventions
# Note: COCO doesn't have headtop, using nose as approximation
SPORTSCENTER_13_KEYPOINTS_ALT = [
    'nose',  # Approximate headtop with nose (COCO has nose but not headtop)
    'right_shoulder_openpose',
    'right_elbow_openpose',
    'right_wrist_openpose',
    'left_shoulder_openpose',
    'left_elbow_openpose',
    'left_wrist_openpose',
    'right_hip_openpose',
    'right_knee_openpose',
    'right_ankle_openpose',
    'left_hip_openpose',
    'left_knee_openpose',
    'left_ankle_openpose',
]
