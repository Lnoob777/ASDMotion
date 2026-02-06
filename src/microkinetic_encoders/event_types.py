EVENT_TYPE_MAP = {
    "name_call_response": 0,
    "joint_attention_point": 1,
    "joint_attention_gaze": 2,
    "eye_contact_attempt": 3,
    "turn_taking": 4,
    "action_imitation": 5,
    "gesture_imitation": 6,
    "repetitive_hand_motion": 7,
    "repetitive_body_motion": 8,
    "object_fixation": 9,
    "emotional_mirroring": 10,
    "sensory_response": 11
}

# Reverse mapping: ID -> event name (for debugging / reports)
ID_TO_EVENT_TYPE = {v: k for k, v in EVENT_TYPE_MAP.items()}

# Number of defined event types
NUM_EVENT_TYPES = len(EVENT_TYPE_MAP)
