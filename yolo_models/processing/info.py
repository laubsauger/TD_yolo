import enum

# Sync with touch designer code


@enum.unique
class BufferStates(enum.IntEnum):
    SERVER = 0
    CLIENT = 1
    SERVER_ALIVE = 2


@enum.unique
class States(bytes, enum.Enum):
    NULL_STATE = b'0'
    READY_SERVER_MESSAGE = b'1'
    READY_CLIENT_MESSAGE = b'2'
    IS_SERVER_ALIVE = b'3'


@enum.unique
class DrawInfo(enum.IntFlag):
    DRAW_TEXT = enum.auto()          # 1
    DRAW_BBOX = enum.auto()          # 2
    DRAW_CONF = enum.auto()          # 4
    DRAW_SKELETON = enum.auto()      # 8
    OVERLAY_ONLY = enum.auto()       # 16 - Draw on black/transparent background
    TRANSPARENT_BG = enum.auto()     # 32 - Use transparent instead of black background
    DRAW_KEYPOINT_CONF = enum.auto() # 64 - Draw keypoint confidence values
    MIRROR_LABELS = enum.auto()      # 128 - Swap left/right labels for mirror view


@enum.unique
class ParamsIndex(enum.IntEnum):
    IOU_THRESH = 0
    SCORE_THRESH = 1
    TOP_K = 2
    ETA = 3
    IMAGE_WIDTH = 4
    IMAGE_HEIGHT = 5
    IMAGE_CHANNELS = 6
    SHARED_ARRAY_MEM_NAME = 7
    SHARD_STATE_MEM_NAME = 8
    IMAGE_DTYPE = 9
    DRAW_INFO = 10
    POSE_THRESHOLD = 11
    FPS_LIMIT = 12  # Max FPS, 0 = unlimited
