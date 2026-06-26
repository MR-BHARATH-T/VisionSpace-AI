# utils/constants.py
"""
VisionSpace AI — Production Constants
======================================
Central registry for all domain constants used across agents:
room taxonomy, object categories, style vocabulary, spatial rules,
commerce signals, and generation constraints.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Room Taxonomy
# ---------------------------------------------------------------------------

ROOM_TYPES: list[str] = [
    "living_room",
    "bedroom",
    "master_bedroom",
    "kids_bedroom",
    "dining_room",
    "kitchen",
    "bathroom",
    "home_office",
    "study",
    "hallway",
    "entryway",
    "laundry_room",
    "garage",
    "balcony",
    "outdoor_patio",
    "sunroom",
    "basement",
    "attic",
    "gym",
    "home_theater",
    "library",
    "playroom",
    "nursery",
    "guest_room",
    "walk_in_closet",
]

# ---------------------------------------------------------------------------
# Object / Furniture Categories
# ---------------------------------------------------------------------------

FURNITURE_CATEGORIES: dict[str, list[str]] = {
    "seating": [
        "sofa", "couch", "sectional", "loveseat", "armchair",
        "accent_chair", "recliner", "chaise_lounge", "bean_bag",
        "dining_chair", "bar_stool", "office_chair", "bench",
        "ottoman", "pouf",
    ],
    "tables": [
        "coffee_table", "side_table", "end_table", "dining_table",
        "console_table", "desk", "writing_desk", "standing_desk",
        "bedside_table", "nightstand", "kitchen_island", "bar_cart",
    ],
    "storage": [
        "bookshelf", "bookcase", "wardrobe", "armoire", "dresser",
        "chest_of_drawers", "sideboard", "buffet", "credenza",
        "tv_unit", "media_console", "shoe_rack", "coat_rack",
        "storage_ottoman", "cabinet", "cupboard",
    ],
    "beds": [
        "single_bed", "twin_bed", "full_bed", "queen_bed", "king_bed",
        "bunk_bed", "loft_bed", "daybed", "murphy_bed", "canopy_bed",
        "platform_bed",
    ],
    "lighting": [
        "floor_lamp", "table_lamp", "pendant_light", "chandelier",
        "wall_sconce", "ceiling_fan", "track_lighting", "led_strip",
        "recessed_light", "arc_lamp", "bedside_lamp",
    ],
    "decor": [
        "rug", "curtain", "blind", "cushion", "throw_pillow",
        "blanket", "throw", "mirror", "artwork", "wall_art",
        "plant", "vase", "sculpture", "candle", "clock",
        "picture_frame", "decorative_tray",
    ],
    "appliances": [
        "refrigerator", "oven", "microwave", "dishwasher", "washer",
        "dryer", "air_conditioner", "heater", "ceiling_fan",
        "television", "monitor", "speaker",
    ],
}

# Flat list of all known object labels
ALL_OBJECT_LABELS: list[str] = [
    obj for objs in FURNITURE_CATEGORIES.values() for obj in objs
]

# ---------------------------------------------------------------------------
# Design Style Vocabulary
# ---------------------------------------------------------------------------

DESIGN_STYLES: list[str] = [
    "modern",
    "contemporary",
    "minimalist",
    "scandinavian",
    "japandi",
    "mid_century_modern",
    "industrial",
    "rustic",
    "farmhouse",
    "coastal",
    "bohemian",
    "eclectic",
    "traditional",
    "transitional",
    "art_deco",
    "hollywood_regency",
    "maximalist",
    "wabi_sabi",
    "mediterranean",
    "tropical",
]

STYLE_KEYWORDS: dict[str, list[str]] = {
    "modern": ["clean lines", "neutral palette", "open space", "minimal decor"],
    "scandinavian": ["light wood", "white walls", "hygge", "functional", "natural light"],
    "japandi": ["wabi-sabi", "zen", "muted tones", "natural materials", "simplicity"],
    "industrial": ["exposed brick", "metal accents", "raw wood", "Edison bulbs", "pipes"],
    "farmhouse": ["shiplap", "barn wood", "vintage", "linen", "galvanized metal"],
    "bohemian": ["layered textiles", "global patterns", "plants", "rattan", "eclectic"],
    "mid_century_modern": ["tapered legs", "organic forms", "bold colors", "retro"],
    "coastal": ["light blue", "sandy tones", "natural fibers", "driftwood", "shells"],
    "art_deco": ["geometric", "gold accents", "velvet", "lacquer", "symmetry"],
    "maximalist": ["bold patterns", "rich colors", "layered", "ornate", "gallery wall"],
}

# ---------------------------------------------------------------------------
# Color Palette Constants
# ---------------------------------------------------------------------------

COLOR_FAMILIES: dict[str, list[str]] = {
    "neutrals": ["white", "off_white", "cream", "beige", "greige", "gray", "charcoal", "black"],
    "earth_tones": ["terracotta", "sienna", "ochre", "rust", "camel", "taupe", "sand"],
    "blues": ["navy", "cobalt", "sky_blue", "powder_blue", "teal", "turquoise", "slate"],
    "greens": ["sage", "olive", "forest", "emerald", "mint", "moss", "avocado"],
    "pinks_reds": ["blush", "dusty_rose", "coral", "terracotta_pink", "burgundy", "crimson"],
    "yellows_oranges": ["mustard", "amber", "gold", "peach", "apricot", "saffron"],
    "purples": ["lavender", "mauve", "plum", "violet", "lilac", "eggplant"],
    "metallics": ["brass", "bronze", "copper", "silver", "chrome", "gold", "matte_black"],
}

# ---------------------------------------------------------------------------
# Spatial Layout Rules
# ---------------------------------------------------------------------------

MIN_CLEARANCE_CM: dict[str, int] = {
    "walkway":           90,   # min passable corridor
    "sofa_to_tv":        200,  # comfortable viewing distance
    "sofa_to_coffee":    45,   # reach distance
    "dining_chair_pull": 90,   # chair pull-out space
    "bed_sides":         70,   # walking space beside bed
    "door_swing":        80,   # door clearance arc
    "desk_to_wall":      60,   # legroom
}

ROOM_AREA_GUIDELINES_SQM: dict[str, dict[str, float]] = {
    "living_room":  {"small": 12, "medium": 20, "large": 30},
    "bedroom":      {"small": 9,  "medium": 14, "large": 20},
    "dining_room":  {"small": 8,  "medium": 14, "large": 20},
    "kitchen":      {"small": 6,  "medium": 10, "large": 16},
    "bathroom":     {"small": 4,  "medium": 6,  "large": 10},
    "home_office":  {"small": 6,  "medium": 10, "large": 15},
}

FOCAL_POINT_BY_ROOM: dict[str, list[str]] = {
    "living_room":  ["television", "fireplace", "window_view"],
    "bedroom":      ["bed_headboard", "window", "dresser_mirror"],
    "dining_room":  ["dining_table", "chandelier"],
    "home_office":  ["desk", "monitor"],
    "kitchen":      ["island", "range_hood"],
}

# ---------------------------------------------------------------------------
# Spatial Analysis Zones
# ---------------------------------------------------------------------------

ZONE_LABELS: list[str] = [
    "seating_zone",
    "dining_zone",
    "sleeping_zone",
    "work_zone",
    "storage_zone",
    "circulation_zone",
    "accent_zone",
    "window_zone",
    "entry_zone",
    "cooking_zone",
]

ZONE_FURNITURE_MAP: dict[str, list[str]] = {
    "seating_zone":     ["sofa", "armchair", "coffee_table", "ottoman"],
    "dining_zone":      ["dining_table", "dining_chair", "sideboard"],
    "sleeping_zone":    ["bed", "nightstand", "wardrobe", "dresser"],
    "work_zone":        ["desk", "office_chair", "bookshelf"],
    "storage_zone":     ["wardrobe", "cabinet", "bookcase", "chest_of_drawers"],
    "cooking_zone":     ["refrigerator", "oven", "kitchen_island", "microwave"],
    "accent_zone":      ["rug", "floor_lamp", "plant", "artwork", "mirror"],
}

# ---------------------------------------------------------------------------
# Commerce / Product Signals
# ---------------------------------------------------------------------------

COMMERCE_CATEGORIES: list[str] = [
    "furniture",
    "lighting",
    "textiles",
    "rugs",
    "curtains_blinds",
    "wall_decor",
    "plants_greenery",
    "storage_solutions",
    "kitchen_decor",
    "bathroom_accessories",
    "outdoor_furniture",
    "smart_home",
]

PRICE_TIERS: dict[str, dict[str, int]] = {
    "budget":    {"min": 0,    "max": 200},
    "mid_range": {"min": 200,  "max": 800},
    "premium":   {"min": 800,  "max": 3000},
    "luxury":    {"min": 3000, "max": 999999},
}

TRUSTED_RETAILERS: list[str] = [
    "IKEA",
    "Wayfair",
    "Amazon",
    "West Elm",
    "Crate & Barrel",
    "CB2",
    "Restoration Hardware",
    "Pottery Barn",
    "Article",
    "Floyd",
    "Burrow",
    "AllModern",
    "Joss & Main",
    "Overstock",
    "Target",
    "Walmart",
    "Home Depot",
    "Lowe's",
    "Anthropologie Home",
    "H&M Home",
]

AFFILIATE_DOMAINS: dict[str, str] = {
    "IKEA":    "ikea.com",
    "Wayfair": "wayfair.com",
    "Amazon":  "amazon.com",
    "Target":  "target.com",
}

# ---------------------------------------------------------------------------
# Generation & Inference Constraints
# ---------------------------------------------------------------------------

MAX_OBJECTS_PER_SCENE: int = 50
MAX_RECOMMENDATIONS_PER_OBJECT: int = 5
MAX_BUDGET_LINE_ITEMS: int = 30

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.65
HIGH_CONFIDENCE_THRESHOLD: float = 0.85

SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif",
)

MAX_IMAGE_DIMENSION: int = 1920   # px — hard cap before resize
PREFERRED_IMAGE_DIMENSION: int = 1024   # px — preferred for inference

# ---------------------------------------------------------------------------
# Agent Task Keys (mirrors ModelManager dependency graph)
# ---------------------------------------------------------------------------

TASK_KEYS: dict[str, str] = {
    "VISION":        "vision",
    "SPATIAL":       "spatial",
    "DESIGN":        "design",
    "COMMERCE":      "commerce",
    "BUDGET":        "budget",
    "EXPLAIN":       "explain",
    "VISUALIZATION": "visualization",
}

TASK_DEPENDENCY_MAP: dict[str, list[str]] = {
    "vision":        [],
    "spatial":       ["vision"],
    "design":        ["vision", "spatial"],
    "commerce":      ["design"],
    "budget":        ["commerce"],
    "explain":       ["vision", "spatial", "design"],
    "visualization": ["design"],
}

# ---------------------------------------------------------------------------
# File / IO Constants
# ---------------------------------------------------------------------------

JSON_INDENT: int = 4
MAX_FILE_SIZE_MB: int = 50
CHECKSUM_ALGORITHM: str = "sha256"

OUTPUT_SUBDIRS: list[str] = [
    "scenes",
    "reports",
    "budgets",
    "visualizations",
    "commerce",
    "exports",
]
