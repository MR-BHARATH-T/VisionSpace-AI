"""
VisionSpace AI Models Package
"""

from typing import Any, Dict, List, Optional
from models.model_manager import ModelManager
from models.text_model import TextModel
from models.vision_model import VisionModel

def build_messages(
    system: Optional[str] = None,
    user: Optional[str] = None,
    assistant: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Helper method to construct a list of standard message objects."""
    from providers.huggingface_provider import HuggingFaceProvider
    return HuggingFaceProvider.build_messages(
        system=system, user=user, assistant=assistant, history=history
    )

__all__ = ["ModelManager", "TextModel", "VisionModel", "build_messages"]