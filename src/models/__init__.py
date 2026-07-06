"""
src/models/__init__.py
Populates the model registry by importing all model definition modules.
get_model / list_models are re-exported here for convenience.
"""
from .model_registry import get_model, list_models, register_model  
import src.models.local_encoder
