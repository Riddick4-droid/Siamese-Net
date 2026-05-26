"""
Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field

class SimilarityResponse(BaseModel):
    """
    Response for the /predict endpoint.
    """
    distance: float = Field(..., description="Euclidean distance between the two face embeddings")
    is_same_person: bool = Field(..., description="Whether the two faces belong to the same person")
    threshold: float = Field(..., description="Decision threshold used (from config)")
    