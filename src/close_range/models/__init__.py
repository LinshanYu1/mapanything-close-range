"""Close-range model extensions."""
from .scale_correction import GlobalLocalScaleCorrection, attach_scale_correction
from .view_fusion import ViewScorer, attach_view_fusion, distance_training_loss
