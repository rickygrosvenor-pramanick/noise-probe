# Expose tabular probes and ensure they are registered
from .probes import (
    GaussianNoiseProbe,
    MaskFeaturesProbe,
    FlipCategoriesProbe,
    ShuffleColumnProbe
)
from noiseprobe.noiseprobe.registry import register_probe

# Centralize registration of all tabular probes
register_probe(GaussianNoiseProbe)
register_probe(MaskFeaturesProbe)
register_probe(FlipCategoriesProbe)
register_probe(ShuffleColumnProbe)