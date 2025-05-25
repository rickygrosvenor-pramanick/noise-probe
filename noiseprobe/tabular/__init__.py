# Expose tabular probes and ensure they are registered
from noiseprobe.tabular.probes import (
    GaussianNoiseProbeTabular,
    MaskFeaturesProbeTabular,
    FlipCategoriesProbeTabular,
    ShuffleColumnProbeTabular
)
from noiseprobe.noiseprobe.registry import register_probe

# Centralize registration of all tabular probes
register_probe(GaussianNoiseProbeTabular)
register_probe(MaskFeaturesProbeTabular)
register_probe(FlipCategoriesProbeTabular)
register_probe(ShuffleColumnProbeTabular)