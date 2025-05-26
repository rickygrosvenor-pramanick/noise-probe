from noiseprobe.text.probes import (
    TextNoiseProbe,
    TextDropoutProbe,
    TextShuffleProbe
)

from noiseprobe.noiseprobe.registry import register_probe

register_probe(TextNoiseProbe)
register_probe(TextDropoutProbe)
register_probe(TextShuffleProbe)


# __all__ = ['TextRobustnessTester'] 