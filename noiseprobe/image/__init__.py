
from noiseprobe.noiseprobe.registry import register_probe

from noiseprobe.image.probes import (
    GaussianNoiseProbeImage,
    SaltAndPepperNoiseProbeImage,
    SpeckleNoiseProbeImage,
    GaussianBlurProbeImage,
    MotionBlurProbeImage,
    MedianBlurProbeImage,
    RotateProbeImage,
    TranslateProbeImage,
    ScaleProbeImage,
    ShearProbeImage,
    ElasticDeformationProbeImage,
    BrightnessProbeImage,
    ContrastProbeImage,
    SaturationProbeImage,
    HueProbeImage,
    GammaCorrectionProbeImage,
    RandomErasingProbeImage,
    PatchOcclusionProbeImage,
    CoarseDropoutProbeImage,
    JpegCompressionProbeImage,
    DownsampleUpsampleProbeImage,
    ColorQuantizationProbeImage,
    PixelationProbeImage,
    FGSMAttackProbeImage,
    PGDAttackProbeImage
)

# Centralize registration of all image probes
register_probe(GaussianNoiseProbeImage)
register_probe(SaltAndPepperNoiseProbeImage)
register_probe(SpeckleNoiseProbeImage)
register_probe(GaussianBlurProbeImage)   
register_probe(MotionBlurProbeImage)
register_probe(MedianBlurProbeImage)
register_probe(RotateProbeImage)
register_probe(TranslateProbeImage)
register_probe(ScaleProbeImage)
register_probe(ShearProbeImage)
register_probe(ElasticDeformationProbeImage)
register_probe(BrightnessProbeImage)
register_probe(ContrastProbeImage)
register_probe(SaturationProbeImage)
register_probe(HueProbeImage)
register_probe(GammaCorrectionProbeImage)
register_probe(RandomErasingProbeImage)
register_probe(PatchOcclusionProbeImage)
register_probe(CoarseDropoutProbeImage)
register_probe(JpegCompressionProbeImage)
register_probe(DownsampleUpsampleProbeImage)
register_probe(ColorQuantizationProbeImage)
register_probe(PixelationProbeImage)
register_probe(FGSMAttackProbeImage)
register_probe(PGDAttackProbeImage)