# noise-probe
NoiseProbe is a Python library that tests the robustness and generalisation of Machine Learning models by systematically perturbing their inputs and measuring their predictive stability.

noiseprobe/
├── noiseprobe/                 # Main package
│   ├── __init__.py             # package marker + version
│   ├── base.py                 # BaseProbe & BaseRobustnessTester interfaces
│   ├── registry.py             # probe-registration logic (typing_probe_registry)
│   ├── utils.py                # helpers (device dispatch, common metrics, etc.)
│   │
│   ├── tabular/                 # Tabular-specific code
│   │   ├── __init__.py
│   │   ├── probes.py           # GaussianNoiseProbe, MaskFeaturesProbe, …
│   │   └── tester.py           # TabularRobustnessTester implementation
│   │
│   └── image/                   # Image-specific code
│       ├── __init__.py
│       ├── probes.py           # e.g. OcclusionProbe, GaussianBlurProbe, ColorJitterProbe
│       └── tester.py           # ImageRobustnessTester implementation
│
├── examples/                    # runnable demos & notebooks
│   ├── tabular_demo.ipynb
│   └── image_demo.ipynb
│
├── tests/                       # unit & integration tests
│   ├── test_tabular_probes.py
│   ├── test_tabular_tester.py
│   ├── test_image_probes.py
│   └── test_image_tester.py
│
├── docs/                        # user-facing docs (Sphinx/Markdown)
│   ├── index.md
│   └── tutorials.md
│
├── pyproject.toml               # build & dependency config
├── setup.cfg                    # packaging metadata
├── README.md                    # project overview + quickstart
└── LICENSE
