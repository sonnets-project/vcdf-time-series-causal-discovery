# External Libraries

This directory contains causal discovery libraries used by the VCDF framework.

## Directory Structure

### `TCDF_master/` *(Required - Local Installation)*
- **Source**: [Temporal Causal Discovery Framework](https://github.com/M-Nauta/TCDF)
- **Commit**: 8242232
- **Reason**: Original repo has no `setup.py` or `pyproject.toml`, cannot be pip installed
- **Paper**: Nauta et al., "Causal Discovery with Attention-Based Convolutional Neural Networks" (2019)
- **⚠️ This directory is REQUIRED and should NOT be deleted**

### `lingam_master/` *(Deprecated - For Reference Only)*
- **Local Version**: 1.9.0
- **Current Usage**: Install via PyPI: `pip install lingam==1.8.3`
- **Source**: [LiNGAM GitHub](https://github.com/cdt15/lingam)
- **Paper**: Shimizu et al., "A Linear Non-Gaussian Acyclic Model for Causal Discovery" (JMLR 2006)
- **This directory is kept for code reference only**

### `tigramite_master/` *(Deprecated - For Reference Only)*
- **Local Version**: 5.2.5.12
- **Current Usage**: Install via PyPI: `pip install tigramite==5.2.5.8`
- **Source**: [Tigramite GitHub](https://github.com/jakobrunge/tigramite)
- **Paper**: Runge et al., "Detecting and quantifying causal associations in large nonlinear time series datasets" (Science Advances 2019)
- **This directory is kept for code reference only**

## Import Guide

```python
# From pip packages (recommended)
import lingam
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import data_processing as pp

# TCDF from local (no pip package available)
from src.models.TCDF_master import TCDF
```

## Notes

- `lingam` and `tigramite` should be installed via `pip install -r requirements.txt`
- Only TCDF requires local import due to lack of official PyPI package
- The `*_master` directories are retained for reference and historical purposes
