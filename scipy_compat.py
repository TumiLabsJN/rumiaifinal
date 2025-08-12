"""
FEAT compatibility patches for scipy >= 1.14.0
Handles removal of deprecated functions in new scipy versions
"""

import scipy
import sys
import logging

logger = logging.getLogger(__name__)

def ensure_feat_compatibility():
    """
    Apply compatibility patches for FEAT to work with scipy >= 1.14.0
    
    Key changes in scipy 1.14:
    - scipy.stats.binom_test -> scipy.stats.binomtest
    - scipy.integrate.simps -> scipy.integrate.simpson
    """
    
    # Check scipy version
    scipy_version = tuple(map(int, scipy.__version__.split('.')[:2]))
    
    if scipy_version >= (1, 14):
        logger.info(f"Applying FEAT compatibility patches for scipy {scipy.__version__}")
        
        # Patch 1: binom_test -> binomtest
        import scipy.stats as stats
        if not hasattr(stats, 'binom_test'):
            def binom_test_compat(x, n=None, p=0.5, alternative='two-sided'):
                """Compatibility wrapper for scipy.stats.binomtest"""
                from scipy.stats import binomtest
                result = binomtest(x, n, p, alternative=alternative)
                return result.pvalue
            
            stats.binom_test = binom_test_compat
            logger.debug("✅ Patched scipy.stats.binom_test")
        
        # Patch 2: simps -> simpson
        import scipy.integrate as integrate
        if not hasattr(integrate, 'simps'):
            # simps was renamed to simpson in scipy 1.14
            integrate.simps = integrate.simpson
            logger.debug("✅ Patched scipy.integrate.simps -> simpson")
    
    else:
        logger.info(f"scipy {scipy.__version__} is compatible, no patches needed")

# Auto-apply patches on import
ensure_feat_compatibility()