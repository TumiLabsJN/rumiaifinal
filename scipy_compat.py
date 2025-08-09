"""
Simple compatibility patch for scipy binom_test → binomtest
Applied before importing FEAT to fix compatibility issue
"""

import sys

def patch_scipy_compatibility():
    """
    Add binom_test to scipy.stats for backward compatibility
    This fixes FEAT/nltools compatibility with newer scipy versions
    """
    try:
        import scipy.stats
        
        # Check if binom_test is already available
        if hasattr(scipy.stats, 'binom_test'):
            return True  # Already patched or available
        
        # Try to import the new function
        try:
            from scipy.stats import binomtest
        except ImportError:
            print("❌ Neither binom_test nor binomtest available in scipy")
            return False
        
        # Create backward compatibility function
        def binom_test(x, n=None, p=0.5, alternative='two-sided'):
            """
            Backward compatibility wrapper for scipy.stats.binomtest
            
            This matches the old binom_test API that FEAT/nltools expects
            """
            result = binomtest(x, n, p, alternative)
            return result.pvalue
        
        # Patch it into scipy.stats
        scipy.stats.binom_test = binom_test
        
        print("✅ Successfully patched scipy.stats.binom_test")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to patch scipy compatibility: {e}")
        return False

def patch_scipy_simps():
    """
    Add simps to scipy.integrate for backward compatibility
    This fixes FEAT/nltools compatibility with newer scipy versions
    """
    try:
        import scipy.integrate
        
        # Check if simps is already available
        if hasattr(scipy.integrate, 'simps'):
            return True  # Already available
        
        # Try to import the new function
        try:
            from scipy.integrate import simpson
        except ImportError:
            print("❌ Neither simps nor simpson available in scipy.integrate")
            return False
        
        # Create backward compatibility function
        def simps(*args, **kwargs):
            """
            Backward compatibility wrapper for scipy.integrate.simpson
            """
            return simpson(*args, **kwargs)
        
        # Patch it into scipy.integrate
        scipy.integrate.simps = simps
        
        print("✅ Successfully patched scipy.integrate.simps")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to patch scipy.integrate compatibility: {e}")
        return False

def patch_lib2to3_compatibility():
    """
    Add lib2to3 compatibility shim for Python 3.12+
    lib2to3 was removed in Python 3.12 but some packages still import it
    """
    try:
        import lib2to3
        return True  # Already available
    except ImportError:
        pass
    
    try:
        # Create a minimal lib2to3 shim
        import sys
        from types import ModuleType
        
        # Create comprehensive lib2to3 module structure
        lib2to3 = ModuleType('lib2to3')
        lib2to3.refactor = ModuleType('lib2to3.refactor')
        lib2to3.pytree = ModuleType('lib2to3.pytree')
        lib2to3.pygram = ModuleType('lib2to3.pygram')
        lib2to3.patcomp = ModuleType('lib2to3.patcomp')
        
        # Add minimal classes that might be imported
        class DummyBase:
            def __init__(self, *args, **kwargs):
                pass
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        
        # Add common classes and functions that lib2to3 users expect
        lib2to3.pytree.Base = DummyBase
        lib2to3.pytree.Node = DummyBase
        lib2to3.pytree.Leaf = DummyBase
        lib2to3.pytree.convert = lambda x: x  # Dummy convert function
        
        # Add to sys.modules
        sys.modules['lib2to3'] = lib2to3
        sys.modules['lib2to3.refactor'] = lib2to3.refactor
        sys.modules['lib2to3.pytree'] = lib2to3.pytree
        sys.modules['lib2to3.pygram'] = lib2to3.pygram
        sys.modules['lib2to3.patcomp'] = lib2to3.patcomp
        
        print("✅ Successfully shimmed lib2to3 for Python 3.12 compatibility")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to shim lib2to3: {e}")
        return False

def ensure_feat_compatibility():
    """
    Apply all compatibility patches needed for FEAT
    """
    success1 = patch_scipy_compatibility()
    success2 = patch_scipy_simps()
    success3 = patch_lib2to3_compatibility()
    
    success = success1 and success2 and success3
    
    if success:
        print("✅ All FEAT compatibility patches applied")
    else:
        print("❌ Some FEAT compatibility patches failed")
    
    return success

if __name__ == "__main__":
    # Test the patch
    ensure_feat_compatibility()
    
    # Test importing FEAT
    try:
        from feat import Detector
        print("✅ FEAT import successful after patching")
    except Exception as e:
        print(f"❌ FEAT import still fails: {e}")