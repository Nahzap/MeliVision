"""
Runner para tests del sistema de anotaciones centralizadas.

Uso:
    python tests/run_centralized_tests.py
    python tests/run_centralized_tests.py --verbose
    python tests/run_centralized_tests.py --single TestMigrationToCentralized
"""
import sys
import subprocess
from pathlib import Path

def run_tests(verbose=False, test_class=None):
    """Ejecuta tests de anotaciones centralizadas con pytest"""
    
    project_root = Path(__file__).resolve().parent.parent
    test_file = project_root / "tests" / "test_centralized_annotations.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Construir comando pytest
    cmd = [sys.executable, "-m", "pytest", str(test_file)]
    
    if verbose:
        cmd.append("-v")
        cmd.append("-s")  # No capture output
    
    if test_class:
        cmd.append(f"-k {test_class}")
    
    cmd.extend([
        "--tb=short",  # Traceback corto
        "--color=yes",
        "-W", "ignore::DeprecationWarning"
    ])
    
    print(f"\n{'='*60}")
    print(f"EJECUTANDO TESTS DE ANOTACIONES CENTRALIZADAS")
    print(f"{'='*60}\n")
    print(f"Comando: {' '.join(cmd)}\n")
    
    # Ejecutar pytest
    result = subprocess.run(cmd, cwd=project_root)
    
    print(f"\n{'='*60}")
    if result.returncode == 0:
        print("✅ TODOS LOS TESTS PASARON")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
    print(f"{'='*60}\n")
    
    return result.returncode == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ejecutar tests de anotaciones centralizadas")
    parser.add_argument("--verbose", "-v", action="store_true", help="Salida verbose")
    parser.add_argument("--single", "-s", type=str, help="Ejecutar solo una clase de tests")
    
    args = parser.parse_args()
    
    success = run_tests(verbose=args.verbose, test_class=args.single)
    sys.exit(0 if success else 1)
