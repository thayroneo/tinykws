"""Diagnóstico mínimo do ambiente e das versões instaladas."""
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ("numpy", "pandas", "librosa", "soundfile", "scikit-learn", "torch", "PyYAML")
for package in PACKAGES:
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NÃO INSTALADO")
