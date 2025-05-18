# tests/conftest.py
import sys, os

# projenin kök dizininde çalıştığını varsayıyoruz
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
