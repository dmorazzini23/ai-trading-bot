import sys
import urllib3

print("In bot.py:")
print("sys.path:", sys.path)
print("urllib3.__file__:", urllib3.__file__)
print("Has util in urllib3:", hasattr(urllib3, "util"))
print("urllib3.util:", getattr(urllib3, "util", None))
