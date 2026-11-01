import platform
from importlib.metadata import version

print("Python:", platform.python_version())

with open("requirements.txt", encoding="utf-8") as file:
    for line in file:
        package = line.strip().split("==")[0]
        if package:
            print(f"{package}: {version(package)}")