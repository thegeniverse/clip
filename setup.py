import setuptools

with open("requirements.txt", "r", encoding="utf-8") as req_file:
    requirements_list = req_file.read().strip().split("\n")

setuptools.setup(
    name="clip",
    packages=["clip"],
    install_requires=requirements_list,
    include_package_data=True,
)