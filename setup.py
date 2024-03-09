from setuptools import setup,find_packages

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str):
    '''
    params: file_path:str
    return: list object
    Method reads requirement.txt file, extract values from the txt file and stores the values in list object.
    This list object is then returned.
    '''
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


# This script is a typical steup script to create local python package. 
# It reads dependency from a requirement files,
# specifies package metadata and uses setuptools to define how the package should be packaged and distributed.
setup(
    name = 'CreditCardFraudDetection',
    version='0.0.1',
    author='Kashyap Kolhe',
    author_email='kpkolhe1998@gmail.com',
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages()
)