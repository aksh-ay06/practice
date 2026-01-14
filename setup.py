from setuptools import setup, find_packages

def get_requirements(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        requirements = file.readlines()

    # Filter out editable installs and comments
    requirements = [req.strip() for req in requirements 
                   if req.strip() and not req.startswith('#') and not req.startswith('-e')]
    return requirements


setup(
    name='MLProject',
    version='0.1.0',
    author='Akshay Patel',
    author_email='ap00143@mix.wvu.edu',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='A machine learning project setup',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
)