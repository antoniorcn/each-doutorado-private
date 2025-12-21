from setuptools import setup, find_packages

setup(
    name='spoofing',
    version='1.0.0',
    author='Antonio Carvalho',
    author_email='antoniorcn@hotmail.com',  # Seu email
    description='Implementação do artigo (A Robust Method with DropBlock for Face Anti Spoofing)',
    long_description=open('README.md').read(),  # Descrição longa (README.md)
    long_description_content_type='text/markdown',  # Tipo do conteúdo do README
    url='https://github.com/antoniorcn/each-doutorado-private',  # URL do repositório
    packages=find_packages(where='src'),  # Encontra os pacotes na pasta src
    package_dir={'': 'src'},              # Define o diretório base para os pacotes
    install_requires=[
        'tensorflow>=2.0.0',  # Versão mínima recomendada do TensorFlow
        'pillow',
        'scikit-learn',
        'opencv-python',      # Biblioteca OpenCV
    ],
    entry_points={
        'console_scripts': [
            'spoofing=spoofing.edu.ic.main:main',
        ],
    },
)