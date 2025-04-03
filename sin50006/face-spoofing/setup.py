from setuptools import setup, find_packages

setup(
    name='spoofing_detector',
    version='1.0.0',
    author='Antonio Carvalho',
    author_email='seu.email@example.com',  # Seu email
    description='Implementação do artigo (A Robust Method with DropBlock for Face Anti Spoofing)',
    long_description=open('README.md').read(),  # Descrição longa (README.md)
    long_description_content_type='text/markdown',  # Tipo do conteúdo do README
    url='https://github.com/antoniorcn/each-doutorado-private',  # URL do repositório
    packages=find_packages(where='src'),  # Encontra os pacotes na pasta src
    package_dir={'': 'src'},              # Define o diretório base para os pacotes
    install_requires=[
        'tensorflow>=2.0.0',  # Versão mínima recomendada do TensorFlow
        'opencv-python',      # Biblioteca OpenCV
    ],
    entry_points={
        'console_scripts': [
            'spoofing=edu.ic.main:main',
        ],
    },
)