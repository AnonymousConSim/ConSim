from setuptools import setup, find_packages

setup(
    name='consim_anonymous_github',
    version='1.0',
    author='Anonymous',
    author_email='consimanonymousgithub@gmail.com',
    description='Framework for NLP automatic concept extraction and evaluation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/yourusername/your_project_name',  # TODO: Update this URL
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[  # TODO: Update this list
        'torch',
        'transformers[torch]',
        'datasets',
        'evaluate',
        'fast_ml',
        'scipy',
        'matplotlib',
        'seaborn',
        'ipykernel',
        'ipywidgets',
        'scikit-learn',
        'sentencepiece',
        'pynvml',
        'nltk',
        'bitsandbytes',
        'peft',
        'tiktoken',
        'openai',
        # 'adapters',
        'tqdm',
        'sentence_transformers'
    ],
    python_requires='>=3.7',
)
