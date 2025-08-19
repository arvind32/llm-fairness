### Arvind Test Code

To get started, create and activate a conda environment and (linux):

    pip install requirements.txt;
    python setup.py develop;
    mkdir data;

To load resume data (in python):

    df = load_resume_data()

To run the example script, loading resume data and producing test outputs (linux):

    cd scripts;
    python test_script.py

There is an LLM playground available at:

    notebooks/resume_scoring.ipynb