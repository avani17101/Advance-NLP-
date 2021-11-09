# ELMo

## Instructions 
* Install requirements
```
pip install -r requirements.txt
```
Libraries used:
* tensorflow
* scikit-learn
* numpy
* re
* pickle

### Running ELMo
* Download the weights from [Drive link](https://drive.google.com/drive/folders/1U0yvaQfz1kx-fzrd7iPZTf4h8oXo-JQM?usp=sharing)
* Put them in ```data/models```
* Run
```python a2.py --mode encode```

### Dataset Used
[switchboard](https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz)
* Downlod the above dataset and put in data/datasets folder

### Training ELMo
* Run
```python a2.py --mode train```






