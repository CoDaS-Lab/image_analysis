# image-analysis

### Python Dependencies
* numpy >= 1.12.0
* scipy >= 0.18.1
* scikit-learn >=0.17
* sk-video >=1.17
* scikit-image >=0.12.0
* python 3.5
* wget

### External Dependencies
* ffmpeg >=3.2.2 or libav (10 or 11)

## Installation

### Requirements (OSX)

Installing ffmpeg:
```
brew install ffmpeg
```

### Requirements (Ubuntu)

## Testing
change directory to directory of test folder:
```
cd /path/to/directory/image-analysis/image-analysis
```
Download testing files:  
```
python test_setup.py
```
Run unittest discover
```
python -m unittest discover
```
