# Image analysis project

## Core Pipeline
The pipeline is organized in a way to allow multiple operations to be defined in a few lines of code. For example:

```
motion_analysis = Pipeline(data=batch_list,
                           ops=[Normalize()],
                           seq=[RGBToGray(), DownSample()],
                           models={'SVM': SVM(), 'PCA': PCA()}
                           save_all=True)
```

The Pipeline class takes 5 inputs: the data, list of operations to run in parallel(without dependencies on each other), list of sequential operations, dictionary of models, and boolean check to save all features to a dictionary or not. The seq list contains all the operations we will run sequentially. The outputs of one operation gets fed into the following operation in the list.  

The models dictionary contains the statistical models to run on the data. The pipeline runs each model, in this case SVM(support vector machine) or PCA(principal component analysis) on the data. By using this structure you can swap out and insert operations at will.


## Load videos

```

```