
## Basic usage

### Input

Critical input variables are at the top of the Jupyter notebook or python script in the `src` folder.
By default, expected input are images and a label file.
`datadir` contains an `images` folder with all images in flat structure, i.e. as direct children.
`datadir` also contains a metadata file, having at least 2 columns: filename and label.
The column names are specified by 2 variables: 
`filecolumn, labelcolumn`

An example is in the link below. `info.json` file is ignored.

https://github.com/ihsanullah2131/metadl_contrib/tree/master/DataFormat/mini_insect_3

<br>

`resultdir`: location to save result

`dataname`: result are saved in this folder inside the resultdir

`resultprefix`: prefix for output file
 
 `random_seed`: can be set to None


### Output

Output is saved in `resultdir/dataname/resultprefix`
`model` contains the saved model and checkpoint

`metric` contains loss and score for training and validation of all epochs

`fig` contains plots
