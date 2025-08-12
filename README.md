# About

This repository contains programs to train a CNN classifier on FashionMNIST, 
generate a visualization of the network. It also contains a file which generates
adversarials, visualizes them using XAI method and extracts information about
the adversarials influence on the visualization

Within the classifier folder there is the FashionMNISTModel.py file, which trains a
CNN on teh FashionMNIST dataset and saves its state_dict to models/FashionMNIST .

In classifier/model_visualization there is the ModelVisualization.py file, which
generates a ONNX file for the trained neural Network with the state_dict from
models/FashionMNIST. This can be used in Netron in order to generate a visualization
of the Network architecture, here saved in a file named classifier_visualization.png.

In models/FashionMNIST there is a XAIAdv.py file, which generates the visualizations of adversarials
amongst other data about the visualizations difference with the visualization
of the original image. All these results get saved in the models/FashionMNIST/results folder.

## Installation

Simply clone the GitHub repository and install the packages from requirements.txt.

Be advised that you will need git lfs installed in order to use the state_dict included.

## Usage

In order to generate the data run XAIAdv.py after deleting the /results folder
(if XAIAdv.py is not changed this is not needed, however the folder does not get manually deleted in the program
and simply overrides existing files).
Using a GPU is heavily advised, as finding the minimal disturbance 
is often very computationally expensive.

In order to generate the .onnx file run the ModelVisualization.py file.

In order to generate the state_dict run the FashionMNISTMode.py file by changing into its 
folder and executing 
```
python -c "import train; train.run()"
```

If the state_dict fails to load, simply retrain the model, it should take about
20-30 minutes on a normal GPU.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)