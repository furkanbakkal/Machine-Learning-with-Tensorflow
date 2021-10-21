# Machine-Learning-with-Tensorflow
Detecting Custom Shapes with Tensorflow using Machine Learning 

![image1](https://user-images.githubusercontent.com/81293327/138339698-3013da75-72a3-4538-9d22-941fe6b92d12.png)

![image2](https://user-images.githubusercontent.com/81293327/138339758-c45498f9-85e9-4548-bfec-c59fab90d520.png)

## Setup Commands

### Installing Tensorflow Object Detection API
- Open a new Anaconda prompt

`
`conda create -n tensorflow pip python=3.8`

`conda activate tensorflow`

`pip install tensorflow`


- Create a folder directly in C: and name it "TensorFlow".


`cd C:\TensorFlow`

`git clone https://github.com/tensorflow/models.git`

`conda install -c anaconda protobuf`

`cd models\research`

`protoc object_detection\protos\*.proto --python_out=.`


- Open a new Anaconda prompt


`conda activate tensorflow`

`pip install cython`

`pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

`cd C:\TensorFlow\models\research`

`copy object_detection\packages\tf2\setup.py .`

`python -m pip install .`

`python object_detection\builders\model_builder_tf2_test.py`

### Installing Other Modules
`conda install -c r r-turtlegraphics`

If you downloaded all modules correctly, just run the main.py file .

https://user-images.githubusercontent.com/81293327/138339970-2828c130-ff40-4574-b694-6624bed29737.mp4


