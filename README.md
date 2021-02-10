# Brain Masking Tool

The aim of this project is to execute an automatic masking of fetal brain images using deep learning.

It takes as an input the path of a target directory and executes the masking of all .nii files.
The output will be an extra .nii file for each one of them, ending with the exact same name ending with "_mask.nii".

## Requirements</b>
- Python3
- pip

## Installation</b>

For an easy use of the tool is highly recommended to create a virtual environment and install the dependencies found in the requirements.txt file.

- ### Setting a virtual environment

<br>Access to your terminal and type the following:</br>
<br><code>$ pip install virtualenv virtualenvwrapper</code></br>
<br>Then create a directory for the virtual environments:</br>
<br><code>$ mkdir ~/python-envs </code>  
<br>Now you'll add to your .bashrc file these two lines:</br>
<br><code>$ export WORKON_HOME=~python-envs</code></br>
<br><code>$ source /usr/share/virtualenvwrapper/virtualenvwrapper.sh</code></br>
<i><br>(If this path to virtualenvwrapper.sh doesn't work, try with:)</br></i>
<br><code>$ /usr/share/virtualenvwrapper/virtualenvwrapper.sh</code></br>
<br>Now you're ready to source your .bashrc and create a Python3 environment:</br>
<br><code>$ source .bashrc</code></br>
<br><code>$ mkvirtualenv --python=python3 python_env</code></br>
<br><code>$ workon python_env</code></br>
<br><i>(Note that "python_env" is a suggested name, you can replace it with any desired name for your environment)</i></br>
<br>And finally when you're done working you can deactivate the environment with:</br>
<br><code>$ deactivate</code></br>

- ### Running the tool

Once you're in your own environment access to the desired location and type the following commands:
<br><code>$ git clone https://github.com/chrisorozco1097/masking_tool</code></br>
<br><code>$ cd masking_tool</code></br>
<br>Now install the requirements</br>
<br><code>$ pip install -r requirements.txt</code></br>
<br>And finally execute the tool:</br>
<br><code>$ python brain_mask.py --target_dir path_to_your_directory</code></br>
<br>If you don't specify a directory the tool will target the .nii files found in the local directory.</br>
<br>You will have to activate the environment every time you want to run the tool.</br>
