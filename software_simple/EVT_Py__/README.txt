EVT_Py
------------
A basic wrapper around eSDK for configuring a camera and acquiring images.

Setup
------------
EVT_Py has been tested on Python 3.10.

Install Python3, pip, and Python dependencies:
Ubuntu:
    sudo apt install python3.10
    sudo apt install python3-pip
    pip install pillow cffi

    Install NUMBA for GPU-direct and using CUDA in Python:
    pip install numba

Windows:
    Download and run python installer: 
        https://www.python.org/downloads/release/python-31010/
        Make sure python.exe is accessible from your PATH environment variable.
    Download and install pip:
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python.exe get-pip.py
        Make sure pip.exe is accessible from your PATH environment variable.
    pip install pillow cffi:
        pip.exe install pillow cffi

Examples
------------
To run an example, open a terminal in the root EVT_Py directory and run one of the following commands to execute a specific example.

EVT_Py_stream:
    A simple application showing how to connect to a camera, configure it, and stream images.
    Run `sudo -E python3 -m examples.EVT_Py_stream`

EVT_Py_convert:
    A simple application showing how to access image data by converting an image and saving it to disk.
    Assumes that the camera is set to a bayer or mono pixel format.
    Run `sudo -E python3 -m examples.EVT_Py_convert`

EVT_Py_multi_cam:
    A application showing how to manage multiple cameras at the same time.
    Run `sudo -E python3 -m examples.EVT_Py_multi_cam`

EVT_Py_numba:
    A application showing how to manage enable GPUDirect and to access GPU image data using numba.
    Assumes that the camera is set to an 8-bit pixel format. 
    Run `sudo -E python3 -m examples.EVT_Py_numba`

Notes
------------
- Only a subset of eSDK functionality has been exposed through the EVT_Py API.
    Additional SDK functionality can be provided by adding the required API calls to `_eSDK_cffi_wrapper.py`, and creating
    the matching EVT_Py function calls.
