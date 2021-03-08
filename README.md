# data_to_mpc


Project requirements listed in requirements.txt

to create a new requirements file the following command can be run

pip freeze > requirements.txt

Note that this will add everything on your python path as == requirements

To install these requirements run

pip install -r requirements.txt

# Notes
For the CPU-only install, it is sufficient to run `pip install jax jaxlib`, or `pip install -r requirements.txt`. This is enough to run the code.

For the Jax+CUDA install, the CUDA drivers and cuDNN need to be installed. Jaxlib for a particular CUDA version can be installed afterwards.

Python version should be less than 3.9.0 -- versions at 3.9.0 or above may work.

# Installing CUDA, cuDNN and jax/jaxlib on Linux

[Nvidia CUDA toolkit documentation and install links](https://developer.nvidia.com/cuda-toolkit-archive)

[Nvidia cuDNN download (requires Nvidia account)](https://developer.nvidia.com/CUDnn)

[Nvidia cuDNN install guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)

[Jax install guide](https://github.com/google/jax#installation)

The instructions beyond here are specific to CUDA 11.1.0, cuDNN v8, and Python 3.8.6 on Ubuntu 20.04 LTS.

## Install prerequisites 
Remove the Nvidia proprietary driver if it is already installed. Install Python - this can be done by running `sudo apt install python3.8`, though it is usually already included. Check that `gcc` is installed with `gcc --version`. If not, install all of `build-essentials` via `sudo apt-get install build-essentials`. It may also be necessary to run `sudo apt-get install linux-headers-$(uname -r)` to ensure the correct kernel headers are installed.

## Install CUDA toolkit via runfile
To install CUDA using the runfile method first download the runfile after selecting a version from [this page](https://developer.nvidia.com/cuda-toolkit-archive) (the runfile is the same for all Linux distros). This command downloads 11.1.0:
```shell
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
```
Do not run this file. Next, disable Nouveau (the open source Nvidia driver). To check if Nouveau is loaded or not, run `lsmod | grep nouveau` - no output means it is not loaded. To disable Nouveau on Ubuntu or Debian create this file: 
```shell
/etc/modprobe.d/blacklist-nouveau.conf
```
In the new file, write:
```shell
blacklist nouveau
options nouveau modeset=0
```
Then run:
```shell
sudo update-initramfs -u
```
Nothing will happen until reboot. After reboot there should be no output from `lsmod | grep nouveau`. 

Once Nouveau is disabled, reboot into console mode/text. For distros using `systemd` including Ubuntu >= 15.04 this can be achieved by running:
```shell
sudo systemctl set-default multi-user.target
```
After this is run, the command `systemctl get-default` should return `multi-user.target`. This will prevent the GUI from loading on next boot. Reboot the computer.

After reboot, log in and navigate to the directory where the runfile was downloaded. Run the file, for the 11.1.0 runfile downloaded previously:
```shell
sudo sh cuda_11.1.0_455.23.05_linux.run
```
After some time an EULA will appear. It may be poorly scaled, type `accept` and hit enter to accept the terms. The next menu should be some installation settings. The default settings are fine. Keep the samples selected as they are useful for validation. Start the installation.

After install these lines need to be added to `~/.bashrc` or equivalent. Run:
```shell
echo 'export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
```
Afterwards run `source ~/.bashrc`. The command `nvcc --version` should work and print the installed version. The install is complete. Restore the GUI by running:
```shell
sudo systemctl set-default graphical.target
```
And `sudo reboot`to reboot the computer.

There is some additional verification to do [here](https://docs.nvidia.com/cuda/archive/11.1.0/cuda-installation-guide-linux/index.html#verify-installation).

## Install cuDNN library via .tgz
Go to the [download site](https://developer.nvidia.com/CUDnn), log in, and choose cuDNN for 11.1, 11.1 and 11.2 and click on "cuDNN Library for Linux (x86_64)" to get a tarball. After that is downloaded unpack it:
```shell
tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
```
Replace the x's with the appropriate version numbers. In the unpacked directory, run: 
```shell
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## Install Jax
```shell
pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.59+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
Change `cuda111` to match the current CUDA version. At the [URL in the command](https://storage.googleapis.com/jax-releases/jax_releases.html) there is a list of available versions.