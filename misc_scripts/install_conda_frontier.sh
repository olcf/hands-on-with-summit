#!/bin/bash

echo ''
echo '~~~Starting Conda Install Script~~~'
echo ''

#Download install script and give it execute permissions
wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod u+x Miniconda3-latest-Linux-x86_64.sh

#Run install script, do NOT update any existing installations (i.e., do NOT use the -u flag)
./Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda-frontier-handson

#Activate conda installation and make sure it doesn't automatically activate it in new sessions
source ~/miniconda-frontier-handson/bin/activate
conda config --set auto_activate_base false

echo ''
echo '~~~Conda Install Script End~~~'
echo ''

if ~/miniconda-frontier-handson/bin/activate ; then
    echo "Conda installation found at:" && which conda
fi
echo ''
