#!/bin/bash

# delete symlinks
find . -maxdepth 1 -type l -delete
ln -s ${1}/* .
