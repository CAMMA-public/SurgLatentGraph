#!/bin/bash

find . -maxdepth 1 -name '*.sh' -type l -delete
ln -s ${1}/*.sh .
