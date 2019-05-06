#!/bin/bash

export matlabroot="/misc/local/matlab-2018b"
export PATH="$PATH:$matlabroot/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:\
$matlabroot/bin/glnxa64:\
$matlabroot/runtime/glnxa64:\
$matlabroot/sys/os/glnxa64:\
$matlabroot/sys/java/jre/glnxa64/jre/lib/amd64/native_threads:\
$matlabroot/sys/java/jre/glnxa64/jre/lib/amd64/server:\
$matlabroot/sys/java/jre/glnxa64/jre/lib/amd64" 
export XAPPLRESDIR="$matlabroot/X11/app-defaults"
export MCR_INHIBIT_CTF_LOCK=1

mcc -m -R -nojvm -v Postprocessing.m
