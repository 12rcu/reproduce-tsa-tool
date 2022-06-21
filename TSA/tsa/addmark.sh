#!/bin/bash

# This script mimics the same functionality as the _addmark() method
# of the Sniffer object in sniffer.py, for contexts where interactions
# are scripted in something other than Python. Any changes to the mark
# conventions on the Python side must be reflected here and vice-versa!

# The special port that we use as both src and dst of marker packets.
MARKPORT=55555

# The special nonpublic IP address that we send markers packets to.
MARKADDR=192.168.100.20

# Except if we are nuc1, in that case we send to nuc2 (.30)
myhostnamefirst4chars=$(echo ${HOSTNAME} | head -c 4)

if [[ ${myhostnamefirst4chars} == "nuc1" ]] ; then
	MARKADDR=192.168.100.30
fi


# Defining marker prefixes.
phase="PHASE_"
intr="INTERACTION_"
prefix=""

if ! [[ $# -eq 2 ]] ; then
	echo 'This script needs exactly two parameters. '$#' parameters provided.'
    echo 'USAGE: ./addmark.sh -i|p <MARKERSTR>'
    echo '-i: Interaction Marker, the payload will be "INTERACTION_<MARKERSTR>"'
    echo '-p: Phase Marker, the payload will be "PHASE_<MARKERSTR>"'
    exit 1
fi

#Checking first flag.
if [[ $1 == "-i" ]]; then
	prefix=$intr
elif [[ $1 == "-p" ]]; then
	prefix=$phase
else
	echo 'First parameter '$1' is the wrong flag.'
	echo 'USAGE: ./addmark.sh -i|p <MARKERSTR>'
    echo '-i: Interaction Marker, the payload will be "INTERACTION_<MARKERSTR>"'
    echo '-p: Phase Marker, the payload will be "PHASE_<MARKERSTR>"'
    exit 1
fi

#String Concatenation of prefix and second variable.
payload=$prefix$2

# Send the marker packet!
echo -n "${payload}" | nc -w 1 -u -p ${MARKPORT} ${MARKADDR} ${MARKPORT}


