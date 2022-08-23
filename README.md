# TSA

TSA is a black-box profiling tool for networked programs for side-channel vulnerability detection and quantification. It works by sniffing the network traffic produced by test runs of target programs, extracting information out of packet metadata as commonly used features such as packet sizes, timings, flags and finding the side-channel vulnerability by calculating the information leakage using Shannon entropy or using machine learning methods to train a model that can find the secret using the traces.

This folder contains Python code and Bash scripts related to executing Docker containers, sniffing the data traffic using Scapy, manipulating data using Python list utilities, visualizing the data using MatPlotLib library, extracting features from sniffed packets.

Information leakage quantification module has some options for probability modeling such as Gaussian distribution, histograms, kernel density estimation, Gaussian mixture models, etc. 
For training a model for the data, we support random forests, k-nearest neighbor classifier, naive Bayes classifier or fully connected neural networks. 
This project also has a trace alignment part written in Mathematica or Python where we try to align the traces and generalize the alignment by dividing the similar parts into phases.

## Installation:

You can install TSA as a package using pip for the current user as moving to the directory, TSA and executing:
```
pip install . --user
```

Dependencies:
* Python 3.7+

Python libraries required:
* docker
* scapy
* numpy
* scipy
* matplotlib
* scikit-learn

If you want to try the Mathematica notebook on alignment or the Python code on alignment, the extra dependencies are:
* Mathematica 
* MAFFT (Multiple Sequence Alignment Library)

If you want to use the classifiers for attack synthesis, _pytorch_ library is required.

If you want to use F-BLEAU for leakage quantification, it needs to be installed in addition to all the other dependencies.

## Files & Folders:
* TSA/alignment-mathematica/: Contains Mathematica implementation of trace alignment methods.
* TSA/examples/: Contains examples for analyzing DARPA STAC benchmark applications with input generation.
* TSA/leakiest-1.4.7: Contains the LeakiEst code for quantification. A binary version is also in TSA/tsa/bin for execution.
* TSA/pcap-tss_examples: Contains an example .pcap and .tss file which are two different representations of the network trace that TSA supports. It also contains a .json file which represents the alignment information for each trace.
* TSA/tsa/: The main library folder for TSA.
* TSA/tsa/sniffer.py: Contains Sniffer, Transform and Utils classes.
	Sniffer class contains utilities related to sniffing the network traffic of target program. It contains methods to start/stop sniffing and extra methods which mark start of a test, export the captured info to .pcap(Packet Capture) format for general usage, import .pcap file into the program.
	Transform contains utilities to process the network traffic data to a different traffic data or to a set of values obtained from the data. This is used to remove packets deemed unnecessary (zero sized packets) and to extract the metadata (size and timings) from the traffic data.
	Utils contains small utilities to process and compare packets, it also contains feature extraction, feature reduction and batch process functions.
* TSA/tsa/visualization.py: Contains Visualize class.
	Visualize contains utilities to visualize network traffic data in different ways for presentation and examining the data.
* TSA/tsa/netrunner.py: Contains the AutoProfit class.
	AutoProfit class takes an App class, a seed input set and a list of mutators. It runs in a loop of running the new input on the application, quantifying information leakage over the captured traces and generating a new set of inputs using the old inputs and mutators. It uses some heuristics on mutator selection and when to stop the loop.
* TSA/tsa/quantification.py: Contains the Quantification class.
	Quantification class contains utilities related to the leakage quantification such as entropy computation, classifier training and running.
	Quantification.process_all() is the main function that takes a trace and options related to quantification and alignment, and analyzes the trace based on the selected options.
* TSA/tsa/alignment.py: Contains Alignment class.
	Alignment class contains utilities related to the trace alignment. It transforms a subset of the trace to an ASCII file, uses MAFFT to align the ASCII traces and maps them to stable and variable phases to be applicable to all other traces.
* TSA/tsa/shaper.py: Contains Shaper class.
	Shaper class contains utilities related to traffic shaping and side-channel mitigation. These utilities synthesize a side-channel mitigation strategy based on an objective function and existing traces where the users can set an objective function on information leakage and network overhead. Other utilities use this mitigation strategy to shape the network traces and relay the modified traces instead.
* TSA/tsa/mutation.py: Contains the base class for input and input mutators to instrument the interaction in a structured way.
* TSA/tsa/new_stac.py: Contains the class responsible for instrumenting the Docker containers and the experiments.
* TSA/tsa/addmark.sh: A bash script for adding markers while capturing packets. This is used to mark the beginning of a test by sending a marker packet or to mark any action in the test run in some settings.
* TSA/tsa/tsa_cmd.py: Contains the command line interface of TSA.
	When using the command line option, the users can pass a network trace file location or folder location containing network traces.

## Usage as a Python Package:
If you want to use TSA as a Python package after installation, you may to call `import tsa` or `from tsa import <Classes>`.
Scapy (and Docker in some setups) require Super User (su) mode to work properly for running experiments or sniffing packets over the network.

Using TSA requires either already captured traces or a target program to execute, seed inputs and mutators to feed into the program imitating a user and generate new inputs. Its main purpose is to capture network traces of execution, process the traces to extract features and quantify side-channel  information leakages of the execution.

While executing a target program, you would use Sniffer class to select ports to sniff, start sniffing and insert packets that denote the start of whole experiment or one interaction. to create a list of interactions which represent independent experiments within a single trace. You can then export this recorded trace to a pcap (Packet CAPture) file to check it in Wireshark or tshark or any other tool that processes pcaps.

For demonstration of processing the .pcap file and use of .json files in leakage computation and plots of features given secrets, you can check TSA/pcap-tss_examples/ folder.

For quantifying the information leakage or training a classification model to demonstrate an attack, you may call `Quantification.process_all(ports, filename, interactions, intervals, show_packets, debug_print, calcSpace, calcTime, ...)` where *ports* is ports the application uses and *filename* is the .pcap or .tss file path to load in memory. Specifying ports lets us filter the traffic even when other applications are running in the background.If you have already loaded the trace in memory, use *interactions* instead.
*intervals* variable is related to sequence alignment where we try to find similar parts over multiple traces. This takes the resulting dictionary from the alignment process.
*show_packets* is for printing/not printing parsed packet information, set usually to False.
*debug_print* is for printing/not printing leakage for every subsection of interaction (multiple directions, multiple intervals, etc.).
*calcSpace* and *calcTime* are the variables to check if the user wants to use space and/or time features in leakage computation.
If you want only time or only space, you can set one of them True, the other False. They both are True by default.

In general you would call it like `x = Quantification.process_all([8443], 'test.pcap', calcSpace=True, calcTime=False)`. Detailed documentation is provided for most of the functions including this one.

There are three ways to use phases. If you manually mark phases during trace capture using `Sniffer.add_phase()` function, you may use *use_phases* flag to enable TSA to use the phases you marked. If *alignment* argument is True, TSA will try to align the traces using MAFFT and use the alignment in its analysis. If you pre-compute the alignment using the Mathematica notebook, you can convert the resulting alignment .json file with to a Python object with `Utils.load_intervals(filename)` then pass it as an argument to `Utils.process_all()`. 

## Usage as a Command-Line Interface:
Using tsa_cmd.py, you can pass the filename, ports to examine and pass several options about whether to analyze space/time features and whether to use trace alignment. Calling `python3 tsa_cmd.py --help` returns the instructions on how to run the command line interface.

