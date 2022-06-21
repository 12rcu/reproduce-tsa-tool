import numpy as np
import random
import os
import sys
import subprocess
import re
import json
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.colors

class Alignment(object):
	def __init__(self, filepath='', interactions=None, secrets=None, subsetSize=0, minLength=3, maxDiversity=0.25, plot_on=False):
		self.filepath = filepath
		if (filepath == ''):
			self.filename = ''
		else:
			self.filename = filepath.split('.tss')[0]
		self.interactions = interactions
		self.secrets = secrets
		self.indexIdx = 0
		self.indexTime = 1
		self.indexSize = 2
		self.indexSrc = 3
		self.indexDst = 4
		self.indexSport = 5
		self.indexDport = 6
		self.indexFlags = 7
		self.numChars = 248
		self.mafftOP = 1.5
		self.mafftEP = 0.0
		self.maxIterate = 1000
		self.threads = 2
		self.spaceRepresentation = -1
		self.subsetSize = subsetSize
		self.minLength = minLength
		self.maxDiversity = maxDiversity
		self.plot_on = plot_on

		self.gridVisualization = True

	def setMafftParams(self, mafftOP, mafftEP, maxIterate, threads):
		"""
		Parameters:
			mafftOP - Gap opening penalty at group-to-group alignment
			mafftEP - Offset value, which works like gap extension penalty, for group-to-group alignment
			maxIterate -  Number cycles of iterative refinement are performed
			threads - Number of threads
		Usage:
			Used to redefine mafft params if desired. If not called, default values will be used (recommended)
		"""
		self.mafftOP = mafftOP
		self.mafftEP = mafftEP
		self.maxIterate = maxIterate
		self.threads = threads

	def setSplitParams(self, minLength, maxDiversity):
		self.minLength = minLength
		self.maxDiversity = maxDiversity

	def setTSSFile(self, file):
		"""
		Parameters:
			file - Name of file to set as target for alignment purposes
		"""
		self.filepath = file
		self.filename = self.filepath.split('.tss')[0]

	def setSubset(self, subsetSize):
		"""
		Parameters:
			subsetSize - Integer size of the subset of sequences to be aligned and then projected onto the unaligned sequences. If the subsetSize is bigger than the actual length, all sequences will be aligned
		"""
		self.subsetSize = subsetSize

	def getSubsetSeqs(self, interactions):
		"""
			Parameters:
				interactions - Array of all interactions to be divided into sets in which one will be aligned, and the other will remain unaligned
			Returns:
				msa - Aligned sequences
				seqs - Unaligned sequences to be projected onto
			Usage:
				If the subset is unset or is greater than the number of interactions, all the traces will be aligned
		"""
		if self.subsetSize == 0 or self.subsetSize > len(interactions):
			return interactions, None

		#indices = list(range(len(interactions)))
		#random.seed()
		#random.shuffle(indices)
		#indices = random.sample(indices, self.subsetSize)
		#msa = []
		#seqs = []
		#for x in range(len(interactions)):
		#	if x in indices:
		#		msa.append(interactions[x])
		#	else:
		#		seqs.append(interactions[x])
		msa = interactions[:self.subsetSize]
		seqs = interactions[self.subsetSize:]
		return msa, seqs

	def align(self):
		"""
		Usage:
			Driver of the mafft alignment given a TSS filepath
			The aligner will parse the TSS file, divide the following sequences into a subset of to be aligned sequences and a subset of
			unaligned sequences (if the entire set is aligned, the resulting aligned sequences will be displayed as a graph).
			The unique sizes present in these sequences are mapped to characters so they can be represented in a way that MAFFT can run alignment on them.
			After aligning the sequences, the subset of aligned sequences is analyzed and a regular expression is extracted representing the phased
			alignment. This regular expression is ran on the unaligned sequences to split them into phases.
			The location of these phases for a particular sequence are then stored in a json file.
			This code produces an output hex file from mafft and a json file of split indices.
		"""
		if self.interactions is None or self.secrets is None:
			self.interactions, self.secrets = self.parseTSSFile()
		unique = self.uniqueSizes(self.interactions)
		sub, seqs = self.getSubsetSeqs(self.interactions)
		sizeToChar, charToSize = self.mapToCharacters(unique)
		inputFile = self.createInputHexFile(sub, self.secrets, sizeToChar)
		alignedHexFile = self.mafft(inputFile)
		msa = self.parseAlignedHexFile(alignedHexFile, charToSize)
		print(len(msa))
		if (seqs is None):
			# Simply displays aligned sequences for now
			if self.plot_on:
				colorfunc = self.makeColorFunction(msa)
				self.plot(msa, colorfunc)
		else:
			if self.plot_on:
				colorfunc = self.makeColorFunction(msa)
				self.plot(msa, colorfunc)

			splitIndices = self.msaSplitSeqsIntoPhases(msa, seqs, self.minLength, self.maxDiversity)
			print(len(splitIndices))
			print(splitIndices[:10])
			json_data = []
			for i, split_index in enumerate(splitIndices):
				json_object = {}
				json_object['interaction_num'] = i
				json_object['interval_list'] = split_index
				json_data.append(json_object)
			
			json_filename = '{}.json'.format(self.filepath)
			
			with open(json_filename, 'w') as outfile:
				json.dump(json_data, outfile)
			
			return json_data

	def parseTSSFile(self):
		"""
		Returns:
			interactions - Array of traces that are unaligned. Each trace is an array of varying length of sizes of the packet
			secrets - An array of the secrets for each trace
		Usage:
			Parses the TSS file specified in the construction of this object.
			Reads data from file into a single string
			Splits string into chunks using two new line characters
		"""
		exists = os.path.isfile(self.filepath)
		if not exists:
			print("Input file to parse does not exist!")
			sys.exit(1)
		f = open(self.filepath, "r")
		data = f.read()
		chunks = data.split("\n\n")
		secrets = []
		interactions = []
		for chunk in chunks:
			secret, interaction = self.parseChunk(chunk)
			if (secret != None):
				secrets.append(secret)
				interactions.append(interaction)
		f.close()
		return interactions, secrets

	def parseChunk(self, chunk):
		"""
		Parameters:
			chunk - A string consisting of multiple lines that contains the information on one trace
		Returns:
			secret - The sercret associated with the trace
			packets - An array of sizes for that trace
		Usage:
			Parses a chunk, a sequence of interactions and its associated secret
			Returns the secret and array of interactions which are also arrays
			Each interaction consists of the packets information
		"""
		lines = chunk.split("\n")
		if "SECRET:" in lines[0]:
			tmp = lines[0].split("SECRET:")
			secret = tmp[1]
			columnNames = lines[1].split(",")
			packets = []
			for x in range(2,len(lines)):
				full_packet = lines[x].split(",")
				packets.append(int(full_packet[self.indexSize]))
			return secret, packets
		return None, None

	def uniqueSizes(self, interactions):
		"""
		Parameters:
			interactions - A matrix of aligned traces
		Returns:
			unique - An array of the uniques sizes present in the interactions
		"""
		seen = set()
		unique = []
		for x in interactions:
			for packet in x:
				if packet not in seen:
					unique.append(packet)
					seen.add(packet)
		unique.sort()
		return unique

	def mapToCharacters(self, unique):
		"""
		Parameters:
			unique - An array of the unique sizes of the sequences
		Returns:
			sizeToChar - Dictionary that maps sizes to chars (keys are sizes, characters are values)
			charToSize - Dictionary that maps chars to sizes (keys are chars, sizes are values)
		Usage:
			Maps all the unique sizes to a character value between 0 and self.numChars
			Returns 2 dictionaries, one to map the sizes to characters, with sizes as indices and characters as values
			The other, to map back to size values with the value being an array of sizes. If the array is of length < 1,
			there was information lost as there were more sizes than available characters
		"""
		length = len(unique)
		sizeToChar = dict()
		charToSize = dict()
		if (length <= self.numChars):
			for x in range(length):
				sizeToChar[unique[x]] = x
				charToSize[x] = [unique[x]]
			return sizeToChar, charToSize
		else:
			maxCollision = int(length / self.numChars) + 1
			lenOfSmallerCollisions = length % self.numChars
			count = 0
			currChar = 0
			switch = True
			for x in range(length):
				if (currChar == lenOfSmallerCollisions and switch):
					maxCollision -= 1
					switch = False

				if (count < maxCollision):
					count += 1
				else:
					currChar += 1
					count = 1

				sizeToChar[unique[x]] = currChar
				if currChar in charToSize:
					charToSize[currChar].append(unique[x])
				else:
					charToSize[currChar] = [unique[x]]

			return sizeToChar, charToSize
				

	def createInputHexFile(self, sequences, secrets, sizeToChar):
		"""
		Parameters:
			sequences - Unaligned traces
			secrets - Array of the secrets for each trace
			sizeToChar - Dictionary that maps sizes to chars
		Usage:
			Maps the unique packet size to a hex value and creates a input.hex file to be converted to ASCII characters
			Returns the name of the newly created hex file
		"""
		inputFilename = self.filename + ".hex"
		f = open(inputFilename, "w+")
		index = 0
		for sequence in sequences:
			f.write("> " + secrets[index] + "\n")
			for packet in sequence:
				size = packet
				f.write(self.toHex(sizeToChar[size]) + " ")
			f.write("\n")
			index += 1
		f.close()
		return inputFilename

	def toHex(self, number):
		"""
		Parameters:
			number - A number to represent a character for mafft
		Returns:
			hex <String> - A hex number to be used in the hex file for input to mafft
		Usage:
			Helper function to convert numbers to hexidecimal notation while ignoring the ASCII excluded by MAFFT
		"""
		hexNum = ''
		if (number > 54):
			hexNum = hex(number + 8)
		elif (number > 40):
			hexNum = hex(number + 5)
		elif (number > 28):
			hexNum = hex(number + 4)
		elif (number > 10):
			hexNum = hex(number + 3)
		elif (number > 8):
			hexNum = hex(number + 2)
		else:
			hexNum = hex(number + 1)
		tokens = hexNum.split('x')
		if (len(tokens[1]) == 1):
			tokens[1] = '0'+tokens[1]
		return tokens[1]

	def toDecimal(self, number):
		"""
		Parameters:
			number - A number to represent a character for mafft
		Returns:
			number - A number that has been manipulated to not include the numbers that are ASCII characters that cannot be aligned by MAFFT
		Usage:
			Helper function to convert hexidecimals back to decimals used in dictionary's character range
		"""
		if (number > 62):
			return number - 8
		elif (number > 45):
			return number - 5
		elif (number > 32):
			return number - 4
		elif (number > 13):
			return number - 3
		elif (number > 10):
			return number - 2
		else:
			return number - 1
	

	def mafft(self, inputFile):
		"""
		Returns:
			alignedHexFile - The name of the resulting aligned hex file
		Usage:
			Calls the mafft for alignment with the default or set mafft Parameters.
			Converts the input hex file to ascii format to be aligned.
			The output is an aligned output ascii file.
			The output aligned ascii file is reconverted to a hex file by a mafft command.
			These files can be used as intermediate points of alignment so redundant steps do not have to be taken.
			Removes input and ascii files and preserves output hex file
		"""

		#Removing the folder prefix, appending the file suffix and prefixes.
		splitFilename = self.filename.split('/')
		splitHexFilename = splitFilename[:]
		splitAsciiFilename = splitFilename[:]
		
		splitHexFilename[-1] = 'output-' + splitHexFilename[-1] + '.hex'
		alignedHexFile = '/'.join(splitHexFilename)
		
		splitAsciiFilename[-1] = splitAsciiFilename[-1] + '.ASCII'
		ASCIIFile = '/'.join(splitAsciiFilename)

		splitAsciiFilename[-1] = 'output-' + splitAsciiFilename[-1]
		outputASCIIFile = '/'.join(splitAsciiFilename)
		
		#alignedHexFile = 'output-' + self.filename + '.hex'
		# exists = os.path.isfile(alignedHexFile)
		# if exists:
		# 	print("Aligned hex file already exists. Aborting alignment")
		# 	return alignedHexFile

		ASCIIconversion = '/usr/local/libexec/mafft/hex2maffttext {} > {}'.format(inputFile, ASCIIFile)
		try:
			output = subprocess.check_output(ASCIIconversion, shell=True)
		except subprocess.CalledProcessError as exc:
			print("Status : FAIL in ascii conversion", exc.returncode, exc.output)
		
		# --genafpair have been replaced in v7.243, use --oldgenafpair in newer versions
		#runMafftCmd = 'mafft --text --op ' + str(self.mafftOP) + ' --ep ' + str(self.mafftEP) + ' --oldgenafpair --maxiterate ' + str(self.maxIterate) + ' --thread ' + str(self.threads) + ' ' + ASCIIFile + ' > ' + outputASCIIFile
		runMafftCmd = 'mafft --text --op {} --ep {} --oldgenafpair --maxiterate {} --thread {} {} > {}'.format(self.mafftOP, self.mafftEP, self.maxIterate, self.threads, ASCIIFile, outputASCIIFile)
		try:
			output = subprocess.check_output(runMafftCmd, shell=True)
		except subprocess.CalledProcessError as exc:
			print("Status : FAIL in mafft cmd", exc.returncode, exc.output)

		hexConversion = '/usr/local/libexec/mafft/maffttext2hex {} > {}'.format(outputASCIIFile, alignedHexFile)
		try:
			output = subprocess.check_output(hexConversion, shell=True)
		except subprocess.CalledProcessError as exc:
			print("Status : FAIL in hex conversion", exc.returncode, exc.output)

		os.remove(inputFile)
		os.remove(ASCIIFile)
		os.remove(outputASCIIFile) 

		return alignedHexFile

	def parseAlignedHexFile(self, filename, charToSize):
		"""
		Parameters:
			filename - Aligned hex file to be parsed
			charToSize - The decionarty of sizes that contain has characters as keys and sizes as values. If there are more sizes than possible charcters, there will be a loss of information.
		Returns:
			sequences - The aligned sequences as a matrix
		"""
		sequences = []
		f = open(filename, 'r')

		#This is for the case where each aligned trace can correspond to two or more lines.
		lines = f.readlines()
		concatLines = []
		tempLine = ''

		for line in lines:
			if (line[0] == '>'):
				if len(tempLine) != 0:
					concatLines.append(tempLine)
					tempLine = ''
			else:
				tempLine = tempLine + line[:-1]


		for line in concatLines:
			if (line[0] != '>'):
				sequence = []
				tokens = line.split(' ')
				for token in tokens:
					if (token == '--'):
						sequence.append(self.spaceRepresentation)
					elif (len(token) > 1):
						num = self.toDecimal(int(token, 16))
						sizes = charToSize[num]
						sequence.append(sizes[0])
				sequences.append(sequence)
		return sequences

	def getColors(self, filename):
		"""
		Parameters:
			filename - File of unique colors
		Usage:
			Uses the file colors.txt to set up 1000 unique colors to deterministically assign to a packet size
		"""
		exists = os.path.isfile(filename)
		if not exists:
			print("Colors text file does not exist. Aborting plotting.")
			sys.exit(1)

		f = open(filename)
		data = f.read()
		lines = data.split("\n")
		del lines[0]
		colors = []
		for line in lines:
			token = line.split("\t")
			colors.append(token[1])
		return colors

	def makeColorFunction(self, sequences):
		"""
		Parameters:
			seqeuences - The sequences of sizes for which a color function is being generated for. Can be unaligned or aligned.
		Returns:
			colors - The color function
		Usage:
			Creates a dictionary for each unique color, size is the index and the entry is the corresponding RGB values
			Packets of size 0 are assigned a random color and the defined spaceRepresentation entries are mapped to white
		"""
		colorSet = self.getColors('colors.txt')
		seen = set()
		colors = []
		colors.append([self.spaceRepresentation, 255, 255, 255])
		colorIterator = 0
		for seq in sequences:
			for packet in seq:
				if packet not in seen:
					color = colorSet[colorIterator].lstrip('#')
					red = int(color[0:2], 16)
					green = int(color[2:4], 16)
					blue = int(color[4:6], 16)
					colors.append([packet, red, green, blue])
					seen.add(packet)
					colorIterator += 1
					if (colorIterator >= len(colorSet)):
						colorIterator = 0
		return colors

	def plot(self, sequences, cf):
		"""
		Parameters:
			sequences - Aligned sequences as a m x n matrix of sizes where m is the number of traces and n is the length the sequences
			cf - Color function used to map a size to a unique color
		Usage:
			Plots the graph of the aligned sequences for visualization
		"""
		img = np.array(sequences)
		ca = np.array(cf)

		u, ind = np.unique(img, return_inverse=True)
		b = ind.reshape((img.shape))

		colors = ca[ca[:,0].argsort()][:,1:]/255.
		cmap = matplotlib.colors.ListedColormap(colors)
		norm = matplotlib.colors.BoundaryNorm(np.arange(len(ca)+1)-0.5, len(ca))
		if self.gridVisualization:
			plt.matshow(b, cmap=cmap, norm=norm, aspect='auto')
		else:
			plt.imshow(b, cmap=cmap, norm=norm, aspect='auto')
		plt.ylabel("Sequence Number")
		plt.xlabel("Packets over Time")
		plt.show()

	def msaSplitSeqsIntoPhases(self, msa, seqs, minLength, maxDiversity):
		"""
		Parameters:
			msa - The subset of aligned sequences
			seqs - The subset of unaligned sequences to be split into phases
			minLength - The minimum length of columns to constitute a stable phase
			maxDiversity - The maximum diversity of a column that allows it to be considered part of a stable phase
		Returns:
			splitIndices - A list of indices that constitute a particular phase that is unique to each the sequence that shares the same index in the list.
		Usage:
			These indices indicate where stable phases are and can be used to interpret gaps between them without the complexity of having to run
			the entire set of sequences through mafft which is often infesible.
		"""

		pattern, numStablePhases = self.msaToPattern(msa, minLength, maxDiversity)
		#print msa
		print(pattern)
		numGroups = 2 * numStablePhases + 1
		strings = self.seqsToString(msa + seqs)

		splitIndices = []
		for x in range(len(strings)):
			p = re.search(pattern, strings[x])
			seq = []
			if (p is None or len(p.groups()) != numGroups):
				# phase did not match => discard
				# discarded phases are represented by an empty list of indices
				print("Not a match")
			else:
				groups = p.groups()
				index = 0
				isStable = True
				for group in groups:
					if (group != ''):
						tokens = group.split(',')
						if isStable:
							phase = [index, index+len(tokens)-2]
							#for x in range(1, len(tokens)):
							#	phase.append(index)
							#	index += 1
							index += len(tokens) - 1 
							seq.append(phase)
							isStable = not isStable
						else:
							phase = [index, index+len(tokens)-2]
							index += len(tokens) - 1
							seq.append(phase)
							isStable = not isStable
			splitIndices.append(seq)
		return splitIndices

	def msaToPattern(self, msa, minLength, maxDiversity):
		"""
		Parameters:
			msa - The aligned subset of sequences
			minLength - The minimum length of columns to constitute a stable phase
			maxDiversity - The maximum diversity of a column that allows it to be considered part of a stable phase
		Returns:
			pattern - A regular expression string that represents the pattern of stable parts in the form V1 S1 V2 S2 ... Sk V(k+1)
			numStablePhases - The number of stable phases in the pattern
		"""
		contiguousIndices = self.msaStableParts(msa, minLength, maxDiversity)
		variableString = '(.*)'
		np_msa = np.array(msa)
		stablePhaseStrings = []
		for phase in contiguousIndices:
			string = '('
			for index in phase:
				column = np_msa[:,index]
				col = self.deleteDuplicates(column)
				if (len(col) > 1):
					string += '(?:'
				for unique in col:
					string += ',' + str(unique)
					if (unique != col[-1]):
						string += '|'
				if (len(col) > 1):
					string += ')'
			string += ')'
			stablePhaseStrings.append(string)

		pattern = '^'
		for string in stablePhaseStrings:
			pattern += variableString
			pattern += string
		pattern += variableString
		return pattern, len(contiguousIndices)

	def msaStableParts(self, msa, msaStableSpread, msaDiversity):
		"""
		Parameters:
			msa - A subset of sequences that have been aligned
			msaStableSpread - The minimum length of sequential columns to constitute a stable phase
			msaDiversity - The maximum diversity that qualifies if a column can be considered part of a stable phase
		Returns:
			contigiousIndices - A list of contigious indices that represent a stable phase for a particular sequence.
		Usage:
			Using the splitting metrics defined before the alignment, this function tests whether a column is fully dense 
			and is lower than the diversity calculated by the consensus function. If there are a sufficient number of these columns
			adjacent to each other, they constitute a stable phase.
		"""
		densities, spreads = self.msaConsensus(msa)
		indices = []
		for i in range(len(densities)):
			if (densities[i] == 1.0 and spreads[i] <= msaDiversity):
				indices.append(i)
		contiguousIndices = []
		tmp = []
		for index in indices:
			if not tmp:
				tmp.append(index)
			else:
				if (tmp[-1] == index - 1):
					tmp.append(index)
				else:
					if (len(tmp) >= msaStableSpread):
						contiguousIndices.append(tmp)
					tmp = []
		return contiguousIndices

	def msaConsensus(self, msa):
		"""
		Parameters:
			msa - Aligned subset of sequences
		Returns:
			densities - Density of each column
			spreads - Scaled standard deviation of each column
		Usage:
			Used to calculate standard deviation and density of each column to find splitting points.
		"""
		np_msa = np.array(msa)
		rows = np_msa.transpose()
		spreads = [None] * len(rows)
		densities = [None] * len(rows)
		index = 0
		for row in rows:
			rowWithoutGaps = self.deleteGaps(row)
			densities[index] = float(len(rowWithoutGaps)) / float(len(row))
			if (len(rowWithoutGaps) > 2):
				spreads[index] = np.std(row, axis=0)
			else:
				spreads[index] = 0
			index += 1
		spreads = np.array(spreads)
		return densities, np.interp(spreads, (spreads.min(), spreads.max()), (0, +1))

	def seqsToString(self, seqs):
		"""
		Parameters:
			seqs - A list of the traces that are unaligned
		Returns:
			strings - The sequence converted into a string
		Usage:
			Converts a list of sequences into a list of strings of sizes seperated by commas.
			This enables us to run a regular expression pattern on the string to detect stable and variable phases.
		"""
		strings = []
		for seq in seqs:
			line = ''
			for val in seq:
				line += (',' + str(val))
			strings.append(line)
		return strings

	def deleteDuplicates(self, mylist):
		"""
		Parameters:
			mylist - A list of numbers with repeating values to be removed
		Returns:
			mylist - A list of the unique numbers present in the paramater
		Usage:
			A helper function to remove duplicate values in a list to yield a list of unique values
		"""
		seen = set()
		unique = []
		for x in mylist:
			if x not in seen:
				unique.append(x)
				seen.add(x)
		return unique

	def deleteGaps(self, row):
		return [x for x in row if x != self.spaceRepresentation]


if __name__ == '__main__':
	alignment = Alignment("grpc_stove-100x2secs-1delay-handshake-pruned.tss", 50)
	alignment.align()
