import sys, re

bitexpr = re.compile('([-]?[0-9]+[.][0-9]+)[/]([0-9]+[.][0-9]+) bits')

def replfun(match):
	leaked = float(match.group(1))
	total = float(match.group(2))
	bitstr = match.group(0)
	percent = 100.0 * leaked / total
	percentstr = '%d%%' % round(percent)
	return '%s  %s' % (percentstr, bitstr)

for line in sys.stdin:
	sys.stdout.write(bitexpr.sub(replfun, line))
