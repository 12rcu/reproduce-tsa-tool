from setuptools import setup

setup(
	name='tsa',
	version='0.9',
	url='http://github.com/kadron/tsa-tool',
	license='None',
	author='Burak Kadron, Nicolas Rosner, Chaofan Shou',
	author_email='kadron@cs.ucsb.edu, nrosner@cs.ucsb.edu, shou@cs.ucsb.edu',
	description='Black box program analysis tool to detect and quantify side channel leakages.',
	packages=['tsa'],
	include_package_data=True,
	zip_safe=False,
	platforms='any',
	python_requires='>=3.6',
	package_data={
	'tsa': ['bin/*.jar']
},
	install_requires=[
		'numpy',
		'scipy',
		'matplotlib',
		'docker',
		'scikit-learn',
		'scapy'
	],
	classifiers=[
		'Development Status :: 2 - Beta',
		'Intended Audience :: Developers',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',
	]
)
