# voltammetry
This repository contains code and demos related to the work "Low Amplitude Burst Sensing (LABS) of Neurotransmitters" (in preparation).

## Setting up environment
The analyses outlined in this repository using Python 3.7. Necessary packages are in the file 

Installing virtual environments using pip: https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/

Installing virtual environments using conda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
 
## Voltammetry package


### Decoding GLMnet error: 
It is possible when using the glment_python package, the following error will appear:
		
    OSError: dlopen(/Users/path/to/package/glmnet_python/GLMnet.so, 6): no suitable image found.  Did find:
	/Users/path/to/package/glmnet_python/GLMnet.so: unknown file type, first eight bytes: 0x7F 0x45 0x4C 0x46 0x02 0x01 0x01 0x00
	    
You will need to find the GLMnet.so file and replace it with the one found in the misc/ directory.
The error and solution are outlined here: https://github.com/bbalasub1/glmnet_python/issues/13


## Further reading:

### GLMnet:

### Older human voltammetry work:
