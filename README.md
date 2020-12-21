# voltammetry
This repository contains code and demos related to the work "Low Amplitude Burst Sensing (LABS) of Neurotransmitters" (in preparation).

## Setting up environment
The analyses outlined in this repository using Python 3.7. Necessary packages to install are in <code>requirements.txt</code>

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

### Elastic net:
<ul>
<li>Elastic Net: http://www.jstatsoft.org/v33/i01/</li>
<li>GLMnet for python: https://web.stanford.edu/~hastie/glmnet_python/</li>
</ul>

### Previous human voltammetry work:
<ul>
<li>Kishida et al. (2016): https://doi.org/10.1073/pnas.1513619112</li>
<li>Moran et al. (2019): https://doi.org/10.1038/npp.2017.304</li>
<li>Bang et al. (2020): https://doi.org/10.1016/j.neuron.2020.09.015</li>
</ul>
