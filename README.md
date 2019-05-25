# fastai_2019_pt2_dev_course

Refactored Fast.ai Pt2 2019 with exported .py files all in a centralised location, exported all notebooks as scripts so that source code goto definition can be used in an IDE.
 
(I found it very difficult to follow the .ipynb's with chained exports and without being able to step to definition when required).

<pre>
Workflow:
    Work through notebooks
    If make any changes: 
        Export as .py to scripts
        Run clean_nb_output.py to remove irrelevant ipynb code
    Step to relevant source in .py file using keybaord shortcuts
    Optionally debug .py file with breakpoint if difficult to follow
    NB debug only works pre .to_device, once on GPU not on CPU to be able to set breakpoints
</pre>

<pre>
dl2/
    .ipynb's with minor changes
    /utils
        python files containing functions and classes all in one place
    /scripts
        exported .ipynb's as .py to use in ide for debugging/stepping to source in ide 
    /exp
        not used
</pre>